"""Declared dual-branch development comparison with an external frozen CORE."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_borrowing_experiment import read_json, read_npz, scalar_metrics, paired_intervals
from .biology_kernel_evaluation import write_json
from .conditional_joint_error_experiment import observable_forward
from .dual_branch_biology import fit_gelu_branch, fit_biology_branch, DualBranchBiologyAdapter
from .dual_branch_features import biology_features, apply_increment, select_strength, calibration_frame_covariance
from .empirical_radial import fit_radial
from .frozen_acquisition_policy import select_frozen_cohort_plan
from .gram_geometry import profiles_to_gram, gram_to_coordinates
from .joint_contrast_scale import contrast_projector, projected_energy
from .lincs_biology_experiment import load_data
from .radial_mixture_evaluation import evaluate_radial_mixtures

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'runs/conditional_residual_information_20260916_v1'
RADIAL = PROJECT/'runs/lincs_empirical_radial_20260916_v1'
SEED = 20260917
SAMPLES = 100000
FULL_ARMS = ('GELU', 'DUAL_GENERIC', 'DUAL_STRUCTURED')
ARMS = ('CORE', *FULL_ARMS, *(a+'_CAL' for a in FULL_ARMS))
PLAN = 'protocols/historical/DUAL_BRANCH_BIOLOGY_PLAN_20260917.md'


def collect_completed_cells(root, ids, cells):
    """Aggregate only fields defined in every cell of an arm.

    CORE copies can carry extra legacy diagnostics which the current scorer
    does not produce. A union of fields would leave unfilled array segments.
    """
    root=Path(root);ids=np.asarray(ids);lookup={v:i for i,v in enumerate(ids)}
    stores={};audit={}
    required={'actual','fold','mean_u','actual_u','predicted','p_null','nll','energy','crps',
              'selected','increment','resource_support','joint_coverage_by_level'}
    for arm in ARMS:
        entries=[];union=set();common=None;seen=np.zeros(len(ids),int)
        for cell in cells:
            value=read_npz(root/f'cell_{cell["fold"]}_{cell["half"]}'/(arm+'.npz'))
            np.testing.assert_array_equal(value['ids'],cell['query_ids'])
            q=np.array([lookup[v] for v in value['ids']])
            keys={k for k,v in value.items() if k!='ids' and v.shape[:1]==(len(q),)}
            if not required<=keys:raise ValueError('Missing core evaluation fields: '+str(required-keys))
            common=keys if common is None else common&keys;union|=keys
            entries.append((q,value));np.add.at(seen,q,1)
        np.testing.assert_array_equal(seen,np.ones(len(ids),int))
        out={}
        for key in sorted(common):
            example=entries[0][1][key]
            buffer=np.empty((len(ids),*example.shape[1:]),dtype=example.dtype)
            for q,value in entries:
                if value[key].shape[1:]!=example.shape[1:]:raise ValueError('Cell field shapes differ')
                buffer[q]=value[key]
            if np.issubdtype(buffer.dtype,np.number) and not np.isfinite(buffer).all():
                raise ValueError('Nonfinite completed evaluation field: '+arm+'/'+key)
            out[key]=buffer
        stores[arm]=out
        audit[arm]=dict(common_fields=sorted(common),omitted_legacy_fields=sorted(union-common),
                        all_objects_assigned_once=True)
    write_json(root/'aggregation_fields.json',audit)
    return stores


def train_fold(data, metadata, fit, test, prior, stats, source, folder, fold):
    """No outer query/calibration outcome is supplied to branch fitting."""
    cache = folder/'predictions.npz'
    if (folder/'fit_complete.json').exists():
        out=read_npz(cache)
        np.testing.assert_array_equal(out['fit_ids'],data['ids'][fit])
        np.testing.assert_array_equal(out['test_ids'],data['ids'][test])
        return out
    nested = read_npz(folder/'nested_distribution/distribution.npz')
    if read_json(folder/'nested_distribution/summary.json')['state'] != 'COMPLETE':
        raise RuntimeError('Nested complete CORE distribution has not finished')
    np.testing.assert_array_equal(nested['ids'],data['ids'][fit])
    transformer=joblib.load(source/f'fold_{fold}/conditioners.joblib')['transformer']
    descriptor=transformer.transform(data,metadata)
    train_bio=biology_features(data,metadata,fit,fit,nested['raw_mean'],nested['raw_covariance'],
        nested['raw_residual'],allowed=nested['biological_reference_mask'])
    dec=contrast_projector(nested['raw_mean'],np.ones(9),nested['raw_covariance'])
    energies=projected_energy(nested['raw_residual'],dec)
    if not np.isfinite(energies).all() or np.any(energies<=0):
        raise ValueError('Invalid full CORE projection energies')
    scale,center=np.asarray(stats['u_scale']),np.asarray(stats['u_center'])
    raw_mean=prior['mean_u'][test]*scale+center
    raw_cov=prior['covariance_u'][test]*scale[None,:,None]*scale[None,None,:]
    test_bio=biology_features(data,metadata,test,fit,raw_mean,raw_cov,nested['raw_residual'])
    n=len(data['ids'])
    output=dict(ids=data['ids'],fit_ids=data['ids'][fit],test_ids=data['ids'][test],
        support=np.zeros(n,bool),support_by_relation=np.zeros((n,2),bool),
        fit_support=train_bio['support'],fit_features=train_bio['values'],
        fit_energies=energies,test_features=test_bio['values'])
    output['support'][test]=test_bio['support']
    output['support_by_relation'][test]=test_bio['support_by_relation']
    models={}
    for arm in FULL_ARMS:
        checkpoint=folder/(arm+'.pt')
        if checkpoint.exists():
            model=DualBranchBiologyAdapter.load(checkpoint)
        else:
            def callback(row):
                print(f'fold={fold} arm={arm} epoch={row["epoch"]} '
                      f'projection_nll={row["training_projection_nll"]:.6f}',flush=True)
            if arm=='GELU':
                model=fit_gelu_branch(descriptor.values[fit],energies,np.ones_like(energies),data['ids'][fit],
                    support=train_bio['support'],seed=SEED+fold,callback=callback)
            else:
                model=fit_biology_branch(models['GELU'],descriptor.values[fit],train_bio['values'],
                    train_bio['names'],train_bio['support'],energies,np.ones_like(energies),data['ids'][fit],
                    mode='generic' if arm=='DUAL_GENERIC' else 'structured',seed=SEED+100+fold,callback=callback)
            model.save(checkpoint)
        models[arm]=model
        write_json(folder/(arm+'_training.json'),model.report)
        bio=None if arm=='GELU' else test_bio['values']
        components=model.predict_components(descriptor.values[test],bio,test_bio['support'])
        output[arm]=np.zeros((n,2));output[arm][test]=components['total']
        output[arm+'_left']=np.zeros((n,2));output[arm+'_left'][test]=components['left']
        output[arm+'_right']=np.zeros((n,2));output[arm+'_right'][test]=components['right']
        np.testing.assert_array_equal(components['total'][~test_bio['support']],
                                      np.zeros((int((~test_bio['support']).sum()),2)))
        np.testing.assert_array_equal(model.predict_increment(descriptor.values[test],bio,
                                      test_bio['support'],enabled=False),np.zeros((len(test),2)))
        if arm!='GELU':
            np.testing.assert_array_equal(components['left'],output['GELU'][test])
            np.testing.assert_array_equal(model.predict_increment(descriptor.values[test],bio,
                test_bio['support'],right_enabled=False),output['GELU'][test])
    a=models['DUAL_GENERIC'].report['right_trainable_parameters']
    b=models['DUAL_STRUCTURED'].report['right_trainable_parameters']
    if a!=b: raise ValueError('Same-input basis parameter budgets differ')
    np.savez_compressed(cache,**output)
    write_json(folder/'fit_complete.json',dict(fold=fold,fit_n=len(fit),test_n=len(test),
        fit_supported=int(train_bio['support'].sum()),test_supported=int(test_bio['support'].sum()),
        descriptor_names=descriptor.names,biological_names=train_bio['names'],
        right_parameters=a,epochs=60,exact_off_checks=True,identical_left_checks=True,
        reference_source='MODEL_FIT honest inner raw errors, same-inner-heldout DIST_FIT for training',
        training_covariance='complete nested AMP_EMP_LOCAL covariance, not inherited RIDGE',
        training_queries_with_target_support=int(train_bio['support_by_relation'][:,0].sum()),
        training_queries_with_moa_support=int(train_bio['support_by_relation'][:,1].sum())))
    return output


def summarize(stores,data,metadata,cells,root,elapsed):
    ids,groups=data['ids'],data['groups']
    layouts=np.asarray([u['layout_block'] for u in metadata['units']])
    actual=stores['CORE']['actual'];all_rows=np.ones(len(ids),bool)
    supported=stores['GELU']['resource_support'].astype(bool)
    metrics={};comparisons={}
    for arm,out in stores.items():
        out['brier']=(out['p_null']-(actual<=0))**2
        out['policy_value']=out['selected']*actual
        out['policy_null']=out['selected']*(actual<=0)
        selected=out['selected'].astype(bool)
        metrics[arm]=dict(full=scalar_metrics(out,actual,all_rows),supported=scalar_metrics(out,actual,supported),
            selected_n=int(selected.sum()),selected_null=int((actual[selected]<=0).sum()),
            selected_mean=float(actual[selected].mean()),
            changed_selected_membership=int(np.count_nonzero(selected!=stores['CORE']['selected'])),
            support_n=int(supported.sum()),active_n=int(np.any(out['increment']!=0,axis=1).sum()),
            by_fold=[dict(fold=f,supported=scalar_metrics(out,actual,supported&(out['fold']==f)),
                full=scalar_metrics(out,actual,out['fold']==f)) for f in range(5)])
        np.savez_compressed(root/(arm+'.npz'),ids=ids,groups=groups,layout=layouts,**out)
    pairs=[(a,'CORE') for a in ARMS if a!='CORE']
    pairs += [('DUAL_GENERIC','GELU'),('DUAL_STRUCTURED','GELU'),('DUAL_STRUCTURED','DUAL_GENERIC')]
    pairs += [('DUAL_GENERIC_CAL','GELU_CAL'),('DUAL_STRUCTURED_CAL','GELU_CAL'),
              ('DUAL_STRUCTURED_CAL','DUAL_GENERIC_CAL')]
    for a,b in pairs:
        comparisons[a+'_minus_'+b]=dict(
            supported=paired_intervals(stores[a],stores[b],supported,groups,layouts),
            full=paired_intervals(stores[a],stores[b],all_rows,groups,layouts),
            policy=paired_intervals(stores[a],stores[b],all_rows,groups,layouts,keys=('policy_value','policy_null')))
    summary=dict(state='COMPLETE',n=len(ids),supported_n=int(supported.sum()),metrics=metrics,
        comparisons=comparisons,cells=cells,samples=SAMPLES,elapsed_seconds=elapsed,
        mean_changed=False,endpoint_changed=False,core_changed=False,protected_data_opened=False,
        primary_contrasts='Fixed full-strength four-arm comparison on supported objects',
        secondary_contrasts='LOO calibration-selected whole-adapter strength; not a certification',
        uncertainty='Chemical-group and layout paired bootstrap, conditional on development folds',
        physical_noise_components_identified=False,jepa_started=False,main_rule_adopted=False)
    write_json(root/'summary.json',summary)
    lines=['# Frozen CORE + dual-branch biological adapter','',
        'Opened LINCS development comparison. CORE and its mean, endpoint, action cost and policy stay fixed.',
        '',f'Supported queries: {int(supported.sum())}/{len(ids)}. Main comparisons are on this pre-outcome subset.',
        '', '| Arm | Supported Γ CRPS | Supported NLL | Full NULL Brier | Selected NULL/n | Selected value |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,m in metrics.items():
        s,f=m['supported']['scores'],m['full']['scores']
        lines.append(f'| {arm} | {s["crps"]:.8f} | {s["nll"]:.5f} | {f["brier"]:.8f} | '
                     f'{m["selected_null"]}/{m["selected_n"]} | {m["selected_mean"]:.8f} |')
    lines+=['','`_CAL` uses only calibration group-LOO scores to select a scalar strength from 0/.25/.5/1.',
        'Generic and structured biological branches share identical inputs, trainable counts and frozen left branch.',
        'The right branch bases encode support/error-scale modelling priors, not established pharmacological laws.',
        'No result here is a new independent validation or automatic permission to enable the module in the main rule.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(output):
    start=time.monotonic();root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    old=read_json(RADIAL/'summary.json')
    manifest=read_json(Path(old['reference_run'])/'run_manifest.json')
    data,metadata=load_data(old['data_directory'])
    ids,groups=data['ids'],data['groups'];n=len(ids)
    if n!=1188 or ids.tolist()!=manifest['ids']: raise ValueError('Opened DEV population differs')
    spec=dict(source=str(SOURCE),radial=str(RADIAL),seed=SEED,samples=SAMPLES,arms=list(ARMS),epochs=60)
    if (root/'run_spec.json').exists() and read_json(root/'run_spec.json')!=spec:
        raise ValueError('Run spec changed')
    write_json(root/'run_spec.json',spec)
    for file in (PROJECT/PLAN,Path(__file__),PROJECT/'opal2/dual_branch_features.py',PROJECT/'opal2/dual_branch_biology.py'):
        target=root/('PROTOCOL.md' if file.name==PLAN else file.name)
        if target.exists() and target.read_bytes()!=file.read_bytes():
            raise ValueError('Declared implementation changed; use another run directory: '+str(file))
        shutil.copy2(file,target)
    core=read_npz(SOURCE/'CORE.npz');prior=read_npz(RADIAL/'AMP_EMP_LOCAL.npz')
    np.testing.assert_array_equal(core['ids'],ids);np.testing.assert_array_equal(prior['ids'],ids)
    raw=gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    actual,observed,difference,_=observable_forward(raw)
    np.testing.assert_allclose(actual,core['actual'],atol=1e-12,rtol=1e-12)
    norm2=np.square(data['Y'][:,0]).mean(1);absolute=np.log1p(difference*norm2[:,None])
    logamp=np.log(np.linalg.norm(data['Y'][:,0],axis=1))
    lookup={v:i for i,v in enumerate(ids)};records={r['fold']:r for r in manifest['folds']}
    stores={a:{} for a in ARMS};trained={};seen=np.zeros(n,int);cells=[]
    for number,cell in enumerate(old['cells']):
        fold,half=cell['fold'],cell['half'];folder=root/f'cell_{fold}_{half}';folder.mkdir(exist_ok=True)
        q=np.asarray([lookup[v] for v in cell['query_ids']]);cal=np.asarray([lookup[v] for v in cell['representative_ids']])
        fit=np.asarray(records[fold]['fit']);test=np.asarray(records[fold]['test'])
        if set(groups[fit])&set(groups[np.r_[q,cal]]) or set(groups[q])&set(groups[cal]):
            raise ValueError('Chemical-group overlap')
        stats=read_json(Path(old['reference_run'])/'folds'/f'fold_{fold}'/'preprocessing.json')
        scale,center=np.asarray(stats['u_scale']),np.asarray(stats['u_center'])
        if fold not in trained:
            write_json(root/'status.json',dict(state='RUNNING',stage='branch_training',fold=fold,cells_complete=number))
            trained[fold]=train_fold(data,metadata,fit,test,prior,stats,SOURCE,root/f'fold_{fold}',fold)
        prediction=trained[fold]
        ref=read_npz(RADIAL/f'cell_{fold}_{half}_radial.npz')
        np.testing.assert_array_equal(ref['cal_ids'],ids[cal]);np.testing.assert_array_equal(ref['query_ids'],ids[q])
        raw_mean=prior['mean_u']*scale+center
        law=fit_radial(ref['amplitude_radii'])
        # CAL objects have opposite roles in the other cell. Their global
        # cached query covariance is not legal for this cell's selection.
        cal_cov=calibration_frame_covariance(ref['cal_amp_scatter'],ref['cal_residual'],
            logamp[cal],groups[cal],cell['local_bandwidth'])
        nested=read_npz(root/f'fold_{fold}/nested_distribution/distribution.npz')
        cal_bio=biology_features(data,metadata,cal,fit,raw_mean[cal],
            cal_cov*scale[None,:,None]*scale[None,None,:],nested['raw_residual'])
        transformer=joblib.load(SOURCE/f'fold_{fold}/conditioners.joblib')['transformer']
        cal_desc=transformer.transform(data,metadata).values[cal]
        cal_prediction={}
        for name in FULL_ARMS:
            model=DualBranchBiologyAdapter.load(root/f'fold_{fold}'/(name+'.pt'))
            cal_prediction[name]=model.predict_increment(cal_desc,
                None if name=='GELU' else cal_bio['values'],cal_bio['support'])
        np.savez_compressed(folder/'honest_calibration_inputs.npz',ids=ids[cal],
            scatter=ref['cal_amp_scatter'],covariance=cal_cov,biological_features=cal_bio['values'],
            support=cal_bio['support'],**cal_prediction)
        calibration={}
        for arm in FULL_ARMS:
            path=folder/(arm+'_selection.json')
            if not path.exists():
                choice=select_strength(raw_mean[cal],scale,ref['cal_amp_scatter'],ref['cal_residual'],
                    logamp[cal],groups[cal],cell['local_bandwidth'],cal_prediction[arm])
                write_json(path,choice)
            calibration[arm]=read_json(path)
        cell_out={}
        for arm in ARMS:
            path=folder/(arm+'.npz')
            if path.exists():
                out={k:v for k,v in read_npz(path).items() if k!='ids'}
            else:
                write_json(root/'status.json',dict(state='RUNNING',stage='joint_sampling',cell=number+1,cells=10,
                    fold=fold,arm=arm,elapsed_seconds=time.monotonic()-start))
                print(f'cell={number+1}/10 arm={arm} joint draws={SAMPLES}',flush=True)
                if arm=='CORE':
                    out={k:v[q].copy() for k,v in core.items() if v.shape[:1]==(n,) and k not in ('ids','groups','layout')}
                    increment=np.zeros((len(q),2))
                else:
                    name=arm.removesuffix('_CAL')
                    strength=calibration[name]['strength'] if arm.endswith('_CAL') else 1.
                    increment=prediction[name][q]*strength
                    changed=np.any(increment!=0,axis=1)
                    if not changed.any():
                        out={k:v.copy() for k,v in cell_out['CORE'].items()}
                    elif arm.endswith('_CAL') and strength==1:
                        out={k:v.copy() for k,v in cell_out[name].items()}
                    else:
                        scatter=apply_increment(raw_mean[q],scale,prior['scatter_u'][q],increment)
                        np.testing.assert_array_equal(scatter[~changed],prior['scatter_u'][q][~changed])
                        scored=evaluate_radial_mixtures(prior['mean_u'][q],scatter,prior['actual_u'][q],stats,
                            actual[q],observed[q],absolute[q],norm2[q],cell['normal_seed'],law=law,
                            endpoint_weights={'CORE':ref['local_weights']},coefficients={arm:np.ones(1)},samples=SAMPLES)[arm]
                        # The zero-increment predictive distribution is identical;
                        # use its original estimator to preserve exact off behavior.
                        for key,value in scored.items():
                            if key in cell_out['CORE']:
                                value[~changed]=cell_out['CORE'][key][~changed]
                        out=scored
                    selection=select_frozen_cohort_plan(ids[q],out['predicted'],out['p_null'],cell['budget'])
                    out['selected']=np.asarray(selection.selected_mask,int)
                out['increment']=increment
                out['resource_support']=prediction['support'][q]
                out['mean_u']=prior['mean_u'][q].copy();out['actual_u']=prior['actual_u'][q].copy()
                out['actual']=actual[q];out['fold']=np.full(len(q),fold,int)
                np.savez_compressed(path,ids=ids[q],**out)
            cell_out[arm]=out
            for key,value in out.items():
                if value.shape[:1]!=(len(q),): continue
                if key not in stores[arm]:stores[arm][key]=np.empty((n,*value.shape[1:]),dtype=value.dtype)
                stores[arm][key][q]=value
        seen[q]+=1
        cells.append(dict(fold=fold,half=half,query_ids=ids[q].tolist(),calibration_ids=ids[cal].tolist(),
            support_n=int(prediction['support'][q].sum()),selections=calibration,budget=cell['budget']))
    np.testing.assert_array_equal(seen,np.ones(n,int))
    stores=collect_completed_cells(root,ids,cells)
    for key in ('mean_u','predicted','p_null','crps','nll','selected'):
        np.testing.assert_array_equal(stores['CORE'][key],core[key])
    for arm in ARMS:
        inactive=np.all(stores[arm]['increment']==0,axis=1)
        for key in ('predicted','p_null','crps','nll'):
            np.testing.assert_array_equal(stores[arm][key][inactive],core[key][inactive])
    summarize(stores,data,metadata,cells,root,time.monotonic()-start)
    write_json(root/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-start))
    print('COMPLETE',root,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--reaggregate',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        if not args.reaggregate:
            run(args.output)
        else:
            root=Path(args.output).resolve()
            summary=read_json(root/'summary.json')
            if summary['state']!='COMPLETE':raise ValueError('Only complete evaluations can be reaggregated')
            data,metadata=load_data(read_json(RADIAL/'summary.json')['data_directory'])
            original={a:read_npz(root/(a+'.npz')) for a in ARMS}
            stores=collect_completed_cells(root,data['ids'],summary['cells'])
            for arm in ARMS:
                for key in ('actual','predicted','p_null','crps','nll','energy','selected','mean_u'):
                    np.testing.assert_array_equal(stores[arm][key],original[arm][key])
            backup=root/'aggregation_before_fix'
            if backup.exists():raise FileExistsError('Original aggregate backup already exists')
            backup.mkdir()
            for name in [*(a+'.npz' for a in ARMS),'summary.json','REPORT.md']:
                (root/name).rename(backup/name)
            summarize(stores,data,metadata,summary['cells'],root,summary['elapsed_seconds'])
            write_json(root/'aggregation_repair.json',dict(state='COMPLETE',
                reason='Copied CORE cells expose extra legacy fields absent from active scorer cells; aggregate common fields only',
                training_repeated=False,sampling_repeated=False,selection_changed=False,
                main_scores_bitwise_unchanged=True,original_files=str(backup)))


if __name__=='__main__': main()
