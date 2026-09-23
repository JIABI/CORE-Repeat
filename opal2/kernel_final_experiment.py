"""Two final bounded architecture contrasts, in one matched 2x2x2 run."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import now, event
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import fit_score_scale, evaluate_and_save
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .conditional_response_kernel import ConditionalResponseKernelMean
from .geometry_kernel_replacement_experiment import load_frozen_fold, predict_branch
from .gamma_supervised_experiment import CONFIG as OLD_CONFIG
from .gamma_supervised_loss import JointGammaCRPS
from .kernel_final_reference import reference_allocation, fit_reference_bank

PROJECT=Path(__file__).resolve().parents[1]
ARMS=('A_HR','O_G_W','O_S_W','O_G_C','O_S_C','D_G_W','D_S_W','D_G_C','D_S_C')
CONFIG=dict(OLD_CONFIG,samples=10000,dual_penalty=1.,dual_step=1.,reference_seed_offset=88001)


def prepare(output, historical):
    root,historical=Path(output).resolve(),Path(historical).resolve()
    if root.exists():raise FileExistsError('Use a fresh final-factorial run directory')
    old=json.loads((historical/'run_manifest.json').read_text())
    if json.loads((historical/'status.json').read_text())['state']!='COMPLETE':
        raise ValueError('Completed prior branch comparison required')
    ds,split,_=_load_study_data(old['data_directory']);_check_cohort(ds,split,old)
    scopes=[]
    for record in old['folds']:
        fit,valid,test=validate_partition(record,len(ds))
        scope=reference_allocation(ds.ids,fit,ds.chem,ds.chem_mask,ds.metadata.get('chemical') or None,
            record['seed']+CONFIG['reference_seed_offset'],CONFIG['max_anchors'])
        scope.update(fold=record['fold'],validation_ids=ds.ids[valid].tolist(),test_ids=ds.ids[test].tolist(),
            old_hr_and_covariance_fitted_on_originalfit=True,reference_separation_scope='new branch supervision only')
        scopes.append(scope)
    root.mkdir(parents=True)
    snapshot=root/'source_snapshot'
    for name in ('opal2','tests'):
        shutil.copytree(PROJECT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml',snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/KERNEL_FINAL_FACTORIAL_PLAN_20260915.md',root/'PROTOCOL.md')
    manifest={key:deepcopy(old[key]) for key in ('ids','folds','reference_run','data_directory','data_shape',
        'feature_names','original_compound_ids','scope')}
    manifest.update(created_utc=now(),source_snapshot=str(snapshot),python_executable=sys.executable,
        historical_reference_run=str(historical),config=deepcopy(CONFIG),arms=list(ARMS),
        scopes=scopes,actual_checkpoint_epoch=30,checkpoint_policy='all arms fixed epoch30, no test-selected fallback',
        final_opened=False,fifth_repeat_opened=False,original_endpoint_changed=False,
        original_contract_changed=False,original_split_files_changed=False,previous_results_preserved=True,
        historical_dev=True,independent_new_holdout=False,formal_certificate=False,
        jepa_active=False,mechanism_annotations_active=False,covariance_updated=False,
        trainable_parameters_matched=True,supervised_ids_matched=True,
        reference_selection='two fixed random disjoint sets of64; both use same selection algorithm',
        architecture='complete conditional local response branch on frozen HR joint geometry distribution',
        sample_precision_prefix=2000,sample_precision_halves=5000,new_branch_fits=40)
    write_json(root/'run_manifest.json',manifest)
    event(root,'PREPARED',new_branch_fits=40,epochs=30,samples=10000,n=639)
    return manifest


def score(folder,mean,ridge,stats,target,grams,ids,train_gains,metric_scale,seed,arm,counts):
    diagnostics=gaussian_coordinate_diagnostics(folder,ids,target,mean,ridge.covariance)
    samples=sample_joint_coordinates(mean,ridge.covariance,CONFIG['samples'],seed+200000)
    draws,audit=decode_draws(restore_target(samples,stats),verify=True)
    return evaluate_and_save(folder,draws,grams,ids,
        metadata=dict(arm=arm,actual_checkpoint_epoch=30 if arm!='A_HR' else None,
            selection='fixed epoch30' if arm!='A_HR' else 'frozen historical HR',
            backbone='immutable historical HR',covariance='exact RIDGE_VALID OOF covariance',
            numerics=audit,u_diagnostics=diagnostics,parameter_counts=counts,
            formal_certificate=False,historical_dev=True),
        train_actual_gains=train_gains,score_scale=metric_scale,seed=seed,
        n_bootstrap=CONFIG['bootstrap'],n_random=CONFIG['random_subsets'])


def execute_fold(root,manifest,ds,record,scope,grams,raw_u,gains):
    from .kernel_final_training import train_branch
    folder=root/'folds'/f"fold_{record['fold']}";folder.mkdir(parents=True)
    write_json(folder/'scope.json',scope)
    originalfit,valid,test=validate_partition(record,len(ds))
    fit=np.asarray(scope['commonbranchfit'],int)
    _,stats,ridge,hr,_=load_frozen_fold(manifest['reference_run'],record)
    x,target=transform_input(ds.Y[:,0],stats),transform_target(raw_u,stats)
    banks={mode:fit_reference_bank(x,ds.chem,ds.chem_mask,ds.ids,scope['originalfit_ids'],
        scope['commonbranchfit_ids'],scope['anchor_ids_by_mode'][mode],ds.metadata.get('chemical') or None)
        for mode in ('O','D')}
    (folder/'banks').mkdir()
    for mode,bank in banks.items():bank.save(folder/'banks'/f'{mode}.pt')
    support={}
    for mode,bank in banks.items():
        support[mode]={}
        for name,ii in (('supervised',fit),('validation',valid),('test',test)):
            with torch.no_grad():
                d=bank(torch.tensor(x[ii]),torch.tensor(ds.chem[ii]),torch.tensor(ds.chem_mask[ii]),ids=ds.ids[ii])
                t=d['tanimoto'].numpy();is_reference=np.isin(ds.ids[ii],scope['anchor_ids_by_mode'][mode])
                energy=np.power(t,8).sum(1)
                support[mode][name]=dict(n=len(ii),self_reference_n=int(is_reference.sum()),
                    nonself_identical_fingerprint_n=int(((t.max(1)==1)&~is_reference).sum()),
                    mean_max_reference_tanimoto=float(t.max(1).mean()),
                    reference_squared_T4_energy_fraction=float(energy[is_reference].sum()/energy.sum()),
                    total_squared_T4_energy=float(energy.sum()),
                    energy_scope='pre-coefficient and pre-gate basis, not final decision contribution')
    write_json(folder/'reference_support.json',support)
    gamma_scale=float(np.std(gains[fit,2],ddof=1))
    objective=JointGammaCRPS(ridge.covariance,stats['u_center'],stats['u_scale'],gamma_scale)
    torch.save(objective.state_dict(),folder/'gamma_objective_state.pt')
    write_json(folder/'gamma_objective.json',dict(gamma_scale=gamma_scale,fitting_ids=ds.ids[fit].tolist(),
        covariance_frozen=True,train_pairs=64,cost_two=.02))
    seed=record['seed']+CONFIG['branch_seed_offset']
    joined=np.r_[fit,valid];fit_local=np.arange(len(fit));valid_local=np.arange(len(fit),len(joined))
    metric_scale=fit_score_scale(grams[originalfit])
    with torch.no_grad():hrmean=hr(torch.as_tensor(x[test],dtype=torch.float64)).numpy()
    score(folder/'arms/A_HR/evaluation',hrmean,ridge,stats,target[test],grams[test],ds.ids[test],
        gains[fit],metric_scale,record['seed'],'A_HR',dict(trainable=0,total=sum(p.numel() for p in hr.parameters())))
    initial_trainable=None
    for arm in ARMS[1:]:
        role,basis,loss=arm.split('_')
        torch.manual_seed(seed)
        model=ConditionalResponseKernelMean(hr,banks[role],mode='conditional_generic' if basis=='G' else 'conditional_structured',
            incremental_penalty=CONFIG['incremental_penalty'],hidden_dim=CONFIG['hidden_dim']).double()
        trainable={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
        if initial_trainable is None:initial_trainable=trainable
        elif any(not torch.equal(p,initial_trainable[name]) for name,p in trainable.items()):
            raise ValueError('Factorial trainable initialization differs')
        with torch.no_grad():
            if not np.array_equal(predict_branch(model,x[test],ds.chem[test],ds.chem_mask[test]),hrmean):
                raise ValueError('A factorial arm did not start at frozen HR')
        event(root,'TRAINING',fold=record['fold'],arm=arm,epochs=30,fit_n=len(fit))
        train_branch(folder/'arms'/arm,model,objective,x[joined],ds.chem[joined],ds.chem_mask[joined],
            target[joined],gains[joined,2],fit_local,valid_local,seed,ds.ids[joined],CONFIG,
            'weighted' if loss=='W' else 'constrained')
        mean=predict_branch(model,x[test],ds.chem[test],ds.chem_mask[test])
        saved=torch.load(folder/'arms'/arm/'epoch30.pt',map_location='cpu',weights_only=True)
        restored=ConditionalResponseKernelMean.from_config(saved['model_config']);restored.load_state_dict(saved['state_dict'])
        if not np.array_equal(mean,predict_branch(restored,x[test],ds.chem[test],ds.chem_mask[test])):
            raise ValueError('Fixed checkpoint does not reproduce its scored prediction')
        with torch.no_grad():
            d=model.diagnostics(torch.tensor(x[test]),torch.tensor(ds.chem[test]),torch.tensor(ds.chem_mask[test]),ids=ds.ids[test])
        arrays={k:v.detach().numpy() for k,v in d.items() if isinstance(v,torch.Tensor)}
        arrays['raw']=arrays['kernel_raw']
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz',ids=ds.ids[test],
            block_names=np.asarray(['chemical','morphology','scalar']),**arrays)
        counts=dict(trainable=sum(p.numel() for p in model.trainable_parameters()),total=sum(p.numel() for p in model.parameters()))
        event(root,'SCORING',fold=record['fold'],arm=arm,samples=10000)
        score(folder/'arms'/arm/'evaluation',mean,ridge,stats,target[test],grams[test],ds.ids[test],
            gains[fit],metric_scale,record['seed'],arm,counts)
        # Read-only fits on references and nonreferences isolate self-landmark leverage.
        groups=dict(supervised=fit,reference=np.asarray([list(ds.ids).index(u) for u in scope['anchor_ids_by_mode'][role]]),
            nonreference=np.asarray([i for i in fit if ds.ids[i] not in set(scope['anchor_ids_by_mode'][role])]),validation=valid)
        audit={}
        readout_indices=np.r_[originalfit,valid]
        positions={int(i):j for j,i in enumerate(readout_indices)}
        with torch.no_grad():
            prediction=model(torch.tensor(x[readout_indices]),torch.tensor(ds.chem[readout_indices]),
                torch.tensor(ds.chem_mask[readout_indices]))
            rng=torch.Generator().manual_seed(seed+51000)
            a=torch.randn((128,len(readout_indices),9),generator=rng,dtype=torch.float64)
            b=torch.randn((128,len(readout_indices),9),generator=rng,dtype=torch.float64)
            out=objective(prediction,torch.tensor(gains[readout_indices,2]),a,b)
            object_u=(prediction-torch.tensor(target[readout_indices])).square().mean(-1).numpy()
            object_crps=out['gamma_crps_per_object'].numpy()
        np.savez_compressed(folder/'arms'/arm/'reference_readout.npz',ids=ds.ids[readout_indices],
            u_mse=object_u,gamma_crps=object_crps)
        for name,ii in groups.items():
            if not len(ii):continue
            select=np.asarray([positions[int(i)] for i in ii])
            audit[name]=dict(n=len(ii),ids=ds.ids[ii].tolist(),u_mse=float(object_u[select].mean()),
                gamma_crps=float(object_crps[select].mean()),readout_only=True,
                common_objectwise_draws=True)
        write_json(folder/'arms'/arm/'reference_fit_diagnostic.json',audit)
        event(root,'ARM_COMPLETE',fold=record['fold'],arm=arm)
    write_json(folder/'complete.json',dict(fold=record['fold'],completed_utc=now(),arms=list(ARMS),actual_checkpoint_epoch=30,
        test_n=len(test),supervised_n=len(fit),reference_n=64,formal_certificate=False))
    event(root,'FOLD_COMPLETE',fold=record['fold'])


def execute(output):
    root=Path(output).resolve();manifest=json.loads((root/'run_manifest.json').read_text())
    if Path(manifest['source_snapshot'])!=PROJECT or manifest['config']!=CONFIG:
        raise ValueError('Use the frozen source and configuration')
    if (root/'folds').exists():raise FileExistsError('This factorial run has already started')
    ds,split,_=_load_study_data(manifest['data_directory']);_check_cohort(ds,split,manifest)
    torch.set_num_threads(CONFIG['threads']);started=time.monotonic()
    event(root,'RUNNING',new_branch_fits=40,epochs=30)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams=profiles_to_gram(torch.as_tensor(ds.Y,dtype=torch.float64)).numpy()
            raw_u=gram_to_coordinates(torch.as_tensor(grams)).numpy();gains=gram_gains(torch.as_tensor(grams)).numpy()
            for record,scope in zip(manifest['folds'],manifest['scopes']):
                execute_fold(root,manifest,ds,record,scope,grams,raw_u,gains)
            event(root,'SUMMARIZING')
            from .kernel_final_summary import summarize
            summarize(root)
        event(root,'COMPLETE',elapsed_seconds=time.monotonic()-started,stopped_at_epoch=30,further_training_started=False)
    except Exception as error:
        event(root,'FAILED',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','execute','summarize'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--historical',default=str(PROJECT/'runs/generic_gamma_epoch30_20260914_v1'))
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output,args.historical)
    elif args.mode=='execute':execute(args.output)
    else:
        from .kernel_final_summary import summarize
        summarize(args.output)

if __name__=='__main__':main()
