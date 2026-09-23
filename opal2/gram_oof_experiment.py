"""Complete paired outer-fold comparison, restricted to the opened four-role DEV."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import plain, write_json
from .gram_experiment import CONFIG as ORIGINAL_G_CONFIG, _schedule
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_forward_diagnostic import coordinates_factor_forward, factor_forward_consistency
from .gram_model import GramConditionalModel
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .gram_simple_models import fit_global, GramSimpleGaussian
from .gram_oof_ridge import fit_preprocessing, transform_input, transform_target, fit_ridge_oof
from .closed_form_baseline import fit_baseline, ClosedFormBaseline
from .gram_reference import ClosedFormGramReference
from .objective_analysis import fair_crps
from .baseline_policy import stable_top_k, _metrics, ACTIONS, WELL_COSTS

PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('GLOBAL_GEOMETRY', 'RIDGE_GEOMETRY', 'G_DIRECT', 'L_GRAM')
CONFIG = dict(seed=20260914, folds=5, strata=10, inner_validation_fraction=.2,
              samples=2000, bootstrap=2000, random_subsets=2000, threads=2,
              object_chunk=2, draw_chunk=32,
              l_fit=dict(k=200, clip=8., noise_shrinkage=.05, variance_floor=1e-6),
              g=deepcopy(ORIGINAL_G_CONFIG))


def now():
    return datetime.now(timezone.utc).isoformat()


def event(root, state, **fields):
    root = Path(root)
    row = plain(dict(utc=now(), state=state, **fields))
    write_json(root/'status.json', row)
    with (root/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def assign_folds(x, ids, *, seed=20260914, n_splits=5, n_strata=10):
    """Assignment depends on X/ID only; one unique object per test fold."""
    x, ids = np.asarray(x, float), np.asarray(ids, str)
    if x.ndim != 2 or len(x) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('One X and one unique ID are required per compound')
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Fold strata require finite nonzero X norms')
    order = np.lexsort((ids, np.log(norms)))
    strata = np.empty(len(x), dtype=int)
    strata[order] = np.minimum(np.arange(len(x))*n_strata//len(x), n_strata-1)
    records, membership = [], np.full(len(x), -1, int)
    for fold, (pool, test) in enumerate(StratifiedKFold(n_splits=n_splits,
            shuffle=True, random_state=seed).split(x, strata)):
        fit_local, valid_local = next(StratifiedShuffleSplit(n_splits=1,
            test_size=CONFIG['inner_validation_fraction'], random_state=seed+10000*fold+91
            ).split(pool, strata[pool]))
        fit, valid = pool[fit_local], pool[valid_local]
        membership[test] = fold
        records.append(dict(fold=fold, fit=fit.tolist(), inner_validation=valid.tolist(),
                            test=test.tolist(), seed=seed+10000*fold))
    return records, membership, strata


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('A fresh OOF run directory is required')
    prior = json.loads((reference/'run_manifest.json').read_text())
    ds, old_split, scope = _load_study_data(prior['data_directory'])
    identities = {k: ds.ids[v].tolist() for k,v in old_split.items()}
    if identities != prior['compound_ids'] or ds.Y.shape != (639,4,3617):
        raise ValueError('Only the complete, previously opened cohort is allowed')
    folds, membership, strata = assign_folds(ds.Y[:,0], ds.ids)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/GRAM_OOF_PLAN_20260914.md', root/'PROTOCOL.md')
    snapshot = root/'source_snapshot'
    for name in ('opal2','tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml',snapshot/'pyproject.toml')
    np.savez_compressed(root/'fold_assignment.npz', ids=ds.ids.astype(str),
                        fold=membership, x_lognorm_stratum=strata)
    for record in folds:
        for key in ('fit','inner_validation','test'):
            record[key+'_ids'] = ds.ids[record[key]].tolist()
    manifest = dict(created_utc=now(), data_directory=prior['data_directory'],
        source_snapshot=str(snapshot), reference_run=str(reference), config=CONFIG,
        original_compound_ids=identities, ids=ds.ids.tolist(), data_shape=list(ds.Y.shape),
        feature_names=ds.feature_names.tolist(), feature_groups=prior['feature_groups'],
        folds=folds, scope=scope, arms=list(ARMS), final_opened=False,
        fifth_repeat_opened=False, original_split_files_changed=False,
        original_endpoint_changed=False, original_contract_changed=False,
        historical_dev=True, formal_certificate=False)
    write_json(root/'run_manifest.json',manifest)
    event(root,'PREPARED',fold_sizes=[{k:len(f[k]) for k in ('fit','inner_validation','test')}
                                   for f in folds],cohort_n=len(ds))


def load_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if Path(manifest['source_snapshot']) != PROJECT or manifest['config'] != CONFIG:
        raise ValueError('Execute the recorded source snapshot/configuration')
    ds, split, _ = _load_study_data(manifest['data_directory'])
    if ds.ids.tolist() != manifest['ids'] or ds.Y.shape != tuple(manifest['data_shape']):
        raise ValueError('Cohort identity or dimensions changed')
    if {k:ds.ids[v].tolist() for k,v in split.items()} != manifest['original_compound_ids']:
        raise ValueError('The original split files changed')
    if ds.feature_names.tolist() != manifest['feature_names']:
        raise ValueError('The endpoint coordinate order changed')
    return root, manifest, ds


def decode_draws(raw_u, *, verify=False):
    tensor = torch.as_tensor(raw_u, dtype=torch.float64)
    gram, audit = coordinates_factor_forward(tensor)
    mask = audit.pop('recovered_schur_failed_mask')
    audit.pop('recovered_schur_info')
    audit.pop('recovered_schur_failed_indices')
    audit['per_object_recovered_schur_failures'] = mask.sum(0).tolist()
    if verify:
        check = factor_forward_consistency(tensor, gram)
        check.pop('gain_absolute_error_per_draw_object')
        check.pop('observable_absolute_error_per_draw_object')
        if check['gains_max_absolute_error'] > 1e-9:
            raise FloatingPointError('Forward utility does not match virtual-vector utility')
        audit['functional_consistency'] = check
    return gram.numpy(), plain(audit)


@torch.no_grad()
def sample_g(model, inputs, stats, samples, seed, *, verify=False):
    model.eval()
    x = torch.tensor(inputs[:,:-1],dtype=torch.float32)
    n = torch.tensor(inputs[:,-1:],dtype=torch.float32)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    output = model(x,n).sample(samples,generator=generator).double().numpy()
    raw = output*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
    return decode_draws(raw,verify=verify)


def train_g(folder, groups, inputs, targets, stats, fit, valid, actual, seed, ids):
    folder.mkdir(parents=True,exist_ok=True)
    cfg = CONFIG['g']
    torch.manual_seed(seed)
    model = GramConditionalModel(groups, **{k:cfg[k] for k in
        ('hidden_dim','attention_heads','attention_layers','covariance_shrinkage','log_std_bound')})
    if (folder/'training_complete.json').exists():
        payload=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(payload['state_dict'])
        return model,payload['epoch']
    if (folder/'history.jsonl').exists():
        raise RuntimeError('An interrupted G fit must not restart into its existing history')
    x=torch.tensor(inputs[:,:-1],dtype=torch.float32)
    n=torch.tensor(inputs[:,-1:],dtype=torch.float32)
    target=torch.tensor(targets,dtype=torch.float32)
    parameters=[list(model.mean_parameters()),list(model.covariance_parameters())]
    optimizers=[torch.optim.AdamW(p,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']) for p in parameters]
    total=cfg['max_epochs']*math.ceil(len(fit)/cfg['batch_size'])
    schedulers=[torch.optim.lr_scheduler.LambdaLR(o,lambda step:_schedule(step,total,cfg)) for o in optimizers]
    rng=torch.Generator(device='cpu').manual_seed(seed+31)
    best,significant,stale,best_epoch,steps=float('inf'),float('inf'),0,0,0
    started=time.monotonic()
    event(folder,'TRAINING_STARTED',parameters=sum(p.numel() for p in model.parameters()),fit_n=len(fit),inner_validation_n=len(valid))
    for epoch in range(cfg['max_epochs']+1):
        means,covs=[],[]
        if epoch:
            model.train()
            order=np.asarray(fit)[torch.randperm(len(fit),generator=rng).numpy()]
            for first in range(0,len(order),cfg['batch_size']):
                ix=order[first:first+cfg['batch_size']]
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                losses=model.loss(x[ix],n[ix],target[ix])
                if not torch.isfinite(losses['loss']): raise FloatingPointError('Nonfinite G loss')
                losses['loss'].backward()
                for p in parameters: torch.nn.utils.clip_grad_norm_(p,cfg['gradient_clip'],error_if_nonfinite=True)
                for opt,schedule in zip(optimizers,schedulers): opt.step(); schedule.step()
                steps+=1
                means.append(losses['mean_mse'].item()); covs.append(losses['covariance_nll'].item())
        row=dict(epoch=epoch,optimizer_steps=steps,train_mean_mse=np.mean(means) if means else None,
                 train_covariance_nll=np.mean(covs) if covs else None,elapsed_seconds=time.monotonic()-started)
        if epoch%cfg['validation_interval']==0:
            draws,audit=sample_g(model,inputs[valid],stats,cfg['validation_samples'],seed+97001)
            gain=gram_gains(torch.tensor(draws)).numpy()
            score=float(fair_crps(gain,actual[valid]).mean(0)[2])
            with torch.no_grad(): losses=model.loss(x[valid],n[valid],target[valid])
            row.update(validation_gamma_crps=score,validation_u_mse=losses['mean_mse'].item(),numerics=audit)
            payload=dict(state_dict=deepcopy(model.state_dict()),model_config=model.config,epoch=epoch,
                         optimizer_steps=steps,validation_gamma_crps=score,
                         fit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist())
            if score<best:
                best,best_epoch=score,epoch
                torch.save(payload,folder/'best.pt')
            if score<significant-cfg['min_delta']: significant,stale=score,0
            elif epoch: stale+=1
            torch.save(payload,folder/'last.pt')
            row.update(best_epoch=best_epoch,checks_without_improvement=stale)
            event(folder,'VALIDATED',**row)
        with (folder/'history.jsonl').open('a') as stream: stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
        if epoch and epoch%cfg['report_interval']==0: write_json(folder/f'report_epoch_{epoch:04}.json',row)
        if epoch>=cfg['min_epochs'] and stale>=cfg['patience_checks']: break
    completion=dict(epoch=epoch,best_epoch=best_epoch,best_validation_gamma_crps=best,
                    elapsed_seconds=time.monotonic()-started,stop_reason='patience' if stale>=cfg['patience_checks'] else 'epoch_cap')
    write_json(folder/'training_complete.json',completion)
    event(folder,'TRAINING_COMPLETE',**completion)
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    return model,best_epoch


def uniform_global_policy(report, actual):
    """Replace arbitrary lexical subsets with exact uniform-subset expectations."""
    report=deepcopy(report)
    policy=report['policy']
    policy.pop('row_trace',None)
    policy['statistical_scope']['score_ties']='uniform-subset expectation; no selected individual IDs'
    for section in ('within_action','common_budget'):
        for row in policy[section]:
            j=ACTIONS.index(row['action']); y=np.asarray(actual)[:,j]
            n,k=len(y),row['selected_n']; q=k/n
            null=y<=0; positive=y>=.005
            row.update(total_net_gain=float(k*y.mean()),
                per_selected_net_gain=float(y.mean()) if k else None,
                per_eligible_net_gain=float(q*y.mean()),selected_null_count=float(k*null.mean()),
                selected_positive_count=float(k*positive.mean()),
                selected_ambiguous_count=float(k*(~null&~positive).mean()),
                fdp=float(null.mean()) if k else None,positive_purity=float(positive.mean()) if k else None,
                fpr=q if null.any() else None,sensitivity=q if positive.any() else None,
                selected_ids=None,uniform_random_subset_exact_expectation=True)
            row.pop('matched_random',None)
    return report


def execute_fold(root, manifest, ds, record):
    fold=record['fold']; seed=record['seed']
    folder=root/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return
    fit,valid,test=[np.asarray(record[k]) for k in ('fit','inner_validation','test')]
    if set(fit)&set(valid) or set(fit)&set(test) or set(valid)&set(test):
        raise ValueError('Fold roles overlap')
    started=time.monotonic()
    actual_g=profiles_to_gram(torch.tensor(ds.Y,dtype=torch.float64)).numpy()
    raw_u=gram_to_coordinates(torch.tensor(actual_g)).numpy()
    actual=gram_gains(torch.tensor(actual_g)).numpy()
    stats=fit_preprocessing(ds.Y[fit],raw_u[fit])
    write_json(folder/'preprocessing.json',dict(**stats,fit_ids=ds.ids[fit].tolist()))
    inputs=transform_input(ds.Y[:,0],stats)
    targets=transform_target(raw_u,stats)
    metric_scale=fit_score_scale(actual_g[fit])
    for arm in ARMS:
        dest=folder/'arms'/arm
        if (dest/'test/metrics.json').exists(): continue
        dest.mkdir(parents=True,exist_ok=True)
        event(root,'MODEL_STARTED',fold=fold,arm=arm,fit_n=len(fit),test_n=len(test))
        model_metadata=dict(arm=arm,fold=fold,fit_n=len(fit),test_n=len(test),
                            input='none' if arm=='GLOBAL_GEOMETRY' else 'X',formal_certificate=False)
        if arm in ('GLOBAL_GEOMETRY','RIDGE_GEOMETRY'):
            if (dest/'fit.npz').exists():
                model=GramSimpleGaussian.load(dest/'fit.npz')
            elif arm=='GLOBAL_GEOMETRY':
                model=fit_global(targets[fit]); model.save(dest/'fit.npz')
            else:
                model,rstats=fit_ridge_oof(ds.Y[fit],raw_u[fit],seed=seed)
                for key in stats:
                    if not np.allclose(stats[key],rstats[key],rtol=0,atol=0):
                        raise ValueError('Ridge and G outer preprocessing differ: '+key)
                model.save(dest/'fit.npz')
            write_json(dest/'fit_summary.json',model.metadata)
            raw=model.sample_coordinates(inputs[test],CONFIG['samples'],seed+200000)
            raw=raw*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
            draws,numerics=decode_draws(raw,verify=True)
            model_metadata.update(numerics=numerics,preprocessing_refit=True)
            del raw,model
        elif arm=='G_DIRECT':
            model,epoch=train_g(dest,manifest['feature_groups'],inputs,targets,stats,fit,valid,actual,seed,ds.ids)
            draws,numerics=sample_g(model,inputs[test],stats,CONFIG['samples'],seed+200000,verify=True)
            model_metadata.update(epoch=epoch,selection='inner-validation Gamma CRPS only',numerics=numerics)
            del model
        else:
            if (dest/'fit.npz').exists(): model=ClosedFormBaseline.load(dest/'fit.npz')
            else:
                model=fit_baseline(ds.Y[fit],random_state=seed,**CONFIG['l_fit'])
                model.metadata['train_ids']=ds.ids[fit].tolist(); model.save(dest/'fit.npz')
            write_json(dest/'fit_summary.json',model.metadata)
            event(root,'SAMPLING_FULL_L',fold=fold,objects=len(test),samples=CONFIG['samples'])
            reference=ClosedFormGramReference(model,provenance=dict(model='refitted L',fit_ids=ds.ids[fit].tolist()))
            sampled=reference.sample(ds.Y[test,0],CONFIG['samples'],seed=seed+200000,
                object_chunk_size=CONFIG['object_chunk'],draw_chunk_size=CONFIG['draw_chunk'])
            draws=sampled.grams; model_metadata.update(sampled.metadata)
            del model,reference
        event(root,'SCORING',fold=fold,arm=arm)
        report=evaluate_and_save(dest/'test',draws,actual_g[test],ds.ids[test],metadata=model_metadata,
            train_actual_gains=actual[fit],score_scale=metric_scale,seed=seed,
            n_bootstrap=CONFIG['bootstrap'],n_random=CONFIG['random_subsets'])
        if arm=='GLOBAL_GEOMETRY':
            write_json(dest/'test/metrics.json',uniform_global_policy(report,actual[test]))
        del draws
        event(root,'MODEL_COMPLETE',fold=fold,arm=arm,elapsed_fold_seconds=time.monotonic()-started)
    write_json(folder/'complete.json',dict(completed_utc=now(),elapsed_seconds=time.monotonic()-started))
    event(root,'FOLD_COMPLETE',fold=fold,elapsed_seconds=time.monotonic()-started)


def selection_mask(scores, ids, folds, fraction, cost, *, global_model=False, within_action=False):
    result=np.zeros(len(ids),float)
    for fold in np.unique(folds):
        ix=np.flatnonzero(folds==fold); k=math.floor(fraction*len(ix))//(1 if within_action else cost)
        result[ix]=k/len(ix) if global_model else stable_top_k(scores[ix],ids[ix],k)
    return result


def summarize(root):
    root=Path(root); manifest=json.loads((root/'run_manifest.json').read_text())
    ids=np.asarray(manifest['ids'],str); n=len(ids)
    allocation=np.full(n,-1,int)
    for r in manifest['folds']: allocation[r['test']]=r['fold']
    all_data,results={},{}
    for arm in ARMS:
        store={}; fold_rows=[]
        for r in manifest['folds']:
            folder=root/'folds'/f"fold_{r['fold']}"/'arms'/arm/'test'
            metric=json.loads((folder/'metrics.json').read_text())
            ix=np.asarray(r['test'])
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:
                if not np.array_equal(z['ids'],ids[ix]): raise ValueError('Prediction IDs do not match fold')
                for key in ('actual','predicted','p_null','utility_crps','geometry_energy'):
                    if key not in store: store[key]=np.empty((n,*z[key].shape[1:]),float)
                    store[key][ix]=z[key]
            fold_rows.append(dict(fold=r['fold'],n=len(ix),actions=metric['action_metrics'],
                                  crps=[a['crps'] for a in metric['utility']]))
        actions=[]
        for j,name in enumerate(ACTIONS):
            row=_metrics(store['actual'][:,j],store['predicted'][:,j],store['p_null'][:,j])
            row.update(action=name,gamma_crps=float(store['utility_crps'][:,j].mean()))
            if arm=='GLOBAL_GEOMETRY': row.update(pearson=None,spearman=None,null_auc=None)
            rank_x=np.zeros(n);rank_y=np.zeros(n)
            for r in manifest['folds']:
                ix=np.asarray(r['test']); a=rankdata(store['actual'][ix,j]);p=rankdata(store['predicted'][ix,j])
                rank_x[ix]=(a-a.mean())/len(ix);rank_y[ix]=(p-p.mean())/len(ix)
            row['within_fold_rank_association']=float(np.corrcoef(rank_x,rank_y)[0,1]) if np.std(rank_y)>0 else None
            actions.append(row)
        policies=[]; masks=[]
        for j,name in enumerate(ACTIONS):
            for q in (.05,.1,.25):
                for kind,scores in (('expected_gain',store['predicted'][:,j]),('lowest_p_null',-store['p_null'][:,j])):
                    for section in ('common_budget','within_action'):
                        mask=selection_mask(scores,ids,allocation,q,WELL_COSTS[j],global_model=arm=='GLOBAL_GEOMETRY',within_action=section=='within_action')
                        null=store['actual'][:,j]<=0;positive=store['actual'][:,j]>=.005
                        k=float(mask.sum());gain=float(mask@store['actual'][:,j]);false=float(mask@null)
                        policies.append(dict(action=name,fraction=q,ranking=kind,section=section,selected_n=int(round(k)),
                            used_wells=int(round(k*WELL_COSTS[j])),per_eligible_net_gain=gain/n,
                            per_selected_net_gain=gain/k if k else None,fdp=false/k if k else None,
                            fpr=false/null.sum() if null.any() else None,
                            sensitivity=float(mask@positive/positive.sum()) if positive.any() else None,
                            budget_scope='sum of foldwise '+('physical-well caps' if section=='common_budget' else 'selected-compound counts'),formal_certificate=False))
                        masks.append(mask)
        store['masks']=np.asarray(masks)
        np.savez_compressed(root/f'{arm}_oof_predictions.npz',ids=ids,fold=allocation,**store)
        all_data[arm]=store
        results[arm]=dict(actions=actions,folds=fold_rows,policies=policies,
                         common_budget=[p for p in policies if p['section']=='common_budget'],
                         within_action=[p for p in policies if p['section']=='within_action'],
                         geometry_energy=float(store['geometry_energy'].mean()))
    rng=np.random.default_rng(CONFIG['seed'])
    boot=np.column_stack([rng.choice(np.flatnonzero(allocation==f),size=(CONFIG['bootstrap'],int((allocation==f).sum())))
                          for f in range(CONFIG['folds'])])
    comparisons={}
    for left,right in itertools.combinations(ARMS,2):
        a,b=all_data[left],all_data[right]
        if not np.array_equal(a['actual'],b['actual']): raise ValueError('Paired outcomes differ')
        delta=a['utility_crps']-b['utility_crps'];brier=(a['p_null']-(a['actual']<=0))**2-(b['p_null']-(b['actual']<=0))**2
        def summary(value):
            return dict(mean=np.mean(value,axis=0).tolist(),interval95=np.quantile(np.mean(value[boot],axis=1),[.025,.975],axis=0).tolist())
        policy=[]
        for k,row in enumerate(results[left]['policies']):
            j=ACTIONS.index(row['action']);weights=a['masks'][k]-b['masks'][k]
            policy.append(dict(action=row['action'],fraction=row['fraction'],ranking=row['ranking'],section=row['section'],
                net_gain_per_eligible=summary(weights*a['actual'][:,j]),
                false_activation_per_eligible=summary(weights*(a['actual'][:,j]<=0))))
        comparisons[left+'__minus__'+right]=dict(gamma_crps=summary(delta),null_brier=summary(brier),policy=policy)
    result=dict(complete=True,completed_utc=now(),n=n,models=results,paired=comparisons,
        final_opened=False,fifth_repeat_opened=False,formal_certificate=False,
        interval_scope='compound bootstrap within outer folds, fixed predictions/masks; does not include model-search, training-overlap or shared-batch uncertainty')
    write_json(root/'summary.json',result)
    lines=['# 639 对象统一折外比较','', '原四孔 DEV；原收益和合同不变。所有对象保留，五折内独立拟合；不是新的独立认证。','',
           '| 模型 | ADD_TWO CRPS↓ | NULL Brier↓ | 合并相关（描述性） | 折内秩关联 |',
           '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        m=results[arm]['actions'][2]
        lines.append(f"| {arm} | {m['gamma_crps']:.6f} | {m['null_brier']:.6f} | {m['spearman']} | {m['within_fold_rank_association']} |")
    lines+=['','## 25% 物理孔预算：ADD_TWO，按期望收益选择','',
            '| 模型 | 对象/孔 | 每选中对象净收益 | FDP | FPR |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        m=next(p for p in results[arm]['common_budget'] if p['action']=='Z1Z2' and p['fraction']==.25 and p['ranking']=='expected_gain')
        lines.append(f"| {arm} | {m['selected_n']}/{m['used_wells']} | {m['per_selected_net_gain']:.6f} | {m['fdp']:.4f} | {m['fpr']:.4f} |")
    lines+=['','完整配对区间、三个动作及各预算见 summary.json；不按最优结果选择模型或预算。',
            'GLOBAL 在每折内是同一分布，选择表为均匀子集期望；不同折的常数不能当成个体排序信息。',
            '全部置信区间是已拟合开发结果的条件式汇总，不覆盖训练集重叠、开发搜索和共享批次不确定性。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    event(root,'COMPLETE',n=n,final_opened=False,fifth_repeat_opened=False)


def execute(root, folds=None):
    root,manifest,ds=load_run(root)
    torch.set_num_threads(CONFIG['threads'])
    started=time.monotonic()
    event(root,'RUNNING',folds=folds if folds is not None else list(range(CONFIG['folds'])))
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            for record in manifest['folds']:
                if folds is None or record['fold'] in folds: execute_fold(root,manifest,ds,record)
            if all((root/'folds'/f'fold_{f}'/'complete.json').exists() for f in range(CONFIG['folds'])):
                summarize(root)
        event(root,'COMPLETE' if (root/'summary.json').exists() else 'REQUESTED_FOLDS_COMPLETE',elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        event(root,'FAILED',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','execute','summarize'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default=str(PROJECT/'runs/gram_probability_20260914_v1'))
    parser.add_argument('--folds',type=int,nargs='*')
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.output,args.reference)
    elif args.mode=='execute': execute(args.output,args.folds)
    else: summarize(args.output)


if __name__=='__main__': main()
