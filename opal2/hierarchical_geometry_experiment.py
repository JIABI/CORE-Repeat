"""Fixed-role conditional geometry: bounded residual mean and joint OOF errors.

This experiment does not identify a source/batch/plate generative decomposition.
It implements the proposed estimation changes on the existing exact Gram task.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import chi2, norm
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_experiment import _load_study_data
from .biology_kernel_evaluation import write_json, plain
from .gram_oof_experiment import now, event, train_g, decode_draws
from .gram_experiment import _schedule
from .gram_model import GramConditionalModel
from .gram_oof_ridge import fit_preprocessing, transform_input, transform_target
from .gram_simple_models import GramSimpleGaussian, _CenteredRidgePath, _fit_error_second_moment
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import evaluate_and_save, fit_score_scale
from .hierarchical_geometry import RidgeResidualMean, sample_joint_coordinates


PROJECT = Path(__file__).resolve().parents[1]
OLD_ARMS = ('GLOBAL_GEOMETRY', 'RIDGE_GEOMETRY', 'G_DIRECT', 'L_GRAM')
NEW_ARMS = ('G_OOF_COV', 'HR_RIDGE_COV', 'HR_OOF_COV')
ARMS = OLD_ARMS + NEW_ARMS
CONFIG = dict(seed=20260914, folds=5, samples=2000, bootstrap=2000,
    random_subsets=2000, threads=2,
    residual=dict(hidden_dim=32, correction_bound=.5, residual_penalty=.1, dropout=.1,
        batch_size=64, learning_rate=.0003, weight_decay=.0001,
        gradient_clip=5., max_epochs=200, min_epochs=40,
        validation_interval=5, patience_checks=8, min_delta=.00001,
        warmup_steps=30, min_learning_rate=.000003, report_interval=20))


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Use a fresh experiment directory')
    previous = json.loads((reference/'run_manifest.json').read_text())
    complete = json.loads((reference/'summary.json').read_text())
    if not complete['complete'] or complete['n'] != 639:
        raise ValueError('The complete preceding five-fold comparison is required')
    ds, split, scope = _load_study_data(previous['data_directory'])
    if (ds.ids.tolist() != previous['ids'] or ds.Y.shape != (639, 4, 3617)
            or ds.feature_names.tolist() != previous['feature_names']
            or {k: ds.ids[v].tolist() for k,v in split.items()} != previous['original_compound_ids']):
        raise ValueError('Opened cohort, coordinate order or original split changed')
    for flag in ('final_opened', 'fifth_repeat_opened', 'original_endpoint_changed', 'original_contract_changed'):
        if previous[flag] is not False:
            raise ValueError('Protected scope differs: '+flag)
    root.mkdir(parents=True)
    shutil.copy2(PROJECT/'protocols/historical/HIERARCHICAL_GEOMETRY_PLAN_20260914.md', root/'PROTOCOL.md')
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(reference/'fold_assignment.npz', root/'fold_assignment.npz')
    manifest = dict(created_utc=now(), reference_run=str(reference), source_snapshot=str(snapshot),
        config=CONFIG, arms=list(ARMS), data_directory=previous['data_directory'],
        ids=previous['ids'], folds=previous['folds'], feature_names=previous['feature_names'],
        feature_groups=previous['feature_groups'], original_compound_ids=previous['original_compound_ids'],
        data_shape=list(ds.Y.shape), scope=scope, final_opened=False, fifth_repeat_opened=False,
        original_endpoint_changed=False, original_contract_changed=False,
        original_split_files_changed=False, historical_dev=True, formal_certificate=False,
        model_scope='fixed-role conditional geometry estimator, not identified physical hierarchical variance',
        original_g_config=previous['config']['g'])
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', arms=list(ARMS), n=639)


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the recorded source snapshot and configuration')
    ds, split, _ = _load_study_data(manifest['data_directory'])
    if (ds.ids.tolist() != manifest['ids'] or ds.Y.shape != tuple(manifest['data_shape'])
            or ds.feature_names.tolist() != manifest['feature_names']
            or {k: ds.ids[v].tolist() for k,v in split.items()} != manifest['original_compound_ids']):
        raise ValueError('The opened cohort or original split changed')
    return root, manifest, ds


def restore_target(prediction, stats):
    result = np.asarray(prediction, float)*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
    if not np.isfinite(result).all():
        raise FloatingPointError('Native-coordinate prediction is nonfinite')
    return result


def covariance_from_native_errors(actual_native, predicted_native, stats):
    residuals = transform_target(actual_native, stats)-transform_target(predicted_native, stats)
    covariance, centered, bias, metadata = _fit_error_second_moment(residuals, include_bias=True)
    return covariance, dict(residuals=residuals, centered_covariance=centered, residual_mean=bias), metadata


def rebuild_inner_ridge(y_fit, raw_u_fit, prior_ridge, record):
    """Reconstruct the exact original nested fit, without the error holdout."""
    ii = np.asarray(record['outer_fit_indices'], int)
    jj = np.asarray(record['outer_validation_indices'], int)
    if set(ii)&set(jj) or sorted(np.r_[ii, jj]) != list(range(len(y_fit))):
        raise ValueError('Internal ridge fit and residual holdout must partition the fit objects')
    stats = fit_preprocessing(y_fit[ii], raw_u_fit[ii])
    path = _CenteredRidgePath(transform_input(y_fit[ii,0], stats), transform_target(raw_u_fit[ii], stats))
    coefficient, intercept = path.coefficients(record['selected_lambda'])
    native = restore_target(transform_input(y_fit[jj,0], stats)@coefficient+intercept, stats)
    expected = prior_ridge.audit_arrays['oof_native_predictions'][jj]
    if not np.allclose(native, expected, rtol=1e-9, atol=1e-10):
        raise ValueError('Reconstructed internal ridge differs from saved nested predictions')
    return ii, jj, stats, coefficient, intercept, native


@torch.no_grad()
def predict_g(model, inputs):
    model.eval()
    result = model(torch.as_tensor(inputs[:,:-1], dtype=torch.float32),
                   torch.as_tensor(inputs[:,-1:], dtype=torch.float32))
    return result.mean.double().numpy(), result.covariance_matrix.double().numpy()


@torch.no_grad()
def predict_hr(model, inputs):
    model.eval()
    return model(torch.as_tensor(inputs, dtype=torch.float64)).numpy()


def load_g(folder):
    payload = torch.load(Path(folder)/'best.pt', map_location='cpu', weights_only=True)
    model = GramConditionalModel(**payload['model_config'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return model, payload


def train_hr(folder, coefficient, intercept, inputs, targets, fit, valid, seed, ids):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG['residual']
    torch.manual_seed(seed)
    model = RidgeResidualMean(inputs.shape[1], coefficient, intercept, **{k:cfg[k] for k in
        ('hidden_dim','correction_bound','residual_penalty','dropout')}).double()
    if (folder/'training_complete.json').exists():
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(payload['state_dict'])
        return model, payload['epoch']
    if (folder/'history.jsonl').exists():
        raise RuntimeError('Interrupted HR training is preserved, not silently restarted')
    x, target = torch.tensor(inputs,dtype=torch.float64), torch.tensor(targets,dtype=torch.float64)
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    total = cfg['max_epochs']*math.ceil(len(fit)/cfg['batch_size'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _schedule(step,total,cfg))
    rng = torch.Generator(device='cpu').manual_seed(seed+31)
    best, significant, stale, best_epoch, steps = float('inf'), float('inf'), 0, 0, 0
    started = time.monotonic()
    event(folder,'TRAINING_STARTED',parameters=sum(p.numel() for p in parameters),fit_n=len(fit),inner_validation_n=len(valid))
    for epoch in range(cfg['max_epochs']+1):
        mse_sum, correction_sum, count = 0., 0., 0
        if epoch:
            model.train()
            order = np.asarray(fit)[torch.randperm(len(fit),generator=rng).numpy()]
            for first in range(0,len(order),cfg['batch_size']):
                ix = order[first:first+cfg['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(x[ix],target[ix])
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError('Nonfinite residual model loss')
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); steps += 1
                mse_sum += len(ix)*losses['mean_mse'].item()
                correction_sum += len(ix)*losses['correction_mse'].item()
                count += len(ix)
        row = dict(epoch=epoch,optimizer_steps=steps,train_mean_mse=mse_sum/count if count else None,
            train_correction_mse=correction_sum/count if count else None,
            learning_rate=optimizer.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started)
        if epoch%cfg['validation_interval'] == 0:
            model.eval()
            with torch.no_grad():
                losses = model.loss(x[valid],target[valid])
                fit_losses = model.loss(x[fit],target[fit])
            score = losses['mean_mse'].item()
            payload = dict(state_dict=deepcopy(model.state_dict()), model_config=model.config,
                epoch=epoch,optimizer_steps=steps,validation_u_mse=score,
                fit_ids=ids[fit].tolist(),validation_ids=ids[valid].tolist())
            if score < best:
                best, best_epoch = score, epoch
                torch.save(payload,folder/'best.pt')
            if score < significant-cfg['min_delta']:
                significant, stale = score, 0
            elif epoch:
                stale += 1
            torch.save(payload,folder/'last.pt')
            row.update(validation_u_mse=score,fit_u_mse=fit_losses['mean_mse'].item(),
                validation_correction_mse=losses['correction_mse'].item(),best_epoch=best_epoch,
                checks_without_improvement=stale)
            event(folder,'VALIDATED',**row)
        with (folder/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(row),allow_nan=False)+'\n')
        if epoch and epoch%cfg['report_interval']==0:
            write_json(folder/f'report_epoch_{epoch:04}.json',row)
        if epoch>=cfg['min_epochs'] and stale>=cfg['patience_checks']:
            break
    completion = dict(epoch=epoch,best_epoch=best_epoch,best_validation_u_mse=best,
        elapsed_seconds=time.monotonic()-started,stop_reason='patience' if stale>=cfg['patience_checks'] else 'epoch_cap')
    write_json(folder/'training_complete.json',completion)
    event(folder,'TRAINING_COMPLETE',**completion)
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True)['state_dict'])
    model.eval()
    return model, best_epoch


def gaussian_coordinate_diagnostics(folder, ids, actual, mean, covariance):
    """Predictive error checks, not a physical-variance or coverage certificate."""
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    actual, mean = np.asarray(actual,float), np.asarray(mean,float)
    if actual.shape != mean.shape or actual.shape != (len(ids),9):
        raise ValueError('One nine-dimensional prediction per compound is required')
    covariance = np.broadcast_to(np.asarray(covariance,float),(len(ids),9,9)).copy()
    scale = np.linalg.cholesky(covariance)
    residual = actual-mean
    whitened = np.linalg.solve(scale,residual[...,None])[...,0]
    mahal = np.square(whitened).sum(-1)
    per_object = np.square(residual).mean(-1)
    denominator = np.square(actual).mean()
    marginal_sd = np.sqrt(np.diagonal(covariance,axis1=-2,axis2=-1))
    logdet = 2*np.log(np.diagonal(scale,axis1=-2,axis2=-1)).sum(-1)
    nll = .5*(9*np.log(2*np.pi)+logdet+mahal)
    coverage = []
    for level in (.5,.8,.9,.95):
        bound = norm.ppf((1+level)/2)
        covered = np.abs(residual)<=bound*marginal_sd
        coverage.append(dict(level=level,marginal_coverage=float(covered.mean()),
            marginal_per_coordinate=covered.mean(0),mean_marginal_width=float((2*bound*marginal_sd).mean()),
            joint_ellipsoid_coverage=float((mahal<=chi2.ppf(level,9)).mean())))
    report = dict(n=len(ids),u_mean_mse=float(per_object.mean()),
        u_r2_vs_fit_mean=float(1-per_object.mean()/denominator) if denominator else None,
        per_coordinate_mse=np.square(residual).mean(0),mean_error=residual.mean(0),
        median_object_mse=float(np.median(per_object)),
        largest_object_error_fraction=float(per_object.max()/per_object.sum()) if per_object.sum() else None,
        mean_predictive_variance=float(np.trace(covariance,axis1=-2,axis2=-1).mean()/9),
        mean_mahalanobis_per_dimension=float(mahal.mean()/9),joint_u_nll=float(nll.mean()),
        whitened_mean=whitened.mean(0),whitened_second_moment=whitened.T@whitened/len(ids),
        coverage=coverage,formal_certificate=False,
        scope='Gaussian-coordinate predictive error; includes mean error, not identified measurement noise')
    np.savez_compressed(folder/'u_predictions.npz',ids=np.asarray(ids,str),actual_u=actual,
        mean_u=mean,covariance_u=covariance,scale_tril_u=scale,mahalanobis=mahal,whitened_errors=whitened)
    write_json(folder/'u_diagnostics.json',report)
    return report


def execute_fold(root,manifest,ds,record):
    f, seed = record['fold'], record['seed']
    folder = root/'folds'/f'fold_{f}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        return
    started = time.monotonic()
    previous = Path(manifest['reference_run'])/'folds'/f'fold_{f}'
    fit, valid, test = [np.asarray(record[k],int) for k in ('fit','inner_validation','test')]
    if (set(fit)&set(valid) or set(fit)&set(test) or set(valid)&set(test)
            or sorted(np.r_[fit,valid,test]) != list(range(len(ds)))):
        raise ValueError('Fit, selection and test must partition the cohort without overlap')
    actual_grams = profiles_to_gram(torch.tensor(ds.Y,dtype=torch.float64)).numpy()
    raw_u = gram_to_coordinates(torch.tensor(actual_grams)).numpy()
    actual = gram_gains(torch.tensor(actual_grams)).numpy()
    stats = fit_preprocessing(ds.Y[fit],raw_u[fit])
    old_stats = json.loads((previous/'preprocessing.json').read_text())
    for key in stats:
        if not np.array_equal(stats[key],old_stats[key]):
            raise ValueError('Outer preprocessing changed: '+key)
    write_json(folder/'preprocessing.json',dict(**stats,fit_ids=ds.ids[fit].tolist()))
    x = transform_input(ds.Y[:,0],stats)
    target = transform_target(raw_u,stats)
    ridge = GramSimpleGaussian.load(previous/'arms/RIDGE_GEOMETRY/fit.npz')
    g, g_payload = load_g(previous/'arms/G_DIRECT')

    # Old reported values are copied, never overwritten or recomputed.
    for arm in OLD_ARMS:
        dest = folder/'arms'/arm/'test'
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('metrics.json','predictions.npz'):
            if not (dest/name).exists():
                shutil.copy2(previous/'arms'/arm/'test'/name,dest/name)
        write_json(dest/'reuse.json',dict(reference=str(previous/'arms'/arm/'test'),reported_values_reused=True))
        if arm=='L_GRAM':
            continue
        if arm=='G_DIRECT':
            mean,covariance = predict_g(g,x[test])
        else:
            base = GramSimpleGaussian.load(previous/'arms'/arm/'fit.npz')
            mean,covariance = base.predict_mean(x[test]),base.covariance
        gaussian_coordinate_diagnostics(dest,ds.ids[test],target[test],mean,covariance)

    # Selection labels are the existing external inner validation only.
    joined = np.r_[fit,valid]
    hr, hr_epoch = train_hr(folder/'mean_fit',ridge.coefficient,ridge.intercept,
        x[joined],target[joined],np.arange(len(fit)),np.arange(len(fit),len(joined)),
        seed+401,ds.ids[joined])
    hr_mean = predict_hr(hr,x[test])
    g_mean,_ = predict_g(g,x[test])
    gaussian_coordinate_diagnostics(folder/'arms/HR_RIDGE_COV/test',ds.ids[test],target[test],hr_mean,ridge.covariance)

    g_oof = np.full((len(fit),9),np.nan)
    hr_oof = np.full_like(g_oof,np.nan)
    count = np.zeros(len(fit),int)
    internal_records = []
    for inner_record in ridge.metadata['outer_cv']:
        j = inner_record['outer_fold']
        inner_root = folder/'oof_fits'/f'inner_{j}'
        ii,jj,istats,coefficient,intercept,base_native = rebuild_inner_ridge(ds.Y[fit],raw_u[fit],ridge,inner_record)
        train_indices, check_indices = fit[ii], fit[jj]
        info = dict(fold=j,fit_ids=ds.ids[train_indices].tolist(),error_holdout_ids=ds.ids[check_indices].tolist(),
            selection_ids=ds.ids[valid].tolist(),selected_ridge_lambda=inner_record['selected_lambda'],
            preprocessing=istats)
        inner_root.mkdir(parents=True,exist_ok=True)
        if (inner_root/'oof_predictions.npz').exists():
            prior_scope = json.loads((inner_root/'fit_scope.json').read_text())
            if any(prior_scope[key] != value for key,value in info.items()):
                raise ValueError('Saved internal fit scope changed')
            info = prior_scope
            with np.load(inner_root/'oof_predictions.npz',allow_pickle=False) as saved:
                if not np.array_equal(saved['ids'],ds.ids[check_indices]):
                    raise ValueError('Saved inner prediction IDs changed')
                g_oof[jj],hr_oof[jj] = saved['g_native'],saved['hr_native']
            count[jj] += 1
            internal_records.append(info)
            continue
        write_json(inner_root/'fit_scope.json',info)
        event(root,'INTERNAL_FIT_STARTED',fold=f,inner_fold=j,fit_n=len(ii),error_holdout_n=len(jj))
        group = np.r_[train_indices,valid]
        ix = transform_input(ds.Y[group,0],istats)
        itarget = transform_target(raw_u[group],istats)
        local_fit, local_valid = np.arange(len(ii)),np.arange(len(ii),len(group))
        inner_seed = seed+1009*(j+1)
        ih, ih_epoch = train_hr(inner_root/'HR',coefficient,intercept,ix,itarget,
            local_fit,local_valid,inner_seed+401,ds.ids[group])
        query_x = transform_input(ds.Y[check_indices,0],istats)
        hr_oof[jj] = restore_target(predict_hr(ih,query_x),istats)
        del ih
        ig, ig_epoch = train_g(inner_root/'G',manifest['feature_groups'],ix,itarget,istats,
            local_fit,local_valid,actual[group],inner_seed,ds.ids[group])
        inner_g_mean,_ = predict_g(ig,query_x)
        g_oof[jj] = restore_target(inner_g_mean,istats)
        del ig
        count[jj] += 1
        info.update(hr_epoch=ih_epoch,g_epoch=ig_epoch)
        write_json(inner_root/'fit_scope.json',info)
        np.savez_compressed(inner_root/'oof_predictions.npz',ids=ds.ids[check_indices].astype(str),
            fit_local_indices=ii,error_local_indices=jj,g_native=g_oof[jj],hr_native=hr_oof[jj],ridge_native=base_native)
        internal_records.append(info)
        event(root,'INTERNAL_FIT_COMPLETE',fold=f,inner_fold=j,hr_epoch=ih_epoch,g_epoch=ig_epoch)
    if not np.array_equal(count,np.ones(len(fit),int)) or not np.isfinite(g_oof).all() or not np.isfinite(hr_oof).all():
        raise ValueError('Every fit compound needs exactly one finite nested error prediction')
    covariances, audits = {}, {}
    for name,native in (('G',g_oof),('HR',hr_oof)):
        covariance,arrays,audit = covariance_from_native_errors(raw_u[fit],native,stats)
        covariances[name],audits[name] = covariance,audit
        np.savez_compressed(folder/f'{name}_oof_error_fit.npz',ids=ds.ids[fit].astype(str),
            predicted_native=native,actual_native=raw_u[fit],covariance=covariance,oof_count=count,**arrays)
        write_json(folder/f'{name}_covariance_fit.json',dict(**audit,physical_variance_identified=False,
            selection='independent internal fits; shared external inner-validation set',folds=internal_records))
    np.savez_compressed(folder/'test_mean_identity.npz',ids=ds.ids[test].astype(str),
        g_original_mean=g_mean,g_oof_cov_mean=g_mean,hr_ridge_cov_mean=hr_mean,hr_oof_cov_mean=hr_mean)
    metric_scale = fit_score_scale(actual_grams[fit])
    for arm in NEW_ARMS:
        dest = folder/'arms'/arm/'test'
        if (dest/'metrics.json').exists():
            continue
        mean = g_mean if arm=='G_OOF_COV' else hr_mean
        covariance = (covariances['G'] if arm=='G_OOF_COV' else
                      ridge.covariance if arm=='HR_RIDGE_COV' else covariances['HR'])
        coordinate = gaussian_coordinate_diagnostics(dest,ds.ids[test],target[test],mean,covariance)
        sampled = sample_joint_coordinates(mean,covariance,CONFIG['samples'],seed+200000)
        draws,numerics = decode_draws(restore_target(sampled,stats),verify=True)
        event(root,'SCORING',fold=f,arm=arm)
        evaluate_and_save(dest,draws,actual_grams[test],ds.ids[test],
            metadata=dict(arm=arm,fold=f,mean_epoch=g_payload['epoch'] if arm=='G_OOF_COV' else hr_epoch,
                input='complete X and its log norm', covariance='original RIDGE' if arm=='HR_RIDGE_COV' else 'full nested OOF error second moment',
                covariance_replaces_original=True,physical_variance_components_identified=False,
                model_scope=manifest['model_scope'],u_diagnostics=coordinate,numerics=numerics,
                formal_certificate=False),
            train_actual_gains=actual[fit],score_scale=metric_scale,seed=seed,
            n_bootstrap=CONFIG['bootstrap'],n_random=CONFIG['random_subsets'])
        del draws,sampled
        event(root,'MODEL_COMPLETE',fold=f,arm=arm)
    write_json(folder/'complete.json',dict(completed_utc=now(),elapsed_seconds=time.monotonic()-started,
        g_mean_epoch=g_payload['epoch'],hr_mean_epoch=hr_epoch,fit_n=len(fit),test_n=len(test)))
    event(root,'FOLD_COMPLETE',fold=f,elapsed_seconds=time.monotonic()-started)


def execute(output,folds=None):
    root,manifest,ds = load_run(output)
    from .gram_oof_experiment import CONFIG as G_EXPERIMENT_CONFIG
    if manifest['original_g_config'] != G_EXPERIMENT_CONFIG['g']:
        raise ValueError('Original G fitting recipe changed')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root,'RUNNING',folds=folds if folds is not None else list(range(CONFIG['folds'])))
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            for record in manifest['folds']:
                if folds is None or record['fold'] in folds:
                    execute_fold(root,manifest,ds,record)
            if all((root/'folds'/f'fold_{f}'/'complete.json').exists() for f in range(CONFIG['folds'])):
                from .hierarchical_geometry_summary import summarize
                summarize(root)
        event(root,'COMPLETE' if (root/'summary.json').exists() else 'REQUESTED_FOLDS_COMPLETE',
            elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        event(root,'FAILED',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','execute','summarize'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default=str(PROJECT/'runs/gram_oof_20260914_v1'))
    parser.add_argument('--folds',type=int,nargs='*')
    args = parser.parse_args()
    if args.mode=='prepare':
        prepare(args.output,args.reference)
    elif args.mode=='execute':
        execute(args.output,args.folds)
    else:
        from .hierarchical_geometry_summary import summarize
        summarize(args.output)


if __name__=='__main__':
    main()
