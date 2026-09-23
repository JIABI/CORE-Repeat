"""Matched TRAIN-only input-scale intervention for independent kernel channels."""
from copy import deepcopy
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import event, now
from .gram_oof_ridge import transform_input, transform_target
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_evaluation import fit_score_scale, evaluate_and_save
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .lincs_biology_experiment import load_data, tensor_biology
from .gamma_supervised_loss import JointGammaCRPS
from .independent_biology_kernel import IndependentBiologyKernelMean
from .independent_biology_training import train_branch
from .independent_biology_experiment import load_base


PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL_ARMS = ('A_FROZEN', 'A_PLUS_OLD', 'A_PLUS_BIO')
MODES = dict(A_PLUS_OLD_SCALED='old_information', A_PLUS_BIO_SCALED='biology')
ARMS = ORIGINAL_ARMS + tuple(MODES)
PLAN = 'protocols/historical/SCALE_CALIBRATION_PLAN_20260915.md'


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Use a new directory; existing results are preserved')
    old = json.loads((reference/'run_manifest.json').read_text())
    if json.loads((reference/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('The unscaled three-arm experiment must be complete')
    if old['arms'] != list(ORIGINAL_ARMS) or len(old['folds']) != 5 or old['fixed_epochs'] != 30:
        raise ValueError('Unexpected unscaled reference experiment')
    if old['config']['samples'] != 10000 or old['config']['stage_epochs'] != 30:
        raise ValueError('Expected fixed epoch30 and10000 prediction draws')
    for record in old['folds']:
        for arm in ORIGINAL_ARMS:
            folder = reference/'folds'/f"fold_{record['fold']}"/'arms'/arm
            if not (folder/'evaluation/predictions.npz').is_file():
                raise FileNotFoundError(folder)
            if arm != 'A_FROZEN' and not (folder/'epoch0.pt').is_file():
                raise FileNotFoundError(folder/'epoch0.pt')
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/PLAN, root/'PROTOCOL.md')
    manifest = {key:deepcopy(old[key]) for key in ('ids','groups','folds','scopes','dataset',
                'data_directory','data_shape','feature_names','config')}
    manifest.update(created_utc=now(), reference_run=str(reference),
        frozen_base_reference_run=old['reference_run'], source_snapshot=str(snapshot),
        python_executable=sys.executable, arms=list(ARMS), modes=MODES,
        fixed_epochs=30, actual_checkpoint_epoch=30, new_branch_fits=10,
        sole_intervention='fixed TRAIN-supported initial local-vector scale, no centering',
        frozen_base=old['frozen_base'], endpoint_changed=False, original_endpoint_changed=False,
        original_contract_changed=False, final_opened=False, fifth_repeat_opened=False,
        jepa_active=False, covariance_updated_by_branch=False, reference_selection_changed=False,
        independent_new_holdout=False, formal_certificate=False,
        model_config={**deepcopy(old['model_config']), 'aggregation_scaling':'train_fixed', 'scale_max_gain':32.})
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=len(manifest['ids']), fits=10, epochs=30)
    return manifest


def copy_original_arm(source, destination):
    destination.mkdir(parents=True)
    (destination/'evaluation').mkdir()
    for name in ('predictions.npz', 'u_predictions.npz', 'metrics.json'):
        shutil.copy2(source/'evaluation'/name, destination/'evaluation'/name)
    for name in ('model_diagnostics.npz', 'history.jsonl', 'gamma_monitoring.jsonl',
                 'training_config.json', 'training_complete.json'):
        if (source/name).exists():
            shutil.copy2(source/name, destination/name)
    write_json(destination/'provenance.json', dict(source=str(source), reused_without_retraining=True,
        checkpoint_epoch=30, original_checkpoints_retained_at_source=True))


def score(folder, mean, covariance, stats, target, grams, ids, train_gains, scale, record, arm, cfg):
    diagnostic = gaussian_coordinate_diagnostics(folder, ids, target, mean, covariance)
    coordinates = sample_joint_coordinates(mean, covariance, cfg['samples'], record['seed']+200000)
    draws, numerical = decode_draws(restore_target(coordinates, stats), verify=True)
    return evaluate_and_save(folder, draws, grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=30,
            covariance='same frozen original RIDGE OOF joint covariance',
            u_diagnostics=diagnostic, numerics=numerical, formal_certificate=False,
            biological_information=arm=='A_PLUS_BIO_SCALED', fixed_TRAIN_input_scale=True),
        train_actual_gains=train_gains, score_scale=scale, seed=record['seed'],
        n_bootstrap=cfg['bootstrap'], n_random=cfg['random_subsets'])


def execute_fold(root, manifest, data, record, scope, grams, raw_u, gains):
    cfg = manifest['config']
    reference = Path(manifest['reference_run'])
    folder = root/'folds'/f"fold_{record['fold']}"
    folder.mkdir(parents=True)
    write_json(folder/'scope.json', scope)
    base, ridge, stats = load_base(manifest['frozen_base_reference_run'], record, scope)
    write_json(folder/'preprocessing.json', stats)
    torch.save(dict(model_config=base.config, state_dict=base.state_dict()), folder/'frozen_A.pt')
    original_fold = reference/'folds'/f"fold_{record['fold']}"
    if stats != json.loads((original_fold/'preprocessing.json').read_text()):
        raise ValueError('Preprocessing differs from unscaled controls')
    ids = data['ids']
    fit, valid, test = np.asarray(scope['commonbranchfit'], int), np.asarray(record['inner_validation'], int), np.asarray(record['test'], int)
    if np.intersect1d(fit, np.r_[valid,test]).size or ids[fit].tolist() != scope['commonbranchfit_ids']:
        raise ValueError('Scale FIT is not the original supervised FIT scope')
    x, u = transform_input(data['Y'][:,0], stats), transform_target(raw_u, stats)
    packed = base.bank.pack_information(torch.tensor(data['chem']), tensor_biology(data)).numpy()
    xt, pt, mt = torch.tensor(x[test]), torch.tensor(packed[test]), torch.tensor(data['chem_mask'][test])
    with torch.no_grad():
        baseline = base(xt,pt,mt).numpy()
    with np.load(original_fold/'arms/A_FROZEN/evaluation/u_predictions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['ids'],ids[test]) or not np.array_equal(z['mean_u'],baseline) or not np.array_equal(z['actual_u'],u[test]):
            raise ValueError('Restored A differs from the original controls')
    for arm in ORIGINAL_ARMS:
        copy_original_arm(original_fold/'arms'/arm, folder/'arms'/arm)
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'],stats['u_scale'],float(np.std(gains[fit,2],ddof=1)))
    score_scale = fit_score_scale(grams[np.asarray(record['fit'],int)])
    joined = np.r_[fit,valid]
    fit_local,valid_local = np.arange(len(fit)),np.arange(len(fit),len(joined))
    active_counts = {}
    for arm, mode in MODES.items():
        original_arm = arm.removesuffix('_SCALED')
        epoch0 = torch.load(original_fold/'arms'/original_arm/'epoch0.pt', map_location='cpu', weights_only=True)
        seed = record['seed']+cfg['branch_seed_offset']
        torch.manual_seed(seed)
        model = IndependentBiologyKernelMean(base, mode=mode, **manifest['model_config']).double()
        # The new fixed buffers cannot change random initialization or active capacity.
        for name,p in model.named_parameters():
            if p.requires_grad and not torch.equal(p,epoch0['state_dict'][name]):
                raise ValueError('Scaled arm initialization differs: '+name)
        if any(not torch.equal(value,epoch0['gamma_objective_state_dict'][key]) for key,value in objective.state_dict().items()):
            raise ValueError('Original joint objective changed')
        scale_metadata = model.fit_aggregation_scale(torch.tensor(x[fit]),torch.tensor(packed[fit]),
                    torch.tensor(data['chem_mask'][fit]),ids=ids[fit])
        fixed_buffers = {name:value.clone() for name,value in model.named_buffers() if name.startswith('aggregation_')}
        active_counts[arm] = sum(p.numel() for p in model.trainable_parameters())
        original_completion = json.loads((original_fold/'arms'/original_arm/'training_complete.json').read_text())
        if active_counts[arm] != original_completion['trainable_parameters']:
            raise ValueError('Scale calibration changed trainable capacity')
        event(root,'TRAINING',fold=record['fold'],arm=arm,epoch_budget=30,active_parameters=active_counts[arm])
        write_json(folder/(arm+'_aggregation_scale_pretrain.json'), scale_metadata)
        train_branch(folder/'arms'/arm,model,objective,x[joined],packed[joined],data['chem_mask'][joined],
                     u[joined],gains[joined,2],fit_local,valid_local,seed,ids[joined],cfg)
        write_json(folder/'arms'/arm/'aggregation_scale.json', scale_metadata)
        if any(not torch.equal(value,dict(model.named_buffers())[name]) for name,value in fixed_buffers.items()):
            raise ValueError('TRAIN-fitted fixed scale moved during training')
        # Same index order and Monte Carlo streams as the paired unscaled training.
        saved = torch.load(folder/'arms'/arm/'epoch30.pt',map_location='cpu',weights_only=True)
        original_last = torch.load(original_fold/'arms'/original_arm/'epoch30.pt',map_location='cpu',weights_only=True)
        for key in ('optimizer_steps','fit_ids','validation_ids','fixed_monitoring_seed'):
            if saved[key] != original_last[key]:
                raise ValueError('Training budget/scope mismatch: '+key)
        for key in ('order_rng_state','training_mc_rng_state','monitoring_rng_state'):
            if not torch.equal(saved[key],original_last[key]):
                raise ValueError('Training random stream mismatch: '+key)
        model.eval()
        with torch.no_grad():
            mean = model(xt,pt,mt).numpy()
            diag = model.diagnostics(xt,pt,mt,ids=ids[test])
            if not np.array_equal(model(xt,pt,mt,branch_enabled=False).numpy(),baseline):
                raise ValueError('Disabling scale-corrected branch does not restore A')
        restored = IndependentBiologyKernelMean.from_config(saved['model_config'])
        restored.load_state_dict(saved['state_dict']);restored.eval()
        with torch.no_grad():
            if not np.array_equal(restored(xt,pt,mt).numpy(),mean):
                raise ValueError('Stored checkpoint differs from evaluated model')
        arrays = {key:value.detach().numpy() for key,value in diag.items() if isinstance(value,torch.Tensor)}
        arrays.update(baseline_mean=baseline,mean=mean,support=arrays['channel_support'])
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz',ids=ids[test],**arrays)
        event(root,'SCORING',fold=record['fold'],arm=arm,samples=cfg['samples'])
        score(folder/'arms'/arm/'evaluation',mean,ridge.covariance,stats,u[test],grams[test],ids[test],
              gains[fit],score_scale,record,arm,cfg)
        event(root,'ARM_COMPLETE',fold=record['fold'],arm=arm,epoch=30)
    write_json(folder/'complete.json',dict(fold=record['fold'],n=len(test),arms=list(ARMS),
        branch_epochs=30,trainable_parameters=active_counts,baseline_reproduced=True))
    event(root,'FOLD_COMPLETE',fold=record['fold'])


def execute(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['arms'] != list(ARMS):
        raise ValueError('Execute the saved source snapshot and complete declared arms')
    if (root/'folds').exists():
        raise FileExistsError('Existing execution is not restarted or overwritten')
    cfg = manifest['config']
    torch.set_num_threads(cfg['threads'])
    started = time.monotonic()
    event(root,'RUNNING',fits=10,epochs=30)
    try:
        data,_ = load_data(manifest['data_directory'])
        if data['ids'].tolist()!=manifest['ids'] or data['groups'].tolist()!=manifest['groups']:
            raise ValueError('The declared identities or groups changed')
        with threadpool_limits(limits=cfg['threads']):
            grams = profiles_to_gram(torch.tensor(data['Y'])).numpy()
            raw_u = gram_to_coordinates(torch.tensor(grams)).numpy()
            gains = gram_gains(torch.tensor(grams)).numpy()
            for record,scope in zip(manifest['folds'],manifest['scopes']):
                execute_fold(root,manifest,data,record,scope,grams,raw_u,gains)
            event(root,'SUMMARIZING')
            from .independent_biology_scale_summary import summarize
            summarize(root)
        event(root,'COMPLETE',elapsed_seconds=time.monotonic()-started,epoch=30)
    except Exception as exc:
        event(root,'FAILED',elapsed_seconds=time.monotonic()-started,error=repr(exc))
        raise


def launch(output):
    root=Path(output).resolve()
    manifest=json.loads((root/'run_manifest.json').read_text())
    if (root/'process.json').exists():
        raise FileExistsError('Already launched')
    command=[manifest['python_executable'],'-u','-m','opal2.independent_biology_scale_experiment','execute','--output',str(root)]
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(manifest['config']['threads']),
             OPENBLAS_NUM_THREADS=str(manifest['config']['threads']),MKL_NUM_THREADS=str(manifest['config']['threads']))
    with (root/'stdout.log').open('xb') as stdout,(root/'stderr.log').open('xb') as stderr:
        process=subprocess.Popen(command,cwd=manifest['source_snapshot'],env=env,
                                 stdout=stdout,stderr=stderr,start_new_session=True)
    awake=None
    if Path('/usr/bin/caffeinate').is_file():
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(process.pid)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True).pid
    write_json(root/'process.json',dict(pid=process.pid,caffeinate_pid=awake,launched_utc=now(),command=command))
    print(json.dumps(dict(pid=process.pid,output=str(root))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','execute','launch'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference')
    args=parser.parse_args()
    if args.command=='prepare':
        if not args.reference: parser.error('prepare requires --reference')
        prepare(args.output,args.reference)
    elif args.command=='launch': launch(args.output)
    else: execute(args.output)
