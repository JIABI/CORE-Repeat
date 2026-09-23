"""Full 50-epoch, five-fold state versus state-biological relation comparison."""
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
from .independent_biology_experiment import load_base
from .independent_biology_scale_experiment import copy_original_arm
from .state_biology_kernel import StateBiologyKernelMean
from .state_biology_training import train_branch


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_FROZEN', 'BIO_TEMPLATE50', 'STATE50', 'STATE_BIO50')
NEW_ARMS = ARMS[1:]
PLAN = 'protocols/historical/STATE_BIOLOGY_PLAN_20260916.md'


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Use a new run directory')
    old = json.loads((reference/'run_manifest.json').read_text())
    if json.loads((reference/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('The fixed-scale source must be complete')
    if len(old['folds']) != 5 or len(old['ids']) != 1188 or old['fixed_epochs'] != 30:
        raise ValueError('Unexpected original LINCS source')
    if old['config']['samples'] != 10000:
        raise ValueError('Original full joint sample count changed')
    # Preserve old artifacts; refuse rather than silently pruning disk contents.
    if shutil.disk_usage(root.parent).free < 8*1024**3:
        raise OSError('At least 8 GiB free space is required for this complete run')
    for record in old['folds']:
        fold = reference/'folds'/f"fold_{record['fold']}"/'arms'
        for name in ('A_FROZEN/evaluation/predictions.npz', 'A_PLUS_BIO_SCALED/epoch0.pt',
                     'A_PLUS_BIO_SCALED/epoch30.pt'):
            if not (fold/name).is_file():
                raise FileNotFoundError(fold/name)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/PLAN, root/'PROTOCOL.md')
    manifest = {k: deepcopy(old[k]) for k in ('ids','groups','folds','scopes','dataset','data_directory','data_shape','feature_names','config')}
    manifest['config']['stage_epochs'] = 50
    common = dict(hidden_dim=16, support_shrinkage=2., raw_increment_bound=1., incremental_penalty=.1, scale_max_gain=32.)
    manifest.update(created_utc=now(), reference_run=str(reference),
        frozen_base_reference_run=old['frozen_base_reference_run'], source_snapshot=str(snapshot),
        python_executable=sys.executable, arms=list(ARMS), fixed_epochs=50, actual_checkpoint_epoch=50,
        new_branch_fits=15, primary_comparison=['STATE_BIO50','STATE50'],
        model_configs={'BIO_TEMPLATE50': dict(common, mode='biology', aggregation_scaling='train_fixed'),
                       'STATE50': dict(common, mode='state_only'), 'STATE_BIO50': dict(common, mode='state_biology')},
        expected_active_parameters={'BIO_TEMPLATE50':1794,'STATE50':6770,'STATE_BIO50':6770},
        frozen_base='complete original A_OLD_GENERIC epoch30', endpoint_changed=False,
        original_endpoint_changed=False, original_contract_changed=False, final_opened=False,
        fifth_repeat_opened=False, jepa_active=False, covariance_updated_by_branch=False,
        reference_selection_changed=False, independent_new_holdout=False, formal_certificate=False,
        biological_comparison='same-capacity state-modulated X relation kernels versus target/MoA relation kernels; not a nested additive information ablation')
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=len(manifest['ids']), fits=15, epochs=50)
    return manifest


def score(folder, mean, covariance, stats, target, grams, ids, gains, scale, record, arm, cfg):
    diagnostic = gaussian_coordinate_diagnostics(folder, ids, target, mean, covariance)
    coordinates = sample_joint_coordinates(mean, covariance, cfg['samples'], record['seed']+200000)
    draws, numerical = decode_draws(restore_target(coordinates, stats), verify=True)
    return evaluate_and_save(folder, draws, grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=cfg['stage_epochs'],
            covariance='unchanged original RIDGE OOF covariance', u_diagnostics=diagnostic,
            numerics=numerical, formal_certificate=False, state_conditioned=arm in ('STATE50','STATE_BIO50'),
            biological_information=arm in ('BIO_TEMPLATE50','STATE_BIO50')),
        train_actual_gains=gains, score_scale=scale, seed=record['seed'],
        n_bootstrap=cfg['bootstrap'], n_random=cfg['random_subsets'])


def execute_fold(root, manifest, data, record, scope, grams, raw_u, gains):
    cfg, source = manifest['config'], Path(manifest['reference_run'])
    folder = root/'folds'/f"fold_{record['fold']}"
    folder.mkdir(parents=True)
    write_json(folder/'scope.json', scope)
    base, ridge, stats = load_base(manifest['frozen_base_reference_run'], record, scope)
    write_json(folder/'preprocessing.json', stats)
    torch.save(dict(model_config=base.config, state_dict=base.state_dict()), folder/'frozen_A.pt')
    ids = data['ids']
    fit, valid, test = np.asarray(scope['commonbranchfit'],int), np.asarray(record['inner_validation'],int), np.asarray(record['test'],int)
    if ids[fit].tolist() != scope['commonbranchfit_ids'] or np.intersect1d(fit, np.r_[valid,test]).size:
        raise ValueError('Original branch scopes changed')
    x, target = transform_input(data['Y'][:,0],stats), transform_target(raw_u,stats)
    packed = base.bank.pack_information(torch.tensor(data['chem']),tensor_biology(data)).numpy()
    xt, pt, mt = torch.tensor(x[test]), torch.tensor(packed[test]), torch.tensor(data['chem_mask'][test])
    with torch.no_grad():
        baseline = base(xt,pt,mt).numpy()
    original_fold = source/'folds'/f"fold_{record['fold']}"
    with np.load(original_fold/'arms/A_FROZEN/evaluation/u_predictions.npz') as z:
        if not np.array_equal(z['ids'],ids[test]) or not np.array_equal(z['mean_u'],baseline) or not np.array_equal(z['actual_u'],target[test]):
            raise ValueError('Frozen baseline reconstruction differs')
    copy_original_arm(original_fold/'arms/A_FROZEN',folder/'arms/A_FROZEN')
    objective = JointGammaCRPS(ridge.covariance,stats['u_center'],stats['u_scale'],float(np.std(gains[fit,2],ddof=1)))
    score_scale = fit_score_scale(grams[np.asarray(record['fit'],int)])
    joined = np.r_[fit,valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit),len(joined))
    index = {v:i for i,v in enumerate(ids)}
    ref_ids = list(base.bank.config['anchor_ids'])
    references = np.asarray([index[v] for v in ref_ids])
    if set(ref_ids) != set(scope['reference_ids']) or np.intersect1d(references,np.r_[fit,valid,test]).size:
        raise ValueError('Original reference set or disjoint scope changed')
    initial_state_parameters, state_buffers, stream_reference = None, None, None
    capacity, matches = {}, {}
    for arm in NEW_ARMS:
        seed = record['seed']+cfg['branch_seed_offset']
        torch.manual_seed(seed)
        klass = IndependentBiologyKernelMean if arm == 'BIO_TEMPLATE50' else StateBiologyKernelMean
        model = klass(base, **manifest['model_configs'][arm]).double()
        if arm != 'BIO_TEMPLATE50':
            state_metadata = model.fit_input_state(torch.tensor(x[fit]), ids=ids[fit],
                                                   reference_x=torch.tensor(x[references]), reference_ids=ref_ids)
            write_json(folder/(arm+'_state_pretrain.json'),state_metadata)
            active = {k:p.clone().detach() for k,p in model.named_parameters() if p.requires_grad}
            if initial_state_parameters is None:
                initial_state_parameters = active
                state_buffers = {k:v.clone() for k,v in model.named_buffers() if k.startswith('state_')}
            else:
                if active.keys() != initial_state_parameters.keys() or any(not torch.equal(v,initial_state_parameters[k]) for k,v in active.items()):
                    raise ValueError('State arms differ in trainable initialization')
                if any(not torch.equal(v,dict(model.named_buffers())[k]) for k,v in state_buffers.items()):
                    raise ValueError('State arms differ in fitted state representation')
        scale_metadata = model.fit_aggregation_scale(torch.tensor(x[fit]),torch.tensor(packed[fit]),torch.tensor(data['chem_mask'][fit]),ids=ids[fit])
        capacity[arm] = sum(p.numel() for p in model.trainable_parameters())
        if capacity[arm] != manifest['expected_active_parameters'][arm]:
            raise ValueError('Unexpected active parameter count')
        event(root,'TRAINING',fold=record['fold'],arm=arm,epoch_budget=50,active_parameters=capacity[arm])
        train_branch(folder/'arms'/arm,model,objective,x[joined],packed[joined],data['chem_mask'][joined],
                     target[joined],gains[joined,2],fit_local,valid_local,seed,ids[joined],cfg)
        write_json(folder/'arms'/arm/'aggregation_scale.json',scale_metadata)
        if arm != 'BIO_TEMPLATE50':
            write_json(folder/'arms'/arm/'input_state.json',state_metadata)
        saved = torch.load(folder/'arms'/arm/'epoch50.pt',map_location='cpu',weights_only=True)
        rng = {k:saved[k] for k in ('order_rng_state','training_mc_rng_state','monitoring_rng_state')}
        if stream_reference is None:
            stream_reference = rng
        elif any(not torch.equal(v,stream_reference[k]) for k,v in rng.items()):
            raise ValueError('Same-budget training streams differ')
        if arm == 'BIO_TEMPLATE50':
            original30 = torch.load(original_fold/'arms/A_PLUS_BIO_SCALED/epoch30.pt',map_location='cpu',weights_only=True)
            current30 = torch.load(folder/'arms'/arm/'epoch30.pt',map_location='cpu',weights_only=True)
            exact = original30['state_dict'].keys() == current30['state_dict'].keys() and all(
                torch.equal(v,current30['state_dict'][k]) for k,v in original30['state_dict'].items())
            if not exact:
                raise ValueError('Template50 did not reproduce the original first30 epochs')
            matches[arm] = dict(original_first30_reproduced=True)
        with torch.no_grad():
            mean = model(xt,pt,mt).numpy()
            diag = model.diagnostics(xt,pt,mt,ids=ids[test])
            if not np.array_equal(model(xt,pt,mt,branch_enabled=False).numpy(),baseline):
                raise ValueError('Disabling branch does not recover A')
        restored = klass.from_config(saved['model_config'])
        restored.load_state_dict(saved['state_dict']); restored.eval()
        with torch.no_grad():
            if not np.array_equal(restored(xt,pt,mt).numpy(),mean):
                raise ValueError('Actual epoch50 checkpoint differs from scored prediction')
        arrays = {k:v.detach().numpy() for k,v in diag.items() if isinstance(v,torch.Tensor)}
        arrays.update(baseline_mean=baseline,mean=mean,support=arrays['channel_support'])
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz',ids=ids[test],**arrays)
        event(root,'SCORING',fold=record['fold'],arm=arm,samples=cfg['samples'])
        score(folder/'arms'/arm/'evaluation',mean,ridge.covariance,stats,target[test],grams[test],ids[test],
              gains[fit],score_scale,record,arm,cfg)
        event(root,'ARM_COMPLETE',fold=record['fold'],arm=arm,epoch=50)
    write_json(folder/'complete.json',dict(fold=record['fold'],arms=list(ARMS),branch_epochs=50,
        trainable_parameters=capacity,baseline_reproduced=True,state_initializations_equal=True,
        state_transforms_equal=True,training_random_streams_equal=True,template_checks=matches))
    event(root,'FOLD_COMPLETE',fold=record['fold'])


def execute(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['arms'] != list(ARMS):
        raise ValueError('Execute the frozen source snapshot and full declared arms')
    if (root/'folds').exists():
        raise FileExistsError('Do not overwrite a prior execution')
    cfg, started = manifest['config'], time.monotonic()
    torch.set_num_threads(cfg['threads'])
    event(root,'RUNNING',fits=15,epochs=50)
    try:
        data,_ = load_data(manifest['data_directory'])
        if data['ids'].tolist() != manifest['ids'] or data['groups'].tolist() != manifest['groups']:
            raise ValueError('Original cohort changed')
        with threadpool_limits(limits=cfg['threads']):
            grams = profiles_to_gram(torch.tensor(data['Y'])).numpy()
            raw_u = gram_to_coordinates(torch.tensor(grams)).numpy()
            gains = gram_gains(torch.tensor(grams)).numpy()
            for record,scope in zip(manifest['folds'],manifest['scopes']):
                execute_fold(root,manifest,data,record,scope,grams,raw_u,gains)
            event(root,'SUMMARIZING')
            from .state_biology_summary import summarize
            summarize(root)
        event(root,'COMPLETE',elapsed_seconds=time.monotonic()-started,epoch=50)
    except Exception as exc:
        event(root,'FAILED',elapsed_seconds=time.monotonic()-started,error=repr(exc))
        raise


def launch(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if (root/'process.json').exists():
        raise FileExistsError('Already launched')
    command = [manifest['python_executable'],'-u','-m','opal2.state_biology_experiment','execute','--output',str(root)]
    env = dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(manifest['config']['threads']),
               OPENBLAS_NUM_THREADS=str(manifest['config']['threads']),MKL_NUM_THREADS=str(manifest['config']['threads']))
    with (root/'stdout.log').open('xb') as stdout,(root/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command,cwd=manifest['source_snapshot'],env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    awake = None
    if Path('/usr/bin/caffeinate').is_file():
        awake = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(process.pid)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True).pid
    write_json(root/'process.json',dict(pid=process.pid,caffeinate_pid=awake,launched_utc=now(),command=command))
    print(json.dumps(dict(pid=process.pid,output=str(root))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','execute','launch'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--reference',default=str(PROJECT/'runs/lincs_independent_biology_scale_20260915_v1'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output,args.reference)
    else:
        globals()[args.command](args.output)
