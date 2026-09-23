"""Independent biological correction using the unchanged LINCS five folds."""
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
from .gram_simple_models import GramSimpleGaussian
from .gram_evaluation import fit_score_scale, evaluate_and_save
from .gram_factor_verified import decode_draws
from .hierarchical_geometry import sample_joint_coordinates
from .hierarchical_geometry_experiment import restore_target, gaussian_coordinate_diagnostics
from .lincs_biology_experiment import load_data, tensor_biology
from .mechanism_response_kernel import MechanismResponseKernelMean
from .gamma_supervised_loss import JointGammaCRPS
from .independent_biology_kernel import IndependentBiologyKernelMean
from .independent_biology_training import train_branch


PROJECT = Path(__file__).resolve().parents[1]
ARMS = ('A_FROZEN', 'A_PLUS_OLD', 'A_PLUS_BIO')
MODES = dict(A_PLUS_OLD='old_information', A_PLUS_BIO='biology')
PLAN = 'protocols/historical/INDEPENDENT_BIOLOGY_PLAN_20260915.md'


def prepare(output, reference):
    root, reference = Path(output).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Use a new run directory')
    old = json.loads((reference/'run_manifest.json').read_text())
    if json.loads((reference/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('A completed original LINCS experiment is required')
    if len(old['folds']) != 5 or old['fixed_epochs'] != 30 or old['config']['samples'] != 10000:
        raise ValueError('Unexpected original experiment')
    for record in old['folds']:
        folder = reference/'folds'/f"fold_{record['fold']}"
        for name in ('arms/A_OLD_GENERIC/epoch30.pt', 'ridge.npz', 'preprocessing.json',
                     'arms/A_OLD_GENERIC/evaluation/predictions.npz'):
            if not (folder/name).is_file():
                raise FileNotFoundError(folder/name)
    cfg = deepcopy(old['config'])
    cfg.update(stage_epochs=30, hidden_dim=16, support_shrinkage=2., raw_increment_bound=1.)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/PLAN, root/'PROTOCOL.md')
    manifest = {k: deepcopy(old[k]) for k in ('ids', 'groups', 'folds', 'scopes', 'dataset',
                'data_directory', 'data_shape', 'feature_names')}
    manifest.update(created_utc=now(), reference_run=str(reference), source_snapshot=str(snapshot),
        python_executable=sys.executable, config=cfg, arms=list(ARMS), modes=MODES,
        fixed_epochs=30, actual_checkpoint_epoch=30, new_branch_fits=10,
        frozen_base='complete original A_OLD_GENERIC epoch30 per fold',
        endpoint_changed=False, original_endpoint_changed=False, original_contract_changed=False, final_opened=False,
        fifth_repeat_opened=False, jepa_active=False, covariance_updated_by_branch=False,
        reference_selection_changed=False, independent_new_holdout=False, formal_certificate=False,
        model_config=dict(hidden_dim=16, support_shrinkage=2., raw_increment_bound=1., incremental_penalty=.1))
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=len(manifest['ids']), fits=10, epochs=30)
    return manifest


def load_base(reference, record, scope):
    folder = Path(reference)/'folds'/f"fold_{record['fold']}"
    payload = torch.load(folder/'arms/A_OLD_GENERIC/epoch30.pt', map_location='cpu', weights_only=True)
    if payload['epoch'] != 30 or payload['fit_ids'] != scope['commonbranchfit_ids'] or payload['validation_ids'] != record['inner_validation_ids']:
        raise ValueError('Frozen A checkpoint scope mismatch')
    base = MechanismResponseKernelMean.from_config(payload['model_config'])
    base.load_state_dict(payload['state_dict']); base.eval().requires_grad_(False)
    if base.mode != 'old_generic':
        raise ValueError('Frozen baseline must be original-information generic A')
    return base, GramSimpleGaussian.load(folder/'ridge.npz'), json.loads((folder/'preprocessing.json').read_text())


def score(folder, mean, covariance, stats, target, grams, ids, train_gains, scale, record, arm, cfg):
    diagnostic = gaussian_coordinate_diagnostics(folder, ids, target, mean, covariance)
    coordinates = sample_joint_coordinates(mean, covariance, cfg['samples'], record['seed']+200000)
    draws, numerical = decode_draws(restore_target(coordinates, stats), verify=True)
    return evaluate_and_save(folder, draws, grams, ids,
        metadata=dict(arm=arm, actual_checkpoint_epoch=30,
            covariance='same frozen original RIDGE OOF joint covariance',
            u_diagnostics=diagnostic, numerics=numerical, formal_certificate=False,
            biological_information=arm=='A_PLUS_BIO'),
        train_actual_gains=train_gains, score_scale=scale, seed=record['seed'],
        n_bootstrap=cfg['bootstrap'], n_random=cfg['random_subsets'])


def execute_fold(root, manifest, data, record, scope, grams, raw_u, gains):
    cfg, reference = manifest['config'], Path(manifest['reference_run'])
    folder = root/'folds'/f"fold_{record['fold']}"
    folder.mkdir(parents=True)
    write_json(folder/'scope.json', scope)
    base, ridge, stats = load_base(reference, record, scope)
    write_json(folder/'preprocessing.json', stats)
    torch.save(dict(model_config=base.config, state_dict=base.state_dict()), folder/'frozen_A.pt')
    ids = data['ids']
    fit, valid, test = np.asarray(scope['commonbranchfit'], int), np.asarray(record['inner_validation'], int), np.asarray(record['test'], int)
    x, u = transform_input(data['Y'][:, 0], stats), transform_target(raw_u, stats)
    packed = base.bank.pack_information(torch.tensor(data['chem']), tensor_biology(data)).numpy()
    xt, pt, mt = torch.tensor(x[test]), torch.tensor(packed[test]), torch.tensor(data['chem_mask'][test])
    with torch.no_grad():
        baseline = base(xt, pt, mt).numpy()
    old_eval = reference/'folds'/f"fold_{record['fold']}"/'arms/A_OLD_GENERIC/evaluation'
    with np.load(old_eval/'u_predictions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['ids'], ids[test]) or not np.array_equal(z['mean_u'], baseline) or not np.array_equal(z['actual_u'], u[test]):
            raise ValueError('Restored A does not reproduce original held-out predictions')
    baseline_folder = folder/'arms/A_FROZEN'
    (baseline_folder/'evaluation').mkdir(parents=True)
    for name in ('predictions.npz', 'u_predictions.npz', 'metrics.json'):
        shutil.copy2(old_eval/name, baseline_folder/'evaluation'/name)
    np.savez_compressed(baseline_folder/'model_diagnostics.npz', ids=ids[test], mean=baseline, baseline_mean=baseline)
    write_json(baseline_folder/'provenance.json', dict(source=str(old_eval), checkpoint_epoch=30, identical_frozen_baseline=True))
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'], float(np.std(gains[fit, 2], ddof=1)))
    original_objective = torch.load(reference/'folds'/f"fold_{record['fold']}"/'gamma_objective_state.pt', map_location='cpu', weights_only=True)
    if any(not torch.equal(v, objective.state_dict()[k]) for k, v in original_objective.items()):
        raise ValueError('Original joint objective differs')
    scale = fit_score_scale(grams[np.asarray(record['fit'], int)])
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    initial, active_counts = None, {}
    for arm, mode in MODES.items():
        seed = record['seed']+cfg['branch_seed_offset']
        torch.manual_seed(seed)
        model = IndependentBiologyKernelMean(base, mode=mode, **manifest['model_config']).double()
        parameters = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
        if initial is None:
            initial = parameters
        elif set(initial) != set(parameters) or any(not torch.equal(v, initial[k]) for k, v in parameters.items()):
            raise ValueError('Independent arms do not have identical parameter initialization')
        active_counts[arm] = sum(p.numel() for p in parameters.values())
        event(root, 'TRAINING', fold=record['fold'], arm=arm, epoch_budget=30, active_parameters=active_counts[arm])
        train_branch(folder/'arms'/arm, model, objective, x[joined], packed[joined], data['chem_mask'][joined],
                     u[joined], gains[joined, 2], fit_local, valid_local, seed, ids[joined], cfg)
        model.eval()
        with torch.no_grad():
            mean = model(xt, pt, mt).numpy()
            diag = model.diagnostics(xt, pt, mt, ids=ids[test])
            if not np.array_equal(model(xt, pt, mt, branch_enabled=False).numpy(), baseline):
                raise ValueError('Disabling a trained independent branch does not restore A')
        saved = torch.load(folder/'arms'/arm/'epoch30.pt', map_location='cpu', weights_only=True)
        restored = IndependentBiologyKernelMean.from_config(saved['model_config'])
        restored.load_state_dict(saved['state_dict']); restored.eval()
        with torch.no_grad():
            if not np.array_equal(restored(xt, pt, mt).numpy(), mean):
                raise ValueError('Saved epoch30 does not reproduce evaluated independent predictions')
        arrays = {k: v.detach().numpy() for k, v in diag.items() if isinstance(v, torch.Tensor)}
        arrays['baseline_mean'] = baseline
        arrays['mean'] = mean
        # The model exports one positive-reference support flag per channel.
        arrays['support'] = arrays['channel_support']
        np.savez_compressed(folder/'arms'/arm/'model_diagnostics.npz', ids=ids[test], **arrays)
        event(root, 'SCORING', fold=record['fold'], arm=arm, samples=cfg['samples'])
        score(folder/'arms'/arm/'evaluation', mean, ridge.covariance, stats, u[test], grams[test], ids[test],
              gains[fit], scale, record, arm, cfg)
        event(root, 'ARM_COMPLETE', fold=record['fold'], arm=arm, epoch=30)
    write_json(folder/'complete.json', dict(fold=record['fold'], n=len(test), arms=list(ARMS),
        branch_epochs=30, trainable_parameters=active_counts, baseline_reproduced=True))
    event(root, 'FOLD_COMPLETE', fold=record['fold'])


def execute(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['arms'] != list(ARMS):
        raise ValueError('Execute the prepared source snapshot and declared arms')
    if (root/'folds').exists():
        raise FileExistsError('Existing execution is not restarted or overwritten')
    cfg = manifest['config']
    torch.set_num_threads(cfg['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', fits=10, epochs=30)
    try:
        data, metadata = load_data(manifest['data_directory'])
        if data['ids'].tolist() != manifest['ids'] or data['groups'].tolist() != manifest['groups']:
            raise ValueError('Original LINCS identities or chemical groups changed')
        with threadpool_limits(limits=cfg['threads']):
            grams = profiles_to_gram(torch.tensor(data['Y'])).numpy()
            raw_u = gram_to_coordinates(torch.tensor(grams)).numpy()
            gains = gram_gains(torch.tensor(grams)).numpy()
            for record, scope in zip(manifest['folds'], manifest['scopes']):
                execute_fold(root, manifest, data, record, scope, grams, raw_u, gains)
            event(root, 'SUMMARIZING')
            from .independent_biology_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started, epoch=30)
    except Exception as exc:
        event(root, 'FAILED', elapsed_seconds=time.monotonic()-started, error=repr(exc))
        raise


def launch(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if (root/'process.json').exists():
        raise FileExistsError('Run has already been launched')
    command = [manifest['python_executable'], '-u', '-m', 'opal2.independent_biology_experiment', 'execute', '--output', str(root)]
    env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(manifest['config']['threads']),
               OPENBLAS_NUM_THREADS=str(manifest['config']['threads']), MKL_NUM_THREADS=str(manifest['config']['threads']))
    with (root/'stdout.log').open('xb') as stdout, (root/'stderr.log').open('xb') as stderr:
        process = subprocess.Popen(command, cwd=manifest['source_snapshot'], env=env,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
    awake = None
    if Path('/usr/bin/caffeinate').is_file():
        awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True).pid
    write_json(root/'process.json', dict(pid=process.pid, caffeinate_pid=awake, launched_utc=now(), command=command))
    print(json.dumps(dict(pid=process.pid, output=str(root))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute', 'launch'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference')
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.reference:
            parser.error('prepare requires --reference')
        prepare(args.output, args.reference)
    elif args.command == 'launch':
        launch(args.output)
    else:
        execute(args.output)
