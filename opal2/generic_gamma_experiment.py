"""Matched generic chemical basis under the existing Gamma-supervised objective."""
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
from .gram_evaluation import fit_score_scale
from .hierarchical_stability_experiment import validate_partition, _check_cohort
from .conditional_response_kernel import LocalResponseBank, ConditionalResponseKernelMean
from .conditional_response_experiment import score_branch
from .geometry_kernel_replacement_experiment import load_frozen_fold, predict_branch
from .gamma_supervised_experiment import CONFIG, train_supervised, monitor_checkpoints
from .gamma_supervised_loss import JointGammaCRPS


PROJECT = Path(__file__).resolve().parents[1]
NEW_ARM = 'M_CONDITIONAL_GENERIC_GAMMA'
ARMS = ('A_HR', 'F_CONDITIONAL_GENERIC', 'J_GEOMETRY_CONTROL', 'K_GEOMETRY_GAMMA_CRPS', NEW_ARM)


def check_initialization(model, generic_saved, structured_saved):
    """Same complete generic starting state and shared learnable initialization."""
    if model.mode != 'conditional_generic' or generic_saved['model_config']['mode'] != 'conditional_generic':
        raise ValueError('The new arm must use the existing generic basis')
    state = model.state_dict()
    if state.keys() != generic_saved['state_dict'].keys() or any(
            not torch.equal(value, generic_saved['state_dict'][key]) for key, value in state.items()):
        raise ValueError('Generic initialization differs from historical F')
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(parameter, structured_saved['state_dict'][name]):
            raise ValueError('Generic and structured arms have different learnable initialization: '+name)
    return dict(exact_historical_generic_state=True, exact_structured_initial_trainable_parameters=True,
                mode=model.mode, trainable_parameters=sum(p.numel() for p in model.trainable_parameters()))


def prepare(output, gamma_reference, sampling_run):
    root, gamma_reference, sampling_run = map(lambda p: Path(p).resolve(), (output, gamma_reference, sampling_run))
    if root.exists():
        raise FileExistsError('Use a fresh generic-Gamma directory')
    prior = json.loads((gamma_reference/'run_manifest.json').read_text())
    if json.loads((gamma_reference/'status.json').read_text())['state'] != 'COMPLETE' or prior['config'] != CONFIG:
        raise ValueError('A completed matching Gamma-supervision run is required')
    sampling = json.loads((sampling_run/'summary.json').read_text())
    if sampling.get('complete') is not True or sampling.get('n') != len(prior['ids']):
        raise ValueError('The complete fixed-cohort sampling summary is required')
    # The sampling checker writes this small explicit continuation record.
    continuation = json.loads((sampling_run/'continuation.json').read_text())
    if continuation.get('source') != str(gamma_reference) or continuation.get('proceed_to_basis_contrast') is not True:
        raise ValueError('The frozen-model sampling check has not met its numerical continuation condition')
    conditional = Path(prior['historical_reference_run'])
    old = json.loads((conditional/'run_manifest.json').read_text())
    for key in ('ids', 'folds', 'reference_run', 'data_directory'):
        if prior[key] != old[key]:
            raise ValueError('Historical allocation differs: '+key)
    ds, split, _ = _load_study_data(prior['data_directory'])
    _check_cohort(ds, split, prior)
    for record in prior['folds']:
        validate_partition(record, len(ds))
        load_frozen_fold(prior['reference_run'], record)
    root.mkdir(parents=True)
    snapshot = root/'source_snapshot'
    for name in ('opal2', 'tests'):
        shutil.copytree(PROJECT/name, snapshot/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(PROJECT/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(PROJECT/'protocols/historical/KERNEL_GAMMA_FOLLOWUP_PLAN_20260914.md', root/'PROTOCOL.md')
    manifest = deepcopy(prior)
    manifest.update(created_utc=now(), source_snapshot=str(snapshot), python_executable=sys.executable,
        gamma_reference_run=str(gamma_reference), conditional_reference_run=str(conditional),
        sampling_stability_run=str(sampling_run), sampling_continuation=continuation,
        arms=list(ARMS), new_arms=[NEW_ARM], config=deepcopy(CONFIG),
        control_reference_arm='F_CONDITIONAL_GENERIC', control_exact_reproduction_required=False,
        architecture_changed=False, chemical_basis_changed_in_M=True,
        inherited_arms=['A_HR', 'F_CONDITIONAL_GENERIC', 'J_GEOMETRY_CONTROL', 'K_GEOMETRY_GAMMA_CRPS'],
        matched_loss_to_K=True, paired_training_rng_to_K=True,
        training_initialization='exact historical F epoch0; trainable parameters match K epoch0',
        comparison_scope='chemical response basis and its fixed TRAIN RMS; all other branches unchanged')
    from .generic_gamma_summary import _validate_manifest
    _validate_manifest(manifest)
    write_json(root/'run_manifest.json', manifest)
    event(root, 'PREPARED', n=639, new_branch_fits=5, stage_epochs=30)
    return manifest


def load_run(output):
    root = Path(output).resolve()
    manifest = json.loads((root/'run_manifest.json').read_text())
    if PROJECT != Path(manifest['source_snapshot']) or manifest['config'] != CONFIG:
        raise ValueError('Execute the frozen source snapshot and configuration')
    prior = json.loads((Path(manifest['gamma_reference_run'])/'run_manifest.json').read_text())
    for key in ('ids', 'folds', 'reference_run', 'data_directory'):
        if manifest[key] != prior[key]:
            raise ValueError('Allocation or dataset changed: '+key)
    from .generic_gamma_summary import _validate_manifest
    _validate_manifest(manifest)
    ds, split, _ = _load_study_data(manifest['data_directory'])
    _check_cohort(ds, split, manifest)
    return root, manifest, ds


def execute_fold(root, manifest, ds, record, grams, raw_u, gains):
    fold = record['fold']
    folder = root/'folds'/f'fold_{fold}'
    gamma_source = Path(manifest['gamma_reference_run'])/'folds'/f'fold_{fold}'
    conditional = Path(manifest['conditional_reference_run'])/'folds'/f'fold_{fold}'
    folder.mkdir(parents=True)
    fit, valid, test = validate_partition(record, len(ds))
    _, stats, ridge, hr, _ = load_frozen_fold(manifest['reference_run'], record)
    x, target = transform_input(ds.Y[:,0], stats), transform_target(raw_u, stats)
    bank = LocalResponseBank.load(gamma_source/'response_bank.pt').double()
    if set(bank.config['fitting_ids']) != set(ds.ids[fit].tolist()):
        raise ValueError('The response bank uses different fitting compounds')
    bank.save(folder/'response_bank.pt')
    scale = float(np.std(gains[fit,2], ddof=1))
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'], scale)
    previous_objective = torch.load(gamma_source/'gamma_objective_state.pt', map_location='cpu', weights_only=True)
    if objective.state_dict().keys() != previous_objective.keys() or any(
            not torch.equal(value, previous_objective[key]) for key, value in objective.state_dict().items()):
        raise ValueError('The Gamma loss distribution/scaling differs from K')
    torch.save(objective.state_dict(), folder/'gamma_objective_state.pt')
    shutil.copy2(gamma_source/'gamma_objective.json', folder/'gamma_objective.json')
    for arm in ARMS[:-1]:
        source = conditional if arm == 'F_CONDITIONAL_GENERIC' else gamma_source
        shutil.copytree(source/'arms'/arm, folder/'arms'/arm)
    seed = record['seed']+CONFIG['branch_seed_offset']
    torch.manual_seed(seed)
    model = ConditionalResponseKernelMean(hr, bank, mode='conditional_generic',
        incremental_penalty=CONFIG['incremental_penalty'], hidden_dim=CONFIG['hidden_dim']).double()
    original_generic = torch.load(conditional/'arms/F_CONDITIONAL_GENERIC/epoch0.pt', map_location='cpu', weights_only=True)
    original_structured = torch.load(gamma_source/'arms/K_GEOMETRY_GAMMA_CRPS/epoch0.pt', map_location='cpu', weights_only=True)
    write_json(folder/'initialization.json', check_initialization(model, original_generic, original_structured))
    joined = np.r_[fit, valid]
    fit_local, valid_local = np.arange(len(fit)), np.arange(len(fit), len(joined))
    event(root, 'GENERIC_GAMMA_TRAINING', fold=fold, arm=NEW_ARM, epochs=30)
    model = train_supervised(folder/'arms'/NEW_ARM, model, objective, x[joined], ds.chem[joined],
        ds.chem_mask[joined], target[joined], gains[joined,2], fit_local, valid_local, seed, ds.ids[joined])
    event(root, 'CHECKPOINT_MONITORING', fold=fold, arm=NEW_ARM)
    monitor_checkpoints(folder/'arms'/NEW_ARM, objective, x[joined], ds.chem[joined], ds.chem_mask[joined],
        target[joined], gains[joined,2], fit_local, valid_local, seed)
    mean = predict_branch(model, x[test], ds.chem[test], ds.chem_mask[test])
    saved = torch.load(folder/'arms'/NEW_ARM/'epoch30.pt', map_location='cpu', weights_only=True)
    restored = ConditionalResponseKernelMean.from_config(saved['model_config'])
    restored.load_state_dict(saved['state_dict'])
    if not np.array_equal(mean, predict_branch(restored, x[test], ds.chem[test], ds.chem_mask[test])):
        raise ValueError('Generic epoch30 checkpoint cannot reproduce its scored mean')
    with torch.no_grad():
        diagnostic = model.diagnostics(torch.as_tensor(x[test], dtype=torch.float64),
            torch.as_tensor(ds.chem[test], dtype=torch.float64), torch.as_tensor(ds.chem_mask[test]), ids=ds.ids[test].tolist())
    values = {key:value.detach().cpu().numpy() for key,value in diagnostic.items() if isinstance(value, torch.Tensor)}
    values['raw'] = values['kernel_raw']
    np.savez_compressed(folder/'arms'/NEW_ARM/'model_diagnostics.npz', ids=ds.ids[test],
        block_names=np.asarray(['chemical', 'morphology', 'scalar']), **values)
    counts = dict(trainable=sum(p.numel() for p in model.trainable_parameters()), total=sum(p.numel() for p in model.parameters()))
    event(root, 'SCORING_EPOCH30', fold=fold, arm=NEW_ARM)
    score_branch(folder/'arms'/NEW_ARM/'evaluation', mean, ridge, stats, target[test], grams[test], ds.ids[test],
        gains[fit], fit_score_scale(grams[fit]), record['seed'], NEW_ARM, counts)
    write_json(folder/'complete.json', dict(fold=fold, test_n=len(test), actual_checkpoint_epoch=30,
        completed_utc=now(), formal_certificate=False, original_covariance_preserved=True))
    event(root, 'FOLD_COMPLETE', fold=fold, actual_checkpoint_epoch=30)


def execute(output):
    root, manifest, ds = load_run(output)
    if (root/'folds').exists():
        raise FileExistsError('This run has already started')
    torch.set_num_threads(CONFIG['threads'])
    started = time.monotonic()
    event(root, 'RUNNING', new_branch_fits=5, epochs=30)
    try:
        with threadpool_limits(limits=CONFIG['threads']):
            grams = profiles_to_gram(torch.as_tensor(ds.Y, dtype=torch.float64)).numpy()
            raw_u = gram_to_coordinates(torch.as_tensor(grams)).numpy()
            gains = gram_gains(torch.as_tensor(grams)).numpy()
            for record in manifest['folds']:
                execute_fold(root, manifest, ds, record, grams, raw_u, gains)
            from .generic_gamma_summary import summarize
            summarize(root)
        event(root, 'COMPLETE', elapsed_seconds=time.monotonic()-started, stopped_at_epoch=30, further_training_started=False)
    except Exception as error:
        event(root, 'FAILED', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic()-started)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'execute', 'summarize'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--gamma-reference', default=str(PROJECT/'runs/gamma_supervised_epoch30_20260914_v1'))
    parser.add_argument('--sampling-run', default=str(PROJECT/'runs/gamma_sampling_stability_20260914_v1'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.gamma_reference, args.sampling_run)
    elif args.mode == 'execute':
        execute(args.output)
    else:
        from .generic_gamma_summary import summarize
        summarize(args.output)


if __name__ == '__main__':
    main()
