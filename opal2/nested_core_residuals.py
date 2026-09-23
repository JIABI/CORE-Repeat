"""Honest MODEL_FIT residuals from the complete existing STATE50 recipe.

This module never loads an earlier fold model. Each error fold is predicted by
a fresh ridge -> HR -> old-generic A (30 epochs) -> STATE50 (50 epochs) fit.
The only population accepted is the caller's MODEL_FIT population. Production
CORE weights and its outer query predictions are not modified.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split

from .biology_kernel_evaluation import write_json
from .gamma_supervised_loss import JointGammaCRPS
from .gram_geometry import gram_to_coordinates, profiles_to_gram, gram_gains
from .gram_oof_ridge import transform_input, transform_target
from .hierarchical_geometry_experiment import train_hr, restore_target
from .hierarchical_stability_ridge import fit_validation_ridge
from .kernel_final_reference import reference_allocation, fit_reference_bank
from .kernel_final_training import train_branch as train_a
from .lincs_biology_experiment import CONFIG as A_CONFIG, tensor_biology
from .mechanism_response_kernel import MechanismResponseBank, MechanismResponseKernelMean
from .state_biology_kernel import StateBiologyKernelMean
from .state_biology_training import train_branch as train_state


ARRAY_KEYS = ('ids', 'groups', 'Y', 'chem', 'chem_mask', 'target',
              'target_mask', 'moa', 'moa_mask')
REFERENCE_COUNT = 64


def _validated_data(data):
    missing = set(ARRAY_KEYS)-set(data)
    if missing:
        raise ValueError('Missing declared MODEL_FIT arrays: '+str(sorted(missing)))
    result = {key: np.asarray(data[key]) for key in ARRAY_KEYS}
    ids, groups = result['ids'].astype(str), result['groups'].astype(str)
    n = len(ids)
    if (ids.shape != (n,) or groups.shape != (n,) or not n or
            len(set(ids)) != n or any(not v for v in ids) or any(not v for v in groups)):
        raise ValueError('Unique MODEL_FIT IDs and known aligned chemical groups required')
    result.update(ids=ids, groups=groups)
    y, chemical = result['Y'], result['chem']
    if y.ndim != 3 or y.shape[:2] != (n, 4) or y.shape[-1] < 8:
        raise ValueError('Complete four-role MODEL_FIT profiles with at least eight features required')
    if chemical.ndim != 2 or len(chemical) != n:
        raise ValueError('Decision-time chemistry must be MODEL_FIT aligned')
    for key in ('chem_mask', 'target_mask', 'moa_mask'):
        if result[key].shape != (n,) or result[key].dtype != bool:
            raise ValueError('Boolean availability mask required: '+key)
    for key in ('Y', 'chem', 'target', 'moa'):
        if len(result[key]) != n or not np.isfinite(result[key]).all():
            raise ValueError('Finite aligned MODEL_FIT values required: '+key)
    for key in ('target', 'moa'):
        if result[key].ndim != 2:
            raise ValueError('Annotation profiles must be two-dimensional')
    return result


def plan_nested_core_folds(data, metadata, seed, n_splits=3):
    """Metadata-only chemical-group split, with the original 64-reference rule."""
    data = _validated_data(data)
    ids, groups = data['ids'], data['groups']
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('A nonnegative integer seed is required')
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError('At least two grouped residual folds required')
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError('Too few chemical groups')
    records = []
    for fold, (pool_groups, held_groups) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(unique)):
        local_seed = int(seed+10000*fold)
        fit_groups, valid_groups = train_test_split(pool_groups, test_size=.2,
                                                    random_state=local_seed+91)
        fit = np.flatnonzero(np.isin(groups, unique[fit_groups]))
        valid = np.flatnonzero(np.isin(groups, unique[valid_groups]))
        held = np.flatnonzero(np.isin(groups, unique[held_groups]))
        allocation = reference_allocation(ids, fit, data['chem'], data['chem_mask'],
            metadata['chemical'], local_seed+A_CONFIG['reference_seed_offset'], REFERENCE_COUNT)
        references = np.flatnonzero(np.isin(ids, allocation['reference_ids']))
        ref_groups = set(groups[references])
        common = fit[~np.isin(groups[fit], list(ref_groups))]
        if len(common) < 10:
            raise ValueError('Insufficient branch-fitting objects after reference group exclusion')
        for a, b in ((fit, valid), (fit, held), (valid, held), (common, references)):
            if set(groups[a]) & set(groups[b]):
                raise RuntimeError('A chemical group crosses nested fitting/validation/reference/query roles')
        record = dict(fold=fold, seed=local_seed, reference_ids=allocation['reference_ids'],
            reference_selection=allocation['allocation_rule'], reference_group_exclusion=True,
            references_also_in_backbone_fit=True)
        for name, rows in (('fit', fit), ('inner_validation', valid), ('heldout', held),
                           ('commonbranchfit', common), ('references', references)):
            record[name] = rows.tolist()
            record[name+'_ids'] = ids[rows].tolist()
        records.append(record)
    counts = np.zeros(len(ids), int)
    for record in records:
        counts[record['heldout']] += 1
    if not np.array_equal(counts, np.ones(len(ids), int)):
        raise RuntimeError('Every MODEL_FIT object must have exactly one honest error prediction')
    return records


def _take(data, rows):
    return {key: data[key][rows] for key in ARRAY_KEYS}


def _train_complete_core(train_data, metadata, record, folder):
    """Fit the full recipe on inner fit+validation only; no query arrays accepted."""
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Preserve nested fits; supply a new output directory')
    folder.mkdir(parents=True)
    ids, y = train_data['ids'], train_data['Y']
    lookup = {v: i for i, v in enumerate(ids)}
    rows = {name: np.asarray([lookup[v] for v in record[name+'_ids']], int)
            for name in ('fit', 'inner_validation', 'commonbranchfit', 'references')}
    originalfit, valid, fit, refs = [rows[k] for k in ('fit', 'inner_validation', 'commonbranchfit', 'references')]
    if set(np.r_[originalfit, valid]) != set(range(len(ids))):
        raise ValueError('Trainer accepts only inner fitting and validation populations')
    if set(ids) & set(record['heldout_ids']):
        raise ValueError('Held-out MODEL_FIT identities reached the trainer')
    grams = profiles_to_gram(torch.tensor(y)).numpy()
    raw_u = gram_to_coordinates(torch.tensor(grams)).numpy()
    gains = gram_gains(torch.tensor(grams)).numpy()
    seed = record['seed']
    ridge, stats = fit_validation_ridge(y[originalfit], raw_u[originalfit], y[valid], raw_u[valid],
                                        seed, groups=train_data['groups'][originalfit])
    ridge.save(folder/'ridge.npz')
    write_json(folder/'preprocessing.json', stats)
    x, u = transform_input(y[:, 0], stats), transform_target(raw_u, stats)
    hr, hr_epoch = train_hr(folder/'HR_fit', ridge.coefficient, ridge.intercept,
                            x, u, originalfit, valid, seed+401, ids)
    hr.eval().requires_grad_(False)
    local = fit_reference_bank(x, train_data['chem'], train_data['chem_mask'], ids,
        ids[originalfit].tolist(), ids[fit].tolist(), record['reference_ids'], metadata['chemical'])
    biology = {k: train_data[k] for k in ('target', 'target_mask', 'moa', 'moa_mask')}
    bank = MechanismResponseBank.fit(local, biology, ids, ids[fit],
        target_names=metadata['target_names'], moa_names=metadata['moa_names'])
    packed = bank.pack_information(torch.tensor(train_data['chem']), tensor_biology(train_data)).numpy()
    objective = JointGammaCRPS(ridge.covariance, stats['u_center'], stats['u_scale'],
                               float(np.std(gains[fit, 2], ddof=1)))
    joined = np.r_[fit, valid]
    local_fit, local_valid = np.arange(len(fit)), np.arange(len(fit), len(joined))
    branch_seed = seed+A_CONFIG['branch_seed_offset']
    torch.manual_seed(branch_seed)
    base = MechanismResponseKernelMean(hr, bank, mode='old_generic',
        incremental_penalty=A_CONFIG['incremental_penalty'], hidden_dim=A_CONFIG['hidden_dim']).double()
    train_a(folder/'A_OLD_GENERIC', base, objective, x[joined], packed[joined],
        train_data['chem_mask'][joined], u[joined], gains[joined, 2], local_fit, local_valid,
        branch_seed, ids[joined], deepcopy(A_CONFIG), 'weighted')
    base.eval()
    cfg = dict(deepcopy(A_CONFIG), stage_epochs=50)
    torch.manual_seed(branch_seed)
    state = StateBiologyKernelMean(base, mode='state_only', hidden_dim=16,
        support_shrinkage=2., raw_increment_bound=1., incremental_penalty=.1,
        aggregation_scaling='train_fixed', scale_max_gain=32.).double()
    # The stored reference order, not numeric row sorting, defines its anchors.
    references = np.asarray([lookup[v] for v in record['reference_ids']], int)
    state.fit_input_state(torch.tensor(x[fit]), ids=ids[fit],
        reference_x=torch.tensor(x[references]), reference_ids=record['reference_ids'])
    state.fit_aggregation_scale(torch.tensor(x[fit]), torch.tensor(packed[fit]),
                                torch.tensor(train_data['chem_mask'][fit]), ids=ids[fit])
    train_state(folder/'STATE50', state, objective, x[joined], packed[joined],
        train_data['chem_mask'][joined], u[joined], gains[joined, 2], local_fit, local_valid,
        branch_seed, ids[joined], cfg)
    state.eval()
    report = dict(recipe='full ridge + validation-selected HR + A_OLD_GENERIC30 + STATE50',
        fitted_ids=ids.tolist(), backbone_fit_ids=ids[originalfit].tolist(),
        validation_ids=ids[valid].tolist(), branch_fit_ids=ids[fit].tolist(),
        reference_ids=record['reference_ids'], hr_best_epoch=int(hr_epoch),
        a_checkpoint_epoch=30, state_checkpoint_epoch=50,
        references_also_in_backbone_fit=True, heldout_outcomes_supplied=False,
        prior_fold_models_loaded=False, main_core_weights_changed=False)
    write_json(folder/'complete.json', report)
    return state, stats, np.asarray(ridge.covariance), report


def predict_raw_geometry(model, stats, first_well, chemistry, chemistry_mask, biology):
    """Prediction API intentionally has no future-well/realized-geometry input."""
    x = transform_input(first_well, stats)
    bio = {key: torch.as_tensor(value, dtype=torch.bool if key.endswith('_mask') else torch.float64)
           for key, value in biology.items()}
    with torch.no_grad():
        packed = model.bank.pack_information(torch.as_tensor(chemistry, dtype=torch.float64), bio)
        mean = model(torch.tensor(x), packed, torch.as_tensor(chemistry_mask, dtype=torch.bool)).numpy()
    return restore_target(mean, stats)


def fit_nested_core_residuals(data, metadata, output, seed, n_splits=3):
    """Return native-coordinate honest residuals for only supplied MODEL_FIT.

    The caller must subset to its original outer MODEL_FIT before calling.
    Native geometry predictions/covariances are retained, because each inner
    model has its own training-only coordinate system. Conversion to the frozen
    outer model's standardized frame is a separate caller operation.
    """
    data = _validated_data(data)
    root = Path(output)
    if root.exists():
        raise FileExistsError('Nested cross-fitting never overwrites earlier fits')
    records = plan_nested_core_folds(data, metadata, seed, n_splits)
    root.mkdir(parents=True)
    write_json(root/'plan.json', dict(ids=data['ids'].tolist(), groups=data['groups'].tolist(),
        seed=int(seed), folds=records, n_splits=n_splits, reference_count=REFERENCE_COUNT,
        a_epochs=30, state_epochs=50, surrogate_mean_used=False,
        data_scope='caller-supplied outer MODEL_FIT only'))
    n, start = len(data['ids']), time.monotonic()
    means, covariances = np.empty((n, 9)), np.empty((n, 9, 9))
    membership, predictions = np.full(n, -1, int), np.zeros(n, int)
    reports, stats_by_fold = [], []
    for record in records:
        fit = np.asarray(record['fit'], int)
        valid, held = np.asarray(record['inner_validation'], int), np.asarray(record['heldout'], int)
        model, stats, covariance, report = _train_complete_core(_take(data, np.r_[fit, valid]),
            metadata, record, root/f"inner_{record['fold']}")
        bio = {k: data[k][held] for k in ('target', 'target_mask', 'moa', 'moa_mask')}
        means[held] = predict_raw_geometry(model, stats, data['Y'][held, 0],
            data['chem'][held], data['chem_mask'][held], bio)
        scale = np.asarray(stats['u_scale'])
        covariances[held] = covariance*scale[:, None]*scale[None, :]
        membership[held] = record['fold']; predictions[held] += 1
        stats_by_fold.append(stats); reports.append(report)
        del model
    if not np.array_equal(predictions, np.ones(n, int)):
        raise RuntimeError('Nested prediction count differs from one')
    # Future held-out profiles are used here only after every prediction exists.
    raw_target = gram_to_coordinates(profiles_to_gram(torch.tensor(data['Y']))).numpy()
    residual = raw_target-means
    result = dict(ids=data['ids'], groups=data['groups'], raw_mean=means,
        raw_target=raw_target, raw_residual=residual, raw_covariance=covariances,
        error_fold=membership, prediction_count=predictions)
    if any(not np.isfinite(result[k]).all() for k in ('raw_mean', 'raw_target', 'raw_residual', 'raw_covariance')):
        raise FloatingPointError('Nonfinite full-recipe nested residuals')
    np.savez_compressed(root/'residuals.npz', **result)
    summary = dict(n=n, n_splits=n_splits, elapsed_seconds=time.monotonic()-start,
        folds=reports, training_preprocessing=stats_by_fold,
        raw_geometry_mse=float(np.square(residual).mean()),
        recipe='complete STATE50; no shortened stages or linear surrogate',
        main_core_weights_changed=False, independent_certification=False)
    write_json(root/'summary.json', summary)
    return dict(**result, summary=summary, records=records)
