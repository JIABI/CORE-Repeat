"""Honest REF-only training records for the optional EU R3 error adapters.

The frozen CORE mean and its original TRAIN-derived base scatter are inputs.
Within REF, each three-fold held-out group gets a complete CORE distribution
fitted with other REF groups. Its future geometry is read only for its training
label, after its distribution and biological reference summaries are fixed.
No CORE network is trained, and no external CAL/QUERY outcome is accessed.
"""
from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from .biology_kernel_evaluation import write_json
from .eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from .gram_oof_ridge import transform_target
from .joint_contrast_scale import contrast_projector, projected_energy


def _result(records, report):
    """Expose complete persisted records plus the adapter-facing named views."""
    names = report['biological_names']
    biology = {key:records['biology_'+key] for key in ('values', 'support', 'support_by_relation')}
    random = {key:records['random_biology_'+key] for key in ('values', 'support', 'support_by_relation')}
    biology['names'], random['names'] = list(names), list(names)
    return dict(records=records, fold_records=report['fold_records'],
                biological_names=names, report=report, energies=records['energies'],
                biology=biology, random_biology=random)


def load_adapter_training(output, *, expected_ids=None):
    """Read a completed fixed input run; caller controls reuse of its protocol."""
    output = Path(output)
    report = json.loads((output/'summary.json').read_text())
    if not report.get('complete'):
        raise ValueError('Reference training records are not complete')
    with np.load(output/'records.npz', allow_pickle=False) as stored:
        records = {key:stored[key].copy() for key in stored.files}
    np.testing.assert_array_equal(records['prediction_count'], np.ones(len(records['ids']), int))
    np.testing.assert_array_equal(records['ids'], np.asarray(report['ids']))
    if expected_ids is not None:
        np.testing.assert_array_equal(records['ids'], np.asarray(expected_ids, str))
    return _result(records, report)


def plan_reference_folds(ids, groups, ref_rows, seed):
    """Three group-held-out folds, with seeded 2:1 covariance/calibration groups."""
    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    supplied = np.asarray(ref_rows)
    if ids.ndim != 1 or groups.shape != ids.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique object ids and aligned chemical groups required")
    if supplied.ndim != 1 or supplied.dtype.kind not in "iu":
        raise ValueError("REF rows must be one-dimensional integer indices")
    rows = supplied.astype(np.int64, copy=True)
    if (len(rows) != len(np.unique(rows)) or np.any(rows < 0)
            or np.any(rows >= len(ids)) or len(np.unique(groups[rows])) < 9):
        raise ValueError("REF requires valid distinct rows and at least nine groups")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("A nonnegative integer seed is required")
    outside = np.ones(len(ids), bool); outside[rows] = False
    if set(groups[rows]) & set(groups[outside]):
        raise ValueError("A chemical group crosses REF and an external role")
    plans = []
    for fold, (remaining, heldout) in enumerate(GroupKFold(3).split(rows, groups=groups[rows])):
        available = np.unique(groups[rows[remaining]])
        permutation = np.random.default_rng(int(seed)+fold).permutation(available)
        cut = 2*len(permutation)//3
        cov_groups, cal_groups = permutation[:cut], permutation[cut:]
        if len(cov_groups) < 3 or len(cal_groups) < 2:
            raise ValueError("Each inner fold needs >=3 covariance and >=2 radial groups")
        cov = rows[remaining[np.isin(groups[rows[remaining]], cov_groups)]]
        cal = rows[remaining[np.isin(groups[rows[remaining]], cal_groups)]]
        h = rows[heldout]
        blocks = [set(groups[k]) for k in (cov, cal, h)]
        if any(blocks[i] & blocks[j] for i in range(3) for j in range(i)):
            raise RuntimeError("A group crosses an inner reference role")
        if set(np.r_[cov, cal, h]) != set(rows):
            raise RuntimeError("Inner reference assignment does not cover REF")
        plans.append(dict(fold=fold, covariance_rows=cov, calibration_rows=cal,
                          heldout_rows=h, heldout_positions=heldout))
    return plans


def build_adapter_training(data, metadata, ref_rows, mean_u, raw_targets, stats,
                           base_scatter, training_logamp_sd, output, seed, *,
                           amplitude_edges):
    """Return all-REF records in the exact supplied ``ref_rows`` order.

    ``amplitude_edges`` are frozen MODEL_TRAIN log-norm quintile boundaries.
    ``base_scatter`` must already come from original MODEL_TRAIN, not the REF
    observations assembled here. Caller owns that external provenance. The
    returned two energies are geometric contrast/remainder targets, not an
    identified physical shared/independent variance decomposition.
    """
    from .eu_r3_biology import build_eu_biology_features

    ids, groups = np.asarray(data['ids'], str), np.asarray(data['groups'], str)
    rows = np.asarray(ref_rows)
    plans = plan_reference_folds(ids, groups, rows, seed)
    rows = rows.astype(np.int64, copy=True)
    means = np.asarray(mean_u, float)
    if means.shape != (len(ids), 9) or not np.isfinite(means[rows]).all():
        raise ValueError("Frozen standardized-u means must align with all objects")
    # Index before conversion: external future targets may be missing/inaccessible.
    raw_ref = np.asarray(raw_targets[rows], float)
    if raw_ref.shape != (len(rows), 9) or not np.isfinite(raw_ref).all():
        raise ValueError("REF needs aligned finite native geometry targets")
    scale, center = np.asarray(stats['u_scale'], float), np.asarray(stats['u_center'], float)
    if (scale.shape != (9,) or center.shape != (9,) or np.any(scale <= 0)
            or not np.isfinite(scale).all() or not np.isfinite(center).all()):
        raise ValueError("Frozen target preprocessing must contain nine valid coordinates")
    base = np.asarray(base_scatter, float)
    if base.shape != (9, 9) or not np.isfinite(base).all() or not np.allclose(base, base.T):
        raise ValueError("Original TRAIN base scatter must be a finite symmetric 9x9 matrix")
    np.linalg.cholesky(base)
    if not np.isfinite(training_logamp_sd) or training_logamp_sd < 0:
        raise ValueError("Frozen TRAIN amplitude spread must be finite and nonnegative")
    edges = np.asarray(amplitude_edges, float)
    if edges.shape != (4,) or not np.isfinite(edges).all() or np.any(np.diff(edges) < 0):
        raise ValueError("Four ordered TRAIN amplitude quintile boundaries are required")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve the existing adapter training records")
    output.mkdir(parents=True)
    position = {int(row): i for i, row in enumerate(rows)}
    target_u = transform_target(raw_ref, stats)
    residual_u = target_u-means[rows]
    raw_residual = raw_ref-(means[rows]*scale+center)
    n = len(rows)
    records = dict(ids=ids[rows], groups=groups[rows], ref_rows=rows,
        mean_u=means[rows].copy(), raw_mean=means[rows]*scale+center,
        raw_residual=raw_residual, energies=np.empty((n, 2)),
        scatter_u=np.empty((n, 9, 9)), covariance_u=np.empty((n, 9, 9)),
        raw_covariance=np.empty((n, 9, 9)), fold=np.full(n, -1, int),
        prediction_count=np.zeros(n, int))
    fold_records, biological_names = [], None

    def legal_inputs(take, include_mean=False):
        value = {key: np.asarray(data[key])[take] for key in ('ids', 'groups', 'chem')}
        value['X'] = np.asarray(data['Y'][take, 0], float)
        if include_mean:
            value['mean_u'] = means[take]
        return value

    for plan in plans:
        fold, cov, cal, h = (plan[k] for k in
            ('fold', 'covariance_rows', 'calibration_rows', 'heldout_rows'))
        cp = np.asarray([position[int(i)] for i in cov])
        rp = np.asarray([position[int(i)] for i in cal])
        hp = plan['heldout_positions']
        folder = output/f'inner_fold_{fold}'; folder.mkdir()
        fitted = fit_eu_distribution(legal_inputs(cov), residual_u[cp],
            legal_inputs(cal), residual_u[rp], base, float(training_logamp_sd))
        predicted = predict_eu_distribution(fitted, legal_inputs(h, True))
        raw_mean = predicted['mean_u']*scale+center
        raw_covariance = predicted['covariance_u']*scale[None, :, None]*scale[None, None, :]
        # Query-frame features are complete before this query's error label is used.
        actual_bio = build_eu_biology_features(data, metadata, h, cov, raw_mean,
            raw_covariance, raw_residual[cp], amplitude_edges=edges)
        random_bio = build_eu_biology_features(data, metadata, h, cov, raw_mean,
            raw_covariance, raw_residual[cp], amplitude_edges=edges,
            random_seed=int(seed)+fold)
        if actual_bio['names'] != random_bio['names']:
            raise RuntimeError("Actual and matched random biological feature schemas differ")
        if biological_names is None:
            biological_names = list(actual_bio['names'])
        elif biological_names != list(actual_bio['names']):
            raise RuntimeError("Biological feature schema changed across inner folds")
        np.testing.assert_array_equal(actual_bio['support'], random_bio['support'])
        np.testing.assert_array_equal(actual_bio['support_by_relation'], random_bio['support_by_relation'])
        # Only now score the held-out observed residual against its frozen prediction.
        decomposition = contrast_projector(raw_mean, np.ones(9), raw_covariance)
        energies = projected_energy(raw_residual[hp], decomposition)
        if not np.isfinite(energies).all() or np.any(energies <= 0):
            raise ValueError("Held-out geometric energies must be finite and positive")
        records['energies'][hp] = energies
        records['scatter_u'][hp] = predicted['scatter_u']
        records['covariance_u'][hp] = predicted['covariance_u']
        records['raw_covariance'][hp] = raw_covariance
        records['fold'][hp] = fold
        records['prediction_count'][hp] += 1
        for prefix, value in (('biology', actual_bio), ('random_biology', random_bio)):
            for key in ('values', 'support', 'support_by_relation'):
                array = np.asarray(value[key]); name = prefix+'_'+key
                if name not in records:
                    records[name] = np.empty((n, *array.shape[1:]), dtype=array.dtype)
                records[name][hp] = array
        record = dict(fold=fold, seed=int(seed)+fold,
            covariance_ids=ids[cov].tolist(), calibration_ids=ids[cal].tolist(),
            heldout_ids=ids[h].tolist(), heldout_positions=hp.tolist(),
            covariance_groups=np.unique(groups[cov]).tolist(),
            calibration_groups=np.unique(groups[cal]).tolist(),
            heldout_groups=np.unique(groups[h]).tolist(),
            base_source='unchanged externally supplied original MODEL_TRAIN scatter',
            distribution=fitted['report'], biology_audit=actual_bio['audit'],
            random_biology_audit=random_bio['audit'], heldout_targets_used_for_prediction=False,
            labels='held-out native geometry residual energy in its own predicted covariance frame')
        fold_records.append(record)
        write_json(folder/'fold_record.json', record)
        joblib.dump(fitted, folder/'distribution.joblib')
        np.savez_compressed(folder/'heldout_records.npz', ids=ids[h], positions=hp,
            mean_u=predicted['mean_u'], scatter_u=predicted['scatter_u'],
            covariance_u=predicted['covariance_u'], raw_mean=raw_mean,
            raw_covariance=raw_covariance, raw_residual=raw_residual[hp], energies=energies,
            radial_weights=predicted['radial_weights'],
            radial_variance_multiplier=predicted['radial_variance_multiplier'],
            biology_values=actual_bio['values'], biology_support=actual_bio['support'],
            random_biology_values=random_bio['values'], random_biology_support=random_bio['support'],
            original_base_scatter=base)
    np.testing.assert_array_equal(records['prediction_count'], np.ones(n, int))
    if np.any(records['fold'] < 0):
        raise RuntimeError("Not every reference received a held-out prediction")
    report = dict(complete=True, n=n, groups=len(np.unique(groups[rows])), folds=3,
        ids=ids[rows].tolist(), biological_names=biological_names,
        seed=int(seed), amplitude_edges=edges.tolist(),
        distribution_recipe='LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL',
        mean_refitted=False, base_scatter_refitted=False, external_query_outcomes_read=False,
        original_core_mean_and_TRAIN_scatter_provenance='required external caller inputs',
        adapter_label_space='native geometry, whitened using each inner-heldout full predictive covariance',
        covariance_is_scatter_times_radial_multiplier=True,
        interpretation='two geometric error blocks, not physical independent/shared noise',
        reference_fit_fraction='floor(2/3 remaining chemical groups); remaining groups calibrate radii',
        fold_records=fold_records)
    np.savez_compressed(output/'records.npz', **records)
    write_json(output/'summary.json', report)
    return _result(records, report)
