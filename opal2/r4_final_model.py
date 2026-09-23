"""Final EU development fit and outcome-free R4 predictions.

The full existing CORE recipe and HistGB selection/calibration procedures are
reused.  Confirmation future observations have no route into ``fit`` or
``score``.  Main-seed Gamma draws are saved exactly (float64, all 100,000) so
the later endpoint pass can compute fair CRPS without repeating integration.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import chi2, norm
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .eu_core_training import fit_complete_eu_core, predict_eu_core
from .eu_core_distribution import fit_eu_distribution, predict_eu_distribution
from .eu_r2_direct_baselines import (
    _select, _bounded_prediction, _probability, _fit_platt,
    empirical_gamma_support, evaluate_direct_distribution,
)
from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_oof_ridge import transform_input, transform_target
from .empirical_radial import draw_radial, radial_ppf, radial_nll, radial_cdf
from .reference_information_diagnostic import gamma_forward
from .objective_analysis import fair_crps
from .state_biology_kernel import StateBiologyKernelMean


SEED = 20260921
SAMPLES = 100000
SEEDS = (SEED, SEED+100000, SEED+200000)
LEVELS = np.array([.5, .8, .9, .95, .99])
ROLE_COUNTS = dict(TRAIN=434, VALIDATION=108, REF_FIT=181, DIST_CAL=181)
CORE_ARM = 'CORE_ORIGINAL'
GAUSSIAN_ARM = 'CORE_LOCAL_GAUSSIAN'
DIRECT_ARM = 'DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL'
QUERY_KEYS = frozenset(('ids', 'groups', 'X', 'chem', 'chem_mask'))


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def normalize_partitions(ids, groups, partitions, *, require_final_counts=True):
    """Resolve a frozen identity manifest; no random allocation is inferred."""
    if isinstance(partitions, (str, Path)):
        partitions = json.loads(Path(partitions).read_text())
    parts = partitions.get('partitions', partitions)
    if set(parts) != set(ROLE_COUNTS):
        raise ValueError('Partitions must contain TRAIN, VALIDATION, REF_FIT, DIST_CAL')
    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    if ids.ndim != 1 or groups.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError('Unique aligned development identities and groups required')
    lookup = {oid: i for i, oid in enumerate(ids)}
    result = {}
    for name, values in parts.items():
        values = np.asarray(values)
        if values.ndim != 1 or not len(values):
            raise ValueError('Each final role must be nonempty')
        if values.dtype.kind in 'iu':
            rows = values.astype(int)
            if np.any(rows < 0) or np.any(rows >= len(ids)):
                raise ValueError('Partition row outside the development dataset')
        else:
            if any(str(value) not in lookup for value in values):
                raise ValueError('Partition identity outside the development dataset')
            rows = np.array([lookup[str(value)] for value in values], int)
        if len(set(rows)) != len(rows):
            raise ValueError('Repeated identity within a final role')
        if require_final_counts and len(rows) != ROLE_COUNTS[name]:
            raise ValueError('Final role count differs from frozen 434/108/181/181')
        result[name] = np.sort(rows)
    joined = np.concatenate(list(result.values()))
    if len(joined) != len(ids) or set(joined) != set(range(len(ids))):
        raise ValueError('Final roles must cover the exact development population once')
    group_sets = [set(groups[rows]) for rows in result.values()]
    if any(group_sets[i] & group_sets[j] for i in range(4) for j in range(i)):
        raise ValueError('Chemical group crosses final development roles')
    return result


def _development(data, metadata):
    if isinstance(data, (str, Path)):
        folder = Path(data)
        metadata = json.loads((folder/'metadata.json').read_text())
        data = _read_npz(folder/'data.npz')
    if metadata is None or metadata.get('confirmation_data_loaded') is not False:
        raise ValueError('Explicit existing development-only metadata is required')
    if metadata.get('biology_active') or metadata.get('representation_active'):
        raise ValueError('R4 final CORE uses no biology or JEPA')
    data = {key: np.asarray(value) for key, value in data.items()}
    required = {'ids', 'groups', 'Y', 'chem', 'chem_mask'}
    if not required <= set(data) or len(data['ids']) != 904:
        raise ValueError('Exactly the 904 opened EU development identities are required')
    if data['Y'].ndim != 3 or data['Y'].shape[:2] != (904, 4) or not np.isfinite(data['Y']).all():
        raise ValueError('Complete finite four-role development profiles are required')
    return data, metadata


def fit(data, partitions, output_dir, *, metadata=None, seed=SEED):
    """Fit the one final model from already opened DEV904 and exact frozen roles."""
    data, metadata = _development(data, metadata)
    part = normalize_partitions(data['ids'], data['groups'], partitions)
    if seed != SEED:
        raise ValueError('Final R4 seed is frozen at 20260921')
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError('Preserve previous final fits; output must be new')
    root.mkdir(parents=True)
    started = time.monotonic()
    t, v, r, c = (part[key] for key in ROLE_COUNTS)
    identity_parts = {key: data['ids'][rows].tolist() for key, rows in part.items()}
    write_json(root/'partitions.json', identity_parts)

    def status(state, stage, **extra):
        write_json(root/'status.json', dict(state=state, stage=stage,
            elapsed_seconds=time.monotonic()-started, **extra))

    def role(rows, reference=False):
        values = {key: data[key][rows] for key in ('ids', 'groups', 'chem', 'chem_mask')}
        values['X' if reference else 'Y'] = data['Y'][rows, 0] if reference else data['Y'][rows]
        return values

    def inputs(rows):
        return dict(ids=data['ids'][rows], groups=data['groups'][rows],
                    X=data['Y'][rows, 0], chem=data['chem'][rows])

    try:
        torch.set_num_threads(2)
        with threadpool_limits(limits=2):
            status('RUNNING', 'full CORE mean fitting')
            core = fit_complete_eu_core(role(t), role(v), role(r, True), metadata,
                                       root/'mean', seed=seed)
            stats = core['stats']
            gram = profiles_to_gram(torch.as_tensor(data['Y'], dtype=torch.float64))
            raw = gram_to_coordinates(gram).numpy()
            gamma = gram_gains(gram).numpy()[:, 2]
            target = transform_target(raw, stats)
            status('RUNNING', 'held-out REF and DIST_CAL distribution fitting')
            reference = predict_eu_core(core['model'], stats, data['Y'][r, 0],
                                        data['chem'][r], data['chem_mask'][r])
            calibration = predict_eu_core(core['model'], stats, data['Y'][c, 0],
                                          data['chem'][c], data['chem_mask'][c])
            ref_residual = target[r]-reference['mean_u']
            cal_residual = target[c]-calibration['mean_u']
            distribution = fit_eu_distribution(inputs(r), ref_residual,
                inputs(c), cal_residual, core['base_covariance'],
                float(np.std(np.log(np.linalg.norm(data['Y'][t, 0], axis=1)))),
                model_training_ids=data['ids'][np.r_[t, v]],
                model_training_groups=data['groups'][np.r_[t, v]])
            joblib.dump(distribution, root/'distribution.joblib')
            write_json(root/'distribution.json', distribution['report'])
            np.savez_compressed(root/'distribution_arrays.npz',
                ref_ids=data['ids'][r], cal_ids=data['ids'][c],
                ref_residual=ref_residual, cal_residual=cal_residual,
                ref_mean_u=reference['mean_u'], cal_mean_u=calibration['mean_u'],
                base_scatter=core['base_covariance'])
            status('RUNNING', 'access-matched HistGB regression and NULL classifier')
            direct_x = np.column_stack((transform_input(data['Y'][:, 0], stats), data['chem']))
            training = np.r_[t, r]
            # Exactly the R2 HistGB grid, validation criterion, and CAL sigmoid.
            # No unused RIDGE/ExtraTrees fits and no query rows are needed.
            with threadpool_limits(limits=1):
                regression, regression_meta = _select('HISTGB', direct_x[training], gamma[training],
                    direct_x[v], gamma[v], seed, classifier=False)
                classifier, classifier_meta = _select('HISTGB', direct_x[training], (gamma[training] <= 0).astype(int),
                    direct_x[v], (gamma[v] <= 0).astype(int), seed, classifier=True)
                cal_point = _bounded_prediction(regression, direct_x[c])
                cal_probability = _probability(classifier, direct_x[c])
                platt, platt_meta = _fit_platt(cal_probability, (gamma[c] <= 0).astype(int), seed)
            direct = dict(regression=regression, classifier=classifier, platt=platt,
                          gamma_residuals=np.sort(gamma[c]-cal_point))
            joblib.dump(direct, root/'histgb.joblib')
            write_json(root/'histgb.json', dict(arm=DIRECT_ARM, seed=seed,
                train_ids=data['ids'][training], validation_ids=data['ids'][v], calibration_ids=data['ids'][c],
                fitting_scope='TRAIN+REF_FIT outcomes; full TRAIN-transformed X/log-norm plus chemistry',
                regression=regression_meta, classification=classifier_meta, platt=platt_meta,
                calibration_used_for_model_selection=False, query_outcomes_used=False,
                calibration_support_scope='CAL fitted the sigmoid; its calibrated probabilities are in-sample, not independent risk validation',
                classifier_probability_is_separate_from_gamma_distribution=True))
            np.savez_compressed(root/'calibration_support.npz', ids=data['ids'][c], groups=data['groups'][c],
                layout=data.get('layout', data['groups'])[c], actual=gamma[c],
                predicted=cal_point, p_null_raw=cal_probability, p_null_calibrated=platt.predict(cal_probability),
                gamma_residuals=direct['gamma_residuals'])
        manifest = dict(recipe='CORE_ORIGINAL: RIDGE + validation-best HR + A30 + STATE50 + AMP_EMP_LOCAL',
            base_scatter='unchanged RIDGE grouped-OOF error second moment', comparator=DIRECT_ARM,
            n_development=904, counts=ROLE_COUNTS, partitions=identity_parts, seed=seed,
            model_training_groups=data['groups'][np.r_[t, v]].tolist(),
            all_development_ids=data['ids'].tolist(), all_development_groups=data['groups'].tolist(),
            cpu_threads_max=2, direct_threads=1, biology=False, jepa=False,
            confirmation_inputs_opened=False, confirmation_outcomes_opened=False,
            calibration_probability_interface='HistGB CAL probability is a separate classifier readout',
            elapsed_seconds=time.monotonic()-started)
        write_json(root/'manifest.json', manifest)
        write_json(root/'complete.json', dict(complete=True, status='COMPLETE',
            confirmation_inputs_opened=False, confirmation_outcomes_opened=False,
            recipe=manifest['recipe'], comparator=DIRECT_ARM,
            elapsed_seconds=time.monotonic()-started))
        status('COMPLETE', 'final model frozen')
        return dict(model=core['model'], stats=stats, distribution=distribution,
                    direct=direct, manifest=manifest, model_dir=str(root))
    except Exception as exc:
        status('FAILED', 'fit', error_type=type(exc).__name__, error=str(exc))
        raise


def load(output_dir):
    """Restore final local checkpoints without fitting or reading measurements."""
    torch.set_num_threads(2)
    root = Path(output_dir).resolve()
    if json.loads((root/'status.json').read_text())['state'] != 'COMPLETE':
        raise ValueError('Final model fitting is incomplete')
    saved = torch.load(root/'mean/STATE50/epoch50.pt', map_location='cpu', weights_only=True)
    model = StateBiologyKernelMean.from_config(saved['model_config']).double()
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return dict(model=model, stats=json.loads((root/'mean/preprocessing.json').read_text()),
        distribution=joblib.load(root/'distribution.joblib'), direct=joblib.load(root/'histgb.joblib'),
        manifest=json.loads((root/'manifest.json').read_text()), model_dir=str(root))


def validate_query(query, fitted):
    if not isinstance(query, Mapping) or not QUERY_KEYS <= set(query) or set(query)-QUERY_KEYS-{'layout'}:
        raise ValueError('Query accepts only ids/groups/X/chem/chem_mask and optional layout; no future outcomes')
    result = {key: np.asarray(value) for key, value in query.items()}
    ids, groups = result['ids'].astype(str), result['groups'].astype(str)
    if ids.ndim != 1 or not len(ids) or groups.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty aligned query identities required')
    if set(ids) & set(fitted['manifest']['all_development_ids']) or set(groups) & set(fitted['manifest']['all_development_groups']):
        raise ValueError('Confirmation identity or chemical group overlaps development')
    x = np.asarray(result['X'], float)
    if x.ndim != 2 or x.shape[0] != len(ids) or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) <= 0):
        raise ValueError('Pass the common finite positive-norm X eligibility subset')
    result.update(ids=ids, groups=groups, X=x)
    if 'layout' in result and result['layout'].shape != ids.shape:
        raise ValueError('Query layouts must align with identities')
    return result


def select(ids, predicted, probability, population_size, lam=.2):
    ids, predicted, probability = np.asarray(ids, str), np.asarray(predicted), np.asarray(probability)
    if predicted.shape != ids.shape or probability.shape != ids.shape or not np.isfinite(predicted).all():
        raise ValueError('Aligned finite policy predictions required')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Finite probabilities in [0,1] required')
    if isinstance(population_size, bool) or int(population_size) != population_size or population_size < len(ids):
        raise ValueError('Original eligible population cannot be smaller than the common X subset')
    selected = np.zeros(len(ids), bool)
    count = min(len(ids), int(population_size)//4//2)
    selected[np.lexsort((ids, -(predicted-lam*probability)))[:count]] = True
    return selected


def integrate(mean, scatter, stats, *, seed, law=None, weights=None, samples=SAMPLES,
              gamma_path=None, chunk_size=8, intervals=True):
    """All decisions and predictive intervals from X-only joint draws.

    The chunk size is part of the frozen stream definition. Arms reset to the
    same normal seed, retaining common random numbers for comparison.
    """
    mean, scatter = np.asarray(mean, float), np.asarray(scatter, float)
    if mean.ndim != 2 or mean.shape[1] != 9 or scatter.shape != (len(mean), 9, 9):
        raise ValueError('Aligned nine-coordinate joint means and scatter required')
    if samples < 4 or samples % 2 or chunk_size < 1 or chunk_size > 8:
        raise ValueError('Even sample count >=4 and chunk size between 1 and 8 required')
    n = len(mean)
    out = {key: np.empty(n) for key in ('predicted', 'p_null', 'gamma_mc_se', 'null_mc_se')}
    if intervals:
        out.update(gamma_lower_by_level=np.empty((n, len(LEVELS))),
            gamma_upper_by_level=np.empty((n, len(LEVELS))),
            coordinate_lower_by_level=np.empty((n, 9, len(LEVELS))),
            coordinate_upper_by_level=np.empty((n, 9, len(LEVELS))))
    cache = None
    if gamma_path is not None:
        gamma_path = Path(gamma_path)
        if gamma_path.exists():
            raise FileExistsError(gamma_path)
        cache = np.lib.format.open_memmap(gamma_path, mode='w+', dtype=np.float64, shape=(n, samples))
    normal_rng, radius_rng = np.random.default_rng(seed), np.random.default_rng(seed+47000)
    chol = np.linalg.cholesky(scatter)
    for begin in range(0, n, chunk_size):
        end = min(begin+chunk_size, n)
        normal = normal_rng.normal(size=(samples, end-begin, 9))
        if law is None:
            eps = np.einsum('nij,snj->sni', chol[begin:end], normal)
        else:
            eps = draw_radial(law, weights[begin:end], scatter[begin:end], normal,
                radius_rng.random((samples, end-begin)), radius_rng.random((samples, end-begin)))
        u = mean[None, begin:end]+eps
        raw = u*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
        gamma = gamma_forward(raw)
        if not np.isfinite(gamma).all():
            raise FloatingPointError('Nonfinite predictive Gamma; no samples dropped')
        p = (gamma <= 0).mean(0)
        out['predicted'][begin:end] = gamma.mean(0)
        out['p_null'][begin:end] = p
        out['gamma_mc_se'][begin:end] = gamma.std(0, ddof=1)/np.sqrt(samples)
        out['null_mc_se'][begin:end] = np.sqrt(p*(1-p)/samples)
        if cache is not None:
            cache[begin:end] = gamma.T
        if intervals:
            out['gamma_lower_by_level'][begin:end] = np.quantile(gamma, (1-LEVELS)/2, axis=0).T
            out['gamma_upper_by_level'][begin:end] = np.quantile(gamma, (1+LEVELS)/2, axis=0).T
            if law is None:
                h = np.sqrt(np.diagonal(scatter[begin:end], axis1=1, axis2=2))[..., None]*norm.ppf((1+LEVELS)/2)
                out['coordinate_lower_by_level'][begin:end] = mean[begin:end, :, None]-h
                out['coordinate_upper_by_level'][begin:end] = mean[begin:end, :, None]+h
            else:
                out['coordinate_lower_by_level'][begin:end] = np.moveaxis(np.quantile(u, (1-LEVELS)/2, axis=0), 0, -1)
                out['coordinate_upper_by_level'][begin:end] = np.moveaxis(np.quantile(u, (1+LEVELS)/2, axis=0), 0, -1)
        print(f'predict seed={seed} rows={end}/{n}', flush=True)
    if cache is not None:
        cache.flush()
        del cache
    return out


def score(fitted, query, output_dir, *, population_size=None, seeds=SEEDS, samples=SAMPLES, chunk_size=8):
    """Freeze every policy from X only; full candidate denominator is explicit."""
    if isinstance(fitted, (str, Path)):
        fitted = load(fitted)
    query = validate_query(query, fitted)
    population_size = len(query['ids']) if population_size is None else population_size
    if tuple(seeds) != SEEDS or samples != SAMPLES or chunk_size != 8:
        raise ValueError('Final integration freezes three seeds, 100000 draws, chunk size 8')
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError('Preserve previous confirmation predictions')
    root.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = dict(state='RUNNING', seeds=list(seeds), primary_seed=seeds[0], samples=samples,
        chunk_size=chunk_size, gamma_cache_dtype='float64', gamma_cache_shape='[eligible_query,100000]',
        model_dir=fitted['model_dir'], population_size=int(population_size), n_query=len(query['ids']),
        query_ids=query['ids'].tolist(), future_outcomes_used=False, cpu_threads_max=2,
        levels=LEVELS.tolist(),
        primary_policy_lambda=.2, secondary_lambda=0., tie_rule='stable object ID ascending',
        arms=[CORE_ARM, GAUSSIAN_ARM, DIRECT_ARM],
        Gaussian_definition='same standardized-u mean; existing LOCAL_SCALE base scatter, before amplitude/radial adjustment',
        gamma_crps='unbiased/fair MC, denominator S*(S-1); all saved draws used',
        random_seed=20260922)
    write_json(root/'manifest.json', manifest)
    try:
        with threadpool_limits(limits=2):
            prediction = predict_eu_core(fitted['model'], fitted['stats'], query['X'], query['chem'], query['chem_mask'])
            inputs = {key: query[key] for key in ('ids', 'groups', 'X', 'chem')}
            distribution = predict_eu_distribution(fitted['distribution'], dict(inputs, mean_u=prediction['mean_u']))
            joblib.dump(distribution, root/'query_distribution.joblib')
            write_json(root/'preprocessing.json', fitted['stats'])
            np.savez_compressed(root/'query_metadata.npz', ids=query['ids'], groups=query['groups'],
                layout=query.get('layout', query['groups']), norm2_per_feature=prediction['norm2_per_feature'])
            all_outputs = {}
            for arm in (CORE_ARM, GAUSSIAN_ARM):
                is_core = arm == CORE_ARM
                scatter = distribution['scatter_u'] if is_core else distribution['base_scatter_u']
                law = distribution['law'] if is_core else None
                weights = distribution['radial_weights'] if is_core else None
                for index, seed in enumerate(seeds):
                    primary = index == 0
                    out = integrate(prediction['mean_u'], scatter, fitted['stats'], seed=seed,
                        law=law, weights=weights, samples=samples, chunk_size=chunk_size, intervals=primary,
                        gamma_path=root/(arm+'_gamma_samples.npy') if primary else None)
                    for lam in (.2, 0.):
                        out[f'selected_lambda_{lam:g}'] = select(query['ids'], out['predicted'], out['p_null'], population_size, lam)
                    if primary:
                        out.update(mean_u=prediction['mean_u'], scatter_u=scatter)
                        out['joint_squared_radius_by_level'] = (radial_ppf(law, weights, LEVELS)**2 if is_core
                            else np.broadcast_to(chi2.ppf(LEVELS, 9), (len(query['ids']), len(LEVELS))).copy())
                    filename = arm+'.npz' if primary else f'{arm}_mc{seed-seeds[0]}.npz'
                    np.savez_compressed(root/filename, ids=query['ids'], **out)
                    if primary:
                        all_outputs[arm] = out
            direct = fitted['direct']
            direct_x = np.column_stack((transform_input(query['X'], fitted['stats']), query['chem']))
            with threadpool_limits(limits=1):
                point = _bounded_prediction(direct['regression'], direct_x)
                raw_probability = _probability(direct['classifier'], direct_x)
                probability = direct['platt'].predict(raw_probability)
            support = empirical_gamma_support(point, direct['gamma_residuals'])
            out = dict(predicted=point, p_null=probability, p_null_raw=raw_probability,
                gamma_residuals=direct['gamma_residuals'], gamma_distribution_mean=support.mean(1),
                p_null_from_gamma=(support <= 0).mean(1),
                gamma_lower_by_level=np.quantile(support, (1-LEVELS)/2, axis=1, method='inverted_cdf').T,
                gamma_upper_by_level=np.quantile(support, (1+LEVELS)/2, axis=1, method='inverted_cdf').T)
            for lam in (.2, 0.):
                out[f'selected_lambda_{lam:g}'] = select(query['ids'], point, probability, population_size, lam)
            np.savez_compressed(root/(DIRECT_ARM+'.npz'), ids=query['ids'], **out)
            all_outputs[DIRECT_ARM] = out
            count = min(len(query['ids']), int(population_size)//4//2)
            random_mask = np.zeros(len(query['ids']), bool)
            random_mask[np.random.default_rng(20260922).choice(len(query['ids']), count, replace=False)] = True
            np.savez_compressed(root/'RANDOM.npz', ids=query['ids'], selected=random_mask)
        manifest.update(state='COMPLETE', elapsed_seconds=time.monotonic()-started)
        write_json(root/'manifest.json', manifest)
        return all_outputs
    except Exception as exc:
        manifest.update(state='FAILED', error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic()-started)
        write_json(root/'manifest.json', manifest)
        raise


def evaluate_saved_predictions(prediction_dir, ids, y):
    """Later outcome-only metrics: no fitting, selection, or repeated integration.

    Missing future rows remain NaN here; the campaign evaluator retains their
    original selection and denominator and computes the predeclared bounds.
    """
    torch.set_num_threads(2)
    root = Path(prediction_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['state'] != 'COMPLETE':
        raise ValueError('Freeze all predictions before exposing evaluation outcomes')
    ids, y = np.asarray(ids, str), np.asarray(y, float)
    expected_ids = np.asarray(manifest['query_ids'], str)
    if not np.array_equal(ids, expected_ids) or y.ndim != 3 or y.shape[:2] != (len(ids), 4):
        raise ValueError('Outcome rows must exactly align with frozen prediction IDs')
    valid = (np.isfinite(y).all(axis=(1, 2)) & (np.linalg.norm(y, axis=2) > 0).all(axis=1)
             & (np.linalg.norm(y[:, :3].mean(1), axis=1) > 0))
    if not valid.any():
        return dict(valid=valid, geometry_valid=valid.copy(), actual=np.full(len(ids), np.nan), arms={})
    gram = profiles_to_gram(torch.as_tensor(y[valid], dtype=torch.float64))
    # A singular joint coordinate is not a missing scalar Gamma endpoint.
    # Preserve its policy outcome while leaving geometry metrics undefined.
    p = gram[:, 1:, 0]
    schur = gram[:, 1:, 1:]-p[..., None]*p[:, None, :]
    _, info = torch.linalg.cholesky_ex((schur+schur.transpose(-1, -2))/2, check_errors=False)
    geometry_valid = np.zeros(len(ids), bool)
    geometry_valid[np.flatnonzero(valid)[info.numpy() == 0]] = True
    raw = gram_to_coordinates(gram[info == 0]).numpy() if geometry_valid.any() else np.empty((0, 9))
    stats = json.loads((root/'preprocessing.json').read_text())
    target = transform_target(raw, stats) if len(raw) else raw
    actual = np.full(len(ids), np.nan)
    actual[valid] = gram_gains(gram).numpy()[:, 2]
    distribution = joblib.load(root/'query_distribution.joblib')
    results = {}
    rows = np.flatnonzero(valid)
    for arm in (CORE_ARM, GAUSSIAN_ARM):
        out = _read_npz(root/(arm+'.npz'))
        residual = target-out['mean_u'][geometry_valid]
        scatter = out['scatter_u'][geometry_valid]
        chol = np.linalg.cholesky(scatter)
        radius2 = np.square(np.linalg.solve(chol, residual[..., None])[..., 0]).sum(1)
        if arm == CORE_ARM:
            nll = (radial_nll(residual, scatter, distribution['law'], distribution['radial_weights'][geometry_valid])
                   if geometry_valid.any() else np.array([]))
            pit = (radial_cdf(distribution['law'], distribution['radial_weights'][geometry_valid], np.sqrt(radius2))
                   if geometry_valid.any() else np.array([]))
        else:
            logdet = 2*np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(1)
            nll = .5*(9*np.log(2*np.pi)+logdet+radius2)
            pit = chi2.cdf(radius2, 9)
        cached = np.load(root/(arm+'_gamma_samples.npy'), mmap_mode='r', allow_pickle=False)
        crps = np.array([fair_crps(np.asarray(cached[row])[:, None], actual[row:row+1])[0] for row in rows])
        lo, hi = out['gamma_lower_by_level'][valid], out['gamma_upper_by_level'][valid]
        clo, chi = out['coordinate_lower_by_level'][geometry_valid], out['coordinate_upper_by_level'][geometry_valid]
        values = dict(crps=crps,
            brier=np.square(out['p_null'][valid]-(actual[valid] <= 0)),
            gamma_coverage_by_level=(actual[valid, None] >= lo) & (actual[valid, None] <= hi),
            gamma_width_by_level=hi-lo)
        results[arm] = {key: _align_missing(value, valid) for key, value in values.items()}
        geometry = dict(nll=nll, radial_pit=pit, mahalanobis2=radius2,
            geometry_mse=np.square(residual).mean(1),
            joint_coverage_by_level=radius2[:, None] <= out['joint_squared_radius_by_level'][geometry_valid],
            coordinate_coverage_by_level=((target[..., None] >= clo) & (target[..., None] <= chi)).mean(1))
        results[arm].update({key: _align_missing(value, geometry_valid) for key, value in geometry.items()})
    direct = _read_npz(root/(DIRECT_ARM+'.npz'))
    legal = dict(predicted=direct['predicted'][valid], gamma_residuals=direct['gamma_residuals'],
                 p_null=direct['p_null_raw'][valid], p_null_calibrated=direct['p_null'][valid])
    values = evaluate_direct_distribution(legal, actual[valid])
    results[DIRECT_ARM] = {key: _align_missing(value, valid) for key, value in values.items()
                           if key != 'gamma_levels' and isinstance(value, np.ndarray) and value.shape[:1] == (valid.sum(),)}
    return dict(valid=valid, geometry_valid=geometry_valid, actual=actual, arms=results)


def _align_missing(values, valid):
    out = np.full((len(valid), *np.asarray(values).shape[1:]), np.nan)
    out[valid] = values
    return out
