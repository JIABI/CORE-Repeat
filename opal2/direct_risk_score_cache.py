"""Honest inner-fold error-energy scores for direct-risk calibration.

The complete inner CORE means and distributions are reused, not retrained.
For each of their six distribution cells, a predictor is fitted on the other
held-out half (DIST_FIT plus DIST_CAL), never on its query outcomes.  Descriptor
coordinates are fitted on that inner mean's MODEL_FIT, which excludes both
halves.  Targets retain the original native RIDGE-covariance energy frame used
by the historical amplitude/descriptor ranking comparison.

The resulting roughly 760 rows are a mapping-training pool inside one outer
MODEL_FIT, not a new independent validation set.  Donor scores are in-sample
reference scores for a declared ECDF transform, not evaluation predictions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .biology_kernel_evaluation import write_json
from .conditional_residual_information import BOOST_CONFIG, ScalePredictor, error_targets
from .lincs_biology_experiment import load_data
from .nested_core_residuals import ARRAY_KEYS
from .residual_descriptors import ResidualDescriptorTransformer, load_normalized_plate_controls


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'runs/conditional_residual_information_20260916_v1'
DISTRIBUTIONS = PROJECT / 'runs/dual_branch_biology_20260917_v2'
RADIAL = PROJECT / 'runs/lincs_empirical_radial_20260916_v1'
SEED = 17091731


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _rows(ids, chosen, name):
    chosen = list(map(str, chosen))
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError(name + ' requires unique nonempty identities')
    lookup = {str(value): i for i, value in enumerate(ids)}
    if not set(chosen) <= set(lookup):
        raise ValueError(name + ' contains identities outside outer MODEL_FIT')
    return np.asarray([lookup[value] for value in chosen], int)


def centered_ecdf(reference, values):
    """Midrank ECDF mapped to [-1,1], using only declared reference scores."""
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    if (reference.ndim != 1 or not len(reference) or values.ndim != 1
            or not np.isfinite(reference).all() or not np.isfinite(values).all()):
        raise ValueError('Finite one-dimensional reference and query scores required')
    ordered = np.sort(reference)
    return (np.searchsorted(ordered, values, side='left')
            + np.searchsorted(ordered, values, side='right')) / len(reference) - 1.


def validate_cell_roles(ids, groups, cell, outer_excluded_ids=()):
    """Resolve saved identities and check the complete outcome lineage."""
    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    if ids.ndim != 1 or groups.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError('Unique aligned outer MODEL_FIT identities required')
    if set(ids) & set(map(str, outer_excluded_ids)):
        raise ValueError('Outer query or validation identities entered score fitting')
    names = ('query', 'fit', 'cal', 'mean_fit', 'mean_validation', 'mean_reference')
    rows = {name: _rows(ids, cell[name + '_ids'], name) for name in names}
    rows['donor'] = np.r_[rows['fit'], rows['cal']]
    if len(set(rows['donor'])) != len(rows['donor']):
        raise ValueError('DIST_FIT and DIST_CAL share an object')
    qgroups = set(groups[rows['query']])
    for name in ('donor', 'mean_fit', 'mean_validation', 'mean_reference'):
        if qgroups & set(groups[rows[name]]):
            raise ValueError('A query chemistry group entered ' + name)
    held_groups = qgroups | set(groups[rows['donor']])
    for name in ('mean_fit', 'mean_validation', 'mean_reference'):
        if held_groups & set(groups[rows[name]]):
            raise ValueError('The shared inner mean saw a score donor/query group')
    if set(groups[rows['fit']]) & set(groups[rows['cal']]):
        raise ValueError('A chemistry group crosses DIST_FIT and DIST_CAL')
    return rows


def fit_cell_scores(data, metadata, nested, cell, *, controls=None, seed=SEED,
                    transformer=None, outer_excluded_ids=()):
    """Fit only small score models; no query residual/target is read here."""
    ids, groups = np.asarray(data['ids'], str), np.asarray(data['groups'], str)
    np.testing.assert_array_equal(nested['ids'], ids)
    np.testing.assert_array_equal(nested['groups'], groups)
    rows = validate_cell_roles(ids, groups, cell, outer_excluded_ids)
    donor, query, mean_fit = rows['donor'], rows['query'], rows['mean_fit']
    if transformer is None:
        transformer = ResidualDescriptorTransformer.fit(
            data, metadata, mean_fit, plate_controls=controls, seed=seed)
    if list(transformer.fit_ids) != ids[mean_fit].tolist():
        raise ValueError('Descriptor coordinates were not fitted on this inner mean MODEL_FIT')
    descriptor = transformer.transform(data, metadata)
    # Slice before computing energies: query targets are not even passed to
    # the energy routine.  Do not substitute global OOF covariance, whose
    # opposite-cell radial law can have used the present query outcomes.
    energies = error_targets(nested['raw_mean'][donor], nested['raw_covariance'][donor],
                            nested['raw_residual'][donor])
    amp_columns = np.asarray(descriptor.blocks['amplitude'] + descriptor.blocks['context'], int)
    full_columns = np.arange(descriptor.values.shape[1])
    result = dict(query_ids=ids[query], donor_ids=ids[donor], query_indices=query,
                  donor_indices=donor, donor_energy=energies)
    for name, columns in (('amplitude', amp_columns), ('descriptors', full_columns)):
        model = ScalePredictor.fit(descriptor.values[donor], energies, columns,
                                   ids[donor], seed=seed + 100 * int(cell['half']))
        if set(model.fit_ids) & set(ids[query]):
            raise RuntimeError('Score predictor fitted an evaluation identity')
        result['query_' + name] = model.predict(descriptor.values[query])
        result['donor_' + name] = model.predict(descriptor.values[donor])
    lognorm = np.log(np.linalg.norm(np.asarray(data['Y'])[:, 0], axis=1))
    result['amplitude_reference_ids'] = ids[mean_fit]
    result['amplitude_reference_log_norm'] = lognorm[mean_fit]
    result['query_log_norm'] = lognorm[query]
    result['query_amplitude_ecdf'] = centered_ecdf(lognorm[mean_fit], lognorm[query])
    result['query_descriptors_rank6_ecdf'] = centered_ecdf(
        result['donor_descriptors'][:, 1], result['query_descriptors'][:, 1])
    report = dict(cell=int(cell['cell']), error_fold=int(cell['error_fold']),
                  half=int(cell['half']), query_ids=ids[query].tolist(),
                  score_fit_ids=ids[donor].tolist(), descriptor_fit_ids=ids[mean_fit].tolist(),
                  query_n=len(query), score_fit_n=len(donor), descriptor_fit_n=len(mean_fit),
                  query_groups=len(set(groups[query])), score_fit_groups=len(set(groups[donor])),
                  feature_names=descriptor.names, amplitude_columns=amp_columns.tolist(),
                  descriptor_columns=full_columns.tolist(), booster=dict(BOOST_CONFIG),
                  target_frame='native inner-mean RIDGE covariance, geometric rank-3/rank-6 energies',
                  predicted_units='energy per degree, denominators 3 and 6',
                  score_fit_rule='same inner held-out fold, opposite half DIST_FIT plus DIST_CAL',
                  donor_score_scope='in-sample training score ECDF reference; not evaluation',
                  ecdf_definition='(count(reference < score)+count(reference <= score))/n_reference - 1',
                  amplitude_rank='raw log first-well norm relative to inner mean MODEL_FIT',
                  descriptor_rank='rank-six raw predictor relative to score-fitting donor predictions',
                  query_outcomes_used=False, outer_outcomes_used=False,
                  complete_core_retrained=False, descriptor_seed=int(seed),
                  score_seed=int(seed + 100 * int(cell['half'])))
    return result, report, transformer


def build_fold(data, metadata, outer_record, source, distributions, output, *, controls=None,
               seed=SEED):
    """Create one new cache using only the supplied outer MODEL_FIT population."""
    start = time.monotonic()
    fold = int(outer_record['fold'])
    source, distributions, output = Path(source), Path(distributions), Path(output)
    folder = output / f'fold_{fold}'
    archive, report_path = folder / 'honest_scores.npz', folder / 'honest_scores.json'
    global_ids = np.asarray(data['ids'], str)
    fit = _rows(global_ids, outer_record['fit_ids'], 'outer MODEL_FIT')
    excluded = outer_record['test_ids'] + outer_record['inner_validation_ids']
    local = {key: np.asarray(data[key])[fit].copy() for key in ARRAY_KEYS}
    local['feature_names'] = np.asarray(data['feature_names']).copy()
    local_metadata = dict(metadata, units=[deepcopy(metadata['units'][i]) for i in fit])
    ids, groups = local['ids'], local['groups']
    nested_folder = source / f'fold_{fold}' / 'nested_core'
    distribution_folder = distributions / f'fold_{fold}' / 'nested_distribution'
    nested = read_npz(nested_folder / 'residuals.npz')
    distribution = read_npz(distribution_folder / 'distribution.npz')
    plan = read_json(distribution_folder / 'plan.json')
    if read_json(distribution_folder / 'summary.json')['state'] != 'COMPLETE':
        raise ValueError('Complete saved inner distributions are required')
    np.testing.assert_array_equal(nested['ids'], ids)
    np.testing.assert_array_equal(nested['groups'], groups)
    np.testing.assert_array_equal(distribution['ids'], ids)
    np.testing.assert_array_equal(distribution['raw_mean'], nested['raw_mean'])
    np.testing.assert_array_equal(nested['prediction_count'], np.ones(len(ids), int))
    spec = dict(seed=int(seed), fold=fold, source=str(source.resolve()),
                distributions=str(distributions.resolve()), score_models=['amplitude', 'descriptors'],
                energy_frame='native inner RIDGE covariance', booster=dict(BOOST_CONFIG))
    if archive.exists() or report_path.exists():
        if not (archive.exists() and report_path.exists()):
            raise FileExistsError('Incomplete cache retained for inspection; use a new output directory')
        report = read_json(report_path)
        if report.get('state') != 'COMPLETE' or report.get('spec') != spec:
            raise ValueError('Existing score cache has a different specification')
        cached = read_npz(archive)
        np.testing.assert_array_equal(cached['ids'], ids)
        np.testing.assert_array_equal(cached['prediction_count'], np.ones(len(ids), int))
        return report
    result = dict(ids=ids, groups=groups, error_fold=nested['error_fold'],
                  construction_cell=distribution['construction_cell'],
                  amplitude=np.empty((len(ids), 2)), descriptors=np.empty((len(ids), 2)),
                  amplitude_ecdf=np.empty(len(ids)), descriptors_rank6_ecdf=np.empty(len(ids)),
                  prediction_count=np.zeros(len(ids), int))
    reports, transformers = [], {}
    for cell in plan['cells']:
        error_fold = int(cell['error_fold'])
        arrays, report, transformer = fit_cell_scores(
            local, local_metadata, nested, cell, controls=controls,
            seed=seed + 100000 * fold + 10000 * error_fold,
            transformer=transformers.get(error_fold), outer_excluded_ids=excluded)
        transformers[error_fold] = transformer
        q = arrays['query_indices']
        result['amplitude'][q] = arrays['query_amplitude']
        result['descriptors'][q] = arrays['query_descriptors']
        result['amplitude_ecdf'][q] = arrays['query_amplitude_ecdf']
        result['descriptors_rank6_ecdf'][q] = arrays['query_descriptors_rank6_ecdf']
        result['prediction_count'][q] += 1
        for name, value in arrays.items():
            result[f"cell_{int(cell['cell']):02d}_" + name] = value
        reports.append(report)
        print(f'score cache fold={fold} cell={cell["cell"] + 1}/6 '
              f'donors={len(arrays["donor_ids"])} query={len(q)} '
              f'elapsed={time.monotonic() - start:.1f}s', flush=True)
    np.testing.assert_array_equal(result['prediction_count'], np.ones(len(ids), int))
    for name in ('amplitude', 'descriptors'):
        if not np.isfinite(result[name]).all() or np.any(result[name] <= 0):
            raise ValueError('Finite positive complete energy predictions required')
    for name in ('amplitude_ecdf', 'descriptors_rank6_ecdf'):
        if not np.isfinite(result[name]).all() or np.any(np.abs(result[name]) > 1):
            raise ValueError('Complete centered ECDF coordinates must be in [-1,1]')
    report = dict(state='COMPLETE', spec=spec, n=len(ids), chemistry_groups=len(set(groups)),
                  cells=reports, elapsed_seconds=time.monotonic() - start,
                  outer_fit_ids=ids.tolist(), outer_excluded_ids=excluded,
                  source_distribution=str(distribution_folder.resolve()),
                  sources=dict(means=str((nested_folder / 'residuals.npz').resolve()),
                               distributions=str((distribution_folder / 'distribution.npz').resolve())),
                  scope='Outer-MODEL_FIT inner-OOF mapping training pool; not an independent validation',
                  assertions=dict(all_objects_predicted_once=True, own_query_outcomes_unused=True,
                                  outer_query_and_validation_outcomes_unused=True,
                                  complete_core_means_and_distributions_unchanged=True),
                  limitations=['Inner score training population is 126-128 objects, versus about 760 for outer deployment.',
                               'Donor predictions are in-sample ECDF references; query scores alone are out of score fit.',
                               'Native RIDGE energy is a predictive-error diagnostic, not identified physical noise.'])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **result)
    write_json(report_path, report)
    return report


def run(output, *, source=SOURCE, distributions=DISTRIBUTIONS, radial=RADIAL, folds=None, seed=SEED):
    old = read_json(Path(radial) / 'summary.json')
    manifest = read_json(Path(old['reference_run']) / 'run_manifest.json')
    data, metadata = load_data(old['data_directory'])
    if np.asarray(data['ids']).tolist() != manifest['ids']:
        raise ValueError('Opened development identities differ from the frozen split')
    controls = load_normalized_plate_controls(metadata, data['feature_names'],
        PROJECT / 'reports/lincs_biology_preflight_20260915/metadata_audit/profile_cache')
    chosen = set(range(5) if folds is None else folds)
    if not chosen or not chosen <= set(range(5)):
        raise ValueError('Select one or more of the five existing outer folds')
    reports = []
    for record in manifest['folds']:
        if int(record['fold']) in chosen:
            reports.append(build_fold(data, metadata, record, source, distributions, output,
                                      controls=controls, seed=seed))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default=str(SOURCE))
    parser.add_argument('--distributions', default=str(DISTRIBUTIONS))
    parser.add_argument('--radial', default=str(RADIAL))
    parser.add_argument('--fold', type=int, action='append', choices=range(5))
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, source=args.source, distributions=args.distributions,
            radial=args.radial, folds=args.fold, seed=args.seed)


if __name__ == '__main__':
    main()
