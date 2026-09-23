"""Diagnose relation covariance of saved 951D RIDGE_RESPONSE OOF residuals.

No training, original-file edits, new measurement access or downstream gating.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from threadpoolctl import threadpool_limits

from opal2.relation_variance_components import (
    KERNEL_NAMES, MODELS, group_dyad_pairs, node_pair_weights,
    relation_grams, unit_moments, solve_moments, partial_design_fraction,
)
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata

OLD_RUN = PROJECT / 'runs/r3_crossdose_response_20260920_v1'
OLD_REPORT = PROJECT / 'reports/r3_crossdose_response_20260920_v1'
OUTPUT = PROJECT / 'reports/r3_signed_program_dose_20260921_v1/variance'
SEED = 2026092117
TASKS = ('SAME', 'CROSS')


def read_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in (z.files if keys is None else keys)}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, default=json_default, allow_nan=False)+'\n')


def mean_group_rows(values, groups):
    labels, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    result = np.zeros((len(labels),) + values.shape[1:], dtype=float)
    np.add.at(result, inverse, values)
    result /= counts.reshape((-1,) + (1,) * (values.ndim-1))
    return labels, result


def load_units():
    metadata_keys = ('ids', 'groups', 'object_ids', 'dose', 'batches',
                     'well_ids', 'plates', 'layout')
    # The prepared compound Y member is deliberately never decoded.
    data = read_npz(PROJECT/'data/rxrx3_r2_20260918/prepared_r2/data.npz', metadata_keys)
    pairing = read_npz(OLD_REPORT/'qualification.npz')
    biology = load_rxrx3_biology_metadata(data)
    all_units, group_fold, all_query_groups = [], {}, []
    for number in range(35):
        folder = OLD_RUN/f'unit_{number:02d}'
        complete = json.loads((folder/'complete.json').read_text())
        if complete['state'] != 'COMPLETE':
            raise ValueError(f'Incomplete saved unit {number}')
        q = read_npz(folder/'query_scores.npz', ('pair_rows', 'groups', 'object_ids',
            'source_ids', 'target_ids', 'batch', 'source_plate', 'source_dose', 'target_dose'))
        pred = read_npz(folder/'predictions.npz', ('source_x', 'SAME_actual',
            'SAME_RIDGE_RESPONSE', 'CROSS_actual', 'CROSS_RIDGE_RESPONSE'))
        rows = q['pair_rows']
        source, target = pairing['source_rows'][rows], pairing['target_rows'][rows]
        fold, pair = divmod(number, 7)
        checks = (
            (q['groups'], data['groups'][source]),
            (q['object_ids'], data['object_ids'][source]),
            (q['source_ids'], data['ids'][source]),
            (q['target_ids'], data['ids'][target]),
            (q['batch'], data['batches'][source]),
            (q['source_plate'], data['plates'][source, 0]),
            (q['source_dose'], pairing['source_dose'][rows]),
            (q['target_dose'], pairing['target_dose'][rows]),
            (data['groups'][source], data['groups'][target]),
        )
        for left, right in checks:
            np.testing.assert_array_equal(left, right)
        assert np.all(pairing['outer_roles'][rows, fold] == 'DEV_EVAL')
        assert np.all(pairing['task_index'][rows] == pair)
        for s, t, target_roles in zip(source, target, pairing['target_roles'][rows]):
            assert data['plates'][s, 0] not in data['plates'][t, target_roles]
            assert set(data['plates'][s, 1:]) == set(data['plates'][t, target_roles])
        for group in np.unique(q['groups']):
            previous = group_fold.setdefault(str(group), fold)
            assert previous == fold, 'OOF chemical group appears in multiple query folds'
        all_query_groups.extend(q['groups'])
        residual = np.stack([pred[task+'_actual'].astype(float) -
                             pred[task+'_RIDGE_RESPONSE'].astype(float) for task in TASKS], axis=1)
        x = pred['source_x'].astype(float)
        assert x.shape[1] == 951 and residual.shape == (len(x), 2, 951)
        if not (np.isfinite(x).all() and np.isfinite(residual).all()):
            raise ValueError('Nonfinite saved response residual or source X')
        annotated = biology['arrays']['target_mask'][source]
        zero_x = np.linalg.norm(x, axis=1) <= 1e-12
        take = annotated & ~zero_x
        groups = q['groups'][take]
        r = residual[take]
        grams = relation_grams(biology['arrays']['target'][source[take]], x[take],
                                q['batch'][take], q['source_plate'][take])
        i, j, alias_weights = group_dyad_pairs(groups)
        if not len(i):
            raise ValueError('No distinct annotated chemical-group pairs')
        covariance = np.stack([(r[:, task] @ r[:, task].T)[i, j] / 951.
                               for task in range(2)], axis=1)
        design = grams[i, j]
        labels, energy = mean_group_rows(np.square(r).mean(axis=2), groups)
        _, group_r = mean_group_rows(r, groups)
        _, energy_all = mean_group_rows(np.square(residual).mean(axis=2), q['groups'])
        _, group_r_all = mean_group_rows(residual, q['groups'])
        mean_energy = energy.mean(axis=0)
        mean_vector_energy = np.square(group_r.mean(axis=0)).mean(axis=1)
        raw_mean_energy_all = energy_all.mean(axis=0)
        mean_vector_energy_all = np.square(group_r_all.mean(axis=0)).mean(axis=1)
        positive = design[:, 0] > 0
        pair_group_left, pair_group_right = groups[i], groups[j]
        pair_codes = np.array(['|'.join(sorted((str(a), str(b))))
                               for a, b in zip(pair_group_left, pair_group_right)])
        positive_groups = np.unique(np.r_[pair_group_left[positive], pair_group_right[positive]])
        def weighted_fraction(mask, subset=None):
            subset = np.ones(len(mask), bool) if subset is None else subset
            return float(np.average(mask[subset], weights=alias_weights[subset])) if subset.any() else None
        info = dict(unit=number, outer_fold=fold, dose_pair_index=pair,
            source_dose=float(q['source_dose'][0]), target_dose=float(q['target_dose'][0]),
            n_query_rows=len(x), n_query_groups=len(np.unique(q['groups'])),
            n_annotated_rows=int(annotated.sum()), n_zero_source_norm=int(zero_x.sum()),
            n_diagnostic_rows=int(take.sum()), n_diagnostic_groups=len(labels),
            n_row_pairs=len(i), n_group_dyads=len(np.unique(pair_codes)),
            n_target_positive_row_pairs=int(positive.sum()),
            n_target_positive_group_dyads=len(np.unique(pair_codes[positive])),
            n_target_edge_groups=len(positive_groups),
            target_edge_groups=positive_groups,
            weighted_target_positive_fraction=weighted_fraction(positive),
            same_batch_fraction=weighted_fraction(design[:, 3] > 0),
            same_plate_fraction=weighted_fraction(design[:, 4] > 0),
            target_positive_same_batch_fraction=weighted_fraction(design[:, 3] > 0, positive),
            target_positive_same_plate_fraction=weighted_fraction(design[:, 4] > 0, positive),
            target_positive_negative_morph_fraction=weighted_fraction(design[:, 1] < 0, positive),
            diagnostic_raw_residual_energy=mean_energy,
            diagnostic_residual_mean_vector_energy=mean_vector_energy,
            diagnostic_centered_diagonal_trace_per_coordinate=mean_energy-mean_vector_energy,
            all_query_raw_residual_energy=raw_mean_energy_all,
            all_query_residual_mean_vector_energy=mean_vector_energy_all,
            all_query_centered_diagonal_trace_per_coordinate=raw_mean_energy_all-mean_vector_energy_all)
        all_units.append(dict(info=info, x=design, y=covariance, w=alias_weights,
            left=pair_group_left, right=pair_group_right, groups=labels,
            energy=energy, pair_codes=pair_codes))
    return all_units, np.unique(all_query_groups), biology['report']


def bootstrap_moments(units, groups, replicates, seed, chunk=32):
    index = {str(group): i for i, group in enumerate(groups)}
    rng = np.random.default_rng(seed)
    multiplicity = np.vstack((np.ones(len(groups), dtype=np.int16),
        rng.multinomial(len(groups), np.full(len(groups), 1./len(groups)),
                        size=replicates))).astype(np.int16)
    count, n_units, k = replicates+1, len(units), len(KERNEL_NAMES)
    gram = np.empty((count, n_units, k, k))
    cross = np.empty((count, n_units, k, len(TASKS)))
    energy = np.empty((count, n_units, len(TASKS)))
    point_sx, point_sy = [], []
    for u, unit in enumerate(units):
        left = np.array([index[str(g)] for g in unit['left']])
        right = np.array([index[str(g)] for g in unit['right']])
        gi = np.array([index[str(g)] for g in unit['groups']])
        for first in range(0, count, chunk):
            stop = min(first+chunk, count)
            m = multiplicity[first:stop]
            weights = node_pair_weights(m, left, right, unit['w'])
            g, h, sx, sy = unit_moments(unit['x'], unit['y'], weights)
            gram[first:stop, u], cross[first:stop, u] = g, h
            node_weight = m[:, gi].astype(float)
            energy[first:stop, u] = node_weight @ unit['energy'] / node_weight.sum(axis=1)[:, None]
            if first == 0:
                point_sx.append(sx[0])
                point_sy.append(sy[0])
        print(json.dumps(dict(stage='node_bootstrap_moments', unit=u,
            groups=len(gi), pairs=len(left), replicates=replicates)), flush=True)
    return gram, cross, energy, np.array(point_sx), np.array(point_sy), multiplicity


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def summarize(units, groups, gram, cross, energy, sx, sy, replicates):
    output, flat = {}, []
    scopes = {'ALL_SEVEN_PAIRS': list(range(35))}
    scopes.update({f'PAIR_{p}': [u for u in range(35) if u % 7 == p] for p in range(7)})
    saved_boot = {}
    for scope, use in scopes.items():
        g, h, e = gram[:, use].mean(axis=1), cross[:, use].mean(axis=1), energy[:, use].mean(axis=1)
        scope_groups = np.unique(np.concatenate([units[u]['groups'] for u in use]))
        edge_groups = np.unique(np.concatenate([units[u]['info']['target_edge_groups'] for u in use]))
        info = dict(n_units=len(use), n_annotated_groups=len(scope_groups),
            n_target_edge_groups=len(edge_groups),
            n_row_pairs=sum(units[u]['info']['n_row_pairs'] for u in use),
            n_unit_group_dyads=sum(units[u]['info']['n_group_dyads'] for u in use),
            n_unique_group_dyads=len(set().union(*(set(units[u]['pair_codes']) for u in use))),
            n_target_positive_row_pairs=sum(units[u]['info']['n_target_positive_row_pairs'] for u in use),
            n_target_positive_unit_group_dyads=sum(units[u]['info']['n_target_positive_group_dyads'] for u in use),
            n_unique_target_positive_group_dyads=len(set().union(*(
                set(units[u]['pair_codes'][units[u]['x'][:, 0] > 0]) for u in use))),
            target_positive_same_batch_fraction=float(np.mean([
                units[u]['info']['target_positive_same_batch_fraction'] for u in use])),
            target_positive_same_plate_fraction=float(np.mean([
                units[u]['info']['target_positive_same_plate_fraction'] for u in use])),
            target_positive_negative_morph_fraction=float(np.mean([
                units[u]['info']['target_positive_negative_morph_fraction'] for u in use])),
            target_design_unexplained_by_morphology_and_technical=partial_design_fraction(g[0], 0, (1, 3, 4)),
            target_design_unexplained_in_signed_joint=partial_design_fraction(g[0], 0, (1, 2, 3, 4)),
            signed_design_unexplained_in_signed_joint=partial_design_fraction(g[0], 2, (0, 1, 3, 4)),
            outcomes={})
        for task, task_name in enumerate(TASKS):
            scales = dict(raw_residual_energy=float(e[0, task]),
                diagnostic_residual_mean_vector_energy=float(np.mean([
                    units[u]['info']['diagnostic_residual_mean_vector_energy'][task] for u in use])),
                diagnostic_centered_diagonal_trace_per_coordinate=float(np.mean([
                    units[u]['info']['diagnostic_centered_diagonal_trace_per_coordinate'][task] for u in use])),
                all_query_raw_residual_energy=float(np.mean([
                    units[u]['info']['all_query_raw_residual_energy'][task] for u in use])),
                all_query_centered_diagonal_trace_per_coordinate=float(np.mean([
                    units[u]['info']['all_query_centered_diagonal_trace_per_coordinate'][task] for u in use])))
            task_result = dict(scales=scales, models={})
            for model, indices in MODELS.items():
                fits = [solve_moments(g[b], h[b, :, task], indices) for b in range(replicates+1)]
                raw = np.array([f['raw'] for f in fits])
                bounded = np.array([f['nonnegative'] for f in fits])
                ref = fits[0]
                summary = dict(rank=ref['rank'], n_components=len(indices),
                    identifiable=ref['identifiable'], condition_number=ref['condition_number'],
                    design_eigenvalues=ref['eigenvalues'], kernel_correlation=ref['kernel_correlation'],
                    bootstrap_rank_deficient_count=sum(not f['identifiable'] for f in fits[1:]),
                    unit_nuisance_intercepts_raw=sy[use, task] - sx[use][:, indices] @ raw[0],
                    unit_nuisance_intercepts_nonnegative=sy[use, task] - sx[use][:, indices] @ bounded[0],
                    components={})
                for j, idx in enumerate(indices):
                    name = KERNEL_NAMES[idx]
                    relative_raw = raw[:, j] / e[:, task]
                    relative_bounded = bounded[:, j] / e[:, task]
                    component = dict(raw=float(raw[0, j]), raw_ci95=interval(raw[1:, j]),
                        nonnegative=float(bounded[0, j]), nonnegative_ci95=interval(bounded[1:, j]),
                        nonnegative_at_boundary=bool(bounded[0, j] <= 1e-12),
                        bootstrap_nonnegative_boundary_fraction=float(np.mean(bounded[1:, j] <= 1e-12)),
                        raw_relative_to_residual_energy=float(relative_raw[0]),
                        raw_relative_ci95=interval(relative_raw[1:]),
                        nonnegative_relative_to_residual_energy=float(relative_bounded[0]),
                        nonnegative_relative_ci95=interval(relative_bounded[1:]))
                    summary['components'][name] = component
                    flat.append(dict(scope=scope, task=task_name, model=model, component=name,
                        n_units=len(use), n_groups=len(scope_groups), n_row_pairs=info['n_row_pairs'],
                        identifiable=ref['identifiable'], raw=component['raw'],
                        raw_low=component['raw_ci95'][0], raw_high=component['raw_ci95'][1],
                        nonnegative=component['nonnegative'], nonnegative_low=component['nonnegative_ci95'][0],
                        nonnegative_high=component['nonnegative_ci95'][1],
                        raw_fraction=component['raw_relative_to_residual_energy'],
                        fraction_low=component['raw_relative_ci95'][0], fraction_high=component['raw_relative_ci95'][1],
                        boundary_fraction=component['bootstrap_nonnegative_boundary_fraction']))
                saved_boot[f'{scope}_{task_name}_{model}_raw'] = raw
                saved_boot[f'{scope}_{task_name}_{model}_nonnegative'] = bounded
                task_result['models'][model] = summary
            info['outcomes'][task_name] = task_result
        info['paired_cross_minus_same_components'] = {}
        for model, indices in MODELS.items():
            a = saved_boot[f'{scope}_CROSS_{model}_raw'] - saved_boot[f'{scope}_SAME_{model}_raw']
            info['paired_cross_minus_same_components'][model] = {
                KERNEL_NAMES[idx]: dict(raw_difference=float(a[0, j]), ci95=interval(a[1:, j]))
                for j, idx in enumerate(indices)}
        output[scope] = info
    return output, flat, saved_boot


def markdown_report(summary):
    aggregate = summary['scopes']['ALL_SEVEN_PAIRS']
    lines = ['# Overlapping-relation covariance of RIDGE_RESPONSE residuals', '',
        'This diagnostic uses saved out-of-fold 951-dimensional response residuals. '
        'It is not a CORE 9-dimensional measurement-noise decomposition.', '',
        'The endpoint is mean response in three held-out plates at the source dose (SAME) '
        'or adjacent higher dose (CROSS). Query responses are used only for this diagnostic.', '',
        f"Support: {aggregate['n_annotated_groups']} annotated chemical groups; "
        f"{aggregate['n_target_edge_groups']} groups incident to a positive target-overlap edge; "
        f"{aggregate['n_row_pairs']:,} row pairs across 35 units; "
        f"{aggregate['n_target_positive_unit_group_dyads']:,} target-positive unit/group dyads "
        f"({aggregate['n_unique_target_positive_group_dyads']:,} distinct target-positive group dyads). "
        'These dyads are not independent observations.', '',
        '## Estimates', '',
        'Coefficients are off-diagonal covariance-moment estimates in average-coordinate '
        'squared response units. Intervals resample chemical groups, not pairs.', '',
        '| Task | Model | Component | Raw estimate [95% interval] | NNLS | Raw / residual energy |',
        '| --- | --- | --- | --- | --- | --- |']
    for task in TASKS:
        for model, values in aggregate['outcomes'][task]['models'].items():
            for component in ('target', 'target_morphology'):
                if component not in values['components']:
                    continue
                c = values['components'][component]
                lo, hi = c['raw_ci95']
                lines.append(f"| {task} | {model} | {component} | {c['raw']:.6g} "
                    f"[{lo:.6g}, {hi:.6g}] | {c['nonnegative']:.6g} | "
                    f"{100*c['raw_relative_to_residual_energy']:.3g}% |")
    lines += ['', '## Residual scales', '',
        '| Task | Raw diagonal energy | Squared unit-mean residual | Centered diagonal trace / 951 |',
        '| --- | --- | --- | --- |']
    for task in TASKS:
        s = aggregate['outcomes'][task]['scales']
        lines.append(f"| {task} | {s['raw_residual_energy']:.6g} | "
            f"{s['diagnostic_residual_mean_vector_energy']:.6g} | "
            f"{s['diagnostic_centered_diagonal_trace_per_coordinate']:.6g} |")
    lines += ['', '## Kernel separability', '',
        '| Scope | Target unexplained by morphology + technical | Target unexplained in signed joint | Signed interaction unexplained in signed joint | Joint condition number |',
        '| --- | --- | --- | --- | --- |']
    for scope, s in summary['scopes'].items():
        cond = s['outcomes']['SAME']['models']['SIGNED_JOINT']['condition_number']
        lines.append(f"| {scope} | {s['target_design_unexplained_by_morphology_and_technical']:.4f} | "
            f"{s['target_design_unexplained_in_signed_joint']:.4f} | "
            f"{s['signed_design_unexplained_in_signed_joint']:.4f} | {cond} |")
    corr = aggregate['outcomes']['SAME']['models']['SIGNED_JOINT']['kernel_correlation'][0, 2]
    lines += ['', f'The unit-demeaned target and target × morphology columns have correlation {corr:.4f}. '
        'The pooled design is full rank, but the overlapping columns widen separate-component intervals. '
        'A positive target association in the adjusted model is not evidence that the nonnegative '
        'target coefficient must remain positive after adding the signed interaction.', '',
        'Coefficient / residual-energy ratios provide a common scale; they are not fractions of '
        'variance explained, because diagonal moments are not fitted and component coefficients '
        'are not constrained to sum to the observed residual energy.']
    lines += ['', '## Estimand and interpretation', '',
        '- Only pairs within one fitted-model/dose unit and from distinct chemical groups are used. '
        'Both endpoints must carry the existing multi-target annotation; missing annotation is not treated as known non-overlap.',
        '- Alias rows receive reciprocal within-unit group-size weights, giving equal group dyads. '
        'Each unit contributes equal total weight. All target, morphology, interaction and technical kernels '
        'are PSD full Gram matrices; the target × morphology Schur kernel retains signed off-diagonal values.',
        '- A freely fitted unit intercept absorbs common residual bias. The nonnegative fit constrains only '
        'kernel coefficients, after intercept removal. It is covariance-moment regression, not a full covariance '
        'likelihood, and free intercepts do not guarantee the assembled covariance is PSD.',
        '- A boundary-zero estimate or interval crossing zero does not prove that the biological component is absent. '
        'Negative raw estimates are retained. Rank deficiency, technical overlap and correlated kernels can prevent '
        'a separate component interpretation.',
        f"- The {summary['bootstrap_replicates']:,} fixed-seed bootstrap draws resample global chemical groups jointly across all doses and tasks. "
        'An edge is weighted by the product of its two node multiplicities. Intervals condition on the fitted '
        'models, fixed five folds and the observed experimental batches; they are not independent verification '
        'or uncertainty over future batches/model fitting.',
        '- No component is converted into an optimal individual borrowing weight or used to tune response models. '
        'The seven pair-specific results and paired CROSS-minus-SAME coefficients are available in summary.json.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError('At least one node-bootstrap replicate is required')
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        units, groups, annotation_report = load_units()
        print(json.dumps(dict(stage='loaded', units=len(units), groups=len(groups),
            annotated_groups=len(np.unique(np.concatenate([u['groups'] for u in units]))))), flush=True)
        g, h, e, sx, sy, multiplicity = bootstrap_moments(units, groups, args.replicates, SEED)
        scopes, flat, bootstrap = summarize(units, groups, g, h, e, sx, sy, args.replicates)
    summary = dict(state='COMPLETE', diagnostic='951D RIDGE_RESPONSE OOF residual relation covariance',
        outcome='Three future-plate mean response at SAME or adjacent higher CROSS dose',
        protected_data_read=False, new_models_trained=False, query_outcomes_used_for_gates=False,
        prepared_Y_decoded=False, coordinate_dimension=951, seed=SEED,
        bootstrap_replicates=args.replicates, bootstrap_unit='global chemical connectivity group',
        bootstrap_global_groups=len(groups), bootstrap_edge_weight='node multiplicity product',
        variance_units='mean squared response coordinate, native common 951D frame',
        uncertainty='conditional on fixed fitted models, folds and observed batches; no independent verification',
        component_interpretation='PSD kernel coefficients fitted to off-diagonal covariance moments; not a full covariance likelihood',
        nuisance='unconstrained unit intercepts; assembled covariance not guaranteed PSD',
        model_indices=MODELS, kernel_names=KERNEL_NAMES,
        model_note='TARGET_ONLY also includes unit intercepts; TARGET_TECHNICAL adds batch/plate; '
                   'ADJUSTED adds source-X cosine; SIGNED_ONLY replaces target with target×morphology; '
                   'SIGNED_JOINT includes both target kernels.',
        annotation_semantics='Existing 0 < reported nM <= 1000 multi-target assay prior, not observed HUVEC target engagement',
        annotation_report=annotation_report, units=[u['info'] for u in units], scopes=scopes,
        elapsed_seconds=time.monotonic()-started)
    write_json(OUTPUT/'summary.json', summary)
    (OUTPUT/'REPORT.md').write_text(markdown_report(summary))
    with (OUTPUT/'component_estimates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    np.savez_compressed(OUTPUT/'bootstrap_components.npz', **bootstrap,
        global_groups=groups, node_multiplicity=multiplicity, residual_energy=e)
    np.savez_compressed(OUTPUT/'moment_sufficient_statistics.npz', gram=g, cross=h,
        unit_mean_design=sx, unit_mean_pair_covariance=sy, kernel_names=np.asarray(KERNEL_NAMES))
    print(json.dumps(dict(state='COMPLETE', output=str(OUTPUT), seconds=summary['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
