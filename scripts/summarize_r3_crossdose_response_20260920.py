"""Group-equal paired summaries of the prespecified 35 response-borrowing units.

No models are fitted here. Twenty random-reference scores are averaged within
each query, never treated as independent biological observations or averaged
as predictions. SAME/CROSS and every arm share group-bootstrap resamples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json

ROOT = PROJECT/'runs/r3_crossdose_response_20260920_v1'
REPORT = PROJECT/'reports/r3_crossdose_response_20260920_v1'
TASKS = ('SAME', 'CROSS')
METRICS = ('profile_mse', 'cosine_loss', 'lognorm_squared_error')
FAMILIES, KINDS, STRENGTHS = ('TARGET', 'MORPH'), ('DIRECT', 'TRANSPORT'), ('CAL', 'A025', 'A100')
REPLICATES, BOOTSTRAPS, SEED = 20, 2000, 2026092012


def read_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def compact_scores(scores, names):
    """Keep real arms and within-query mean random scores, without refitting."""
    names = list(names)
    fields = ['RIDGE_RESPONSE', 'TRAIN_MEAN', 'SOURCE_X']
    values = [scores[:, :, names.index(name), :] for name in fields]
    for family in FAMILIES:
        for kind in KINDS:
            for strength in STRENGTHS:
                name = f'{family}_{kind}_{strength}'
                fields.append(name)
                values.append(scores[:, :, names.index(name), :])
                random = [names.index(f'{family}_R{rep:02d}_{kind}_{strength}')
                          for rep in range(1, REPLICATES+1)]
                fields.append(f'{family}_RANDOM_MEAN_{kind}_{strength}')
                values.append(scores[:, :, random, :].mean(axis=2))
    return np.stack(values, axis=2), fields


def group_mean_scores(scores, groups, mask):
    """First average dose-pairs/aliases within chemical group, then across groups."""
    mask = np.asarray(mask, bool)
    labels, inverse, count = np.unique(np.asarray(groups)[mask], return_inverse=True, return_counts=True)
    rows = np.moveaxis(np.asarray(scores)[:, mask], 1, 0)
    shape = (len(labels),)+rows.shape[1:]
    sums = np.zeros(shape, dtype=float)
    np.add.at(sums, inverse, rows)
    means = sums/count.reshape((-1,)+(1,)*(rows.ndim-1))
    return labels, means, count


def bootstrap_group_risks(group_values, *, replicates=BOOTSTRAPS, seed=SEED):
    """One count matrix generates paired resamples for every task/arm/metric."""
    g = len(group_values)
    if g < 2:
        raise ValueError('At least two chemical groups required for bootstrap intervals')
    rng = np.random.default_rng(seed)
    indices = rng.integers(g, size=(replicates, g))
    counts = np.zeros((replicates, g), dtype=float)
    np.add.at(counts, (np.arange(replicates)[:, None], indices), 1.)
    flat = np.asarray(group_values).reshape(g, -1)
    sampled = (counts @ flat)/g
    return sampled.reshape((replicates,)+group_values.shape[1:])


def scalar_interval(value, draws):
    return dict(estimate=float(value), ci95=np.quantile(draws, [.025, .975]).tolist())


def contrast(mean, boot, a, b):
    """Negative risk difference and positive relative improvement favour A."""
    result = {}
    for metric_index, metric in enumerate(METRICS):
        difference = mean[:, a, metric_index]-mean[:, b, metric_index]
        draws = boot[:, :, a, metric_index]-boot[:, :, b, metric_index]
        denominator = mean[:, b, metric_index]
        if np.any(denominator <= 0) or np.any(boot[:, :, b, metric_index] <= 0):
            raise ValueError('Relative improvement requires positive baseline risks')
        relative = 1-mean[:, a, metric_index]/denominator
        rd = 1-boot[:, :, a, metric_index]/boot[:, :, b, metric_index]
        result[metric] = {
            task: dict(risk_difference=scalar_interval(difference[k], draws[:, k]),
                       relative_improvement=scalar_interval(relative[k], rd[:, k]))
            for k, task in enumerate(TASKS)}
        result[metric]['CROSS_minus_SAME_relative_improvement'] = scalar_interval(
            relative[1]-relative[0], rd[:, 1]-rd[:, 0])
    return result


def summary_panel(scores, groups, mask, names, *, seed=SEED):
    labels, gm, counts = group_mean_scores(scores, groups, mask)
    if len(labels) < 2:
        return dict(n_rows=int(mask.sum()), n_chemical_groups=len(labels), unavailable=True)
    means = gm.mean(0)
    boot = bootstrap_group_risks(gm, seed=seed)
    risks = {name: {task: {metric: float(means[t, i, m]) for m, metric in enumerate(METRICS)}
                    for t, task in enumerate(TASKS)} for i, name in enumerate(names)}
    comparisons = {}
    ridge = names.index('RIDGE_RESPONSE')
    for family in FAMILIES:
        for kind in KINDS:
            for strength in STRENGTHS:
                arm = f'{family}_{kind}_{strength}'
                random = f'{family}_RANDOM_MEAN_{kind}_{strength}'
                comparisons[arm+' minus RIDGE_RESPONSE'] = contrast(means, boot, names.index(arm), ridge)
                comparisons[arm+' minus '+random] = contrast(means, boot, names.index(arm), names.index(random))
    # The generic-source-information comparison on exactly the TARGET subset.
    comparisons['TARGET_TRANSPORT_CAL minus MORPH_TRANSPORT_CAL'] = contrast(
        means, boot, names.index('TARGET_TRANSPORT_CAL'), names.index('MORPH_TRANSPORT_CAL'))
    return dict(n_rows=int(mask.sum()), n_chemical_groups=len(labels),
        group_row_count=dict(min=int(counts.min()), median=float(np.median(counts)), max=int(counts.max())),
        group_equal_mean_risks=risks, comparisons=comparisons)


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), min=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), max=float(a.max())) if len(a) else dict(n=0)


def unit_statistics(unit):
    folder = ROOT/f'unit_{unit:02d}'
    complete = json.loads((folder/'complete.json').read_text())
    calibrations = json.loads((folder/'calibration.json').read_text())
    fitting = json.loads((folder/'ridge_fitting.json').read_text())
    refs = read_npz(folder/'references.npz')
    saved_predictions = read_npz(folder/'predictions.npz')
    ncal = len(refs['cal_ids'])
    statistics = dict(unit=unit, complete=complete, ridge_fitting=fitting, calibration=calibrations,
                      query_reference_diagnostics={},
                      saved_profile_zero_norm_audit={key: int((np.linalg.norm(value.astype(float), axis=1) <= 1e-12).sum())
                          for key, value in saved_predictions.items()},
                      zero_norm_audit_scope='Saved float32 source, actual and main CAL predictions only; random and forced-strength prediction vectors were not persisted')
    for family in FAMILIES:
        sup = refs[family+'_support'][ncal:]
        statistics['query_reference_diagnostics'][family] = dict(
            n_supported=int(sup.sum()), ess=describe(refs[family+'_ess'][ncal:][sup]),
            same_source_plate_weight=describe(refs[family+'_same_plate_weight'][ncal:][sup]),
            donor_count=describe((refs[family+'_weights'][0, ncal:] > 0).sum(1)[sup]))
    return statistics


def descriptive_layout(scores, population, names, scopes):
    """No bootstrap confidence claims for four reused experimental layouts."""
    result = {}
    main = names.index('TARGET_TRANSPORT_CAL')
    random = names.index('TARGET_RANDOM_MEAN_TRANSPORT_CAL')
    morph = names.index('MORPH_TRANSPORT_CAL')
    morph_random = names.index('MORPH_RANDOM_MEAN_TRANSPORT_CAL')
    ridge = names.index('RIDGE_RESPONSE')
    for scope, support in scopes.items():
        entries = []
        for batch in np.unique(population['batch']):
            for mode, batchmask in (('within_batch', population['batch'] == batch),
                                    ('leave_batch_out', population['batch'] != batch)):
                mask = support & batchmask
                labels, gm, _ = group_mean_scores(scores, population['groups'], mask)
                if not len(labels):
                    continue
                mean = gm.mean(0)
                row = dict(batch=str(batch), mode=mode, n_rows=int(mask.sum()), n_groups=len(labels))
                for t, task in enumerate(TASKS):
                    row[task] = dict(ridge_mse=float(mean[t, ridge, 0]), target_mse=float(mean[t, main, 0]),
                        random_mean_mse=float(mean[t, random, 0]),
                        target_relative_improvement_vs_ridge=float(1-mean[t, main, 0]/mean[t, ridge, 0]),
                        target_minus_random=float(mean[t, main, 0]-mean[t, random, 0]),
                        morphology_mse=float(mean[t, morph, 0]), morphology_random_mean_mse=float(mean[t, morph_random, 0]),
                        morphology_relative_improvement_vs_ridge=float(1-mean[t, morph, 0]/mean[t, ridge, 0]),
                        morphology_minus_random=float(mean[t, morph, 0]-mean[t, morph_random, 0]))
                entries.append(row)
        result[scope] = entries
    return dict(interpretation='Descriptive within/leave-batch-out sensitivity, not independent-layout inference', scopes=result)


def random_draw_dispersion(full_scores, full_names, groups, scopes):
    result = {}
    for scope, mask in scopes.items():
        _, gm, _ = group_mean_scores(full_scores, groups, mask)
        mean = gm.mean(0)
        result[scope] = {}
        for family in FAMILIES:
            for kind in KINDS:
                for strength in STRENGTHS:
                    real = full_names.index(f'{family}_{kind}_{strength}')
                    indices = [full_names.index(f'{family}_R{r:02d}_{kind}_{strength}') for r in range(1, 21)]
                    record = {}
                    for t, task in enumerate(TASKS):
                        record[task] = {}
                        for m, metric in enumerate(METRICS):
                            vals = mean[t, indices, m]
                            record[task][metric] = dict(mean=float(vals.mean()), std_between_draws=float(vals.std(ddof=1)),
                                min=float(vals.min()), max=float(vals.max()),
                                random_draws_lower_risk_than_real=int((vals < mean[t, real, m]).sum()),
                                interpretation='Random reference assignment dispersion, not a biological sample confidence interval')
                    result[scope][f'{family}_{kind}_{strength}'] = record
    return result


def alpha_summary(units):
    result = {}
    for task in TASKS:
        result[task] = {}
        for family in FAMILIES:
            for kind in KINDS:
                real = [u['calibration'][task][f'{family}_{kind}'] for u in units]
                random = [u['calibration'][task][f'{family}_R{r:02d}_{kind}']
                          for u in units for r in range(1, 21)]
                result[task][f'{family}_{kind}'] = dict(
                    real_alpha=describe([v['alpha'] for v in real]),
                    real_nonzero_units=sum(v['alpha'] > 0 for v in real),
                    real_supported_CAL_groups=describe([v['n_groups'] for v in real]),
                    real_reasons=dict(Counter(v['reason'] for v in real)),
                    random_alpha=describe([v['alpha'] for v in random]),
                    random_nonzero_unit_draws=sum(v['alpha'] > 0 for v in random),
                    random_count_note='35 units x 20 control draws, not 700 independent datasets')
    return result


def retrieval_context_summary(units):
    """Descriptive source-plate association; no inference of biological mechanism."""
    result = {}
    for family in FAMILIES:
        counts = np.array([u['query_reference_diagnostics'][family]['n_supported'] for u in units], float)
        real = np.array([u['query_reference_diagnostics'][family]['same_source_plate_weight'].get('mean', 0.) for u in units])
        random = np.array([[v['query_same_plate_weight'] for v in u['complete']['supports'][family]['random_audit']]
                           for u in units])
        retained = np.array([[v['retained_weight_mass'] for v in u['complete']['supports'][family]['random_audit']]
                             for u in units])
        real_mean = float(np.average(real, weights=counts))
        random_means = np.average(random, weights=counts, axis=0)
        result[family] = dict(supported_query_rows=int(counts.sum()),
            real_same_source_plate_weight=real_mean,
            random_same_source_plate_weight_mean=float(random_means.mean()),
            random_same_source_plate_weight_across_draws=describe(random_means),
            real_minus_random_same_plate_weight=float(real_mean-random_means.mean()),
            random_retained_original_weight_mass_unit_mean=float(retained.mean()),
            aggregation='Descriptive supported-query-row-weighted plate association; retained weight mass is an unweighted mean over unit/control draws')
    return result


def interval_text(value, scale=1.):
    lo, hi = value['ci95']
    return f"{value['estimate']*scale:.4f} [{lo*scale:.4f}, {hi*scale:.4f}]"


def make_report(result):
    panel = result['panels']['pooled']['target_supported']
    risks = panel['group_equal_mean_risks']
    comp = panel['comparisons']
    text = ['# R3: same-dose and cross-dose response borrowing', '',
        f"All 35 prespecified units completed. {result['population']['n_rows']} paired condition rows, "
        f"{result['population']['n_groups']} chemical groups; TARGET-supported readout: "
        f"{panel['n_rows']} rows in {panel['n_chemical_groups']} groups.", '',
        'The endpoint is response-profile prediction, not Gamma, NULL, acquisition value or a change to CORE.', '',
        '## Primary comparison: TARGET-supported query objects', '',
        '| Model | SAME profile MSE | CROSS profile MSE |', '|---|---:|---:|']
    for name in ('RIDGE_RESPONSE', 'TRAIN_MEAN', 'SOURCE_X', 'TARGET_TRANSPORT_CAL',
                 'TARGET_RANDOM_MEAN_TRANSPORT_CAL', 'MORPH_TRANSPORT_CAL', 'MORPH_RANDOM_MEAN_TRANSPORT_CAL',
                 'TARGET_TRANSPORT_A025', 'TARGET_TRANSPORT_A100', 'TARGET_DIRECT_CAL'):
        text.append(f"| {name} | {risks[name]['SAME']['profile_mse']:.7f} | {risks[name]['CROSS']['profile_mse']:.7f} |")
    text += ['', '| Primary contrast | SAME risk difference [95% CI] | CROSS risk difference [95% CI] |', '|---|---:|---:|']
    for baseline in ('RIDGE_RESPONSE', 'TARGET_RANDOM_MEAN_TRANSPORT_CAL'):
        rec = comp['TARGET_TRANSPORT_CAL minus '+baseline]['profile_mse']
        text.append(f"| TARGET_TRANSPORT_CAL − {baseline} | {interval_text(rec['SAME']['risk_difference'])} | {interval_text(rec['CROSS']['risk_difference'])} |")
    rec = comp['TARGET_TRANSPORT_CAL minus RIDGE_RESPONSE']['profile_mse']
    text += ['', 'Relative MSE improvement versus Ridge (positive is better):', '',
        f"- SAME: {interval_text(rec['SAME']['relative_improvement'], 100)}%.",
        f"- CROSS: {interval_text(rec['CROSS']['relative_improvement'], 100)}%.",
        f"- CROSS minus SAME improvement: {interval_text(rec['CROSS_minus_SAME_relative_improvement'], 100)} percentage points.",
        '', '## Seven prespecified dose pairs', '',
        '| Dose pair µM | Supported rows / groups | SAME improvement % | CROSS improvement % | CROSS−SAME pp |',
        '|---|---:|---:|---:|---:|']
    for key, block in result['panels'].items():
        if key == 'pooled':
            continue
        p = block['target_supported']
        rec = p['comparisons']['TARGET_TRANSPORT_CAL minus RIDGE_RESPONSE']['profile_mse']
        text.append(f"| {key} | {p['n_rows']} / {p['n_chemical_groups']} | {interval_text(rec['SAME']['relative_improvement'],100)} | {interval_text(rec['CROSS']['relative_improvement'],100)} | {interval_text(rec['CROSS_minus_SAME_relative_improvement'],100)} |")
    full = result['panels']['pooled']['all']
    morph = full['comparisons']['MORPH_TRANSPORT_CAL minus RIDGE_RESPONSE']
    text += ['', '## Morphology borrowing: full-cohort secondary comparison', '',
        f"Morphology support covers all {full['n_rows']} query rows in {full['n_chemical_groups']} chemical groups. Positive percentages below are lower loss; negative percentages are deterioration.", '',
        '| Metric | SAME relative improvement % [95% CI] | CROSS relative improvement % [95% CI] |',
        '|---|---:|---:|']
    for metric in METRICS:
        text.append(f"| {metric} | {interval_text(morph[metric]['SAME']['relative_improvement'],100)} | {interval_text(morph[metric]['CROSS']['relative_improvement'],100)} |")
    random = full['comparisons']['MORPH_TRANSPORT_CAL minus MORPH_RANDOM_MEAN_TRANSPORT_CAL']['profile_mse']
    text += ['', 'MORPH TRANSPORT CAL versus its matched-random mean MSE:', '',
        f"- SAME risk difference: {interval_text(random['SAME']['risk_difference'])}.",
        f"- CROSS risk difference: {interval_text(random['CROSS']['risk_difference'])}.",
        '', 'The small profile-MSE gain does not imply uniformly better response prediction: the log-norm squared-error readout worsens. The protocol requires reporting this amplitude trade-off alongside profile MSE and cosine loss.',
        '', 'Forced-strength deterioration is specific to the candidate: TARGET TRANSPORT at alpha=.25 and 1 is substantially worse, while DIRECT has its own saved contrasts and must not be described using the TRANSPORT result.',
        '', '## Scope and saved outputs', '',
        '- Group-equal risks first average all dose pairs and aliases within each chemical group. Relative improvement is a ratio of those mean risks, not a mean of per-object percentages.',
        '- Intervals use 2,000 paired chemical-group bootstrap resamples, sharing each resample across SAME, CROSS and every model. They are development-cohort intervals conditional on the saved fitted procedures.',
        '- Matched-random means average scores of 20 separately calibrated reference draws. Draws are not independent biological observations and this is not an ensemble prediction.',
        '- `summary.json` includes all-query and morphology-supported panels; all strengths, DIRECT candidates and three metrics; batch/layout descriptive sensitivity; random-draw dispersion and calibration strengths.',
        '- Cosine is defined as zero (loss one) if either profile norm is at most 1e-12; log norms use log(max(norm,1e-12)). Zero-norm counts are audited for saved actual and main CAL profiles only; random and forced prediction vectors were not saved.',
        '- `aggregate_scores.npz` retains compact per-query scores, identities, folds, doses and support for reuse. `unit_statistics.json` retains every unit calibration and retrieval summary.',
        '- No inference about Gamma, NULL or the frozen acquisition strategy follows directly from this response endpoint.']
    return '\n'.join(text)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    missing = [u for u in range(35) if not (ROOT/f'unit_{u:02d}/complete.json').exists()]
    if missing:
        raise SystemExit('Aggregation waits for all 35 completed units; missing '+str(missing))
    qualification = read_npz(REPORT/'qualification.npz')
    collected, population, unit_reports = [], [], []
    for u in range(35):
        block = read_npz(ROOT/f'unit_{u:02d}/query_scores.npz')
        if u == 0:
            full_names = block['arm_names'].tolist()
        assert block['arm_names'].tolist() == full_names
        assert block['task_names'].tolist() == list(TASKS)
        assert block['metric_names'].tolist() == list(METRICS)
        assert block['scores'].shape == (2, len(block['groups']), len(full_names), 3)
        assert np.isfinite(block['scores']).all()
        rows = block['pair_rows']
        for key in ('object_ids', 'groups', 'source_ids', 'target_ids', 'source_dose', 'target_dose'):
            np.testing.assert_array_equal(block[key], qualification[key][rows])
        np.testing.assert_array_equal(qualification['outer_roles'][rows, u//7], np.full(len(rows), 'DEV_EVAL'))
        assert np.all(qualification['task_index'][rows] == u % 7)
        collected.append(block['scores'])
        fields = ('pair_rows', 'object_ids', 'groups', 'source_ids', 'target_ids', 'source_dose',
                  'target_dose', 'batch', 'layout', 'source_plate', 'target_support', 'morph_support')
        p = {k: block[k] for k in fields}
        p.update(unit=np.full(len(rows), u), outer_fold=np.full(len(rows), u//7), task_index=np.full(len(rows), u%7))
        population.append(p)
        unit_reports.append(unit_statistics(u))
    pop = {k: np.concatenate([p[k] for p in population]) for k in population[0]}
    np.testing.assert_array_equal(np.sort(pop['pair_rows']), np.arange(len(qualification['source_rows'])))
    full = np.concatenate(collected, axis=1)
    scores, names = compact_scores(full, full_names)
    scopes = dict(all=np.ones(len(pop['groups']), bool), target_supported=pop['target_support'],
                  morphology_supported=pop['morph_support'])
    panels = {}
    with threadpool_limits(limits=args.threads):
        contexts = [('pooled', np.ones(len(pop['groups']), bool))]
        for i in range(7):
            subset = pop['task_index'] == i
            first = np.flatnonzero(subset)[0]
            contexts.append((f"{pop['source_dose'][first]:g} → {pop['target_dose'][first]:g}", subset))
        for ci, (context, subset) in enumerate(contexts):
            panels[context] = {}
            evaluated_masks = []
            for si, (scope, mask) in enumerate(scopes.items()):
                current_mask = subset & mask
                equivalent = next((previous for previous, oldmask in evaluated_masks
                                   if np.array_equal(oldmask, current_mask)), None)
                if equivalent is not None:
                    # Identical populations are the same estimand; do not give
                    # them different endpoints merely from bootstrap RNG jitter.
                    panels[context][scope] = panels[context][equivalent]
                else:
                    panels[context][scope] = summary_panel(scores, pop['groups'], current_mask, names,
                        seed=SEED+100*ci+si)
                    evaluated_masks.append((scope, current_mask))
            print('summarized '+context, flush=True)
        random = random_draw_dispersion(full, full_names, pop['groups'], scopes)
        layouts = descriptive_layout(scores, pop, names, scopes)
    result = dict(state='COMPLETE', experiment='paired same/cross-dose response borrowing',
        population=dict(n_rows=len(pop['groups']), n_objects=len(np.unique(pop['object_ids'])),
            n_groups=len(np.unique(pop['groups'])), n_units=35, n_dose_pairs=7,
            target_supported_rows=int(pop['target_support'].sum()),
            target_supported_groups=len(np.unique(pop['groups'][pop['target_support']])),
            morphology_supported_rows=int(pop['morph_support'].sum())),
        metrics=list(METRICS), tasks=list(TASKS), compact_arm_names=names,
        primary='TARGET_TRANSPORT_CAL on TARGET-supported query objects versus both RIDGE_RESPONSE and matched-random mean score',
        inference=dict(bootstrap_replicates=BOOTSTRAPS, seed=SEED, unit='chemical connectivity group',
            estimand='group-equal mean over observed eligible query condition pairs and aliases',
            relative_improvement='1 - group-equal arm risk / group-equal comparator risk',
            interaction='CROSS relative improvement minus SAME relative improvement on identical query rows',
            random_controls='20 reference draws are averaged within query before group aggregation, not independent subjects',
            scope='development; saved fitted models; no multiplicity-adjusted discovery claim'),
        panels=panels, alpha_selection=alpha_summary(unit_reports), random_draw_dispersion=random,
        retrieval_context=retrieval_context_summary(unit_reports),
        zero_norm_audit=dict(
            counts={key: sum(u['saved_profile_zero_norm_audit'].get(key, 0) for u in unit_reports)
                    for key in unit_reports[0]['saved_profile_zero_norm_audit']},
            scope='Saved float32 source, actual and main CAL predictions; not random or forced-strength vectors',
            epsilon=1e-12, cosine_convention='cosine=0 (loss1) if either norm<=epsilon',
            lognorm_convention='log(max(norm,epsilon))'),
        batch_layout_sensitivity=layouts, frozen_core_changed=False, gamma_or_null_evaluated=False,
        protected_measurements_opened=False)
    write_json(REPORT/'summary.json', result)
    write_json(REPORT/'unit_statistics.json', unit_reports)
    np.savez_compressed(REPORT/'aggregate_scores.npz', scores=scores,
        arm_names=np.asarray(names), task_names=np.asarray(TASKS), metric_names=np.asarray(METRICS), **pop)
    (REPORT/'RESULTS.md').write_text(make_report(result))
    print(REPORT/'summary.json')


if __name__ == '__main__':
    main()
