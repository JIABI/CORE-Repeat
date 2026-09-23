"""Recompute R4 FDP block sensitivity from saved scores, lists and outcomes.

No model loading, fitting, measurement access, Monte Carlo integration or
changes to the realized selected lists occur in this repair.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from opal2.r4_evaluation import campaign_resampling, load_selections

SOURCE = ROOT/'reports/r4_execution_20260921_v1/primary/campaign'
SELECTION = ROOT/'runs/r4_confirmation_20260921_v1/selections'
OUTPUT = ROOT/'reports/r4_statistical_closure_20260921_v1'
UTILITY_KEYS = ('difference_low', 'difference_high',
                'CORE_random_difference_low', 'CORE_random_difference_high')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    if (OUTPUT/'complete.json').exists():
        raise FileExistsError('Completed statistical closure exists; preserve it')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    before = {path:path.read_bytes() for path in (
        SOURCE/'summary.json', SOURCE/'object_results.csv',
        SELECTION/'selections.npz', SELECTION/'SELECTIONS_FROZEN.json')}
    original = json.loads(before[SOURCE/'summary.json'])
    ids, eligible, policies = load_selections(SELECTION)
    with (SOURCE/'object_results.csv').open() as handle:
        objects = list(csv.DictReader(handle))
    np.testing.assert_array_equal(ids, [row['object_id'] for row in objects])
    np.testing.assert_array_equal(eligible, [row['eligible_x']=='True' for row in objects])
    for policy, values in policies.items():
        np.testing.assert_array_equal(values['selected'],
            [row[policy+'_selected']=='True' for row in objects])
    gamma = np.array([float(row['gamma']) for row in objects])
    uncertainty, rows, audits = {}, [], []
    for block_name, column in (('chemical_connectivity','group'), ('library_layout','layout')):
        previous = original['uncertainty'][block_name]
        print(f'Resampling {block_name}: {previous["repeats"]} replicates', flush=True)
        current = campaign_resampling(ids, gamma, eligible, policies,
            np.array([row[column] for row in objects]),
            seed=previous['seed'], repeats=previous['repeats'])
        assert current['blocks'] == previous['blocks']
        for mode, contents in current['intervals'].items():
            for key in UTILITY_KEYS:
                if contents[key] != previous['intervals'][mode][key]:
                    raise AssertionError(f'Unrelated value result changed: {block_name}/{mode}/{key}')
            for policy in ('CORE','HISTGB_CAL'):
                interval = contents[policy+'_fdp']
                old = previous['intervals'][mode][policy+'_fdp']
                if interval['valid_replicates']+interval['undefined_replicates'] != current['repeats']:
                    raise AssertionError('Unaccounted FDP resampling replicate')
                point = original['policies'][policy]
                rows.append(dict(block_unit=block_name, mode=mode, policy=policy,
                    identification_lower=point['fdp_lower'], identification_upper=point['fdp_upper'],
                    sensitivity_lower=interval['lower'], sensitivity_upper=interval['upper'],
                    repeats=current['repeats'], defined_replicates=interval['valid_replicates'],
                    undefined_replicates=interval['undefined_replicates'],
                    replicates_with_missing_selected=interval['replicates_with_missing_selected'],
                    old_retained_replicates=old['valid_replicates'],
                    old_invalid_lower=old['lower'], old_invalid_upper=old['upper'],
                    interval_type=interval['interval_kind']))
        uncertainty[block_name] = current
        audits.append(dict(block_unit=block_name, unchanged_utility_intervals=True,
                           same_seed=True, same_repeats=True))
    corrected = dict(original)
    corrected['uncertainty'] = uncertainty
    corrected['statistical_correction'] = dict(
        date='2026-09-21', source='reports/r4_execution_20260921_v1/primary/campaign/summary.json',
        scope='FDP block sensitivity only; marginal missing-label bounds in each replicate',
        models_refitted=False, realized_selections_changed=False,
        old_results_preserved=True,
        interpretation='missing-label percentile envelopes are dependence sensitivity, not finite-sample certificates')
    # Everything outside the explicitly corrected uncertainty record is exact.
    assert {k:v for k,v in corrected.items() if k not in ('uncertainty','statistical_correction')} == {
        k:v for k,v in original.items() if k != 'uncertainty'}
    assert all(path.read_bytes()==content for path,content in before.items())
    write_json(OUTPUT/'corrected_campaign_summary.json', corrected)
    with (OUTPUT/'fdp_intervals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_json(OUTPUT/'audit.json', dict(
        identity_order_exact=True, eligible_set_exact=True, realized_lists_exact=True,
        original_inputs_bytewise_unchanged=True, all_non_fdp_statistics_exact=True,
        controls=audits, n=len(ids), observed_outcomes=int(np.isfinite(gamma).sum()),
        computation='saved-table resampling only; no new measurements or model fit'))
    text = ['# R4 FDP missing-outcome statistical correction', '',
        'This correction reuses the original frozen lists, scores, outcomes, block units, '
        '10,000 replicates and seed 20260923. Only the FDP uncertainty summary changes.', '',
        '## Corrected estimates', '',
        'The finite-cohort identification bounds describe unknown selected labels. '
        'The sensitivity envelopes span the 2.5th percentile of resampled lower bounds '
        'to the 97.5th percentile of resampled upper bounds. These are different quantities; '
        'the latter are approximate dependence sensitivity, not binomial confidence intervals.', '',
        '| Block unit | Target | Policy | Missing-label bound (%) | Sensitivity envelope (%) | Defined / total | With selected missing |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row["block_unit"]} | {row["mode"]} | {row["policy"]} | '
            f'{100*row["identification_lower"]:.3f}–{100*row["identification_upper"]:.3f} | '
            f'{100*row["sensitivity_lower"]:.3f}–{100*row["sensitivity_upper"]:.3f} | '
            f'{row["defined_replicates"]}/{row["repeats"]} | {row["replicates_with_missing_selected"]} |')
    text += ['', '## What was corrected', '',
        'The old scalar FDP summary discarded replicates containing selected missing outcomes. '
        'The correction retains them using [observed NULL/k, (observed NULL + missing)/k]. '
        'Replicates with no selected objects are explicitly undefined, not assigned FDP zero.', '',
        'The fixed-list analysis resamples contributions from the realized frozen lists. '
        'The reconstructed-campaign analysis reapplies the already-frozen score and budget rule '
        'inside each bootstrap sample, as specified in the original protocol. It does not '
        'replace the realized list or select a new model.', '',
        '## Invariants', '',
        '- All original selection and outcome artifacts are unchanged.',
        '- All value, random-comparator, count, probability and missing-identification results are unchanged.',
        '- Every pre-existing non-FDP resampling summary is reproduced exactly.',
        '- Original FDP bootstrap intervals are superseded by this report and remain archived for audit.',
        '- No training, joint sampling or new data access was performed.', '']
    (OUTPUT/'REPORT.md').write_text('\n'.join(text))
    record = dict(status='COMPLETE', completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started, output=str(OUTPUT.relative_to(ROOT)),
        original_selection_unchanged=True, non_fdp_results_unchanged=True)
    write_json(OUTPUT/'complete.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
