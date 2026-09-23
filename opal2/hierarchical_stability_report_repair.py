"""Generate a separate final report from completed, unchanged stability scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .biology_kernel_evaluation import write_json
from .hierarchical_stability_summary import summarize, OUTCOME_ROUNDOFF_ATOL


def build(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('The report and source must use separate sibling trees')
    if (output/'summary.json').exists():
        raise FileExistsError('The completed report is preserved')
    status = json.loads((source/'status.json').read_text())
    if (status.get('state') != 'FAILED'
            or status.get('error') != 'Repeated partitions must preserve each ID original outcome'):
        raise ValueError('This report repairs only the recorded summary consistency failure')
    counts = dict(folds=len(list(source.glob('repetitions/repeat_*/folds/fold_*/complete.json'))),
        hr_fits=len(list(source.glob('repetitions/repeat_*/folds/fold_*/arms/HR_*/training_complete.json'))),
        arm_scores=len(list(source.glob('repetitions/repeat_*/folds/fold_*/arms/*/test/metrics.json'))))
    if counts != dict(folds=15, hr_fits=45, arm_scores=90):
        raise ValueError('All planned training and scoring artifacts must already be complete')
    numerical_rows = []
    for path in sorted(source.glob('repetitions/repeat_*/folds/fold_*/arms/*/test/metrics.json')):
        model = json.loads(path.read_text())['model']
        audit = model['numerics']
        if 'numerical_policy' not in audit:
            continue
        if (audit['rows_dropped'] != 0 or audit['draws_resampled'] != 0
                or audit['jitter_added'] != 0 or audit['diagonal_floor_added'] != 0
                or audit['direct_H_refactorization_failure_count']
                    != audit['failed_factor_draws_high_precision_verified']
                or audit['high_precision_used_to_modify_predictions']
                or not audit['functional_consistency']['every_draw_checked']):
            raise ValueError('The declared forward numerical repair was not retained')
        numerical_rows.append(dict(path=str(path.relative_to(source)),
            draw_object_count=audit['draw_object_count'],
            high_precision_verified=audit['failed_factor_draws_high_precision_verified'],
            gamma_vector_error=audit['functional_consistency']['gains_max_absolute_error'],
            observable_scaled_error=audit['functional_consistency']['observables_max_scaled_error']))
    if len(numerical_rows) != 35:
        raise ValueError('Expected 55 reused and 35 newly scored arms')
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'source_snapshot'
    if snapshot.exists():
        raise FileExistsError('A report snapshot already exists')
    project = Path(__file__).resolve().parents[1]
    shutil.copytree(project/'opal2', snapshot/'opal2',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(project/'pyproject.toml', snapshot/'pyproject.toml')
    shutil.copy2(source/'run_manifest.json', output/'SOURCE_RUN_MANIFEST.json')
    shutil.copy2(source/'PROTOCOL.md', output/'PROTOCOL.md')
    shutil.copy2(source/'NUMERICAL_REPAIR.md', output/'NUMERICAL_REPAIR.md')
    write_json(output/'REPORT_REPAIR.json', dict(source_run=str(source),
        source_failed_status=status, source_status_preserved=True, completed_artifact_counts=counts,
        original_predictions_modified=False, original_scores_modified=False,
        no_retraining=True, no_resampling=True, no_new_model_selection=True,
        only_code_change='cross-repeat bounded-utility floating equality permits machine roundoff with identical labels; all output redirected',
        outcome_absolute_tolerance=OUTCOME_ROUNDOFF_ATOL, outcome_relative_tolerance=0.,
        null_and_positive_labels_must_match_exactly=True,
        source_snapshot=str(snapshot), python_executable=sys.executable))
    write_json(output/'NUMERICAL_AUDIT.json', dict(old_scores_reused=55, new_scores=35,
        checked_draw_object_evaluations=sum(r['draw_object_count'] for r in numerical_rows),
        high_precision_verified=sum(r['high_precision_verified'] for r in numerical_rows),
        max_gamma_vector_error=max(r['gamma_vector_error'] for r in numerical_rows),
        max_observable_tolerance_ratio=max(r['observable_scaled_error'] for r in numerical_rows),
        rows=numerical_rows, dropped=0, resampled=0, jitter=0))
    result = summarize(source, output=output)
    write_json(output/'status.json', dict(state='COMPLETE', stage='comparison_report',
        completed_utc=result['completed_utc'], source_run=str(source),
        source_execution_status_preserved='FAILED', no_retraining=True))
    print(json.dumps(dict(complete=True, output=str(output), n=result['n_unique_compounds'],
        outcome_roundoff_audit=result['repeated_outcome_roundoff_audit']), indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.source, args.output)
