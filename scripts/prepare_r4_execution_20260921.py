"""Record the final development recipe and deterministic R4 fit roles.

Reads the already-open DEV identity arrays, not confirmation measurements.
Qualification completion is a separate prerequisite for confirmation export.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from opal2.biology_kernel_evaluation import write_json


def main():
    root = PROJECT / 'reports/r4_execution_20260921_v1'
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'development_partitions.json'
    if output.exists():
        raise FileExistsError('Preserve the existing final development allocation')
    dataset = PROJECT / 'reports/eu_core_development_20260917_v1/prepared_data_cc904'
    with np.load(dataset / 'data.npz', allow_pickle=False) as arrays:
        ids = arrays['ids'].astype(str)
        groups = arrays['groups'].astype(str)
    if len(ids) != 904 or len(set(ids)) != 904 or len(set(groups)) != 904:
        raise ValueError('Expected the existing 904 unique development groups')
    order = np.argsort(ids, kind='stable')
    order = order[np.random.default_rng(20260921).permutation(904)]
    cuts = (('TRAIN', 434), ('VALIDATION', 108), ('REF_FIT', 181), ('DIST_CAL', 181))
    parts, group_parts, cursor = {}, {}, 0
    for role, size in cuts:
        selected = order[cursor:cursor + size]
        parts[role] = sorted(ids[selected].tolist())
        group_parts[role] = sorted(groups[selected].tolist())
        cursor += size
    write_json(output, dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User requested all R4 experiments; final fit uses only already-open DEV904.',
        dataset=str(dataset.relative_to(PROJECT)), seed=20260921,
        allocation='Stable ID sort, PCG64 permutation, consecutive fixed role quotas',
        partitions=parts, group_partitions=group_parts,
        counts={k: len(v) for k, v in parts.items()},
        confirmation_identities_used_for_fitting=False,
        development_data_used_for_prior_model_selection=True,
    ))
    config = dict(
        run='r4_confirmation_20260921_v1', date='2026-09-21',
        user_request='开始做r4所有的实验',
        state='DEVELOPMENT_RECIPE_LOCKED_CONFIRMATION_PENDING_QUALIFICATION',
        dataset='EU_OPENSCREEN', site='FMP', cell_type='HepG2',
        dose_uM=10, exposure_h=24,
        core='CORE_ORIGINAL: RIDGE -> validation-best HR -> A_OLD_GENERIC30 -> STATE50; AMP_EMP_LOCAL',
        base_scatter='unchanged RIDGE grouped-OOF second moment',
        comparator='DIRECT_ACCESS_MATCHED_HISTGB_CLASSIFIER_CAL',
        additional_core_distribution='matched-mean Gaussian',
        task='ADD_TWO', cost_per_action_well=0.01,
        gamma='0.5*(cos((X+Z1+Z2)/3,V)-cos(X,V))-0.02',
        primary_comparison='CORE minus ACCESS_MATCHED HistGB CAL',
        primary_endpoint='sum selected realized Gamma divided by metadata-qualified N',
        risk_penalty=0.2, secondary_risk_penalty=0.0,
        budget='B=N//4 additional wells; k=min(B//2, valid_X_count)',
        tie_rule='ascending stable unique object ID',
        seeds=[20260921, 20360921, 20460921],
        samples_per_object=100000, random_selection_seed=20260922,
        bootstrap_seed=20260923, bootstrap_replicates=10000,
        scientific_margin_selected_mean_gamma=0.01,
        inferential_units=['chemical connectivity', 'library layout'],
        new_campaign_bootstrap='resample blocks, then rerun frozen top-k at reconstructed N',
        fixed_list_sensitivity_reported_separately=True,
        main_cost_scenario='existing training/reference/calibration resource pool',
        cost_sensitivity=['new REF', 'new REF+CAL', 'new all fitting resources'],
        amortization_campaigns=[1,2,5,10,20],
        count_action_cost_twice=False,
        feature_space='existing DEV904 DMSO-only per-plate frozen control space',
        cpu_threads=2, device='cpu',
        numerical_cache='main-seed 100000 Gamma samples float64; additional-seed decision moments',
        missing_X='retain N and budget, common eligible-X set for all methods, no replacements',
        missing_outcome='retain selections; report bounded Gamma/risk identification intervals',
        secondary_biological_use='MEDINA and USC morphology-neighbourhood reproducibility, developer anchors only',
        primary_task_changed_by_secondary_endpoint=False,
        confirmation_X_opened=False, confirmation_outcomes_opened=False,
        formal_finite_sample_risk_certificate=False,
    )
    write_json(root / 'algorithm_config.json', config)
    write_json(root / 'status.json', dict(
        state=config['state'], updated_utc=datetime.now(timezone.utc).isoformat(),
        final_development_fit='READY_FOR_IMPLEMENTATION_TEST',
        confirmation_X_opened=False, confirmation_outcomes_opened=False,
        qualification='1541 candidate proposal; four historical identities under follow-up',
        recipe=str((root/'algorithm_config.json').relative_to(PROJECT)),
    ))
    print(json.dumps({'partitions': str(output), 'counts': {k:len(v) for k,v in parts.items()},
                      'state':config['state']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
