"""Metadata-only adjacent-dose response pairing qualification, no fitting.

Does not read the prepared Y array or inspect any response value. Existing
approved condition/well/identity metadata and the frozen split are reused.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata


def main():
    prepared = ROOT / "data/rxrx3_r2_20260918/prepared_r2"
    with np.load(prepared / "data.npz", allow_pickle=False) as z:
        # Y deliberately absent: no compound outcome array is decoded.
        names = ('ids', 'groups', 'object_ids', 'dose', 'batches', 'well_ids', 'plates', 'layout', 'feature_names')
        data = {key: z[key].copy() for key in names}
    meta = json.loads((prepared / "metadata.json").read_text())
    manifest = json.loads((ROOT / "runs/rxrx3_r2_completion_20260918_v1/run_manifest.json").read_text())
    biology = load_rxrx3_biology_metadata(data)
    target = biology['arrays']['target']
    annotated = biology['arrays']['target_mask']
    lookup = {str(value): i for i, value in enumerate(data['ids'])}
    roles = ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL')
    role_arrays, role_check = {}, []
    for fold in range(5):
        per_group, per_object = defaultdict(set), defaultdict(set)
        assignment = np.full(len(data['ids']), '', dtype='<U10')
        for cell, part in zip(manifest['cells'], manifest['parts']):
            if cell['outer_fold'] != fold:
                continue
            for role, ids in part.items():
                for cid in ids:
                    row = lookup[cid]
                    assert assignment[row] == ''
                    assignment[row] = role
                    per_group[data['groups'][row]].add(role)
                    per_object[data['object_ids'][row]].add(role)
        assert np.all(assignment != '')
        group_bad = [str(k) for k, value in per_group.items() if len(value) != 1]
        object_bad = [str(k) for k, value in per_object.items() if len(value) != 1]
        assert not group_bad and not object_bad
        role_arrays[fold] = assignment
        role_check.append(dict(outer_fold=fold, groups_with_cross_role_doses=len(group_bad),
                               object_ids_with_cross_role_doses=len(object_bad)))
    key_count = Counter(zip(data['object_ids'], data['dose']))
    group_dose = Counter(zip(data['groups'], data['dose']))
    assert max(key_count.values()) == 1
    doses = sorted(np.unique(data['dose']).tolist())
    by_dose = {dose: {str(data['object_ids'][i]): int(i)
                       for i in np.flatnonzero(data['dose'] == dose)} for dose in doses}
    tasks, pair_manifest, primary_pairs = [], [], []
    for task_index, (lower, upper) in enumerate(zip(doses[:-1], doses[1:])):
        objects = sorted(set(by_dose[lower]) & set(by_dose[upper]))
        sources = np.array([by_dose[lower][oid] for oid in objects])
        destinations = np.array([by_dose[upper][oid] for oid in objects])
        assert np.array_equal(data['groups'][sources], data['groups'][destinations])
        assert np.array_equal(target[sources], target[destinations])
        same_batch = data['batches'][sources] == data['batches'][destinations]
        old_target_shares_plate = np.array([data['plates'][s, 0] in data['plates'][d, 1:]
                                           for s, d in zip(sources, destinations)])
        identical_plate_set = np.array([set(data['plates'][s]) == set(data['plates'][d])
                                       for s, d in zip(sources, destinations)])
        strict_primary = same_batch & identical_plate_set
        nonoverlap_target_roles = []
        for oid, s, d in zip(objects, sources, destinations):
            assert data['well_ids'][s, 0] not in data['well_ids'][d]
            indices = np.flatnonzero(data['plates'][d] != data['plates'][s, 0])
            assert len(indices) >= 3
            # Exactly 3 wells, choose by physical well ID if four are eligible.
            indices = sorted(indices.tolist(), key=lambda j: str(data['well_ids'][d, j]))[:3]
            nonoverlap_target_roles.append(indices)
            eligible = bool(data['batches'][s] == data['batches'][d] and set(data['plates'][s]) == set(data['plates'][d]))
            pair_manifest.append(dict(object_id=oid, chemical_group=str(data['groups'][s]),
                source_dose_uM=lower, target_dose_uM=upper,
                source_row=int(s), target_row=int(d), source_id=str(data['ids'][s]),
                target_id=str(data['ids'][d]), same_batch=bool(data['batches'][s] == data['batches'][d]),
                source_X_role=0, original_target_future_roles=[1, 2, 3],
                candidate_plate_disjoint_target_roles=indices,
                same_dose_control_future_roles=[1, 2, 3], primary_eligible=eligible))
            if eligible:
                assert len(np.flatnonzero(data['plates'][d] != data['plates'][s, 0])) == 3
                assert len(set(data['plates'][d, indices])) == 3
                assert data['plates'][s, 0] not in data['plates'][d, indices]
                primary_pairs.append(dict(task_index=task_index, object_id=oid,
                    group=str(data['groups'][s]), source_row=int(s), target_row=int(d),
                    source_id=str(data['ids'][s]), target_id=str(data['ids'][d]),
                    source_dose=lower, target_dose=upper, target_roles=indices,
                    source_roles=[1, 2, 3], outer_roles=[str(role_arrays[f][s]) for f in range(5)]))
        per_fold = []
        for fold, assignments in role_arrays.items():
            assert np.array_equal(assignments[sources], assignments[destinations])
            row = dict(outer_fold=fold, roles={})
            for eligibility, take in [('all_paired', np.ones(len(sources), bool)), ('same_batch_candidate', same_batch),
                                      ('primary_same_batch_same_four_plates', strict_primary)]:
                ss, dd = sources[take], destinations[take]
                local_roles = assignments[ss]
                donors = ss[local_roles == 'REF_FIT']
                relation = target[ss] @ target[donors].T > 0
                relation &= data['groups'][ss, None] != data['groups'][None, donors]
                # Same task endpoints give the same source/target doses,
                # platform, HUVEC and nominal protocol. References may be from
                # other batches, matching the existing biology context policy.
                support = relation.any(1)
                counts = relation.sum(1)
                row['roles'][eligibility] = {}
                for role in roles:
                    rr = local_roles == role
                    row['roles'][eligibility][role] = dict(
                        n_identities=int(rr.sum()), n_chemical_groups=len(np.unique(data['groups'][ss[rr]])),
                        annotated_identities=int(annotated[ss[rr]].sum()),
                        target_supported_identities=int(support[rr].sum()),
                        target_supported_chemical_groups=len(np.unique(data['groups'][ss[rr]][support[rr]])),
                        support_donor_count_histogram={str(k): int(v) for k, v in sorted(Counter(counts[rr].tolist()).items())})
            per_fold.append(row)
        tasks.append(dict(source_dose_uM=lower, target_dose_uM=upper,
            paired_identities=len(objects), paired_chemical_groups=len(np.unique(data['groups'][sources])),
            same_batch_identities=int(same_batch.sum()), cross_batch_identities=int((~same_batch).sum()),
            original_future_target_shares_source_X_plate=int(old_target_shares_plate.sum()),
            identical_four_plate_sets=int(identical_plate_set.sum()),
            same_batch_identical_four_plate_sets=int((same_batch & identical_plate_set).sum()),
            same_batch_target_annotation_n=int(annotated[sources[same_batch]].sum()),
            primary_identities=int(strict_primary.sum()),
            primary_chemical_groups=len(np.unique(data['groups'][sources[strict_primary]])),
            primary_target_annotation_n=int(annotated[sources[strict_primary]].sum()),
            plate_disjoint_target_can_supply_three_wells=True,
            per_fold=per_fold))
    object_batches = {str(o): len(set(data['batches'][data['object_ids'] == o])) for o in np.unique(data['object_ids'])}
    output = ROOT / 'reports/r3_crossdose_response_20260920_v1'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(
        state='METADATA_QUALIFIED_WITH_ROLE_DESIGN_CHOICES', compound_outcomes_read=False,
        protected_evaluation_read=False, identities=len(np.unique(data['object_ids'])),
        chemical_groups=len(np.unique(data['groups'])), conditions=len(data['ids']),
        grouping=dict(pair_key='object_ids + source dose + target dose',
            independent_unit='chemical connectivity groups; all doses and aliases remain together',
            duplicate_object_dose_keys=sum(v > 1 for v in key_count.values()),
            duplicate_group_dose_keys=sum(v > 1 for v in group_dose.values()),
            max_identities_per_group_dose=max(group_dose.values()),
            object_ids_with_multiple_batches=sum(v > 1 for v in object_batches.values())),
        all_doses_role_isolation=role_check,
        feature_space=dict(n_features=len(data['feature_names']),
            coordinates_shared_across_doses=True,
            measurement_space=meta['measurement_space'],
            assay_space_fitted_from_compound_outcomes=meta['assay_space_fitted_from_compound_outcomes'],
            source_preprocessing_provenance=meta['source_preprocessing_provenance'],
            training_note='Fit any model input/response transforms only on new task TRAIN. Do not mix separately fitted dose-specific R2 standardized coordinates.'),
        annotations=dict(coverage=biology['report']['eligible_target_coverage'],
            moa_available=False, vocabulary=biology['report']['eligible_target_vocabulary'],
            context='same HUVEC/platform/nominal 18-24h protocol and intron background; exact times and HUVEC potency unknown',
            reference_support_note='Per-role counts are full paired REF availability excluding own chemical group, not nested fitting support. Match source and target task conditions, do not call the old same-dose reference mask on source-query versus target-donor rows.'),
        role_options=dict(
            original_target_roles='Original target-dose Z1/Z2/V are physically distinct wells from source X but commonly share its plate. Confounds a same-dose-vs-cross-dose comparison of dependence.',
            recommended_primary='Use same-batch and identical-four-plate-set object pairs. Source input is source-dose X only. Target response is mean of exactly the 3 target-dose wells on plates distinct from source-X plate, selected by metadata from the existing 4 roles. For same-dose matched control use source-dose Z1/Z2/V. Target-dose X may legally be used as an outcome; it is not an input.',
            alternative='Retain original target Z1/Z2/V with a declared plate-overlap sensitivity analysis. Do not silently claim independence.',
            no_new_endpoint_claim='This is a new response-borrowing task, not the original same-condition ADD_TWO Gamma endpoint.'),
        tasks=tasks,
        files=dict(pair_manifest=str(output/'pairs_metadata.json'), paired_arrays=str(output/'qualification.npz'), prepared=str(prepared)))
    arrays = {}
    for key in ('task_index', 'source_row', 'target_row', 'target_roles', 'source_roles'):
        arrays[key if key in ('task_index', 'target_roles', 'source_roles') else key+'s'] = np.asarray([r[key] for r in primary_pairs], dtype=int)
    for key in ('source_dose', 'target_dose'):
        arrays[key] = np.asarray([r[key] for r in primary_pairs], dtype=float)
    for old, new in (('object_id', 'object_ids'), ('group', 'groups'), ('source_id', 'source_ids'),
                     ('target_id', 'target_ids'), ('outer_roles', 'outer_roles')):
        arrays[new] = np.asarray([r[old] for r in primary_pairs], dtype=str)
    arrays['role_names'] = np.asarray(roles, dtype=str)
    arrays['source_X_role'] = np.zeros(len(primary_pairs), dtype=int)
    np.savez_compressed(output/'qualification.npz', **arrays)
    (output/'pairs_metadata.json').write_text(json.dumps(pair_manifest, indent=2)+'\n')
    (output/'qualification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('state', 'identities', 'chemical_groups', 'grouping', 'all_doses_role_isolation')}, indent=2))
    print(json.dumps([{k: task[k] for k in ('source_dose_uM','target_dose_uM','paired_identities','same_batch_identities','cross_batch_identities','original_future_target_shares_source_X_plate','identical_four_plate_sets','same_batch_target_annotation_n')} for task in tasks], indent=2))
    print(output/'qualification.json')


if __name__ == '__main__':
    main()
