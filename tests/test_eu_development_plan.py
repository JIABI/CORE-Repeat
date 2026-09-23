from collections import Counter, defaultdict
import copy
import json

import pytest

from opal2.eu_development_plan import (
    EUDevelopmentPlanError, ROLE_BY_REPLICATE, allocate_groups,
    prepare_population, build_split_plan, make_phase_manifest, write_plan,
)


def fixture_metadata(n=30):
    assignments, metadata = [], []
    for i in range(n):
        oid = f'EOS{i:04d}'
        # Two IDs deliberately share a conservative chemistry identity.
        group = 'SHARED' if i in (0, 1) else f'GROUP{i:04d}'
        assignments.append(dict(dataset='EU_OPENSCREEN', object_id=oid, name=oid,
            connectivity=group, identity_role='EU_TARGET_FIT_CANDIDATE', reserved_for_EU='False'))
        library = 'B1001' if i < n // 2 else 'B1002'
        address = chr(ord('A') + (i % 16)) + f'{1 + i // 16:02d}'
        for rep in ['R1', 'R2', 'R3', 'R4']:
            metadata.append(dict(plate_uid=f'FMP|Batch{rep}|{library}_{rep}', site='FMP',
                cell_line_protocol='HepG2', batch_id='Batch' + rep, library_plate=library,
                replicate=rep, well_position=address, object_id_raw=oid,
                in_external_identity_table='True', concentration_metadata_value='10',
                concentration_unit_protocol='uM', exposure_hours_protocol='24',
                metadata_protocol_dose_conflict='False', source_url='official_metadata'))
    plates = {row['plate_uid']: row for row in metadata}
    for row in plates.values():
        control = dict(row, object_id_raw='DMSO', well_position='A23',
            in_external_identity_table='False', concentration_metadata_value='0',
            dose_or_vehicle_protocol_value='0.1', dose_or_vehicle_protocol_unit='% v/v')
        metadata.append(control)
        metadata.append(dict(control, object_id_raw='Nocodazole', well_position='B23'))
    return assignments, metadata


def test_complete_grouped_plan_has_no_role_leakage_and_preserves_four_wells():
    assignments, metadata = fixture_metadata()
    ids, wells, dmso = prepare_population(assignments, metadata, {'PROTECTED'}, expected_ids=None)
    splits, roles, summaries = build_split_plan(ids, wells)
    assert len(splits) == 5 * len(ids)
    assert len(roles) == 5 * 4 * len(ids)
    assert len(dmso) == 8
    eval_folds = defaultdict(set)
    for fold in range(5):
        rows = [r for r in splits if r['outer_fold'] == fold]
        group_roles = defaultdict(set)
        group_subroles = defaultdict(set)
        for row in rows:
            group_roles[row['connectivity']].add(row['phase_role'])
            group_subroles[row['connectivity']].add(row['model_fit_subrole'])
            if row['phase_role'] == 'DEV_EVAL':
                eval_folds[row['connectivity']].add(fold)
        assert all(len(value) == 1 for value in group_roles.values())
        assert all(len(value) == 1 for value in group_subroles.values())
        assert set(row['phase_role'] for row in rows) == {'MODEL_FIT', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL'}
        for oid in {row['object_id'] for row in ids}:
            four = [r for r in roles if r['outer_fold'] == fold and r['object_id'] == oid]
            assert len({r['plate_uid'] for r in four}) == 4
            assert {r['replicate']: r['measurement_role'] for r in four} == ROLE_BY_REPLICATE
    assert all(len(folds) == 1 for folds in eval_folds.values())
    # Same metadata in a different row order gives the same deterministic plan.
    ids2, wells2, _ = prepare_population(list(reversed(assignments)), list(reversed(metadata)),
                                         {'PROTECTED'}, expected_ids=None)
    assert build_split_plan(ids2, wells2) == (splits, roles, summaries)


@pytest.mark.parametrize('failure', ['missing_repeat', 'condition_changed', 'same_plate'])
def test_four_physical_same_condition_requirement(failure):
    assignments, metadata = fixture_metadata()
    metadata = copy.deepcopy(metadata)
    if failure == 'missing_repeat':
        metadata.pop(0)
    elif failure == 'condition_changed':
        metadata[0]['exposure_hours_protocol'] = '48'
    else:
        metadata[1]['plate_uid'] = metadata[0]['plate_uid']
    with pytest.raises(EUDevelopmentPlanError):
        prepare_population(assignments, metadata, {'PROTECTED'}, expected_ids=None)


def test_protected_and_other_condition_not_in_release(tmp_path):
    assignments, metadata = fixture_metadata()
    protected_row = dict(assignments[0], object_id='PROTECTED_ID', connectivity='PROTECTED',
                         identity_role='EU_CONFIRMATION_RESERVED', reserved_for_EU='True')
    assignments.append(protected_row)
    metadata += [dict(row, object_id_raw='PROTECTED_ID') for row in metadata[:4]]
    metadata += [dict(row, cell_line_protocol='U2OS', plate_uid='U2OS|' + row['plate_uid'])
                 for row in metadata[:4]]
    source_manifest = {'reserved_connectivity_groups': ['PROTECTED'], 'measurement_access_released': False}
    summary = write_plan(tmp_path, assignments, metadata, source_manifest, expected_ids=30)
    manifest = json.loads((tmp_path / 'phase_manifest.json').read_text())
    assert 'PROTECTED_ID' not in manifest['allowed_compound_ids']
    assert manifest['site'] == 'FMP' and manifest['cell'] == 'HepG2'
    assert manifest['allowed_compound_wells'] == 120
    assert manifest['allowed_dmso_wells'] == 8
    assert not manifest['positive_controls_allowed']
    assert not manifest['original_images_or_pretrained_embeddings_allowed']
    assert manifest['measurement_access_released']
    assert not source_manifest['measurement_access_released']
    assert not summary['outcomes_read']
    assert not summary['independent_final_evaluation']
    # A protected identity mislabeled FIT is rejected, not silently accepted.
    assignments[-1]['identity_role'] = 'EU_TARGET_FIT_CANDIDATE'
    with pytest.raises(EUDevelopmentPlanError, match='protected'):
        prepare_population(assignments, metadata, {'PROTECTED'}, expected_ids=None)


def test_group_quota_rounding_is_exact():
    groups = {f'G{i:04d}': f'plate{i % 7}:quadrant{i % 4}' for i in range(911)}
    outer = allocate_groups(groups, list(range(5)), [1] * 5, 20260917)
    assert Counter(outer.values()) == {0: 183, 1: 182, 2: 182, 3: 182, 4: 182}
    roles = allocate_groups({g: s for g, s in groups.items() if outer[g] != 0},
                            ['MODEL_FIT', 'REF_FIT', 'DIST_CAL'], [.6, .2, .2], 20261017)
    assert Counter(roles.values()) == {'MODEL_FIT': 437, 'REF_FIT': 146, 'DIST_CAL': 145}
