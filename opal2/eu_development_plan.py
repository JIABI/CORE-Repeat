"""Metadata-only grouped EU development plans and scoped measurement allowlists.

The original identity-reservation manifest remains closed.  This module emits a
separate authorization scope for the preselected EU FIT pool and matched DMSO,
not for the protected confirmation population.  It never loads measurements.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260917
ROLE_BY_REPLICATE = {'R1': 'X', 'R2': 'Z1', 'R3': 'Z2', 'R4': 'V'}
CONDITION_FIELDS = ('site', 'cell_line_protocol', 'concentration_metadata_value',
                    'concentration_unit_protocol', 'exposure_hours_protocol')


class EUDevelopmentPlanError(ValueError):
    pass


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def _true(value):
    return value in (True, 'True', 'true')


def _position(address):
    match = re.fullmatch(r'([A-P])(\d{2})', address)
    if not match or not 1 <= int(match.group(2)) <= 24:
        raise EUDevelopmentPlanError(f'Unexpected EU 384-well address: {address}')
    row_number = ord(match.group(1)) - ord('A') + 1
    column_number = int(match.group(2))
    quadrant = ('TOP' if row_number <= 8 else 'BOTTOM') + '_' + (
        'LEFT' if column_number <= 11 else 'RIGHT')
    return row_number, column_number, quadrant


def prepare_population(assignments, metadata, protected_groups, *, expected_ids=911):
    """Select only FMP HepG2 FIT identities and same-plate DMSO metadata."""
    protected_groups = set(protected_groups)
    eligible = {}
    for row in assignments:
        if row['dataset'] != 'EU_OPENSCREEN' or row['identity_role'] != 'EU_TARGET_FIT_CANDIDATE':
            continue
        oid = row['object_id']
        if oid in eligible:
            raise EUDevelopmentPlanError(f'Duplicate FIT identity: {oid}')
        if not row['connectivity'] or row['connectivity'] in protected_groups or _true(row['reserved_for_EU']):
            raise EUDevelopmentPlanError(f'FIT identity is unresolved or protected: {oid}')
        eligible[oid] = row
    if expected_ids is not None and len(eligible) != expected_ids:
        raise EUDevelopmentPlanError(f'Expected {expected_ids} FIT identities, found {len(eligible)}')
    if not eligible:
        raise EUDevelopmentPlanError('No FIT population')
    local = [row for row in metadata if row['site'] == 'FMP' and row['cell_line_protocol'] == 'HepG2']
    byobject = defaultdict(list)
    for row in local:
        if row['object_id_raw'] in eligible:
            byobject[row['object_id_raw']].append(row)
    identities, compound_wells = [], []
    for oid in sorted(eligible):
        identity = eligible[oid]
        wells = byobject[oid]
        reps = Counter(row['replicate'] for row in wells)
        if reps != Counter({'R1': 1, 'R2': 1, 'R3': 1, 'R4': 1}):
            raise EUDevelopmentPlanError(f'Exactly R1-R4 physical wells required: {oid}, {dict(reps)}')
        if len({row['plate_uid'] for row in wells}) != 4:
            raise EUDevelopmentPlanError(f'Four distinct physical plates required: {oid}')
        for field in CONDITION_FIELDS:
            if len({row[field] for row in wells}) != 1 or not wells[0][field]:
                raise EUDevelopmentPlanError(f'Four-well condition mismatch/missing {field}: {oid}')
        for row in wells:
            if _true(row.get('metadata_protocol_dose_conflict', False)):
                raise EUDevelopmentPlanError(f'Dose conflict in chosen condition: {oid}')
        # The selected metadata define the declared first developmental condition.
        if (float(wells[0]['concentration_metadata_value']) != 10 or
                wells[0]['concentration_unit_protocol'] != 'uM' or
                float(wells[0]['exposure_hours_protocol']) != 24):
            raise EUDevelopmentPlanError(f'Unexpected first-task condition for {oid}')
        libraries = {row['library_plate'] for row in wells}
        addresses = {row['well_position'] for row in wells}
        if len(libraries) != 1:
            raise EUDevelopmentPlanError(f'Library layout changes across repeats: {oid}')
        xrow = next(row for row in wells if row['replicate'] == 'R1')
        rr, cc, quadrant = _position(xrow['well_position'])
        library = xrow['library_plate']
        identities.append(dict(dataset='EU_OPENSCREEN', object_id=oid,
            connectivity=identity['connectivity'], name=identity['name'],
            library_plate=library, x_well_position=xrow['well_position'],
            x_row=rr, x_column=cc, layout_quadrant=quadrant,
            stratification_key=library + ':' + quadrant,
            site='FMP', cell='HepG2', dose_uM=10.0, exposure_h_protocol=24.0,
            exact_execution_verified=False, fixed_position_across_repeats=len(addresses) == 1))
        for row in sorted(wells, key=lambda r: r['replicate']):
            rr, cc, _ = _position(row['well_position'])
            compound_wells.append(dict(dataset='EU_OPENSCREEN', object_id=oid,
                connectivity=identity['connectivity'], resource_kind='FIT_COMPOUND',
                well_id=row['plate_uid'] + '|' + row['well_position'],
                plate_uid=row['plate_uid'], site=row['site'], cell=row['cell_line_protocol'],
                batch_id=row['batch_id'], library_plate=row['library_plate'],
                replicate=row['replicate'], measurement_role=ROLE_BY_REPLICATE[row['replicate']],
                well_position=row['well_position'], well_row=rr, well_column=cc,
                dose_record=row['concentration_metadata_value'], dose_unit=row['concentration_unit_protocol'],
                exposure_h_protocol=row['exposure_hours_protocol'], exact_execution_verified=False,
                source_url=row['source_url']))
    plates = {row['plate_uid'] for row in compound_wells}
    dmso = []
    for row in local:
        if row['object_id_raw'] != 'DMSO' or row['plate_uid'] not in plates:
            continue
        if _true(row['in_external_identity_table']):
            raise EUDevelopmentPlanError('DMSO is unexpectedly a library identity')
        rr, cc, _ = _position(row['well_position'])
        dmso.append(dict(dataset='EU_OPENSCREEN', object_id='DMSO', connectivity='',
            resource_kind='DMSO_CONTROL', well_id=row['plate_uid'] + '|' + row['well_position'],
            plate_uid=row['plate_uid'], site=row['site'], cell=row['cell_line_protocol'],
            batch_id=row['batch_id'], library_plate=row['library_plate'], replicate=row['replicate'],
            measurement_role='CONTROL', well_position=row['well_position'], well_row=rr, well_column=cc,
            dose_record=row.get('dose_or_vehicle_protocol_value', ''),
            dose_unit=row.get('dose_or_vehicle_protocol_unit', ''),
            exposure_h_protocol=row['exposure_hours_protocol'], exact_execution_verified=False,
            source_url=row['source_url']))
    if not dmso or plates != {row['plate_uid'] for row in dmso}:
        raise EUDevelopmentPlanError('Every chosen plate must have listed DMSO controls')
    all_wells = compound_wells + sorted(dmso, key=lambda row: row['well_id'])
    if len(all_wells) != len({row['well_id'] for row in all_wells}):
        raise EUDevelopmentPlanError('Duplicate physical well in allowlist')
    return identities, compound_wells, sorted(dmso, key=lambda row: row['well_id'])


def allocate_groups(group_strata, labels, weights, seed):
    """Seeded metadata-stratified group allocation with exact global group quotas.

    Within each stratum, select the bucket with largest proportional deficit;
    global quotas use largest-remainder rounding. No measured value is an input.
    """
    if len(labels) != len(weights) or any(w <= 0 for w in weights):
        raise EUDevelopmentPlanError('Invalid allocation weights')
    total = sum(weights)
    weights = [w / total for w in weights]
    n = len(group_strata)
    expected = [n * w for w in weights]
    targets = [math.floor(x) for x in expected]
    remainder_order = sorted(range(len(labels)), key=lambda k: (-(expected[k] - targets[k]), k))
    for k in remainder_order[:n - sum(targets)]:
        targets[k] += 1
    rng = random.Random(seed)
    strata = defaultdict(list)
    for group, stratum in sorted(group_strata.items()):
        strata[stratum].append(group)
    stratum_order = sorted(strata)
    rng.shuffle(stratum_order)
    used = [0] * len(labels)
    result = {}
    for stratum in stratum_order:
        groups = sorted(strata[stratum])
        rng.shuffle(groups)
        local = [0] * len(labels)
        tie_order = list(range(len(labels)))
        rng.shuffle(tie_order)
        tie_rank = {label: rank for rank, label in enumerate(tie_order)}
        for j, group in enumerate(groups):
            candidates = [k for k in range(len(labels)) if used[k] < targets[k]]
            k = max(candidates, key=lambda k: (
                round((j + 1) * weights[k] - local[k], 12),
                (targets[k] - used[k]) / max(targets[k], 1), -tie_rank[k]))
            result[group] = labels[k]
            local[k] += 1
            used[k] += 1
    if used != targets:
        raise EUDevelopmentPlanError('Allocation did not meet group quotas')
    return result


def build_split_plan(identities, compound_wells, *, seed=SEED, folds=5):
    if folds != 5:
        raise EUDevelopmentPlanError('This declared development plan has five outer folds')
    group_members = defaultdict(list)
    for row in identities:
        group_members[row['connectivity']].append(row)
    group_strata = {group: '|'.join(sorted({r['stratification_key'] for r in members}))
                    for group, members in group_members.items()}
    outer = allocate_groups(group_strata, list(range(folds)), [1] * folds, seed)
    byid = {row['object_id']: row for row in identities}
    splits, well_roles, summaries = [], [], []
    for fold in range(folds):
        train_strata = {group: stratum for group, stratum in group_strata.items() if outer[group] != fold}
        inner = allocate_groups(train_strata, ['MODEL_FIT', 'REF_FIT', 'DIST_CAL'],
                                [0.6, 0.2, 0.2], seed + 100 + fold)
        model_strata = {group: group_strata[group] for group in inner if inner[group] == 'MODEL_FIT'}
        model = allocate_groups(model_strata, ['TRAIN', 'VALIDATION'], [0.8, 0.2], seed + 1000 + fold)
        byobject_role = {}
        for oid in sorted(byid):
            identity = byid[oid]
            group = identity['connectivity']
            role = 'DEV_EVAL' if outer[group] == fold else inner[group]
            subrole = model.get(group, '')
            row = dict(outer_fold=fold, object_id=oid, connectivity=group,
                development_holdout_fold=outer[group], phase_role=role, model_fit_subrole=subrole,
                library_plate=identity['library_plate'], stratification_key=identity['stratification_key'],
                x_well_position=identity['x_well_position'], site='FMP', cell='HepG2',
                dose_uM=10.0, exposure_h_protocol=24.0,
                independent_final_evaluation=False)
            splits.append(row)
            byobject_role[oid] = row
        for well in compound_wells:
            split = byobject_role[well['object_id']]
            well_roles.append(dict(outer_fold=fold, phase_role=split['phase_role'],
                                   model_fit_subrole=split['model_fit_subrole'], **well,
                                   independent_final_evaluation=False))
        fold_rows = list(byobject_role.values())
        summaries.append(dict(outer_fold=fold,
            identity_counts=dict(Counter(r['phase_role'] for r in fold_rows)),
            group_counts={role: len({r['connectivity'] for r in fold_rows if r['phase_role'] == role})
                          for role in ['MODEL_FIT', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL']},
            model_subrole_counts=dict(Counter(r['model_fit_subrole'] for r in fold_rows if r['phase_role'] == 'MODEL_FIT')),
            stratification_counts={stratum: dict(Counter(r['phase_role'] for r in fold_rows
                                                        if r['stratification_key'] == stratum))
                                   for stratum in sorted({r['stratification_key'] for r in fold_rows})}))
    return splits, well_roles, summaries


def make_phase_manifest(identities, compound_wells, dmso, reservation_manifest, *, seed=SEED):
    return dict(schema='opal2.eu_development_phase.v1', phase='FMP_HEPG2_FIT911_DEVELOPMENT',
        scope='EU internal development only; not independent confirmation',
        seed=seed, site='FMP', cell='HepG2', expected_dose_uM=10.0,
        exposure_h_protocol=24.0, exact_execution_verified=False,
        measurement_access_released=True,
        release_limit='Explicit allowlisted physical wells only, checked before reading values',
        allowed_identity_role='EU_TARGET_FIT_CANDIDATE',
        allowed_compound_ids=sorted(r['object_id'] for r in identities),
        allowed_connectivity_groups=sorted({r['connectivity'] for r in identities}),
        protected_connectivity_groups=reservation_manifest['reserved_connectivity_groups'],
        reserved_confirmation_measurements_released=False,
        source_reservation_manifest='../new_data_assignment_20260917_v1/reservation_manifest.json',
        source_reservation_manifest_modified=False,
        well_allowlist='measurement_allowlist.csv', identity_split_plan='identity_split_plan.csv',
        four_well_role_plan='four_well_role_plan.csv',
        allowed_compound_wells=len(compound_wells), allowed_dmso_wells=len(dmso),
        controls_allowed='DMSO on the same FMP HepG2 physical plates only',
        positive_controls_allowed=False, other_sites_or_cells_allowed=False,
        original_images_or_pretrained_embeddings_allowed=False,
        role_by_replicate=ROLE_BY_REPLICATE,
        outer_folds=5, outer_development_evaluation_name='DEV_EVAL',
        independent_final_evaluation=False,
        phase_purposes={
            'MODEL_FIT:TRAIN': ['preprocessing_fit', 'model_fit'],
            'MODEL_FIT:VALIDATION': ['checkpoint_selection'],
            'REF_FIT': ['reference_fit', 'distribution_reference_fit'],
            'DIST_CAL': ['distribution_calibration'],
            'DEV_EVAL': ['development_evaluation_x', 'development_evaluation_outcomes'],
            'DMSO_CONTROL': ['plate_control_preprocessing']},
        cross_role_rule='Each outer fold uses only its own designated roles; no DEV_EVAL identity in any training, reference or calibration pool',
        measured_values_read_by_preparation=False, trained_models=False)


def write_plan(output, assignments, metadata, reservation_manifest, *, expected_ids=911, seed=SEED):
    output = Path(output)
    identities, compound_wells, dmso = prepare_population(assignments, metadata,
        reservation_manifest['reserved_connectivity_groups'], expected_ids=expected_ids)
    splits, well_roles, fold_summaries = build_split_plan(identities, compound_wells, seed=seed)
    manifest = make_phase_manifest(identities, compound_wells, dmso, reservation_manifest, seed=seed)
    summary = dict(status='COMPLETE_METADATA_ONLY', task='FMP HepG2 internal development', seed=seed,
        compound_ids=len(identities), chemical_identity_groups=len({r['connectivity'] for r in identities}),
        compound_wells=len(compound_wells), dmso_wells=len(dmso),
        allowed_physical_wells=len(compound_wells) + len(dmso),
        physical_plates=len({r['plate_uid'] for r in compound_wells}),
        metadata_batches=sorted({r['batch_id'] for r in compound_wells}),
        library_counts=dict(Counter(r['library_plate'] for r in identities)),
        fixed_position_identity_count=sum(r['fixed_position_across_repeats'] for r in identities),
        identity_split_rows=len(splits), four_well_role_rows=len(well_roles),
        outer_folds=fold_summaries, reserved_protected_groups=len(reservation_manifest['reserved_connectivity_groups']),
        protected_identity_intersection=sorted(set(manifest['allowed_connectivity_groups']) &
                                               set(reservation_manifest['reserved_connectivity_groups'])),
        new_phase_scope_released=True, old_global_reservation_release_changed=False,
        independent_final_evaluation=False, outcomes_read=False, training_run=False,
        stratification='library plate and 384-well quadrant; groups allocated by fixed seeded proportional deficits',
        layout_limit='Original compound positions are fixed across repeats; this is not held-out-layout or held-out-batch evidence')
    if summary['protected_identity_intersection']:
        raise EUDevelopmentPlanError('Protected identities intersect the development plan')
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [('development_identities.csv', identities), ('identity_split_plan.csv', splits),
                           ('four_well_role_plan.csv', well_roles),
                           ('measurement_allowlist.csv', compound_wells + dmso)]:
        with (output / filename).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'phase_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
