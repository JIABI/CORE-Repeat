"""Metadata-only RxRx3 compound--gene assay priors for the frozen R3 cohort.

The prior uses the predeclared source-value rule 0 < nM <= 1000. It is not
an assertion of human/HUVEC target activity or mechanism of action. No source
measurement array, image, profile or embedding is opened by this module.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROLES = ('X', 'Z1', 'Z2', 'V')
ANNOTATION_REVISION = '2935f21f84898fc26e2a9a40bdaee685c8e63fb6'
ANNOTATION_URL = ('https://raw.githubusercontent.com/recursionpharma/EFAAR_benchmarking/'
                  + ANNOTATION_REVISION + '/efaar_benchmarking/benchmark_annotations/compound_gene_interactions.csv')
PROTOCOL_ID = 'RxRx3_COMPOUND_HUVEC_INTRON_18_24H'
PLATFORM = 'RxRx3-core:released_CellProfiler'
ACTIVITY_CUTOFF_NM = 1000.
ASSAY_TYPES = {'ic50', 'ec50', 'ec50, ic50'}
BACKGROUND_LABELS = {'Query Compounds + Intron control', 'Control Compounds + Intron control'}


def _rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def _unique(rows, key, label):
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate '+label)
    return result


def _true(value):
    return str(value).lower() == 'true'


def _vector(data, key, n=None, *, numeric=False):
    value = np.asarray(data[key], float if numeric else str)
    if value.ndim != 1 or not len(value) or (n is not None and len(value) != n):
        raise ValueError('Unaligned metadata field: '+key)
    if numeric and not np.isfinite(value).all():
        raise ValueError('Nonfinite metadata field: '+key)
    if not numeric and np.any(value == ''):
        raise ValueError('Empty metadata field: '+key)
    return value


def load_rxrx3_biology_metadata(data, *, root=PROJECT, conditions_csv=None,
                                roles_csv=None, annotation_csv=None,
                                r2_manifest=None, identity_csv=None):
    """Load only local metadata for supplied, allowlisted R2 condition IDs.

    Required data keys: ids, groups, object_ids, dose, layout, plates and
    well_ids (``wells`` is accepted as an alias). All arrays must align with
    the saved R2 manifest and approved four-role export. Additional data keys,
    including Y, are never accessed.

    Return the EU biology schema: arrays target/target_mask/moa/moa_mask;
    metadata units/target_names/moa_names; source relation rows; and report.
    ``moa`` has zero columns and its mask is false for every object. The
    fixed 1000-nM rule is a weak source-specific prior, not a tuned parameter.
    """
    root = Path(root)
    qualification = root/'reports/new_data_qualification_20260917_v1/rxrx3'
    export = root/'data/rxrx3_r2_20260918/approved_export'
    conditions_csv = Path(conditions_csv or export/'conditions.csv')
    roles_csv = Path(roles_csv or export/'roles.csv')
    annotation_csv = Path(annotation_csv or qualification/'source/external_compound_gene_interactions.csv')
    identity_csv = Path(identity_csv or qualification/'identity_raw.csv')
    r2_manifest = Path(r2_manifest or root/'runs/rxrx3_r2_completion_20260918_v1/run_manifest.json')
    ids = _vector(data, 'ids')
    n = len(ids)
    if len(set(ids)) != n:
        raise ValueError('Unique condition IDs required')
    groups, object_ids, layout = (_vector(data, key, n) for key in ('groups', 'object_ids', 'layout'))
    dose = _vector(data, 'dose', n, numeric=True)
    well_key = 'well_ids' if 'well_ids' in data else 'wells'
    wells, plates = np.asarray(data[well_key], str), np.asarray(data['plates'], str)
    if wells.shape != (n, 4) or plates.shape != (n, 4):
        raise ValueError('Four aligned X/Z1/Z2/V wells and plates required')
    if 'well_ids' in data and 'wells' in data and not np.array_equal(wells, np.asarray(data['wells'], str)):
        raise ValueError('Conflicting well_ids and wells aliases')
    if len(set(wells.ravel())) != 4*n:
        raise ValueError('Physical role wells must be unique')

    manifest = json.loads(r2_manifest.read_text())
    mid, mg = manifest['ids'], manifest['groups']
    if len(mid) != len(mg) or len(set(mid)) != len(mid):
        raise ValueError('Invalid frozen R2 identity manifest')
    manifest_groups = dict(zip(mid, mg))
    if not set(ids) <= set(mid):
        raise ValueError('Annotation request outside frozen R2 development allowlist')
    if any(manifest_groups[oid] != group for oid, group in zip(ids, groups)):
        raise ValueError('Chemical grouping differs from frozen R2 manifest')
    wanted = set(ids)
    conditions = _unique([r for r in _rows(conditions_csv) if r['condition_id'] in wanted],
                         'condition_id', 'approved condition')
    if set(conditions) != wanted:
        raise ValueError('Missing approved condition metadata')
    wanted_objects = set(object_ids)
    identities = _unique([r for r in _rows(identity_csv) if r['object_id'] in wanted_objects],
                         'object_id', 'source compound identity')
    if set(identities) != wanted_objects:
        raise ValueError('Missing source compound identity')
    role_rows = {oid: {} for oid in ids}
    for row in _rows(roles_csv):
        cid = row['condition_id']
        if cid not in wanted:
            continue
        role = row['role']
        if role not in ROLES or role in role_rows[cid]:
            raise ValueError('Ambiguous four-role metadata')
        role_rows[cid][role] = row

    units, treatment_by_object = [], {}
    for i, cid in enumerate(ids):
        row = conditions[cid]
        identity = identities[object_ids[i]]
        if (row['identity_role'] != 'RXRX3_MODULE_DEV_CANDIDATE'
                or not _true(row['measurement_access'])
                or not _true(row['roles_X_Z1_Z2_V_assigned'])
                or row['perturbation_type'] != 'COMPOUND'):
            raise ValueError('Condition is not an approved chemical development object')
        if (row['object_id'] != object_ids[i] or row['group_id'] != groups[i]
                or float(row['dose_record']) != dose[i] or dose[i] <= 0
                or row['batch'] != layout[i]):
            raise ValueError('Supplied condition metadata differs from approved export')
        if (identity['original_id'] != row['treatment'] or identity['smiles'] != row['smiles']
                or identity['perturbation_class'] != 'compound'):
            raise ValueError('Source identity/name/SMILES differs from approved export')
        if row['cell_record'] != 'HUVEC' or row['well_type_label'] not in BACKGROUND_LABELS:
            raise ValueError('Unrecognized RxRx3 drug cell/background condition')
        treatment_by_object[object_ids[i]] = row['treatment']
        rr = role_rows[cid]
        if set(rr) != set(ROLES) or len(set(plates[i])) != 4:
            raise ValueError('Four distinct physical role plates required')
        for j, role in enumerate(ROLES):
            r = rr[role]
            if (r['well_id'] != wells[i, j] or r['physical_plate_id'] != plates[i, j]
                    or any(r[key] != row[key] for key in
                           ('object_id', 'group_id', 'identity_role', 'batch', 'cell_record',
                            'dose_record', 'treatment', 'perturbation_type'))
                    or r['well_type_label'] not in BACKGROUND_LABELS
                    or not _true(r['measurement_access'])
                    or not _true(r['roles_X_Z1_Z2_V_assigned'])):
                raise ValueError('Physical role differs from approved condition metadata')
        units.append(dict(id=cid, compound_id=object_ids[i], cell_line='HUVEC',
            platform=PLATFORM, site=None,
            site_evidence='dataset-level single site; no per-well site field',
            protocol_id=PROTOCOL_ID, exposure_hours_protocol_nominal=None,
            exposure_hours_protocol_range=[18., 24.], protocol_min_h=18., protocol_max_h=24.,
            time_exact_h=None, exact_execution_verified=False,
            background='intron_targeting_CRISPR_control',
            background_guide_identity=None, background_editing_time_hours=None,
            actual_dose_uM=float(dose[i]), dose_evidence='official recorded concentration in uM',
            layout_block=str(layout[i]), well_type_label=row['well_type_label'],
            roles={role: dict(plate=str(plates[i, j]), well_id=str(wells[i, j]),
                             well=str(wells[i, j]).rsplit('_', 1)[-1], cell_count=None,
                             well_type_label=rr[role]['well_type_label'])
                   for j, role in enumerate(ROLES)}))

    by_treatment = {name: set() for name in treatment_by_object.values()}
    exact_matched, relations, eligible_rows = set(), [], []
    annotations = _rows(annotation_csv)
    required = {'gene_symbol', 'nM_value', 'measurement_type', 'database', 'treatment'}
    if not annotations or not required <= set(annotations[0]):
        raise ValueError('Unsupported external annotation schema')
    for source_row, row in enumerate(annotations, start=2):
        if row['treatment'] not in by_treatment:
            continue
        exact_matched.add(row['treatment'])
        try:
            potency = float(row['nM_value'])
        except (TypeError, ValueError):
            potency = np.nan
        eligible = (np.isfinite(potency) and 0 < potency <= ACTIVITY_CUTOFF_NM
                    and row['measurement_type'] in ASSAY_TYPES and bool(row['gene_symbol'].strip()))
        if eligible:
            by_treatment[row['treatment']].add(row['gene_symbol'])
            eligible_rows.append(row)
        relations.append(dict(source_row=source_row, raw=row, eligible_target=bool(eligible),
                              human_status=None, action_direction=None,
                              assay_condition_match_verified=False,
                              original_assay_inequality_operator=None))
    target_names = sorted(set().union(*by_treatment.values()))
    vocabulary = {name: j for j, name in enumerate(target_names)}
    target = np.zeros((n, len(vocabulary)))
    for i, oid in enumerate(object_ids):
        for name in by_treatment[treatment_by_object[oid]]:
            target[i, vocabulary[name]] = 1.
    target_mask = target.any(1)
    raw_condition_mask = np.asarray([treatment_by_object[oid] in exact_matched for oid in object_ids])
    moa = np.zeros((n, 0))
    moa_mask = np.zeros(n, bool)
    semantics = 'Reported compound-gene assay prior: finite 0 < nM <= 1000; includes the 1000-nM boundary'
    metadata = dict(units=units, target_names=target_names, moa_names=[],
                    reference_context_policy='rxrx3_protocol_range_v1',
                    target_semantics=semantics,
                    moa_semantics='Unavailable: no source action direction or MoA; never copied from target',
                    context_semantics='Same platform, HUVEC, intron-control background, declared 18-24h protocol and recorded dose; experiment need not match')
    report = dict(n=n, n_compound_identities=len(set(object_ids)), n_chemical_groups=len(set(groups)),
        exact_name_matched_compound_identities=len(exact_matched),
        exact_name_matched_conditions=int(raw_condition_mask.sum()),
        matched_annotation_rows=len(relations), eligible_annotation_rows=len(eligible_rows),
        eligible_target_coverage=int(target_mask.sum()),
        eligible_target_compound_identities=len({oid for oid, valid in zip(object_ids, target_mask) if valid}),
        eligible_target_chemical_groups=len(set(groups[target_mask])),
        eligible_target_vocabulary=len(target_names), eligible_moa_coverage=0,
        target_semantics=semantics, moa_semantics=metadata['moa_semantics'],
        activity_cutoff_nM=ACTIVITY_CUTOFF_NM, activity_rule_predeclared=True,
        eligible_measurement_types=sorted(ASSAY_TYPES),
        matched_measurement_type_counts=dict(Counter(r['raw']['measurement_type'] for r in relations)),
        eligible_measurement_type_counts=dict(Counter(r['measurement_type'] for r in eligible_rows)),
        match_quality='exact official treatment string after approved object/name/SMILES validation; no fuzzy aliases',
        unknown_relation_semantics='false mask means missing/ineligible prior, never no target or no activity',
        unknown_fields=['human assay evidence', 'action direction', 'MoA', 'assay cell/time context',
                        'original assay inequality operator', 'exact per-well exposure time',
                        'intron guide/editing time', 'per-well cell count'],
        source=str(annotation_csv), source_url=ANNOTATION_URL, source_revision=ANNOTATION_REVISION,
        identity_source=str(identity_csv), conditions_source=str(conditions_csv),
        well_metadata_source=str(roles_csv), phase_manifest=str(r2_manifest),
        protocol_source=str(root/'reports/new_data_assignment_20260917_v1/rxrx3_protocol_resolution/PROTOCOL_LICENSE_PRETRAINING.md'),
        sources=dict(annotations=str(annotation_csv), identities=str(identity_csv),
                     approved_conditions=str(conditions_csv), approved_roles=str(roles_csv), r2_manifest=str(r2_manifest)),
        context_evidence='Published drug protocol range 18-24h; exact per-well execution unknown',
        no_HUVEC_potency_claim=True, no_human_target_claim=True,
        annotation_license_status='Existing local source directory license does not specify this compound-gene file; no redistribution authorization inferred',
        measurements_read=False, outcomes_used=False, potency_used_only_for_fixed_prior_eligibility=True)
    return dict(arrays=dict(target=target, target_mask=target_mask, moa=moa, moa_mask=moa_mask),
                metadata=metadata, relations=relations, report=report)


def rxrx3_context_mask(metadata, query, donors):
    """Explicit nominal-context match; caller must also enforce fold/REF/group permissions."""
    units = metadata['units']
    query, donors = np.asarray(query, int), np.asarray(donors, int)
    if (query.ndim != 1 or donors.ndim != 1 or
            np.any(np.r_[query, donors] < 0) or np.any(np.r_[query, donors] >= len(units))):
        raise ValueError('Unaligned context indices')

    def key(i):
        unit = units[int(i)]
        values = (unit.get('platform'), unit.get('cell_line'), unit.get('background'),
                  unit.get('protocol_id'), unit.get('protocol_min_h'),
                  unit.get('protocol_max_h'), unit.get('actual_dose_uM'))
        if any(v is None or v == '' for v in values):
            return None
        numeric = np.asarray(values[-3:], float)
        if not np.isfinite(numeric).all() or np.any(numeric <= 0) or numeric[0] > numeric[1]:
            return None
        return values[:4] + tuple(numeric)

    qkeys, dkeys = [key(i) for i in query], [key(i) for i in donors]
    return np.asarray([[a is not None and b is not None and a == b for b in dkeys]
                       for a in qkeys], bool).reshape(len(query), len(donors))
