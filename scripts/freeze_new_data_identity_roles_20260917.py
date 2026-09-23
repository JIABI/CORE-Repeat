"""Freeze EU-first identity reservations using approved metadata only.

No split fitting, profile download, image access or outcome-based filtering.
Later identity corrections may narrow this reservation but never silently
release reserved groups into development.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'reports/new_data_qualification_20260917_v1'
OUTPUT = ROOT / 'reports/new_data_assignment_20260917_v1'
OVERLAY_RELATIVE = 'identity_resolution/affected_EU_original1549.csv'
INELIGIBLE_ROLE = 'EU_RESERVED_HISTORICAL_OVERLAP_INELIGIBLE'


def build_assignments(source, proposed, *, previous_manifest=None,
                      previous_assignments=(), overlap_rows=()):
    """Keep historical reservations and apply an exclusion-only identity overlay.

    Connectivity aliases accumulate rather than disappear. Previously reserved
    object IDs also retain their reservation if normalization changes their key.
    An overlap exclusion cannot be removed merely by dropping its overlay row.
    """
    previous_manifest = previous_manifest or {}
    bykey = {(r['dataset'], r['object_id']): r for r in source}
    if len(bykey) != len(source):
        raise ValueError('Duplicate source identity')
    current = {r['connectivity'] for r in source
               if r['dataset'] == 'EU_OPENSCREEN'
               and r['strict_new_identity_screen'] == 'no_detected_overlap_provisional'}
    current.discard('')
    previous = set(previous_manifest.get('reserved_connectivity_groups', []))
    original = set(previous_manifest.get('original_reserved_connectivity_groups',
                                        previous or current))
    reserved = current | previous
    previously_reserved_keys = {
        (r['dataset'], r['object_id']) for r in previous_assignments
        if r.get('reserved_for_EU') in (True, 'True') or r['connectivity'] in previous
    }
    for key in previously_reserved_keys:
        if key in bykey and bykey[key]['connectivity']:
            reserved.add(bykey[key]['connectivity'])
    excluded_ids = set(previous_manifest.get('EU_historical_overlap_excluded_ids', []))
    excluded_groups = set(previous_manifest.get('EU_historical_overlap_excluded_groups', []))
    for row in previous_assignments:
        if row['dataset'] == 'EU_OPENSCREEN' and row['identity_role'] == INELIGIBLE_ROLE:
            excluded_ids.add(row['object_id'])
            if row['connectivity']:
                excluded_groups.add(row['connectivity'])
    for row in overlap_rows:
        if row['dataset'] != 'EU_OPENSCREEN':
            raise ValueError('EU overlap overlay contains a different dataset')
        key = (row['dataset'], row['object_id'])
        if key not in bykey:
            raise ValueError(f'Overlay identity not present: {key}')
        if row['candidate_connectivity'] != bykey[key]['connectivity']:
            raise ValueError(f'Overlay connectivity disagrees: {key}')
        if row['action'] != 'exclude_from_new_evaluation_conservative_identity_overlap':
            raise ValueError(f'Unexpected overlay action: {key}')
        excluded_ids.add(row['object_id'])
        excluded_groups.add(row['candidate_connectivity'])
    reserved |= excluded_groups
    assignments = []
    for r in source:
        key = (r['dataset'], r['object_id'])
        proposal = proposed[key]['proposed_use']
        role = {
            'protected_confirmation_candidate': 'EU_CONFIRMATION_RESERVED',
            'possible_dataset_specific_FIT_reference_not_new_identity_confirmation': 'EU_TARGET_FIT_CANDIDATE',
            'module_discovery_candidate': 'RXRX3_MODULE_DEV_CANDIDATE',
            'reserve_for_EU_confirmation_exclude_from_other_discovery_proposed': 'EU_IDENTITY_RESERVED_ELSEWHERE',
            'supplementary_reference_or_transfer_candidate': 'JUMP_SUPPLEMENTARY_CANDIDATE',
            'identity_unresolved_no_measurement_release': 'IDENTITY_QUARANTINE',
            'protected_old_JUMP_identity_no_reassignment': 'OLD_JUMP_PROTECTED',
        }[proposal]
        protected = r['connectivity'] in reserved or key in previously_reserved_keys
        historical_overlap = (r['dataset'] == 'EU_OPENSCREEN' and
                              (r['object_id'] in excluded_ids or r['connectivity'] in excluded_groups))
        if historical_overlap:
            role = INELIGIBLE_ROLE
            protected = True
        elif protected and r['dataset'] != 'EU_OPENSCREEN':
            role = 'EU_IDENTITY_RESERVED_ELSEWHERE'
        elif protected and role != 'EU_CONFIRMATION_RESERVED':
            role = 'EU_RESERVED_PENDING_REVIEW'
        assignments.append(dict(dataset=r['dataset'], object_id=r['object_id'],
            name=r['name'], connectivity=r['connectivity'], identity_role=role,
            historical_identity_status=r['strict_new_identity_screen'],
            reserved_for_EU=protected, source_identity_status=r['identity_status'],
            confirmation_candidate_eligible=role == 'EU_CONFIRMATION_RESERVED',
            historical_overlap_exclusion=historical_overlap, measurement_access=False))
    state = dict(reserved_connectivity_groups=sorted(reserved),
                 original_reserved_connectivity_groups=sorted(original),
                 EU_historical_overlap_excluded_ids=sorted(excluded_ids),
                 EU_historical_overlap_excluded_groups=sorted(excluded_groups),
                 previous_reserved_groups_retained=previous <= reserved)
    return assignments, state


def build_target_support(assignments, annotations):
    """Only eligible confirmation candidates receive metadata support rows."""
    fit_ids = {r['object_id'] for r in assignments if r['identity_role'] == 'EU_TARGET_FIT_CANDIDATE'}
    inverted = defaultdict(set)
    for oid in fit_ids:
        for label in annotations[oid]['target_names'].split(';'):
            if label:
                inverted[label].add(oid)
    support = []
    for r in assignments:
        if r['identity_role'] != 'EU_CONFIRMATION_RESERVED':
            continue
        labels = [x for x in annotations[r['object_id']]['target_names'].split(';') if x]
        donors = set().union(*(inverted[x] for x in labels)) if labels else set()
        support.append(dict(object_id=r['object_id'], connectivity=r['connectivity'],
            target_label_count=len(labels), candidate_donor_id_count=len(donors),
            donor_ids=';'.join(sorted(donors)), support_type='exact_external_target_label_METADATA_ONLY',
            realized_measurement_support_verified=False, module_enabled=False))
    return support


def preserve_pre_fix_outputs(output):
    """Archive prior generated tables once; never overwrite an earlier snapshot."""
    snapshot = output / 'before_identity_resolution_fix'
    filenames = ['identity_assignments.csv', 'reservation_manifest.json', 'assignment_summary.json',
                 'rxrx3_well_access_plan.csv', 'eu_openscreen_well_access_plan.csv',
                 'jump_target_well_access_plan.csv', 'eu_candidate_target_support.csv',
                 'rxrx3_dose_feasibility.csv']
    for name in filenames:
        source = output / name
        destination = snapshot / name
        if source.exists() and not destination.exists():
            snapshot.mkdir(exist_ok=True)
            shutil.copy2(source, destination)


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = read(INPUT / 'candidate_identity_overlap.csv')
    proposed = {(r['dataset'], r['object_id']): r for r in read(INPUT / 'proposed_identity_uses.csv')}
    manifest_path = OUTPUT / 'reservation_manifest.json'
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_path = OUTPUT / 'identity_assignments.csv'
    previous_assignments = read(previous_path) if previous_path.exists() else []
    overlay_path = OUTPUT / OVERLAY_RELATIVE
    if not overlay_path.exists():
        raise FileNotFoundError(f'Required exclusion-only identity overlay missing: {overlay_path}')
    assignments, reservation_state = build_assignments(source, proposed,
        previous_manifest=previous_manifest, previous_assignments=previous_assignments,
        overlap_rows=read(overlay_path))
    reserved = set(reservation_state['reserved_connectivity_groups'])
    assert len(assignments) == len({(r['dataset'], r['object_id']) for r in assignments})
    preserve_pre_fix_outputs(OUTPUT)
    write(OUTPUT / 'identity_assignments.csv', assignments)
    byid = {(r['dataset'], r['object_id']): r for r in assignments}
    target_aliases = defaultdict(list)
    for r in source:
        if r['dataset'] == 'JUMP_Target':
            for alias in r['production_object_ids'].split('|'):
                if alias: target_aliases[alias].append(byid[(r['dataset'], r['object_id'])])
    well_counts = {}
    for ds, sub in [('RxRx3_core', 'rxrx3'), ('EU_OPENSCREEN', 'eu_openscreen'), ('JUMP_Target', 'jump_target')]:
        wells = []
        for w in read(INPUT / sub / 'well_metadata.csv'):
            oid = w.get('object_id', w.get('object_id_raw', ''))
            row = byid.get((ds, oid))
            if ds == 'JUMP_Target':
                aliases = target_aliases.get(oid, [])
                roles = {a['identity_role'] for a in aliases}
                if 'EU_IDENTITY_RESERVED_ELSEWHERE' in roles:
                    row = next(a for a in aliases if a['identity_role'] == 'EU_IDENTITY_RESERVED_ELSEWHERE')
                elif aliases:
                    # Quarantine rather than guess in the event of conflicting aliases.
                    row = aliases[0] if len(roles) == 1 else None
            if row:
                role = row['identity_role']; conn = row['connectivity']
            elif sub == 'rxrx3' and w['perturbation_class'] != 'compound':
                role = 'GENETIC_OR_CONTROL_RESOURCE_NOT_ASSIGNED'; conn = ''
            elif sub == 'eu_openscreen' and w['in_external_identity_table'] == 'False':
                role = 'CONTROL_CONTEXT_PENDING_PROTOCOL'; conn = ''
            elif sub == 'jump_target' and w['canonical_control_type'] == 'negcon':
                role = 'CONTROL_CONTEXT_PENDING_PROTOCOL'; conn = ''
            else:
                role = 'UNRESOLVED_WELL_IDENTITY'; conn = ''
            if sub == 'rxrx3':
                wid=w['well_id']; plate=w['physical_plate_id']; site=w['lab_site']; batch=w['experiment']
                condition=w['condition_id']; cell=w['cell']; dose=w['dose_value']; time=w['time_hours']
                group=conn or (('GENE:'+w['gene']) if w['gene'] else oid)
            elif sub == 'eu_openscreen':
                wid=w['plate_uid']+'|'+w['well_position']; plate=w['plate_uid']; site=w['site']; batch=w['batch_id']
                cell=w['cell_line_protocol']; dose=w['concentration_metadata_value']; time=w['exposure_hours_protocol']
                condition='|'.join([oid,cell,dose,time,site]); group=conn or oid
            else:
                wid=w['physical_well_id']; plate=w['physical_plate_id']; site=w['source']; batch=w['batch']
                cell=w['nominal_cell_background']; dose=w['nominal_compound_dose_uM']; time=w['nominal_exposure_hours']
                condition='|'.join([oid,cell,dose,time,site]); group=conn or oid
            wells.append(dict(dataset=ds,well_id=wid,object_id=oid,group_id=group,
                identity_role=role,physical_plate_id=plate,site=site,batch=batch,condition_id=condition,
                cell_record=cell,dose_record=dose,time_record=time,
                roles_X_Z1_Z2_V_assigned=False,measurement_access=False))
        assert len(wells)==len({w['well_id'] for w in wells})
        write(OUTPUT / (sub+'_well_access_plan.csv'), wells)
        well_counts[ds]=dict(rows=len(wells),roles=dict(Counter(w['identity_role'] for w in wells)))
    manifest=dict(version='20260917-v1', status='identity_reservations_active_measurements_not_released',
        identity_assignments='identity_assignments.csv', measurement_access_released=False,
        **reservation_state, released_role_purposes={},
        identity_exclusion_overlay=OVERLAY_RELATIVE,
        EU_original_protected_groups=len(reservation_state['original_reserved_connectivity_groups']),
        EU_eligible_confirmation_candidates=sum(r['identity_role']=='EU_CONFIRMATION_RESERVED' for r in assignments),
        original_qualification='../new_data_qualification_20260917_v1/identity_overlap_summary.json',
        old_JUMP_protected_wells='../new_data_qualification_20260917_v1/protected_JUMP_wells.csv',
        fitted_models_changed=False, evaluation_roles_assigned=False,
        guard='opal2.identity_reservations.IdentityReservations; explicit loader check, not OS access control',
        invariants=['No evaluation identities in any reference bank or representation pretraining',
                    'All doses and stereo/salt connectivity siblings share identity reservation',
                    'All guides of a gene share future grouped split',
                    'Previously reserved connectivity groups and object identities remain protected across reruns',
                    'Historical-overlap exclusions do not re-enter development or confirmation automatically',
                    'No metadata qualification result opens measurements automatically'])
    (OUTPUT/'reservation_manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=dict(identity_roles={ds:dict(Counter(r['identity_role'] for r in assignments if r['dataset']==ds))
        for ds in sorted({r['dataset'] for r in assignments})}, wells=well_counts,
        EU_reserved_connectivity_groups=len(reserved),
        EU_original_protected_groups=len(reservation_state['original_reserved_connectivity_groups']),
        EU_eligible_confirmation_candidates=sum(r['identity_role']=='EU_CONFIRMATION_RESERVED' for r in assignments),
        EU_reserved_historical_overlap_ineligible=sum(r['identity_role']==INELIGIBLE_ROLE for r in assignments),
        identity_exclusion_overlay=OVERLAY_RELATIVE,
        previous_reserved_groups_retained=reservation_state['previous_reserved_groups_retained'],
        measurement_access_released=False)
    (OUTPUT/'assignment_summary.json').write_text(json.dumps(summary,indent=2))
    # Resource support only: shared exact target labels, not response borrowing.
    # Generic action words (e.g. "inhibitor") are NOT treated as a shared MoA.
    annotations={r['object_id']:r for r in read(INPUT/'eu_openscreen/annotation_support.csv')}
    support=build_target_support(assignments, annotations)
    write(OUTPUT/'eu_candidate_target_support.csv',support)
    summary['eu_metadata_target_support']={str(n):sum(r['candidate_donor_id_count']>=n for r in support)
        for n in [1,3,5,10]}
    counts=[]
    conditions=read(INPUT/'rxrx3/condition_repeat_counts.csv')
    for dose in sorted({c['dose_value_uM'] for c in conditions if c['perturbation_class']=='compound'},key=float):
        rr=[c for c in conditions if c['perturbation_class']=='compound' and c['dose_value_uM']==dose
            and byid[('RxRx3_core',c['object_id'])]['identity_role']=='RXRX3_MODULE_DEV_CANDIDATE'
            and 'Query Compounds + Intron control' in c['well_type_labels']
            and c['four_distinct_plates_in_one_experiment']=='True']
        counts.append(dict(dose_uM=dose,complete_metadata_conditions=len(rr),
            object_ids=len({r['object_id'] for r in rr}),
            connectivity_groups=len({byid[('RxRx3_core',r['object_id'])]['connectivity'] for r in rr}),
            protocol_time='18-24 h; individual execution time unknown',outcomes_read=False))
    write(OUTPUT/'rxrx3_dose_feasibility.csv',counts)
    (OUTPUT/'assignment_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
