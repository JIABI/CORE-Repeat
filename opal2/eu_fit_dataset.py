"""Build the declared EU development assay from an authorized FIT-only export."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .biology_kernel_evaluation import write_json
from .eu_assay_preprocessing import fit_control_space, apply_control_space


ROLES = ('X', 'Z1', 'Z2', 'V')


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def validate_export_metadata(metadata_rows, allow_rows):
    """Check physical identities before opening the allowed measurement CSV."""
    allow = {r['well_id']: r for r in allow_rows}
    if len(allow) != len(allow_rows):
        raise ValueError('Duplicate well in allowlist')
    keys = [r['well_id'] for r in metadata_rows]
    if len(keys) != len(set(keys)) or set(keys) != set(allow):
        raise ValueError('Export and declared physical-well allowlist differ')
    for index, row in enumerate(metadata_rows):
        if int(row['export_row_index']) != index:
            raise ValueError('Export metadata row order changed')
        expected = allow[row['well_id']]
        for key in ('plate_uid', 'object_id', 'resource_kind', 'measurement_role'):
            if row[key] != expected[key]:
                raise ValueError('Export identity/role mapping disagrees with frozen plan')
    return allow


def chemistry_for(ids, source):
    source_rows = read_rows(source)
    if any('dataset' in row for row in source_rows):
        source_rows = [r for r in source_rows if r.get('dataset') == 'EU_OPENSCREEN']
    index = {r['object_id']: r for r in source_rows}
    if len(index) != len(source_rows):
        raise ValueError('Duplicate EU identity records; no silent chemistry overwrite')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    out = np.zeros((len(ids), 513), np.float64)
    mask = np.zeros(len(ids), bool)
    for i, oid in enumerate(ids):
        if oid not in index:
            raise ValueError('FIT identity missing from chemistry metadata')
        mol = Chem.MolFromSmiles(index[oid]['smiles'])
        if mol is None:
            raise ValueError('Invalid FIT SMILES; do not silently remove an identity')
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), out[i, :512])
        out[i, 512], mask[i] = 1., True
    return out, mask


def build_dataset(plan_dir, ingest_dir, identity_source, output, *, complete_case_policy=None):
    plan, ingest, output = map(Path, (plan_dir, ingest_dir, output))
    if output.exists():
        raise FileExistsError(output)
    inventory = json.loads((ingest/'audit.json').read_text())
    if inventory.get('dataset_incomplete_training_blocked', True) and complete_case_policy is None:
        raise ValueError('Authorized export incomplete: '+str(inventory.get('missing_allowed_wells','unknown'))
                         +' planned wells absent. Population and missingness policy need resolution before training.')
    if inventory.get('protected_numeric_rows_decoded') != 0 or inventory.get('protected_outcome_rows_saved') != 0:
        raise ValueError('Export violates protected-outcome restrictions')
    manifest = json.loads((plan/'phase_manifest.json').read_text())
    if manifest['phase'] != 'FMP_HEPG2_FIT911_DEVELOPMENT' or manifest['reserved_confirmation_measurements_released']:
        raise ValueError('Wrong or expanded measurement phase')
    allow_rows = read_rows(plan/'measurement_allowlist.csv')
    meta = read_rows(ingest/'row_metadata.csv')
    excluded_ids = set()
    policy = None
    if complete_case_policy is not None:
        policy = json.loads(Path(complete_case_policy).read_text())
        missing = read_rows(ingest/'missing_allowed_wells.csv')
        if policy.get('mode') != 'development_complete_cases_keep_original_splits':
            raise ValueError('Unrecognized complete-case policy')
        excluded_ids = set(policy['excluded_compound_ids'])
        missing_keys = {r['well_id'] for r in missing}
        if (excluded_ids != {r['object_id'] for r in missing} or len(excluded_ids) != 7
                or len(missing_keys) != 13 or len(missing_keys) != len(missing)
                or any(r['resource_kind'] != 'FIT_COMPOUND' for r in missing)
                or missing_keys != ({r['well_id'] for r in allow_rows} - {r['well_id'] for r in meta})):
            raise ValueError('Complete-case authorization differs from audited missing wells')
        actual_allow = [r for r in allow_rows if r['well_id'] not in missing_keys]
        validate_export_metadata(meta, actual_allow)
        allow = {r['well_id']:r for r in allow_rows}
    else:
        allow = validate_export_metadata(meta, allow_rows)
    allowed_ids = set(manifest['allowed_compound_ids'])
    drug_ids = {r['object_id'] for r in allow_rows if r['resource_kind'] == 'FIT_COMPOUND'}
    if allowed_ids != drug_ids or len(allowed_ids) != 911 or len(allow) != 4428:
        raise ValueError('FIT-only scope changed')
    drug = np.array([r['resource_kind'] == 'FIT_COMPOUND' for r in meta])
    expected_drug_rows = 3631 if policy is not None else 3644
    if int(drug.sum()) != expected_drug_rows or int((~drug).sum()) != 784:
        raise ValueError('Declared drug/control counts differ')
    active = np.array([r['object_id'] not in excluded_ids for r in meta])
    # Only the allowed export is opened. Excluded source rows never reach here.
    with (ingest/'raw_allowed_profiles.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header[:4] != ['Metadata_Batch', 'Metadata_Plate', 'Metadata_Well', 'Metadata_Object_Count']:
            raise ValueError('Source schema differs from audited 2981-column header')
        features = header[4:]
        values = np.empty((len(meta), len(features)), np.float64)
        cell_count = np.empty(len(meta), np.float64)
        count = 0
        for count, row in enumerate(reader, 1):
            if count > len(meta):
                raise ValueError('Too many rows in allowed export')
            m = meta[count-1]
            for src, key in (('Metadata_Batch','source_csv_batch'), ('Metadata_Plate','source_csv_plate'),
                             ('Metadata_Well','source_csv_well')):
                if row[src] != m[key]:
                    raise ValueError('Measurement row identity differs from audited export metadata')
            def number(value):
                return float(value) if value.strip() else np.nan
            if not active[count-1]:
                # The excluded partial object's values do not enter numeric QC or fitting.
                values[count-1] = np.nan
                cell_count[count-1] = np.nan
                continue
            values[count-1] = [number(row[k]) for k in features]
            cell_count[count-1] = number(row['Metadata_Object_Count'])
        if count != len(meta):
            raise ValueError('Missing allowed export rows')
    values, cell_count, drug = values[active], cell_count[active], drug[active]
    meta = [m for m,keep in zip(meta,active) if keep]
    active_ids = allowed_ids - excluded_ids
    if int(drug.sum()) != 4*len(active_ids):
        raise ValueError('Complete-case population is not four wells per identity')
    plates = np.array([r['plate_uid'] for r in meta])
    space = fit_control_space(values, plates, ~drug, features)
    standardized, assay_audit = apply_control_space(values, plates, space, required_wells=drug)
    ids = np.array(sorted(active_ids))
    lookup = {oid: i for i, oid in enumerate(ids)}
    y = np.full((len(ids), 4, standardized.shape[1]), np.nan)
    counts = np.full((len(ids), 4), np.nan)
    well_ids = np.full((len(ids), 4), '', dtype='<U160')
    groups, layout = {}, {}
    for j, m in enumerate(meta):
        if not drug[j]:
            continue
        identity = allow[m['well_id']]
        i, slot = lookup[m['object_id']], ROLES.index(m['measurement_role'])
        if well_ids[i, slot]:
            raise ValueError('Duplicate compound role')
        y[i, slot], counts[i, slot], well_ids[i, slot] = standardized[j], cell_count[j], m['well_id']
        groups[m['object_id']] = identity['connectivity']
        layout[m['object_id']] = identity['library_plate']
    if not np.isfinite(y).all() or np.any(well_ids == ''):
        raise ValueError('Incomplete four-well object; no automatic exclusion')
    chem, mask = chemistry_for(ids, identity_source)
    output.mkdir(parents=True)
    np.savez_compressed(output/'data.npz', ids=ids, groups=np.array([groups[v] for v in ids]),
        Y=y, chem=chem, chem_mask=mask, cell_count=counts, well_ids=well_ids,
        layout=np.array([layout[v] for v in ids]))
    chemical = dict(kind='Morgan binary fingerprint', radius=2, bits=512,
                    final_coordinate='valid_SMILES_indicator', rdkit_version=rdBase.rdkitVersion,
                    chemistry_input='source reagent SMILES',
                    grouping='frozen standardized connectivity, not raw reagent InChIKey')
    metadata = dict(dataset='EU_OPENSCREEN', scope='FIT911 FMP HepG2 internal development',
        n=len(ids), morphology_dimension=y.shape[-1], role_order=ROLES, phase_manifest=str(plan/'phase_manifest.json'),
        identity_split_plan=str(plan/'identity_split_plan.csv'), chemical=chemical,
        measurement_space='DMSO-only per-plate median / unscaled MAD, common positive-MAD coordinates, clip +/-10',
        source_morphology_dimension=len(features), target_names=[], moa_names=[], biology_active=False,
        representation_active=False, cell_count_retained_but_not_core_input=True,
        same_layout_across_replicates=True, role_equals_replicate_batch=True,
        dose_uM_protocol=10., exposure_h_protocol=24., exact_execution_verified=False,
        confirmation_data_loaded=False, qc_objects_removed=False,
        original_planned_n=911, complete_case_population=policy is not None,
        excluded_incomplete_ids=sorted(excluded_ids), original_split_membership_preserved=True)
    write_json(output/'metadata.json', metadata)
    write_json(output/'control_space.json', space)
    write_json(output/'assay_audit.json', assay_audit)
    if policy is not None:
        write_json(output/'complete_case_policy.json',policy)
    return metadata
