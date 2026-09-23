"""Read-only adapter for the previously opened source_5 development allocation.

No network calls, raw-profile discovery, fifth-repeat lookup or FINAL reads are
implemented here. Outcome normalization is inherited unchanged; prediction-time
reference exposure is independently controlled by the episode builder.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .data import MeasurementDataset, cellprofiler_groups


ROLE_ORDER = ("X", "Z1", "Z2", "V")
DMSO = "JCP2022_033924"


def morgan_fingerprints(smiles, n_bits=512, radius=2):
    """RDKit Morgan radius-2 bits plus explicit valid-structure indicator.

    A failed/missing SMILES has an all-zero bit vector and valid=0, not a
    fabricated molecule. The final input coordinate carries this missingness.
    """
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    if n_bits < 1 or radius < 0:
        raise ValueError("Invalid Morgan fingerprint size/radius")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    matrix = np.zeros((len(smiles), n_bits + 1), dtype=np.float64)
    valid = np.zeros(len(smiles), dtype=bool)
    reasons = []
    for i, value in enumerate(smiles):
        text = str(value).strip()
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(text) if text and text.lower() not in {"nan", "none"} else None
        if molecule is None or molecule.GetNumAtoms() == 0:
            reasons.append({"row": i, "reason": "missing_or_unparseable_SMILES"})
            continue
        bits = generator.GetFingerprint(molecule)
        array = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bits, array)
        matrix[i, :n_bits] = array
        matrix[i, -1] = 1
        valid[i] = True
    return matrix, valid, {"kind": "RDKit Morgan fingerprint", "radius": radius,
                           "bits": n_bits, "final_coordinate": "valid_SMILES_indicator",
                           "valid": int(valid.sum()), "missing": int((~valid).sum()),
                           "parse_failures": reasons, "rdkit_version": rdBase.rdkitVersion}


def _legacy_loader(root: Path):
    path = root / "scripts/source5_stage2_data.py"
    if not path.is_file():
        raise FileNotFoundError("Expected existing source5_stage2_data.py")
    # Its dependencies perform no download on import.  load_development itself
    # executes the existing DEV identity and reserved-allocation guards.
    scripts = str(path.parent)
    inserted = scripts not in sys.path
    old_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    if inserted:
        sys.path.insert(0, scripts)
    try:
        for name in ("source5_dev_profiles", "source5_dmso_diagnostics"):
            if name in sys.modules:
                module_path = Path(sys.modules[name].__file__).resolve()
                if module_path.parent != path.parent:
                    raise RuntimeError(f"Ambiguous legacy module already imported: {name}")
        spec = importlib.util.spec_from_file_location("_opal2_legacy_stage2_data", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = old_bytecode_setting
        if inserted:
            sys.path.remove(scripts)


def _reference_summary(values, plates, feature_groups, group_count):
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Reference summary requires genuine finite control measurements")
    center = np.median(values, axis=0)
    scale = np.sqrt(np.mean(np.square(values - center), axis=0))
    summary = [np.log1p(len(values)), np.log1p(len(set(map(str, plates))))]
    for group in range(group_count):
        use = feature_groups == group
        summary.extend((float(np.mean(center[use])),
                        float(np.log1p(np.sqrt(np.mean(np.square(center[use]))))),
                        float(np.log1p(np.sqrt(np.mean(np.square(scale[use])))))))
    return np.asarray(summary, dtype=np.float64)


def load_source5(space="primary", legacy_root=None, fingerprint_bits=512) -> MeasurementDataset:
    """Return exactly the already-open 639 compounds and four assigned outcomes.

    ``legacy_root`` is explicit to make this adapter portable. Existing outcome
    normalization uses controls on outcome plates; those controls are not thereby
    granted to the predictor. No target-only measured QC enters ``cond``.
    """
    if legacy_root is None:
        raise ValueError("Provide the existing JUMP study directory as legacy_root")
    root = Path(legacy_root).expanduser().resolve()
    legacy = _legacy_loader(root)
    loaded = legacy.load_development(space)
    y = loaded["Y"]
    ids = loaded["ids"]
    names = loaded["feature_names"]
    if y.shape != (639, 4, 3617) or len(set(ids)) != 639:
        raise ValueError("This adapter is restricted to the existing 639 x 4 x 3617 development allocation")
    manifest = json.loads((root / "protocol/source5_access_manifest_r1.json").read_text())
    if loaded["info"]["manifest_created_utc"] != manifest["created_utc"]:
        raise ValueError("Legacy allocation changed during the read")
    role_path = (root / manifest["roles_path"]).resolve()
    if root not in role_path.parents:
        raise ValueError("Role allocation must remain inside the existing study")
    roles = pd.read_csv(role_path, sep="\t", dtype=str, keep_default_na=False)
    role_column = next((key for key in ("role", "role_r1", "role_preview") if key in roles), None)
    if role_column is None:
        raise ValueError("No existing role allocation")
    roles = roles.rename(columns={role_column: "role"})
    roles = roles.loc[roles.Metadata_JCP2022.isin(ids) & roles.role.isin(ROLE_ORDER)].copy()
    roles["_role_index"] = roles.role.map({name: j for j, name in enumerate(ROLE_ORDER)})
    roles = roles.sort_values(["Metadata_JCP2022", "_role_index"])
    if len(roles) != 639 * 4 or roles.groupby("Metadata_JCP2022").size().ne(4).any():
        raise ValueError("Expected exactly four already assigned wells per DEV compound")
    for compound, rows in roles.groupby("Metadata_JCP2022", sort=True):
        if tuple(rows.role) != ROLE_ORDER:
            raise ValueError(f"Duplicate/missing assigned role for {compound}")
    if roles.Metadata_JCP2022.to_numpy().reshape(639, 4)[:, 0].tolist() != ids.tolist():
        raise ValueError("Metadata does not match the legacy measurement order")

    feature_group_index, feature_group_names = cellprofiler_groups(names)
    group_count = len(feature_group_names)
    group_counts = Counter(feature_group_names[i] for i in feature_group_index)
    raw_path = root / "data/processed/source5_dev_all_wells_r1.npz"
    preflight = json.loads((root / "reports/source5_development/stage2_preflight_r1.json").read_text())
    selected = np.asarray(preflight["retained_feature_indices"], dtype=int)
    scales = np.asarray(preflight["initial_reference_fixed_scales"], dtype=float)
    with np.load(raw_path, allow_pickle=False) as opened:
        # The old loader has already checked both reserved-compound and batch
        # exclusion for this exact opened numeric object. Repeat identifiers only.
        if set(opened["jcp"]) & set(manifest["reserved_final_compound_ids"]):
            raise ValueError("Reserved compound in purported opened DEV file")
        if set(opened["batch"]) & set(manifest["reserved_final_batches"]):
            raise ValueError("Reserved batch in purported opened DEV file")
        if not np.array_equal(opened["feature_names"][selected], names):
            raise ValueError("Control coordinates differ from the fixed endpoint coordinates")
        dmso = opened["jcp"] == DMSO
        # Slice genuine controls only; additional opened treatment rows are not
        # exported or used to form references.
        controls = opened["values"][dmso][:, selected] / scales
        control_plate = opened["plate"][dmso].copy()
        control_batch = opened["batch"][dmso].copy()
        control_type = opened["plate_type"][dmso].copy()
        control_well_id = opened["physical_well_ids"][dmso].copy()
        control_well = opened["well"][dmso].copy()
    # Match the declared endpoint coordinate system for full-profile reference
    # anchoring. Only genuine controls enter this transformation. In spatial
    # mode, fit the already declared row/column correction on whole-DMSO plates.
    panel_controls = controls.copy()
    if space == "spatial":
        pos = np.asarray([legacy.position(well) for well in control_well])
        for batch in sorted(set(roles.Metadata_Batch)):
            ref = (control_batch == batch) & (control_type == "DMSO")
            local = control_batch == batch
            design = legacy.model_matrix(control_plate[ref],pos[ref,0],pos[ref,1],True)
            if np.linalg.matrix_rank(design) != design.shape[1]:
                raise ValueError("Reference-only spatial design is rank deficient")
            coeff = np.linalg.lstsq(design,controls[ref],rcond=None)[0]
            nr = len(set(control_plate[ref]))
            rl,cl = sorted(set(pos[ref,0]))[1:],sorted(set(pos[ref,1]))[1:]
            spatial = np.column_stack([(pos[local,0] == x).astype(float) for x in rl]
                                    + [(pos[local,1] == x).astype(float) for x in cl])
            panel_controls[local] -= spatial @ coeff[nr:]
    for plate in sorted(set(control_plate)):
        use = control_plate == plate
        panel_controls[use] -= np.median(panel_controls[use],axis=0)
    ref_names = ["log1p_control_wells", "log1p_control_plates"]
    for group in feature_group_names:
        ref_names += [group + "::median_coordinate_mean", group + "::log1p_median_coordinate_RMS",
                      group + "::log1p_control_residual_RMS"]
    rdim = len(ref_names)
    reference = np.zeros((639, 4, 3, rdim), dtype=np.float64)
    reference_mask = np.zeros((639, 4, 3), dtype=bool)
    batch_cache, plate_cache, reference_objects = {}, {}, []
    for batch in sorted(set(roles.Metadata_Batch)):
        use = (control_batch == batch) & (control_type == "DMSO")
        if use.any():
            batch_cache[batch] = _reference_summary(controls[use], control_plate[use], feature_group_index, group_count)
            reference_objects.append({"level": "batch", "identifier": batch, "kind": "whole_DMSO_plate_controls",
                                      "control_wells": int(use.sum()), "physical_well_ids": control_well_id[use].tolist()})
    for plate in sorted(set(roles.Metadata_Plate)):
        use = (control_plate == plate) & (control_type == "COMPOUND")
        if use.any():
            plate_cache[plate] = _reference_summary(controls[use], control_plate[use], feature_group_index, group_count)
            reference_objects.append({"level": "plate", "identifier": plate, "kind": "compound_plate_DMSO_controls",
                                      "control_wells": int(use.sum()), "physical_well_ids": control_well_id[use].tolist()})
    source_names = sorted(set(roles.Metadata_Source))
    batch_names = sorted(set(roles.Metadata_Source + "::" + roles.Metadata_Batch))
    plate_names = sorted(set(roles.Metadata_Source + "::" + roles.Metadata_Batch + "::" + roles.Metadata_Plate))
    lookup = [{name: i for i, name in enumerate(values)} for values in (source_names, batch_names, plate_names)]
    catalog_y,catalog_ids,catalog_groups,catalog_members = [],[],[],[]
    catalog_lookup = {}
    physical_lookup = {}
    for item in reference_objects:
        level,key = item["level"],item["identifier"]
        members = set(item["physical_well_ids"])
        use = np.asarray([well in members for well in control_well_id])
        batches = sorted(set(control_batch[use]))
        if len(batches) != 1:
            raise ValueError("Each source5 catalog profile must belong to one real batch")
        batch_name = source_names[0]+"::"+batches[0]
        plate_group = -1 if level == "batch" else lookup[2][batch_name+"::"+key]
        selected_rows = []
        for row in np.flatnonzero(use):
            physical = str(control_well_id[row])
            if physical not in physical_lookup:
                physical_lookup[physical] = len(catalog_y)
                catalog_y.append(panel_controls[row])
                catalog_ids.append("physical_reference::"+physical)
                catalog_groups.append([lookup[0][source_names[0]],lookup[1][batch_name],plate_group])
                catalog_members.append([physical])
            selected_rows.append(physical_lookup[physical])
        catalog_lookup[(level,key)] = selected_rows
    max_members = max(map(len,catalog_members),default=0)
    member_matrix = np.full((len(catalog_members),max_members),"",dtype=object)
    for row,members in enumerate(catalog_members):
        member_matrix[row,:len(members)] = members
    max_panel_size = max(map(len,catalog_lookup.values()),default=0)
    panel_index = np.full((639,4,3,max_panel_size),-1,int)
    groups = np.zeros((639, 4, 3), dtype=np.int64)
    condition_names = ["log1p_nominal_concentration_uM", "planned_row_fraction", "planned_column_fraction",
                       "concentration_from_protocol_indicator"]
    # A planned batch ID is known without its outcome.  Explicit metadata-only
    # one-hot descriptors distinguish candidate batches in this within-source
    # diagnostic. They are not evidence of adaptation to unseen batch identities.
    condition_names += ["known_source::" + name for name in source_names]
    condition_names += ["known_batch::" + name for name in batch_names]
    condition_names += ["known_plate::" + name for name in plate_names]
    condition_names += ["unseen_source_indicator", "unseen_batch_indicator", "unseen_plate_indicator"]
    condition_offsets = np.cumsum([4, len(source_names), len(batch_names)]).tolist()
    cond = np.zeros((639, 4, len(condition_names)), dtype=np.float64)
    well_ids = np.empty((639, 4), dtype=object)
    for k, row in enumerate(roles.itertuples(index=False)):
        i, j = divmod(k, 4)
        keys = (row.Metadata_Source, row.Metadata_Source + "::" + row.Metadata_Batch,
                row.Metadata_Source + "::" + row.Metadata_Batch + "::" + row.Metadata_Plate)
        groups[i, j] = [lookup[level][key] for level, key in enumerate(keys)]
        r = ord(row.Metadata_Well[0].upper()) - ord("A")
        c = int(row.Metadata_Well[1:]) - 1
        if not 0 <= r < 16 or not 0 <= c < 24:
            raise ValueError("Unexpected source_5 plate geometry")
        cond[i, j, :4] = (np.log1p(10.0), r / 15.0, c / 23.0, 1.0)
        for level, offset in enumerate(condition_offsets):
            cond[i, j, offset + groups[i, j, level]] = 1.0
        well_ids[i, j] = row.Metadata_Source + "::" + row.Metadata_Plate + "::" + row.Metadata_Well
        if row.Metadata_Batch in batch_cache:
            reference[i, j, 1] = batch_cache[row.Metadata_Batch]
            reference_mask[i, j, 1] = True
            rows = catalog_lookup[("batch",row.Metadata_Batch)]
            panel_index[i,j,1,:len(rows)] = rows
        if row.Metadata_Plate in plate_cache:
            reference[i, j, 2] = plate_cache[row.Metadata_Plate]
            reference_mask[i, j, 2] = True
            rows = catalog_lookup[("plate",row.Metadata_Plate)]
            panel_index[i,j,2,:len(rows)] = rows
    initial = loaded["initial_metadata"]
    if initial.Metadata_JCP2022.tolist() != ids.tolist():
        raise ValueError("Chemical structures are misaligned with compounds")
    chemical, chemical_mask, chemical_info = morgan_fingerprints(initial.SMILES.tolist(), fingerprint_bits)
    unique_ref_ids = sorted({well for item in reference_objects for well in item["physical_well_ids"]})
    metadata = {
        "dataset": "JUMP source_5 previously opened DEV", "space": space, "scope": "639_DEV_FOUR_ROLES_ONLY",
        "roles": list(ROLE_ORDER), "legacy_info": loaded["info"],
        "outcome_normalization": "Exact existing load_development output; no new clipping or coordinate selection",
        "outcome_normalization_controls_are_not_prediction_inputs_by_default": True,
        "reference_summary_coordinate_basis": "genuine DMSO raw retained coordinates / fixed initial-X MAD; no treatment values",
        "reference_summary_names": ref_names, "reference_levels": ["source", "batch", "plate"],
        "reference_kind": "DMSO_ONLY", "target2_reference_available": False,
        "reference_availability": {
            "source": "missing: no separately declared source-level panel",
            "batch": "whole-DMSO controls of that well's batch; allowed only after its context is observed unless explicitly declared in advance",
            "plate": "DMSO controls of that well's plate; allowed only after its context is observed unless explicitly declared in advance",
            "default_target_mask": "all false, even if a source group matches context",
            "timing": "context-associated control availability is a declared retrospective assumption, not verified execution timing",
        },
        "reference_objects": reference_objects, "unique_reference_wells": len(unique_ref_ids),
        "reference_cost": "shared physical-control cost not included in per-compound optional-well price; count once at campaign level",
        "condition_names": condition_names,
        "condition_category_scope": "one-hot planned source/batch/plate identities from this opened allocation; no role names or outcome values",
        "condition_generalization": "known-batch within-source diagnostic only; unseen IDs need frozen-vocabulary unknown indicators and informative numerical descriptors/references, not a claim of zero-shot biological transport",
        "nominal_concentration_uM": 10.0,
        "concentration_status": "Protocol-level ordinary-library concentration; not independently verified per executed well",
        "condition_scope": "known planned conditions only; no target cell count/intensity/observed QC or Gamma",
        "group_names": {"source": source_names, "batch": batch_names, "plate": plate_names},
        "feature_grouping": "CellProfiler compartment :: measurement family :: recognized channel tokens; no asserted pathway labels",
        "reference_profile_catalog": "Every genuine individual DMSO control retained with identity and physical ID; compact shared catalog, not aggregated pseudo-wells or Target-2",
        "reference_profile_coordinates": "Same retained coordinates, fixed MAD and DMSO plate centering as endpoint; spatial correction fitted only on whole-DMSO controls when declared",
        "reference_template_policy": "Fitted from training-owned catalog rows; physically overlapping controls excluded from each identity template",
        "cell_counts_available": False,
        "physical_identity_verified": True,
        "feature_group_counts": dict(sorted(group_counts.items())),
        "chemical": chemical_info, "smiles": initial.SMILES.astype(str).tolist(),
        "roles_establish_chronology": False,
        "final_profiles_read": False, "fifth_repeat_read": False,
        "development_expansion_is_not_independent_sample_expansion": True,
    }
    return MeasurementDataset(y.copy(), ids.copy(), names.copy(), feature_group_index, feature_group_names,
                              cond, reference, reference_mask, groups, chemical,
                              well_ids=well_ids.astype(str), chem_mask=chemical_mask, metadata=metadata,
                              panel_y=np.asarray(catalog_y),panel_ids=np.asarray(catalog_ids),
                              panel_identity=np.repeat(DMSO,len(catalog_y)),panel_groups=np.asarray(catalog_groups),
                              panel_members=member_matrix.astype(str),panel_index=panel_index)
