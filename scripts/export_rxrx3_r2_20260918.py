#!/usr/bin/env python3
"""Export explicitly authorized RxRx3 roles after identity-only batch filtering.

Mixed parquet batches are decoded in memory under the 2026-09-18 authorization.
Only approved identities reach NumPy, numerical checks, or derived files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT / "reports/r2_rxrx3_access_20260918_v1"
OUTPUT = ROOT / "data/rxrx3_r2_20260918/approved_export"
REPORT = ROOT / "reports/r2_rxrx3_export_20260918_v1"
SOURCE = ROOT / "data/rxrx3_r2_20260918/source/CellProfiler_features_rxrx3_core.parquet"
METADATA = ROOT / "reports/new_data_qualification_20260917_v1/rxrx3/source/metadata_rxrx3_core.csv"
ALLOWED_ROLE = "RXRX3_MODULE_DEV_CANDIDATE"
ROLES = ("X", "Z1", "Z2", "V")
CONTROL_TREATMENT = "EMPTY_control"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fixed_role_order(roles: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize existing assignments; never select or reassign any role."""
    require(not roles.empty, "Empty role assignment")
    require(not roles.well_id.duplicated().any(), "Duplicate role well")
    require(not roles.duplicated(["condition_id", "role"]).any(), "Duplicate condition role")
    require(roles.identity_role.eq(ALLOWED_ROLE).all(), "Role outside development allowlist")
    require(roles.role.isin(ROLES).all(), "Unrecognized role")
    require(roles.groupby("condition_id").size().eq(4).all(), "Incomplete four-role condition")
    for field in ("object_id", "group_id", "batch", "dose_record", "cell_record"):
        require(roles.groupby("condition_id")[field].nunique(dropna=False).eq(1).all(), f"Mixed {field} within condition")
    require(roles.groupby("condition_id").physical_plate_id.nunique().eq(4).all(), "Roles do not use four distinct plates")
    result = roles.assign(_role_order=roles.role.map(dict(zip(ROLES, range(4)))))
    return result.sort_values(["condition_id", "_role_order"]).drop(columns="_role_order").reset_index(drop=True)


def authorized_controls(candidates: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Only author controls on actual development-role plates are eligible."""
    eligible = (
        candidates.treatment.eq(CONTROL_TREATMENT)
        & candidates.perturbation_type.eq("COMPOUND")
        & candidates.batch.str.startswith("compound-")
        & candidates.physical_plate_id.isin(roles.physical_plate_id)
        & candidates.profile_present.eq("True")
    )
    controls = candidates.loc[eligible].copy().sort_values("well_id").reset_index(drop=True)
    require(not controls.well_id.duplicated().any(), "Duplicate control well")
    require(not set(controls.well_id) & set(roles.well_id), "Control and role wells overlap")
    require(set(controls.physical_plate_id) == set(roles.physical_plate_id), "Some development plates lack eligible controls")
    controls["measurement_authorized"] = "True"
    controls["export_role"] = "AUTHOR_NEGATIVE_CONTROL"
    return controls


def validate_metadata(roles: pd.DataFrame, controls: pd.DataFrame, allowlist: pd.DataFrame, metadata: pd.DataFrame) -> None:
    require(not allowlist.well_id.duplicated().any(), "Duplicate allowlist well")
    require(not metadata.well_id.duplicated().any(), "Duplicate official metadata well")
    require(allowlist.identity_role.eq(ALLOWED_ROLE).all(), "Non-development identity in allowlist")
    allowed = allowlist.set_index("well_id")
    require(set(roles.well_id).issubset(allowed.index), "Role not present in allowlist")
    compare = allowed.loc[roles.well_id].reset_index()
    for column in ("object_id", "group_id", "condition_id", "identity_role", "physical_plate_id", "batch", "dose_record", "treatment"):
        require(np.array_equal(roles[column].to_numpy(), compare[column].to_numpy()), f"Role/allowlist mismatch: {column}")
    official = metadata.set_index("well_id")
    for selected in (roles, controls):
        require(set(selected.well_id).issubset(official.index), "Selected well absent from official metadata")
        matched = official.loc[selected.well_id].reset_index()
        require(matched.perturbation_type.eq("COMPOUND").all(), "Non-compound measurement selected")
        require(np.array_equal(selected.treatment.to_numpy(), matched.treatment.to_numpy()), "Treatment mismatch")
        require(np.array_equal(selected.batch.to_numpy(), matched.experiment_name.to_numpy()), "Experiment mismatch")
        require(np.array_equal(selected.physical_plate_id.to_numpy(), (matched.experiment_name + "_" + matched.plate).to_numpy()), "Physical plate mismatch")
        require(np.array_equal(selected.dose_record.to_numpy(), matched.concentration.to_numpy()), "Dose mismatch")


def extract_approved_rows(source: Path, requested_wells: list[str], features: list[str], batch_size: int = 4096) -> tuple[np.ndarray, dict]:
    """Filter each decoded Arrow batch by identity before numerical conversion.

    No numeric expression, summary, comparison, imputation, or conversion touches
    the excluded rows. Parquet decoding necessarily processes their values.
    """
    require(len(requested_wells) == len(set(requested_wells)), "Requested well IDs are not unique")
    lookup = {well: index for index, well in enumerate(requested_wells)}
    seen = np.zeros(len(requested_wells), dtype=bool)
    result = np.empty((len(requested_wells), len(features)), dtype=np.float64)
    parquet = pq.ParquetFile(source)
    require(all(pa.types.is_float64(parquet.schema_arrow.field(name).type) for name in features), "Non-float64 source feature")
    batches = 0
    for decoded in parquet.iter_batches(batch_size=batch_size, columns=["well_id", *features], use_threads=False):
        batches += 1
        keys = decoded.column("well_id").to_pylist()
        positions = [index for index, key in enumerate(keys) if key in lookup]
        if not positions:
            continue
        # The first operation involving measurement rows is identity-only take.
        approved = decoded.take(pa.array(positions, type=pa.int64()))
        approved_keys = approved.column("well_id").to_pylist()
        destinations = np.array([lookup[key] for key in approved_keys], dtype=np.int64)
        require(len(destinations) == len(set(destinations)), "Duplicate approved key within batch")
        require(not seen[destinations].any(), "Duplicate approved key across batches")
        for column_index, name in enumerate(features):
            result[destinations, column_index] = approved.column(name).to_numpy(zero_copy_only=False)
        seen[destinations] = True
        del approved, approved_keys, destinations
    require(seen.all(), f"Missing {int((~seen).sum())} requested approved wells")
    return result, {"decoded_batches": batches, "batch_size": batch_size, "exported_wells": int(seen.sum())}


def approved_quality_counts(values: np.ndarray) -> dict:
    """Counts only: retain every approved row and original finite/nonfinite value."""
    rows = values.reshape(-1, values.shape[-1])
    finite = np.isfinite(rows)
    return {
        "rows": len(rows),
        "nonfinite_values": int((~finite).sum()),
        "rows_with_nonfinite_values": int((~finite.all(axis=1)).sum()),
        "features_with_nonfinite_values": int((~finite.all(axis=0)).sum()),
        "exact_zero_rows": int((rows == 0).all(axis=1).sum()),
    }


def strings(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=str)


def export(source: Path = SOURCE, output: Path = OUTPUT, report: Path = REPORT, batch_size: int = 4096) -> dict:
    started = time.perf_counter()
    authorization = json.loads((ACCESS / "phase_authorization.json").read_text())
    require(authorization.get("numeric_decode_before_filter_authorized") is True, "Mixed-batch numeric decoding not authorized")
    require(authorization.get("negative_control_treatment") == CONTROL_TREATMENT, "Author control not authorized")
    schema = json.loads((ACCESS / "schema.json").read_text())
    require(source.stat().st_size == schema["file_bytes"], "Source size changed since metadata audit")
    features = schema["feature_names"]
    require(len(features) == 951, "Unexpected feature count")
    require(pq.ParquetFile(source).schema_arrow.names == [field["name"] for field in schema["fields"]], "Source schema changed")
    roles = fixed_role_order(read_csv(ACCESS / "role_assignments.csv"))
    controls = authorized_controls(read_csv(ACCESS / "candidate_negative_controls.csv"), roles)
    metadata = read_csv(METADATA)
    validate_metadata(roles, controls, read_csv(ACCESS / "allowlisted_wells.csv"), metadata)
    conditions = roles.drop_duplicates("condition_id").copy().reset_index(drop=True)
    official = metadata.set_index("well_id")
    conditions["smiles"] = official.loc[conditions.well_id, "SMILES"].to_numpy()
    conditions = conditions.drop(columns=["well_id", "physical_plate_id", "role"])
    conditions["measurement_access"] = "True"
    conditions["roles_X_Z1_Z2_V_assigned"] = "True"
    roles["measurement_access"] = "True"
    roles["roles_X_Z1_Z2_V_assigned"] = "True"
    control_names = controls.well_id.tolist()
    requested_wells = roles.well_id.tolist() + control_names
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    # Persist the exact permitted identities before opening numerical columns.
    roles.to_csv(report / "role_export_allowlist.csv", index=False)
    controls.to_csv(report / "control_export_allowlist.csv", index=False)
    conditions.to_csv(output / "conditions.csv", index=False)
    roles.to_csv(output / "roles.csv", index=False)
    controls.to_csv(output / "controls.csv", index=False)
    metadata_seconds = time.perf_counter() - started
    print(f"Identity allowlists fixed: {len(roles)} role wells and {len(controls)} controls", flush=True)
    values, extraction = extract_approved_rows(source, requested_wells, features, batch_size)
    decode_seconds = time.perf_counter() - started - metadata_seconds
    Y = values[:len(roles)].reshape(len(conditions), 4, len(features))
    C = values[len(roles):]
    common = {"feature_names": np.array(features), "source_revision": np.array(schema["source_revision"])}
    np.savez(output / "measurements.npz", Y=Y, ids=strings(conditions.condition_id), groups=strings(conditions.group_id),
             object_ids=strings(conditions.object_id), well_ids=strings(roles.well_id).reshape(-1, 4),
             plates=strings(roles.physical_plate_id).reshape(-1, 4), dose=conditions.dose_record.to_numpy(dtype=float),
             dose_record=strings(conditions.dose_record), smiles=strings(conditions.smiles), treatments=strings(conditions.treatment),
             batches=strings(conditions.batch), cells=strings(conditions.cell_record), roles=np.array(ROLES), **common)
    np.savez(output / "controls.npz", Y=C, well_ids=strings(controls.well_id), plates=strings(controls.physical_plate_id),
             batches=strings(controls.batch), treatments=strings(controls.treatment), control_semantics=np.array("author negative control; not established as DMSO"), **common)
    quality_roles = approved_quality_counts(Y)
    quality_controls = approved_quality_counts(C)
    result = {
        "stage": "authorized_identity_blind_export_complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": schema["source_revision"], "source_file": str(source),
        "measurements_file": str(output / "measurements.npz"), "controls_file": str(output / "controls.npz"),
        "shape": list(Y.shape), "controls_shape": list(C.shape), "dtype": str(Y.dtype),
        "condition_count": len(conditions), "object_count": int(conditions.object_id.nunique()),
        "connectivity_groups": int(conditions.group_id.nunique()), "physical_plates": int(roles.physical_plate_id.nunique()),
        "allowed_identity_role": ALLOWED_ROLE, "role_order": list(ROLES), "role_assignments_changed": False,
        "all_exported_role_wells_in_original_allowlist": True, "all_requested_profiles_found": True,
        "all_control_wells_on_role_plates": True, "control_treatment": CONTROL_TREATMENT,
        "control_interpretation": "Author negative/non-targeting control; vehicle identity is not verified as DMSO",
        "mixed_source_values_decoded_in_memory": True, "excluded_values_decoded_in_memory": True,
        "identity_filter_before_numpy_conversion_or_numeric_operations": True,
        "excluded_numeric_statistics_computed": False, "excluded_measurements_exported": False,
        "all_feature_columns_retained": True, "source_numeric_values_transformed": False,
        "imputation_performed": False, "numeric_value_based_row_removal": False,
        "scaling_or_pca_fitted": False, "model_training_performed_by_export": False,
        "approved_role_quality": quality_roles, "approved_control_quality": quality_controls,
        "extraction": extraction,
        "timing_seconds": {"metadata_validation": metadata_seconds, "decode_then_filter": decode_seconds, "total": time.perf_counter() - started},
    }
    write_json(report / "export_audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--batch-size", type=int, default=4096)
    arguments = parser.parse_args()
    export(arguments.source, arguments.output, arguments.report, arguments.batch_size)
