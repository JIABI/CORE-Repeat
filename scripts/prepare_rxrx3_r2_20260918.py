#!/usr/bin/env python3
"""Prepare RxRx3 chemical-development identities without opening measurement columns.

The default audit reads only parquet schema and well_id, then official metadata.
The authorized numerical export is a separate operation implemented in
export_rxrx3_r2_20260918.py; this metadata-only audit never opens features.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/r2_rxrx3_access_20260918_v1"
SOURCE = ROOT / "data/rxrx3_r2_20260918/source/CellProfiler_features_rxrx3_core.parquet"
ASSIGNMENT = ROOT / "reports/new_data_assignment_20260917_v1/rxrx3_well_access_plan.csv"
METADATA = ROOT / "reports/new_data_qualification_20260917_v1/rxrx3/source/metadata_rxrx3_core.csv"
ALLOWED_ROLE = "RXRX3_MODULE_DEV_CANDIDATE"
SEED = 20260918
EXPECTED_BYTES = 1460689326
ROLES = ("X", "Z1", "Z2", "V")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def feature_names(schema: pa.Schema) -> list[str]:
    """Treat source pandas index as bookkeeping, never as a biological feature."""
    result = []
    for field in schema:
        if field.name in {"well_id", "__index_level_0__"}:
            continue
        if not pa.types.is_floating(field.type):
            raise ValueError(f"Unexpected measurement field type: {field.name}: {field.type}")
        result.append(field.name)
    if not result or len(result) != len(set(result)):
        raise ValueError("No features or duplicate feature names")
    return result


def role_assignments(available: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Select four physical plates in one experiment using metadata only.

    Chemical identity and dose remain separate from the biological split group.
    The seeded ordering is determined before seeing any measurement values.
    """
    records = []
    required = {"condition_id", "batch", "physical_plate_id", "well_id", "object_id", "group_id"}
    if not required.issubset(available.columns):
        raise ValueError(f"Missing role fields: {required - set(available.columns)}")
    for condition, condition_rows in available.groupby("condition_id", sort=True):
        candidates = []
        for batch, batch_rows in condition_rows.groupby("batch", sort=True):
            if batch_rows.physical_plate_id.nunique() >= 4:
                candidates.append((batch, batch_rows))
        if not candidates:
            continue
        batch, rows = candidates[0]
        rows = rows.copy()
        rows["_order"] = rows.well_id.map(
            lambda well: hashlib.blake2b(f"{seed}|{condition}|{well}".encode(), digest_size=16).hexdigest()
        )
        selected = rows.sort_values(["_order", "well_id"]).drop_duplicates("physical_plate_id").head(4)
        for role, (_, row) in zip(ROLES, selected.iterrows(), strict=True):
            record = row.drop(labels=["_order"]).to_dict()
            record.update({"role": role, "role_seed": seed, "role_batch": batch})
            records.append(record)
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        assert result.groupby("condition_id").size().eq(4).all()
        assert result.groupby("condition_id").physical_plate_id.nunique().eq(4).all()
        assert result.groupby("condition_id").batch.nunique().eq(1).all()
        assert result.groupby("condition_id").object_id.nunique().eq(1).all()
        assert result.groupby("condition_id").dose_record.nunique(dropna=False).eq(1).all()
        assert result.identity_role.eq(ALLOWED_ROLE).all()
    return result


def audit(source: Path = SOURCE, output: Path = REPORT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size != EXPECTED_BYTES:
        raise ValueError("Source download is incomplete or differs from the verified release size")
    parquet = pq.ParquetFile(source)
    features = feature_names(parquet.schema_arrow)
    schema = {
        "source_file": str(source), "file_bytes": source.stat().st_size,
        "source_revision": "89aedc798bf33e6f51cc8c6363009d9ffed69e31",
        "rows": parquet.metadata.num_rows, "row_groups": parquet.num_row_groups,
        "row_group_sizes": [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)],
        "created_by": parquet.metadata.created_by,
        "fields": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
        "feature_count": len(features), "feature_names": features,
        "excluded_bookkeeping_columns": ["well_id", "__index_level_0__"],
        "numeric_columns_read": [], "numeric_statistics_inspected": False,
    }
    write_json(output / "schema.json", schema)

    # Projection is applied by the parquet reader; no measurement column is read.
    keys = parquet.read(columns=["well_id"]).column("well_id").to_pylist()
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("Missing/non-string well key")
    key_counts = Counter(keys)
    duplicates = {key: count for key, count in key_counts.items() if count > 1}
    if duplicates:
        raise ValueError(f"Duplicate source well keys: {len(duplicates)}")
    key_set = set(keys)

    assignments = pd.read_csv(ASSIGNMENT, dtype=str, keep_default_na=False)
    metadata = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    if assignments.well_id.duplicated().any() or metadata.well_id.duplicated().any():
        raise ValueError("Metadata or assignment has duplicate well IDs")
    if not key_set.issubset(set(metadata.well_id)):
        raise ValueError("CP file contains wells absent from qualified metadata")
    joined = assignments.merge(
        metadata[["well_id", "treatment", "perturbation_type", "well_type_label"]],
        on="well_id", how="left", validate="one_to_one",
    )
    allowed = joined[joined.identity_role.eq(ALLOWED_ROLE)].copy()
    assert allowed.perturbation_type.eq("COMPOUND").all()
    allowed["profile_present"] = allowed.well_id.isin(key_set)
    allowed.to_csv(output / "allowlisted_wells.csv", index=False)
    available = allowed[allowed.profile_present].copy()
    available.to_csv(output / "available_development_wells.csv", index=False)
    roles = role_assignments(available)
    roles.to_csv(output / "role_assignments.csv", index=False)

    # Candidates only: no control numerical values are read or admitted.
    controls = joined[
        joined.treatment.eq("EMPTY_control")
        & joined.batch.isin(set(allowed.batch))
        & joined.batch.str.startswith("compound-")
    ].copy()
    controls["profile_present"] = controls.well_id.isin(key_set)
    controls["measurement_authorized"] = False
    controls.to_csv(output / "candidate_negative_controls.csv", index=False)

    role_conditions = roles.drop_duplicates("condition_id") if not roles.empty else roles
    condition_hist = (
        role_conditions.groupby("object_id").size().value_counts().sort_index().to_dict()
        if not roles.empty else {}
    )
    per_dose = (
        role_conditions.groupby("dose_record").agg(
            conditions=("condition_id", "nunique"),
            object_ids=("object_id", "nunique"),
            connectivity_groups=("group_id", "nunique"),
        ).reset_index()
        if not roles.empty else pd.DataFrame()
    )
    per_dose.to_csv(output / "available_dose_conditions.csv", index=False)
    audit_data = {
        "stage": "metadata_only_after_authorized_download",
        "source_rows": len(keys), "source_unique_wells": len(key_set),
        "source_duplicate_well_ids": len(duplicates), "feature_count": len(features),
        "official_metadata_wells": len(metadata),
        "metadata_wells_without_cp_profile": len(set(metadata.well_id) - key_set),
        "allowed_development_identities": int(allowed.object_id.nunique()),
        "allowed_development_wells": len(allowed),
        "available_development_identities": int(available.object_id.nunique()),
        "available_development_wells": len(available),
        "development_wells_missing_profile": int((~allowed.profile_present).sum()),
        "available_well_labels": available.well_type_label.value_counts().to_dict(),
        "complete_four_plate_conditions": len(role_conditions),
        "identities_with_complete_condition": int(roles.object_id.nunique()) if not roles.empty else 0,
        "complete_condition_count_per_identity_histogram": {str(k): int(v) for k, v in condition_hist.items()},
        "role_rows": len(roles), "role_seed": SEED,
        "roles_use_four_distinct_plates_within_one_experiment": True,
        "role_assignment_uses_measurements": False,
        "candidate_negative_controls": len(controls),
        "candidate_negative_controls_with_profile": int(controls.profile_present.sum()),
        "literal_dmso_metadata_wells": int(metadata.treatment.str.upper().eq("DMSO").sum()),
        "candidate_negative_control_treatment": "EMPTY_control",
        "numeric_columns_read": [], "protected_numeric_values_read": False,
        "numeric_preparation_started": False, "model_training_started": False,
        "next_requirement": "Parent approval of mixed-row-group decode-then-filter and independent preprocessing contract",
    }
    write_json(output / "preparation_audit.json", audit_data)
    return audit_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--dry-run", action="store_true", help="Metadata-only audit (also the default)")
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
