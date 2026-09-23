"""Synthetic verification of RxRx3 identity-first export, with no source profiles."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


path = Path(__file__).resolve().parents[1] / "scripts/export_rxrx3_r2_20260918.py"
spec = importlib.util.spec_from_file_location("export_rxrx3_r2", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def role_rows():
    return pd.DataFrame([
        {"condition_id": condition, "role": role, "well_id": f"{condition}_{role}",
         "object_id": f"object_{condition}", "group_id": f"group_{condition}", "batch": "compound-001",
         "dose_record": "0.01", "cell_record": "HUVEC", "identity_role": module.ALLOWED_ROLE,
         "physical_plate_id": f"compound-001_{index}"}
        for condition in ("cond_a", "cond_b") for index, role in enumerate(module.ROLES)
    ])


def test_excluded_value_changes_cannot_change_approved_export(tmp_path):
    wells = ["excluded_1", "cond_b_V", "cond_a_X", "excluded_2", "control", "cond_a_Z1"]
    requested = ["cond_a_X", "cond_a_Z1", "cond_b_V", "control"]
    base = np.array([[12., 20.], [1., 2.], [3., np.nan], [-5., 0.], [8., 9.], [np.inf, 4.]])
    outputs = []
    for index in range(2):
        data = base.copy()
        if index:
            data[[0, 3]] = [[np.nan, np.inf], [-1e200, -np.inf]]
        source = tmp_path / f"source_{index}.parquet"
        pq.write_table(pa.table({"well_id": wells, "f1": data[:, 0], "f2": data[:, 1], "__index_level_0__": list(range(6))}), source)
        outputs.append(module.extract_approved_rows(source, requested, ["f1", "f2"], batch_size=index + 1)[0])
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], base[[2, 5, 1, 4]])
    assert module.approved_quality_counts(outputs[0])["nonfinite_values"] == 2


def test_fixed_role_order_preserves_mapping_under_input_permutation():
    original = role_rows()
    shuffled = original.sample(frac=1, random_state=41)
    canonical = module.fixed_role_order(shuffled)
    pd.testing.assert_frame_equal(canonical, original)
    assert canonical.role.tolist() == list(module.ROLES) * 2


def test_control_whitelist_rejects_crispr_other_plates_and_missing_profiles():
    roles = role_rows()
    records = [{"well_id": f"control_{i}", "treatment": "EMPTY_control", "perturbation_type": "COMPOUND",
                "batch": "compound-001", "physical_plate_id": f"compound-001_{i}", "profile_present": "True"}
               for i in range(4)]
    records.extend([
        dict(records[0], well_id="crispr", perturbation_type="CRISPR"),
        dict(records[0], well_id="crispr_control", treatment="CRISPR_control"),
        dict(records[0], well_id="wrong_plate", physical_plate_id="compound-001_99"),
        dict(records[0], well_id="no_profile", profile_present="False"),
    ])
    controls = module.authorized_controls(pd.DataFrame(records), roles)
    assert controls.well_id.tolist() == [f"control_{i}" for i in range(4)]
    assert controls.measurement_authorized.eq("True").all()


@pytest.mark.parametrize("duplicate", [True, False])
def test_missing_or_duplicate_approved_profiles_fail(tmp_path, duplicate):
    source = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"well_id": ["a", "a" if duplicate else "excluded"], "f": [1., 2.]}), source)
    with pytest.raises(ValueError, match="Duplicate|Missing"):
        module.extract_approved_rows(source, ["a"] if duplicate else ["a", "b"], ["f"], batch_size=1)


def test_protected_identity_cannot_enter_fixed_roles():
    roles = role_rows()
    roles.loc[0, "identity_role"] = "EU_CONFIRMATION_RESERVED"
    with pytest.raises(ValueError, match="allowlist"):
        module.fixed_role_order(roles)
