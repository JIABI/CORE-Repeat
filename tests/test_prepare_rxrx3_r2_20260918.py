"""Test metadata role safety without reading any real measurement profile."""

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa


path = Path(__file__).resolve().parents[1] / "scripts/prepare_rxrx3_r2_20260918.py"
spec = importlib.util.spec_from_file_location("prepare_rxrx3_r2", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_index_is_not_a_measurement_feature():
    schema = pa.schema([("well_id", pa.string()), ("AreaShape_Area", pa.float64()), ("__index_level_0__", pa.int64())])
    assert module.feature_names(schema) == ["AreaShape_Area"]


def test_roles_require_four_plates_in_same_experiment():
    rows = []
    for condition, batches in [("complete", [("b1", 4)]), ("split", [("b1", 2), ("b2", 2)]), ("sameplate", [("b1", 4)])]:
        for batch, count in batches:
            for i in range(count):
                rows.append({"condition_id": condition, "batch": batch, "physical_plate_id": f"{batch}_{0 if condition == 'sameplate' else i}", "well_id": f"{condition}_{batch}_{i}", "object_id": "o1", "group_id": "g1", "dose_record": "0.01", "identity_role": module.ALLOWED_ROLE})
    roles = module.role_assignments(pd.DataFrame(rows))
    assert roles.condition_id.unique().tolist() == ["complete"]
    assert roles.physical_plate_id.nunique() == 4
    assert roles.role.tolist() == list(module.ROLES)
    pd.testing.assert_frame_equal(roles, module.role_assignments(pd.DataFrame(rows).sample(frac=1, random_state=4)))
