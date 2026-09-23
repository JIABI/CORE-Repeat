"""Pure diagnostic behavior, not evidence about any real extreme object."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from opal2.statistical_anomaly_audit import audit_opened_data


def fixture():
    y = np.array([[[1., 4.], [3., 8.]], [[2., 3.], [4., 9.]],
                  [[1001., 2.], [101., 3.]]])
    dataset = SimpleNamespace(Y=y, ids=np.array(["train-a", "train-b", "focus"]),
        feature_names=np.array(["Nuclei_Intensity_DNA", "Cells_AreaShape_Area"]),
        observed_mask=np.ones((3, 2), bool), n_cells_mask=np.zeros((3, 2), bool),
        well_ids=np.array([["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]),
        metadata={"roles": ["X", "V"], "cell_counts_available": False})
    train = y[:2].reshape(-1, 2)
    baseline = SimpleNamespace(center=train.mean(0), scale=train.std(0))
    return dataset, baseline, {"train": [0, 1], "evaluation": [2]}


def test_readonly_json_and_original_vs_affine_units():
    data, base, split = fixture()
    original, original_base = copy.deepcopy(data), copy.deepcopy(base)
    report = audit_opened_data(data, base, split, focus_ids=("focus", "absent"))
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(data.Y, original.Y)
    np.testing.assert_array_equal(base.scale, original_base.scale)
    assert report["all_objects"]["exported_endpoint_Y"]["maximum"]["absolute_value"] == 1001
    assert report["partitions"]["evaluation"]["exported_endpoint_Y"]["threshold_counts"] == [
        {"absolute_threshold": 100, "coordinate_entries": 2, "physical_wells": 2, "objects": 1},
        {"absolute_threshold": 200, "coordinate_entries": 1, "physical_wells": 1, "objects": 1},
        {"absolute_threshold": 1000, "coordinate_entries": 1, "physical_wells": 1, "objects": 1}]
    feature = report["focus_objects"][0]["top_coordinates_by_exported_endpoint_Y"][0]
    assert feature["feature_name"] == "Nuclei_Intensity_DNA"
    assert feature["L_affine_Y_by_role"][0] == (1001 - base.center[0]) / base.scale[0]
    assert feature["scale_equals_train_sd"]
    assert report["focus_objects"][1]["present_in_supplied_export"] is False
    assert report["objects_excluded"] == []


def test_known_ids_supported_and_overlap_rejected():
    data, base, _ = fixture()
    output = audit_opened_data(data, base, {"train": ["train-a", "train-b"], "evaluation": ["focus"]})
    assert len(output["per_object"]) == 3
    with pytest.raises(ValueError, match="partition"):
        audit_opened_data(data, base, {"train": [0, 1], "evaluation": [1]})


def test_missing_well_not_counted_or_imputed():
    data, base, split = fixture()
    data.observed_mask[2, 0] = False
    data.Y[2, 0] = np.nan
    output = audit_opened_data(data, base, split, focus_ids=("focus",))
    json.dumps(output, allow_nan=False)
    assert output["partitions"]["evaluation"]["exported_endpoint_Y"]["maximum"]["absolute_value"] == 101
    assert np.isnan(data.Y[2, 0]).all()
    assert output["focus_objects"][0]["top_coordinates_by_exported_endpoint_Y"][0]["exported_Y_by_role"][0] is None
