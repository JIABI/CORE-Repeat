"""Exact algebra and information limits of observed-role diagnostics."""
from dataclasses import replace
import json
import numpy as np
import pytest
from test_data import records
from opal2.diagnostics import (RoleAssignment,finite_two_way_ss,observed_role_diagnostics,
                               write_role_diagnostics,fixed_x_three_roles)
from opal2.evaluation import actual_utilities


def test_balanced_ss_identity_and_known_additive_components():
    row=np.array([-2.,0.,2.])
    col=np.array([-1.,1.])
    y=4+row[:,None]+col[None,:]
    result=finite_two_way_ss(y)
    assert result["sum_of_squares"] == {"compound_average":16.,"role_column":6.,"interaction_plus_residual":0.}
    assert result["total_sum_of_squares"] == 22.
    assert result["identity_absolute_error"] == 0.
    assert result["prediction_ceiling"] is None
    assert not result["interaction_and_measurement_noise_separable"]
    interaction=np.array([[1.,-1.],[-2.,2.],[1.,-1.]])
    changed=finite_two_way_ss(y+interaction)
    assert changed["sum_of_squares"]["interaction_plus_residual"] == 12.
    assert np.isclose(sum(changed["fraction_of_total_SS"].values()),1.)


def test_original_role_endpoint_preserved_and_table_complete():
    ds=records(n=12,w=4)
    summary,table,layout=observed_role_diagnostics(ds)
    old=actual_utilities(ds,np.arange(len(ds))).samples[0,:,-1]
    np.testing.assert_allclose(table["V_slot_3__gamma"],old,rtol=0,atol=1e-15)
    assert summary["complete_compounds"]==12
    assert len(layout)==12*3*2
    assert summary["physical_condition_layout"]["batch"]["compound_condition_complete_balanced"]
    assert summary["physical_condition_layout"]["batch"]["aligned_finite_condition_SS"] is not None
    assert not summary["original_registered_endpoint_changed"]
    assert summary["null_flip_count"]==int(table["null_label_flips"].sum())


def test_actual_batch_combination_incomplete_is_not_role_ss():
    ds=records(n=12,w=4)
    ds.groups[0,1,1]=19
    report,_,_=observed_role_diagnostics(ds)
    layout=report["physical_condition_layout"]["batch"]
    assert not layout["compound_condition_complete_balanced"]
    assert not layout["role_column_is_one_fixed_condition"]
    assert layout["aligned_finite_condition_SS"] is None
    assert layout["full_factorial_cells_required"]>layout["observed_compound_condition_cells"]
    assert not layout["identified_biological_or_technical_variance"]


def test_missing_results_are_excluded_not_relabelled_and_output_roundtrip(tmp_path):
    ds=records(n=12,w=4)
    ds.observed_mask[0,2]=False
    ds.Y[0,2]=np.nan
    report=write_role_diagnostics(ds,tmp_path/"observed")
    assert report["complete_compounds"]==11
    assert report["excluded_missing_compound_ids"]==[ds.ids[0]]
    saved=json.loads((tmp_path/"observed"/"diagnostic.json").read_text())
    assert saved["finite_compound_by_role_SS"]["prediction_ceiling"] is None
    assert (tmp_path/"observed"/"compound_role_values.tsv").is_file()
    with pytest.raises(FileExistsError):
        write_role_diagnostics(ds,tmp_path/"observed")


def test_invalid_role_assignments_and_constant_rank_handling():
    with pytest.raises(ValueError,match="exact"):
        RoleAssignment("bad",.2,(1,2),3)
    with pytest.raises(ValueError,match="distinct"):
        RoleAssignment("bad",0,(0,2),3)
    ds=records(n=4,w=4)
    ds.Y[:]=1.
    summary,table,_=observed_role_diagnostics(ds)
    assert not summary["null_flip_count"]
    assert all(item["spearman_rank_correlation"] is None for item in summary["rank_stability"])
    assert all(value is None for value in summary["finite_compound_by_role_SS"]["fraction_of_total_SS"].values())
    with pytest.raises(ValueError,match="holds X"):
        observed_role_diagnostics(ds,assignments=(fixed_x_three_roles()[0],RoleAssignment("different_X",1,(0,2),3)))
    with pytest.raises(ValueError,match="training transform"):
        observed_role_diagnostics(replace(ds,metadata={"train_scaler_applied":True}))
