import numpy as np
import pytest

from opal2.conditional_joint_error_experiment import observable_forward
from opal2.variance_headroom_experiment import (gamma_fast, global_calibration_choice,
    calibrate_rank_map, scalar_grid)


def test_fast_forward_is_original_endpoint():
    raw=np.random.default_rng(37).normal(size=(19,7,9))
    np.testing.assert_allclose(gamma_fast(raw),observable_forward(raw)[0],atol=2e-15,rtol=2e-14)


def test_fast_forward_rejects_squared_diagonal_underflow():
    raw=np.ones((1,9));raw[:,3]=-400.
    with pytest.raises(ValueError,match='Invalid Cholesky diagonal'):
        gamma_fast(raw)


def test_global_choice_retains_zero_without_calibration_gain():
    grid=scalar_grid();scores=np.broadcast_to(np.abs(grid),(20,len(grid))).copy()
    eta,record=global_calibration_choice(scores,np.arange(20),grid)
    assert eta==0 and record['reason']=='no_one_SE_gain'


def test_global_choice_uses_only_group_calibration_scores():
    grid=scalar_grid();scores=np.broadcast_to((grid-grid[6])**2,(20,len(grid))).copy()
    eta,record=global_calibration_choice(scores,np.repeat(np.arange(10),2),grid)
    assert eta==grid[6] and record['index']==6


def test_rank_mapping_handles_ties_and_uses_calibration_distribution():
    cal=np.array([1.,1.,2.,3.,4.,5.]);target=np.linspace(-.5,.5,6)
    query=np.array([0.,1.,2.5,6.])
    values,info=calibrate_rank_map(cal,query,target)
    changed,_=calibrate_rank_map(cal,np.r_[query,1000.],target)
    np.testing.assert_array_equal(values,changed[:len(query)])
    assert np.all(np.diff(values)>=0) and np.max(np.abs(values))<=np.log(4.)
    assert info['unique_scores']==5


def test_rank_map_can_represent_constant_calibration_effect():
    result,_=calibrate_rank_map(np.arange(8.),np.arange(12.),np.full(8,.2))
    np.testing.assert_allclose(result,.2)


def test_rank_map_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):calibrate_rank_map([0.,1.,np.nan],[0.],[0.,.1,.2])
