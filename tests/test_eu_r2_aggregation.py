"""Regression test for signed paired false-activation comparisons."""
import importlib.util
from pathlib import Path

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'scripts/run_r2_core_comparison_20260917.py'
spec=importlib.util.spec_from_file_location('r2_aggregate_test',SOURCE)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_false_activation_pair_is_signed_numeric_not_xor():
    actual=np.array([-.1,-.2,.3,-.4])
    a=np.array([True,False,True,False])
    b=np.array([False,True,True,True])
    aa=runner.false_activation_outcomes(actual,a)
    bb=runner.false_activation_outcomes(actual,b)
    np.testing.assert_array_equal(aa-bb,[1.,-1.,0.,-1.])
    assert aa.dtype==np.float64
    result=runner.bootstrap_difference(aa,bb,np.array(['a','a','b','b']))
    assert result['difference']==-.25
