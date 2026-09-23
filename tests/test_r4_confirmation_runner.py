"""Check final-stage prediction expansion without source-data access."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from opal2.r4_final_model import CORE_ARM, GAUSSIAN_ARM, DIRECT_ARM

spec = importlib.util.spec_from_file_location('r4_runner', Path(__file__).parents[1]/
    'scripts/run_r4_confirmation_20260921.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_full_population_alignment_preserves_missing_X_and_predeclared_scores(tmp_path):
    query = dict(ids=np.array(['a','b','c']), eligible=np.array([True,False,True]))
    for arm in (CORE_ARM,GAUSSIAN_ARM,DIRECT_ARM):
        np.savez(tmp_path/(arm+'.npz'), ids=np.array(['c','a']),
                 predicted=np.array([.3,.1]), p_null=np.array([.4,.2]))
    predictions = runner.predictors_from_saved(query,tmp_path)
    assert set(predictions) == {'CORE','CORE_LAMBDA0','GAUSSIAN',
        'GAUSSIAN_LAMBDA0','HISTGB_CAL','HISTGB_CAL_LAMBDA0'}
    for name, value in predictions.items():
        np.testing.assert_allclose(value['expected'],[.1,np.nan,.3],equal_nan=True)
        np.testing.assert_allclose(value['p_null'],[.2,np.nan,.4],equal_nan=True)
        assert value['lambda'] == (0. if name.endswith('LAMBDA0') else .2)


def test_predictor_cannot_change_the_common_eligible_population(tmp_path):
    query = dict(ids=np.array(['a','b','c']), eligible=np.array([True,False,True]))
    np.savez(tmp_path/(CORE_ARM+'.npz'),ids=np.array(['a','b']),
             predicted=np.array([.1,.2]),p_null=np.array([.2,.3]))
    with pytest.raises(ValueError,match='eligibility population'):
        runner.predictors_from_saved(query,tmp_path)
