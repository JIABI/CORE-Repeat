import numpy as np
import pytest

from opal2.r4_evaluation import (campaign_resampling, evaluate_campaign,
                                freeze_selections, load_selections)


def predictors(n):
    return {
        'CORE':dict(expected=np.arange(n)/n, p_null=np.full(n,.2)),
        'HISTGB_CAL':dict(expected=np.arange(n)[::-1]/n, p_null=np.full(n,.3)),
    }


def test_freeze_preserves_N_missing_X_and_no_outcome_input(tmp_path):
    n=17
    ids=np.array([f'id{i:03}' for i in range(n)])
    eligible=np.ones(n,bool);eligible[-1]=False
    pred=predictors(n)
    for value in pred.values():
        value['expected'][-1]=np.nan;value['p_null'][-1]=np.nan
    policies,record=freeze_selections(ids,eligible,pred,tmp_path)
    assert record['budget']['population_n']==17
    assert record['budget']['activations']==2
    assert not record['outcomes_used']
    assert set(ids[policies['CORE']['selected']])=={'id014','id015'}
    with pytest.raises(FileExistsError):
        freeze_selections(ids,eligible,pred,tmp_path)
    _,_,saved=load_selections(tmp_path)
    np.testing.assert_array_equal(saved['CORE']['selected'],policies['CORE']['selected'])


def test_evaluate_missing_endpoints_retains_bounds_and_lists(tmp_path):
    n=24;ids=np.array([f'id{i:03}' for i in range(n)])
    selection=tmp_path/'selection'
    policies,_=freeze_selections(ids,np.ones(n,bool),predictors(n),selection)
    gamma=np.linspace(-.1,.3,n);gamma[-1]=np.nan
    result=evaluate_campaign(selection,ids,gamma,ids,np.array([str(i%3) for i in range(n)]),
                             tmp_path/'outcome',repeats=20)
    assert result['n']==24 and result['observed_outcome_n']==23
    assert result['policies']['CORE']['selected_n']==3
    assert result['policies']['CORE']['unknown_selected_n']==1
    assert result['primary']['lower']<result['primary']['upper']
    assert result['policies']['CORE']['total_value'] is None
    np.testing.assert_array_equal(load_selections(selection)[2]['CORE']['selected'],policies['CORE']['selected'])


def test_campaign_reselection_matches_when_primary_predictors_identical(tmp_path):
    n=32;ids=np.array([f'id{i:03}' for i in range(n)])
    pred=predictors(n);pred['HISTGB_CAL']=pred['CORE'].copy()
    policies,_=freeze_selections(ids,np.ones(n,bool),pred,tmp_path)
    gamma=np.linspace(-.2,.4,n)
    result=campaign_resampling(ids,gamma,np.ones(n,bool),policies,
                              np.array([str(i%4) for i in range(n)]),repeats=20)
    for mode in result['intervals'].values():
        assert mode['difference_low']['lower']==0
        assert mode['difference_high']['upper']==0


def test_failed_predictor_is_not_allowed_to_delete_eligible_query(tmp_path):
    pred=predictors(16);pred['CORE']['p_null'][0]=np.nan
    with pytest.raises(ValueError,match='eligible prediction failed'):
        freeze_selections(np.array([str(i) for i in range(16)]),np.ones(16,bool),pred,tmp_path)
