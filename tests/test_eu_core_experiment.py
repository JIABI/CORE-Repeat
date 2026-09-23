import numpy as np
import pytest
from opal2.eu_core_experiment import select, partitions, policy_summary


def test_fixed_budget_ties_and_null_accounting():
    ids=np.array([f'i{x:03}' for x in range(183)])[::-1]
    chosen=select(ids,np.zeros(183),np.zeros(183),.2)
    assert chosen.sum()==22 and sorted(ids[chosen])[0]=='i000' and max(ids[chosen])=='i021'
    values=np.ones(183);values[np.flatnonzero(chosen)[:3]]=0.
    out=policy_summary(values,chosen)
    assert out['null_selected']==3 and out['extra_wells']==44 and out['fdp']==3/22


def test_split_refuses_missing_object_and_overlapping_group():
    ids=np.array(list('abcde'));groups=np.array(list('ABCDE'))
    roles=[('MODEL_FIT','TRAIN'),('MODEL_FIT','VALIDATION'),('REF_FIT',''),('DIST_CAL',''),('DEV_EVAL','')]
    plan=[dict(outer_fold='0',object_id=oid,connectivity=group,phase_role=role,model_fit_subrole=sub)
          for oid,group,(role,sub) in zip(ids,groups,roles)]
    assert partitions(ids,groups,plan,0)['REF_FIT'].tolist()==[2]
    with pytest.raises(ValueError):partitions(ids[:-1],groups[:-1],plan,0)
    groups[2]=groups[0];plan[2]['connectivity']=groups[0]
    with pytest.raises(ValueError):partitions(ids,groups,plan,0)


def test_authorized_exclusion_preserves_every_remaining_assignment():
    ids=np.array(list('abcdef'));groups=np.array(list('ABCDEF'))
    roles=[('MODEL_FIT','TRAIN'),('MODEL_FIT','VALIDATION'),('REF_FIT',''),('DIST_CAL',''),('DEV_EVAL',''),('REF_FIT','')]
    plan=[dict(outer_fold='0',object_id=oid,connectivity=group,phase_role=role,model_fit_subrole=sub)
          for oid,group,(role,sub) in zip(ids,groups,roles)]
    out=partitions(ids[:-1],groups[:-1],plan,0,excluded_ids=['f'])
    assert out['REF_FIT'].tolist()==[2] and out['DEV_EVAL'].tolist()==[4]
    with pytest.raises(ValueError):partitions(ids[:-1],groups[:-1],plan,0,excluded_ids=['z'])
