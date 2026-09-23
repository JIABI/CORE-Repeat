import numpy as np
import pandas as pd
import pytest

from opal2.lkcp_module_data import select_plates, parse_profile, aligned_layout, ROLES


def test_metadata_role_selection_excludes_fifth():
    rows=[dict(Plate_Map_Name='L_A549_24H',Assay_Plate_Barcode=f'P{i}') for i in (4,2,0,3,1)]
    selected=select_plates(rows,('A549_24H',))
    assert [r['plate'] for r in selected]==['P0','P1','P2','P3']
    assert [r['role'] for r in selected]==list(ROLES)
    assert {r['excluded_fifth_plate'] for r in selected}=={'P4'}
    with pytest.raises(ValueError,match='five distinct'):
        select_plates(rows[:-1],('A549_24H',))


def frame_for(plate):
    return pd.DataFrame(dict(Metadata_Plate=[plate]*384,
        Metadata_Well=[f'W{i:03}' for i in range(384)],
        Metadata_broad_sample=['DMSO']*24+['BRD-K00000001-001-01-1']*360,
        Metadata_mmoles_per_liter=[0.]*24+[10.]*360,
        Metadata_cell_line=['A549']*384,Metadata_time_point=['24H']*384,
        Cells_AreaShape_Area=np.linspace(-11,11,384)))


def test_per_well_validation_and_four_role_alignment(tmp_path):
    inputs=[]
    for i,role in enumerate(ROLES):
        record=dict(plate=f'P{i}',condition='A549_24H',role=role)
        frame=frame_for(f'P{i}')
        if i==1:
            frame=frame.iloc[::-1].reset_index(drop=True)
        path=tmp_path/f'p{i}.csv.gz';frame.to_csv(path,index=False,compression='gzip')
        parsed,mask,values,info=parse_profile(path,record,['Cells_AreaShape_Area'])
        assert info['rows']==384 and info['treatment_wells']==360
        assert values.max()==10 and values.min()==-10
        inputs.append((parsed,mask,values,record))
    values,rows=aligned_layout(*[list(x) for x in zip(*inputs)])
    assert values.shape==(360,4,1)
    np.testing.assert_array_equal(values[:,0],values[:,1])
    inputs[2][0].loc[24,'Metadata_broad_sample']='BRD-K99999999'
    with pytest.raises(ValueError,match='Compound differs'):
        aligned_layout(*[list(x) for x in zip(*inputs)])


def test_reject_consensus_missing_feature_and_bad_numeric(tmp_path):
    path=tmp_path/'plate.csv.gz';frame=frame_for('P0')
    record=dict(plate='P0',condition='A549_24H',role='X')
    frame.iloc[:330].to_csv(path,index=False,compression='gzip')
    with pytest.raises(ValueError,match='every physical well'):
        parse_profile(path,record,['Cells_AreaShape_Area'])
    frame.to_csv(path,index=False,compression='gzip')
    with pytest.raises(ValueError,match='Missing declared columns'):
        parse_profile(path,record,['Cells_Missing'])
    frame.loc[24,'Cells_AreaShape_Area']=np.nan
    frame.to_csv(path,index=False,compression='gzip')
    with pytest.raises(ValueError,match='5% nonfinite'):
        parse_profile(path,record,['Cells_AreaShape_Area'])
