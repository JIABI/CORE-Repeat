import numpy as np
import pytest
from rdkit import Chem
from opal2.lkcp_biology_join import resolve_sample, encode_annotations, connectivity_groups


def source(sample,smiles='CCO',**kwargs):
    mol=Chem.MolFromSmiles(smiles)
    return dict(broad_id=sample,smiles=smiles,InChIKey=Chem.MolToInchiKey(mol),target='A|B',moa='inhibitor',**kwargs)


def test_exact_identity_formats_conflicts_and_missing():
    sample='BRD-K00000001-001-01-1';row=source(sample)
    row['InChIKey']=Chem.MolToInchi(Chem.MolFromSmiles('CCO'))
    result=resolve_sample(sample,[row],[])
    assert result['chemistry_available'] and not result['identity_conflict']
    with pytest.raises(ValueError,match='exact full sample'):
        resolve_sample('BRD-K00000001-002-01-1',[row],[])
    conflict=resolve_sample(sample,[row],[source(sample,'CC')])
    assert conflict['identity_conflict'] and not conflict['chemistry_available']
    assert not resolve_sample(sample,[],[])['chemistry_available']


def test_frozen_vocabulary_masks_truncation_without_erasing_raw_terms():
    result=resolve_sample('BRD-K00000001-001-01-1',[source('BRD-K00000001-001-01-1')],[])
    vector,mask,detail=encode_annotations(result,['A'],'target')
    np.testing.assert_array_equal(vector,[1.])
    assert not mask and detail['oov_terms']==['B'] and detail['full_term_count']==2
    assert encode_annotations(result,['A','B'],'target')[1]


def test_union_same_broad_and_identical_known_connectivity():
    records={
        's1':dict(broad_id='B1',chemistry_available=True,inchikey14='K1'),
        's2':dict(broad_id='B1',chemistry_available=False,inchikey14=None),
        's3':dict(broad_id='B2',chemistry_available=True,inchikey14='K1'),
        's4':dict(broad_id='B3',chemistry_available=False,inchikey14=None),
    }
    groups,summary=connectivity_groups(records)
    assert groups['B1']==groups['B2'] and groups['B3']!=groups['B1']
    assert summary['union_groups']==2
