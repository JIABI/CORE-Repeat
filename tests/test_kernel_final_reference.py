"""Reference-role engineering tests; not assay evidence."""
import numpy as np
import torch
from opal2.kernel_final_reference import reference_allocation, fit_reference_bank
from opal2.conditional_response_kernel import LocalResponseBank


def fixture():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(18,7))
    chem = np.asarray([[(i>>j)&1 for j in range(8)]+[1] for i in range(1,19)],float)
    ids = np.asarray([f'test_{i:02d}' for i in range(18)])
    return x,chem,np.ones(18,bool),ids,dict(fingerprint_indices=list(range(8)),validity_index=8,kind='synthetic')


def test_matched_reference_roles_have_same_supervised_labels_and_random_rule():
    x,c,m,ids,meta=fixture()
    scope=reference_allocation(ids,np.arange(15),c,m,meta,81,count=4)
    assert len(scope['commonbranchfit'])==11
    assert set(scope['anchor_ids_by_mode']['O']).issubset(scope['commonbranchfit_ids'])
    assert set(scope['reference_ids']).isdisjoint(scope['commonbranchfit_ids'])
    assert set(scope['reference_ids'])|set(scope['commonbranchfit_ids'])==set(ids[:15])
    assert scope==reference_allocation(ids,np.arange(15),c,m,meta,81,count=4)
    # Row-order changes leave identity assignment unchanged.
    other=reference_allocation(ids[::-1],np.arange(3,18),c[::-1],m[::-1],meta,81,count=4)
    assert other['anchor_ids_by_mode']==scope['anchor_ids_by_mode']


def test_bank_scalers_use_supervised_only_and_roundtrip(tmp_path):
    x,c,m,ids,meta=fixture()
    scope=reference_allocation(ids,np.arange(15),c,m,meta,81,count=4)
    args=(scope['originalfit_ids'],scope['commonbranchfit_ids'])
    for mode in ('O','D'):
        bank=fit_reference_bank(x,c,m,ids,*args,scope['anchor_ids_by_mode'][mode],meta)
        rows=scope['commonbranchfit']
        d=bank(torch.tensor(x[rows]),torch.tensor(c[rows]),torch.tensor(m[rows]))
        assert int(np.sum(np.max(d['tanimoto'].numpy(),axis=1)==1))==(4 if mode=='O' else 0)
        for bmode in ('generic','structured'):
            for values in bank.basis_blocks(d,bmode).values():
                torch.testing.assert_close(values.square().mean(),torch.tensor(1.,dtype=torch.float64))
        changed_x=x.copy();changed_x[15:]=np.nan
        changed_c=c.copy();changed_c[15:]=np.nan
        other=fit_reference_bank(changed_x,changed_c,m,ids,*args,scope['anchor_ids_by_mode'][mode],meta)
        for key,v in bank.state_dict().items():assert torch.equal(v,other.state_dict()[key])
        path=tmp_path/(mode+'.pt');bank.save(path);restored=LocalResponseBank.load(path)
        for key,v in bank.state_dict().items():assert torch.equal(v,restored.state_dict()[key])
