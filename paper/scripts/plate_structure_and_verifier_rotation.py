#!/usr/bin/env python3
"""Reproduce the dataset-level role-rotation table from packaged four-well data.

This is the A1 calculation from the original exploratory diagnostic, scoped to
Supplementary Table 2. The cosine, cost, rank/tie and aggregation definitions
are unchanged. No fitting, ranking of policy candidates or selection occurs.
Role rotations reuse the same wells and are not independent remeasurements.

Usage:
  python paper/scripts/plate_structure_and_verifier_rotation.py --check
  python paper/scripts/plate_structure_and_verifier_rotation.py --output out.csv
Set OPAL2_DATA_ROOT or pass --data-root to the extracted zenodo_data directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

COST = 0.02
DEV = {
    'EU': ('reports/eu_core_development_20260917_v1/prepared_data_cc904/data.npz',904),
    'JUMP': ('data/source5_primary_fullcontrols/measurements.npz',639),
    'LINCS': ('data/lincs_pilot1_biology_20260915/data.npz',1188),
    'RxRx3': ('data/rxrx3_r2_20260918/prepared_r2/data.npz',10410),
}


def cos(a,b):
    return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum(-1))


def gam(x,z1,z2,v):
    return 0.5*(cos((x+z1+z2)/3.0,v)-cos(x,v))-COST


def rank(a):
    a=np.asarray(a,float)
    result=np.empty(len(a)); order=np.argsort(a,kind='mergesort')
    result[order]=np.arange(len(a))
    sorted_values=a[order]; i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sorted_values[j+1]==sorted_values[i]:j+=1
        if j>i:result[order[i:j+1]]=(i+j)/2.0
        i=j+1
    return result


def spearman(a,b):
    a,b=rank(a),rank(b)
    if a.std()==0 or b.std()==0:return np.nan
    return np.corrcoef(a,b)[0,1]


def calculate(data_root):
    with (data_root/'source_data/measurement_fig2_object_predictions.csv').open(newline='') as f:
        saved={}
        for row in csv.DictReader(f):
            saved.setdefault(row['dataset'],{})[row['candidate_id']]=float(row['actual_gamma'])
    result=[];max_differences={}
    for ds,(relative,expected_n) in DEV.items():
        with np.load(data_root/'research'/relative,allow_pickle=False) as z:
            y=z['Y'];ids=z['ids'].astype(str)
            if ds=='EU':plate=np.array([w.split('|')[2] for w in z['well_ids'][:,0]])
            elif ds=='JUMP':plate=np.array([w.split('::')[1] for w in z['well_ids'][:,0]])
            elif ds=='LINCS':plate=np.array([w.split('::')[0] for w in z['well_ids'][:,0]])
            else:plate=z['plates'][:,0]
        assert y.shape[0]==len(ids)==expected_n and y.shape[1]==4
        assert np.isfinite(y).all()
        x,z1,z2,v=y[:,0],y[:,1],y[:,2],y[:,3]
        g0=gam(x,z1,z2,v);g1=gam(x,z2,v,z1);g2=gam(x,z1,v,z2)
        known=np.array([saved[ds][i] for i in ids])
        difference=float(np.abs(g0-known).max());assert difference<1e-9
        max_differences[ds]=difference
        cxv=cos(x,v);cz12=cos(z1,z2);cxz1=cos(x,z1);cxz2=cos(x,z2);cz1v=cos(z1,v);cz2v=cos(z2,v)
        pair=(cxz1+cxz2+cxv+cz12+cz1v+cz2v)/6
        normx=np.linalg.norm(x,axis=1)
        result.append(dict(dataset=ds,n=len(ids),D=y.shape[2],n_x_plates=len(np.unique(plate)),
            mean_cosXV=cxv.mean(),median_cosXV=np.median(cxv),frac_cosXV_gt0_3=(cxv>0.3).mean(),
            mean_pairwise_cos=pair.mean(),mean_gamma=g0.mean(),sd_gamma=g0.std(),null_frac=(g0<=0).mean(),
            spear_normX_gamma=spearman(normx,g0),spear_cosXV_gamma=spearman(cxv,g0),
            mean_gamma_V_is_Z1=g1.mean(),mean_gamma_V_is_Z2=g2.mean(),
            spear_g0_g1=spearman(g0,g1),spear_g0_g2=spearman(g0,g2),
            null_agree_g0_g1=((g0<=0)==(g1<=0)).mean(),null_agree_g0_g2=((g0<=0)==(g2<=0)).mean()))
    return result,max_differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default=Path(__file__).resolve().parents[3]/'zenodo_data'
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('OPAL2_DATA_ROOT',default)))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Compare all cells with the released original table.')
    args=parser.parse_args()
    root=args.data_root.expanduser().resolve()
    rows,diffs=calculate(root)
    check_max=0.0
    if args.check:
        with (root/'supplemental_diagnostics/ps_dataset_reproducibility.csv').open(newline='') as f:
            original=list(csv.DictReader(f))
        assert len(original)==len(rows)==4
        for generated,stored in zip(rows,original):
            assert generated['dataset']==stored['dataset'] and set(generated)==set(stored)
            for key in generated:
                if key=='dataset':continue
                delta=abs(float(generated[key])-float(stored[key]));check_max=max(check_max,delta)
                np.testing.assert_allclose(float(generated[key]),float(stored[key]),rtol=0,atol=1e-12,err_msg=generated['dataset']+'/'+key)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(resources=4,objects=sum(r['n'] for r in rows),
        gamma_max_abs_difference=diffs,checked_against_saved=args.check,
        table_max_abs_difference=check_max if args.check else None,
        fitting=False,selection=False)))


if __name__=='__main__':main()
