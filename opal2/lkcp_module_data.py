"""Bounded official LKCP per-well adapter for declared module-development roles.

Does not fit an OPAL model, calculate utility, infer dose units, or read a fifth
plate. The remote MD5 names are DVC's actual storage addressing, not new audit
hash requirements. Existing Pilot1 data are consulted only for feature names.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REVISION = '061870127481dcd73c29df85ebcfddeac2ed0828'
RAW = f'https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/{REVISION}/'
REMOTE = 'https://cellpainting-gallery.s3.amazonaws.com/cpg0004-lincs/broad/workspace/software/lincs-cell-painting_DVC/'
CONDITIONS = ('A549_24H', 'A549_48H')
ROLES = ('X', 'Z1', 'Z2', 'V')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def select_plates(rows, conditions=CONDITIONS):
    maps=defaultdict(list)
    for row in rows:
        name=row['Plate_Map_Name']
        if any(name.endswith('_'+condition) for condition in conditions):
            maps[name].append(row['Assay_Plate_Barcode'])
    result=[]
    for name, plates in sorted(maps.items()):
        plates=sorted(plates)
        if len(plates)!=5 or len(set(plates))!=5:
            raise ValueError('Each selected layout must have five distinct designed plates')
        for role,plate in zip(ROLES,plates[:4]):
            result.append(dict(plate=plate,layout=name,role=role,
                condition=next(c for c in conditions if name.endswith('_'+c)),
                excluded_fifth_plate=plates[4]))
    if not result:
        raise ValueError('No selected layouts')
    return result


def parse_profile(path, record, features):
    frame=pd.read_csv(path, compression='gzip', low_memory=False)
    required=['Metadata_Plate','Metadata_Well','Metadata_broad_sample',
        'Metadata_mmoles_per_liter','Metadata_cell_line','Metadata_time_point']
    missing=set(required+list(features))-set(frame.columns)
    if missing:
        raise ValueError('Missing declared columns: '+','.join(sorted(missing)))
    if len(frame)!=384 or frame['Metadata_Well'].duplicated().any():
        raise ValueError('Expected one unique record for every physical well')
    if not frame['Metadata_Plate'].eq(record['plate']).all():
        raise ValueError('Plate identity mismatch')
    cell,hours=record['condition'].split('_')
    if not frame['Metadata_cell_line'].eq(cell).all() or not frame['Metadata_time_point'].eq(hours).all():
        raise ValueError('Recorded condition mismatch')
    treatment=frame['Metadata_broad_sample'].fillna('').str.startswith('BRD-')
    control=frame['Metadata_broad_sample'].eq('DMSO') | frame['Metadata_broad_sample'].isna()
    if not treatment.any() or not control.any() or not (treatment|control).all():
        raise ValueError('Unexpected treatment/control identity encoding')
    values=frame[list(features)].to_numpy(dtype=float)
    finite=np.isfinite(values)
    treatment_missing=(~finite[treatment]).mean(axis=1)
    if np.any(treatment_missing>.05) or np.any(~finite[treatment].any(axis=1)):
        raise ValueError('Treatment well exceeds declared 5% nonfinite threshold; no rows removed')
    filled=np.where(finite,values,0.)
    clipped=np.clip(filled,-10.,10.)
    if np.any(np.linalg.norm(clipped[treatment],axis=1)==0):
        raise ValueError('Zero treatment norm after fixed fill/clip; no rows removed')
    info=dict(**record,rows=len(frame),treatment_wells=int(treatment.sum()),dmso_wells=int(control.sum()),
        measurement_columns=sum(c.startswith(('Cells_','Cytoplasm_','Nuclei_')) for c in frame),
        pilot_features_present=len(features),nonfinite_treatment_coordinates=int((~finite[treatment]).sum()),
        clipped_treatment_coordinates=int((np.abs(filled[treatment])>10).sum()),
        control_all_finite=bool(finite[control].all()),
        recorded_concentrations=sorted(frame.loc[treatment,'Metadata_mmoles_per_liter'].unique().tolist()),
        target_known_rows=int(frame.loc[treatment,'Metadata_target'].notna().sum()) if 'Metadata_target' in frame else 0,
        moa_known_rows=int(frame.loc[treatment,'Metadata_moa'].notna().sum()) if 'Metadata_moa' in frame else 0)
    return frame,treatment.to_numpy(),clipped,info


def aligned_layout(frames, masks, arrays, records):
    """Match the same physical position/sample/concentration across four plates."""
    if [r['role'] for r in records]!=list(ROLES):
        raise ValueError('Four ordered physical roles required')
    first=frames[0]
    selected=np.flatnonzero(masks[0])
    columns=['Metadata_Well','Metadata_broad_sample','Metadata_mmoles_per_liter']
    reference=first.iloc[selected][columns].reset_index(drop=True)
    output=[]
    for frame,mask,values in zip(frames,masks,arrays):
        indexed=frame.set_index('Metadata_Well',drop=False)
        aligned=indexed.loc[reference['Metadata_Well']]
        if not aligned['Metadata_broad_sample'].reset_index(drop=True).equals(reference['Metadata_broad_sample']):
            raise ValueError('Compound differs across physical roles')
        if not np.allclose(aligned['Metadata_mmoles_per_liter'],reference['Metadata_mmoles_per_liter'],rtol=1e-7,atol=1e-10):
            raise ValueError('Recorded concentration differs across physical roles')
        indices=frame.index.get_indexer(aligned.index) if frame.index.name=='Metadata_Well' else np.array([
            np.flatnonzero(frame['Metadata_Well'].eq(w))[0] for w in reference['Metadata_Well']])
        if not np.all(mask[indices]):
            raise ValueError('Treatment/control role mismatch')
        output.append(values[indices])
    return np.stack(output,axis=1),first.iloc[selected].reset_index(drop=True)


def acquire(output):
    output=Path(output).resolve()
    if not (output/'MODULE_DEV_DESIGN.md').is_file():
        raise ValueError('Record the developmental design before reading profiles')
    resource=output/'resources';resource.mkdir(exist_ok=True)
    meta=PROJECT/'runs/lincs_core_freeze_20260916_v1/cohort_metadata/public_metadata'
    selected=select_plates(list(csv.DictReader((meta/'barcode_platemap.csv').open())))
    if len(selected)!=24:
        raise ValueError('Declared two-condition24-plate scope changed')
    dvc_text=urllib.request.urlopen(RAW+'profiles/2017_12_05_Batch2.dvc',timeout=30).read().decode()
    address=re.search(r'md5:\s*([a-f0-9]{32}\.dir)',dvc_text).group(1)
    directory_url=REMOTE+address[:2]+'/'+address[2:]
    directory=json.load(urllib.request.urlopen(directory_url,timeout=30))
    bypath={d['relpath']:d for d in directory}
    for record in selected:
        name=record['plate']+'/'+record['plate']+'_normalized_dmso.csv.gz'
        d=bypath[name];key=d['md5']
        record.update(profile_relative_path=name,url=REMOTE+key[:2]+'/'+key[2:])
        with urllib.request.urlopen(urllib.request.Request(record['url'],method='HEAD'),timeout=30) as stream:
            record['compressed_bytes']=int(stream.headers['Content-Length'])
    total=sum(r['compressed_bytes'] for r in selected)
    if total>80_000_000 or any(r['compressed_bytes']>5_000_000 for r in selected):
        raise ValueError('Bounded61MB-profile scope grew; ask coordinator before further acquisition')
    write_json(output/'download_manifest.json',dict(recorded_before_profile_download=datetime.now(timezone.utc).isoformat(),
        source_revision=REVISION,directory_manifest_url=directory_url,total_compressed_bytes=total,profiles=selected,
        data_level='4a per-plate DMSO mad_robustize; no feature selection/spherizing/consensus',
        sample_roles=list(ROLES),formal_evaluation=False,all_examined_objects_developmental=True))
    def download(record):
        path=resource/(record['plate']+'_normalized_dmso.csv.gz')
        if not path.exists():
            with urllib.request.urlopen(record['url'],timeout=45) as stream:
                payload=stream.read(record['compressed_bytes']+1)
            if len(payload)!=record['compressed_bytes']:
                raise ValueError('Unexpected download size')
            path.write_bytes(payload)
        if path.stat().st_size!=record['compressed_bytes']:
            raise ValueError('Existing resource size mismatch')
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(download,selected))
    feature_source=PROJECT/'runs/lincs_state_biology_20260916_v1/run_manifest.json'
    features=json.loads(feature_source.read_text())['feature_names']
    bylayout=defaultdict(list);plate_info=[]
    for record,path in zip(selected,paths):
        frame,mask,values,info=parse_profile(path,record,features)
        bylayout[record['layout']].append((frame,mask,values,record))
        plate_info.append(info)
    datasets={};all_units=[]
    for condition in CONDITIONS:
        ys=[];units=[]
        for layout,entries in sorted(bylayout.items()):
            if not layout.endswith('_'+condition):
                continue
            entries.sort(key=lambda x: ROLES.index(x[3]['role']))
            values,rows=aligned_layout(*[list(x) for x in zip(*entries)])
            ys.append(values)
            for _,row in rows.iterrows():
                sample=str(row['Metadata_broad_sample']);compound=sample[:13]
                unit_id=layout+'::'+str(row['Metadata_Well'])+'::'+sample
                item=dict(id=unit_id,compound_id=compound,broad_sample=sample,condition=condition,
                    layout=layout,well=str(row['Metadata_Well']),recorded_concentration=float(row['Metadata_mmoles_per_liter']),
                    concentration_field='Metadata_mmoles_per_liter',assay_dose_uM=None,
                    dose_units_verified=False,target=None if pd.isna(row.get('Metadata_target')) else str(row['Metadata_target']),
                    moa=None if pd.isna(row.get('Metadata_moa')) else str(row['Metadata_moa']),
                    roles={r['role']:dict(plate=r['plate'],well=str(row['Metadata_Well'])) for *_,r in entries})
                units.append(item)
        y=np.concatenate(ys)
        assert len(y)==len(units)
        np.savez_compressed(output/(condition+'_module_dev_measurements.npz'),Y=y,
            ids=np.array([u['id'] for u in units]),compound_ids=np.array([u['compound_id'] for u in units]),
            feature_names=np.array(features),role_names=np.array(ROLES),
            recorded_concentrations=np.array([u['recorded_concentration'] for u in units]),
            well_ids=np.array([[u['roles'][r]['plate']+'::'+u['well'] for r in ROLES] for u in units]))
        datasets[condition]=dict(shape=list(y.shape),unique_compound_ids=len({u['compound_id'] for u in units}),
            role='MODULE_DEV_FIT' if condition.endswith('24H') else 'MODULE_DEV_HELD_CONDITION',
            duplicate_compound_episodes=len(units)-len({u['compound_id'] for u in units}),
            target_known_episodes=sum(u['target'] is not None for u in units),moa_known_episodes=sum(u['moa'] is not None for u in units))
        all_units.extend(units)
    paired_design=[]
    for condition in CONDITIONS:
        paired_design.append({(u['layout'].split('_')[0],u['well'],u['broad_sample'],u['recorded_concentration'])
            for u in all_units if u['condition']==condition})
    if paired_design[0]!=paired_design[1]:
        raise ValueError('Source/held condition sample-position-recorded-concentration design differs')
    write_json(output/'measurement_metadata.json',dict(units=all_units,conditions=datasets,feature_names=features,
        source_revision=REVISION,preprocessing=dict(source='official per-plate DMSO level4a',
            feature_selection='unchanged existing Pilot1 feature names, all present; no held-condition selection',
            feature_source=str(feature_source),nonfinite_fill=0.,clip=[-10.,10.],nonfinite_fraction_stop=.05),
        actual_dose_verified=False,formal_evaluation=False,consensus_used=False,fifth_plate_read=False,
        chemical_group_assignment='not inferred from compound IDs; molecular-structure grouping remains required',
        endpoint_calculated=False))
    summary=dict(conditions=datasets,profiles=plate_info,total_compressed_bytes=total,
        all_pilot1694_features_present=True,individual_wells_verified=True,
        source_held_metadata_positions_exactly_matched=True,
        assay_dose_units_verified=False,chemical_identity_grouping_pending=True,
        new_training=False,formal_evaluation=False,fifth_plate_read=False,
        source_directory_profile_plate_count=len({x['relpath'].split('/')[0] for x in directory}),
        declared135_missing_profile_plates=sorted({r['Assay_Plate_Barcode'] for r in csv.DictReader((meta/'barcode_platemap.csv').open())}-{x['relpath'].split('/')[0] for x in directory}))
    write_json(output/'compatibility_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='profiles'},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args();acquire(args.output)
