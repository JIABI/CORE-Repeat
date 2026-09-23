"""Join the predeclared public chemistry cohort to four physical LINCS wells."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .biology_kernel_evaluation import write_json
from .gram_oof_experiment import now


def build(raw_directory, annotations_file, output):
    raw_directory, output = Path(raw_directory), Path(output)
    if output.exists():
        raise FileExistsError('Prepared LINCS inputs are not overwritten')
    annotations = json.loads(Path(annotations_file).read_text())
    metadata = json.loads((raw_directory/'metadata.json').read_text())
    with np.load(raw_directory/'measurements.npz', allow_pickle=False) as z:
        raw = {key:z[key].copy() for key in z.files}
    lookup = {str(unit):i for i,unit in enumerate(raw['ids'])}
    selected = sorted(annotations)
    if not set(selected)<=set(lookup) or len(selected)!=1188:
        raise ValueError('Metadata-only chemistry cohort must match the declared 1188 objects')
    rows = np.asarray([lookup[unit] for unit in selected])
    if [u['id'] for u in metadata['units']] != raw['ids'].tolist():
        raise ValueError('Raw physical-well metadata is not aligned')
    units = [deepcopy(metadata['units'][i]) for i in rows]
    if raw['role_names'].tolist()!=['X','Z1','Z2','V']:
        raise ValueError('Unexpected role ordering')
    target_names = sorted({t for unit in selected for t in annotations[unit]['target_set']})
    moa_names = sorted({t for unit in selected for t in annotations[unit]['moa_set']})
    target_index, moa_index = ({t:i for i,t in enumerate(names)} for names in (target_names,moa_names))
    n = len(selected)
    target, moa = np.zeros((n,len(target_names))), np.zeros((n,len(moa_names)))
    target_mask, moa_mask = np.zeros(n,bool),np.zeros(n,bool)
    chemistry = np.zeros((n,513),float)
    groups=[]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512)
    for i,unit in enumerate(selected):
        annotation = annotations[unit]
        if units[i]['broad_sample']!=annotation['broad_sample']:
            raise ValueError('Biology must join the exact physical sample')
        mol = Chem.MolFromSmiles(annotation['smiles'])
        if mol is None or Chem.MolToInchiKey(mol)!=annotation['full_inchikey']:
            raise ValueError('Declared exact chemistry cannot be reproduced: '+unit)
        bits=np.zeros(512,dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),bits)
        if not bits.any():
            raise ValueError('Declared chemical input has no fingerprint bits')
        chemistry[i,:512]=bits;chemistry[i,512]=1.
        groups.append(annotation['inchikey14'])
        for key,values,index,mask in (('target',target,target_index,target_mask),('moa',moa,moa_index,moa_mask)):
            known=annotation[key+'_known'];terms=annotation[key+'_set']
            if not isinstance(known,bool) or bool(terms)!=known:
                raise ValueError('Separate known/missing annotation flags disagree with curated payload')
            if known:
                values[i,[index[t] for t in terms]]=1.
                mask[i]=True
        units[i].update(chemistry=annotation,layout_block='|'.join(units[i]['layout_blocks']))
    output.mkdir(parents=True)
    np.savez_compressed(output/'data.npz',Y=raw['Y'][rows],ids=np.asarray(selected),
        feature_names=raw['feature_names'],well_ids=raw['well_ids'][rows],
        groups=np.asarray(groups),chem=chemistry,chem_mask=np.ones(n,bool),target=target,moa=moa,
        target_mask=target_mask,moa_mask=moa_mask,actual_dose_uM=raw['actual_dose_uM'][rows])
    result=dict(created_utc=now(),ids=selected,units=units,source=metadata['source'],
        source_directory=str(raw_directory.resolve()),annotation_file=str(Path(annotations_file).resolve()),
        preprocessing=metadata['preprocessing_report'],target_names=target_names,moa_names=moa_names,
        chemical=dict(kind='Morgan binary fingerprint',radius=2,bits=512,
            final_coordinate='valid_SMILES_indicator',rdkit_version=rdBase.rdkitVersion),
        availability=dict(raw_metadata_eligible=len(raw['ids']),chemistry_eligible=n,
            excluded_missing_chemistry=len(raw['ids'])-n,target_known=int(target_mask.sum()),
            moa_known=int(moa_mask.sum()),both_known=int((target_mask&moa_mask).sum())),
        public_vocabulary='fixed annotation strings, not learned from outcomes; novel TRAIN-unseen terms remain in profile norms',
        actual_dose_uM_range=[float(raw['actual_dose_uM'][rows].min()),float(raw['actual_dose_uM'][rows].max())],
        observation_source='four distinct physical wells; not consensus',cell_count_used_as_model_input=False,
        no_outcome_based_cohort_selection=True,old_JUMP_accessed=False)
    write_json(output/'metadata.json',result)
    print(json.dumps(dict(shape=list(raw['Y'][rows].shape),availability=result['availability'],
                         prepared_directory=str(output.resolve())),indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-directory',required=True);parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.raw_directory,args.annotations,args.output)
