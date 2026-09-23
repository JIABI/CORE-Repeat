"""Exact-public-sample metadata join for the opened LKCP module-development data.

Unknown structures, conflicting identities and out-of-vocabulary annotation
sets are explicitly masked. No name/connectivity-based biological imputation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

PROJECT=Path(__file__).resolve().parents[1]
BIO=PROJECT/'reports/lincs_biology_preflight_20260915/biology_audit'


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def table(path):
    with Path(path).open(encoding='utf-8',errors='replace') as stream:
        return list(csv.DictReader((line for line in stream if not line.startswith('!')),delimiter='\t'))


def terms(value):
    return sorted({t.strip() for t in (value or '').split('|') if t.strip() and t.strip().lower() not in {'na','nan'}})


def normalized_key(value):
    value=(value or '').strip()
    if not value:
        return None
    if value.startswith('InChI='):
        return Chem.InchiToInchiKey(value) or None
    return value if re.fullmatch(r'[A-Z]{14}-[A-Z]{10}-[A-Z]',value) else None


def resolve_sample(sample, native_rows, sample_rows, pilot_entry=None):
    sources=[]
    for name,rows in [('repurposing_info_exact_sample',native_rows),('repurposing_samples_20170327_exact_sample',sample_rows)]:
        for row in rows:
            if row['broad_id']!=sample:
                raise ValueError('Only exact full sample identity is allowed')
            smiles=(row.get('smiles') or '').strip()
            mol=Chem.MolFromSmiles(smiles) if smiles else None
            value=(row.get('InChIKey') or '').strip()
            sources.append(dict(source=name,raw_smiles=smiles,raw_identity=value,
                computed_full_inchikey=Chem.MolToInchiKey(mol) if mol is not None else None,
                canonical_smiles=Chem.MolToSmiles(mol,isomericSmiles=True) if mol is not None else None,
                declared_full_inchikey=normalized_key(value),
                malformed_declared_identity=bool(value and normalized_key(value) is None)))
    if pilot_entry is not None and pilot_entry.get('chemistry_available'):
        if pilot_entry['broad_sample']!=sample:
            raise ValueError('Pilot metadata fallback must match exact full sample')
        mol=Chem.MolFromSmiles(pilot_entry['smiles'])
        sources.append(dict(source='existing_Pilot_exact_sample_chemistry',raw_smiles=pilot_entry['smiles'],
            raw_identity=pilot_entry['full_inchikey'],computed_full_inchikey=Chem.MolToInchiKey(mol),
            canonical_smiles=Chem.MolToSmiles(mol,isomericSmiles=True),
            declared_full_inchikey=normalized_key(pilot_entry['full_inchikey']),malformed_declared_identity=False))
    keys={v for s in sources for v in (s['computed_full_inchikey'],s['declared_full_inchikey']) if v}
    valid=[s for s in sources if s['canonical_smiles']]
    conflict=len(keys)>1 or any(s['malformed_declared_identity'] for s in sources)
    available=bool(valid) and len(keys)==1 and not conflict
    target=sorted({v for row in native_rows for v in terms(row.get('target'))})
    moa=sorted({v for row in native_rows for v in terms(row.get('moa'))})
    key=next(iter(keys)) if available else None
    return dict(broad_sample=sample,broad_id=sample[:13],exact_native_rows=len(native_rows),
        exact_2017_sample_rows=len(sample_rows),sources=sources,chemistry_available=available,
        identity_conflict=conflict,smiles=valid[0]['canonical_smiles'] if available else None,
        full_inchikey=key,inchikey14=key.split('-')[0] if key else None,
        target_set=target,moa_set=moa,target_annotation_known=bool(target),moa_annotation_known=bool(moa),
        biological_identity_verified=available and not conflict,
        biology_source='repurposing_info.tsv exact full Broad sample only')


def encode_annotations(record, vocabulary, kind):
    index={term:i for i,term in enumerate(vocabulary)}
    full=record[kind+'_set'];oov=sorted(set(full)-set(index))
    vector=np.zeros(len(vocabulary),dtype=np.float64)
    for term in full:
        if term in index:
            vector[index[term]]=1.
    mask=bool(full) and record['biological_identity_verified'] and not oov
    return vector,mask,dict(full_terms=full,full_term_count=len(full),oov_terms=oov,
        annotation_known=bool(full),complete_vocabulary=not oov,model_mask=mask,
        interpretation='Model mask is false for incomplete vocabulary; no truncated-set cosine is used')


def connectivity_groups(records):
    """Union same Broad compound and all known identical connectivity keys."""
    ids=sorted({r['broad_id'] for r in records.values()});parent={i:i for i in ids}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    bykey=defaultdict(set)
    for r in records.values():
        if r['chemistry_available']:
            bykey[r['inchikey14']].add(r['broad_id'])
    for members in bykey.values():
        first=min(members)
        for cid in members:
            union(first,cid)
    mapping={cid:'LKCP_GROUP::'+find(cid) for cid in ids}
    return mapping,dict(unique_broad_ids=len(ids),union_groups=len(set(mapping.values())),
        known_connectivity_groups=len(bykey),connectivities_with_multiple_broad_ids={k:sorted(v) for k,v in bykey.items() if len(v)>1})


def build(raw_directory, output):
    raw_directory,output=Path(raw_directory),Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh annotated output; existing data are not replaced')
    source=load(raw_directory/'measurement_metadata.json')
    pilot=load(PROJECT/'data/lincs_pilot1_biology_20260915/metadata.json')
    native_path=BIO/'public_metadata/metadata/moa/repurposing_info.tsv'
    sample_path=BIO/'public_metadata/metadata/moa/clue/repurposing_samples_20170327.txt'
    indexes=[]
    for path in (native_path,sample_path):
        index=defaultdict(list)
        for row in table(path):
            index[row['broad_id']].append(row)
        indexes.append(index)
    prior={v['broad_sample']:v for v in load(BIO/'metadata_only_chemistry_available_map.json').values()}
    samples=sorted({u['broad_sample'] for u in source['units']})
    rdBase.DisableLog('rdApp.warning')
    resolved={s:resolve_sample(s,indexes[0][s],indexes[1][s],prior.get(s)) for s in samples}
    groups,group_summary=connectivity_groups(resolved)
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512)
    encoded={}
    for sample,r in resolved.items():
        chem=np.zeros(513,dtype=np.float64)
        if r['chemistry_available']:
            mol=Chem.MolFromSmiles(r['smiles']);bits=np.zeros(512,dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),bits)
            if not bits.any():
                raise ValueError('Available exact structure produced empty fingerprint')
            chem[:512]=bits;chem[-1]=1.
        target,tm,ts=encode_annotations(r,pilot['target_names'],'target')
        moa,mm,ms=encode_annotations(r,pilot['moa_names'],'moa')
        encoded[sample]=dict(chem=chem,target=target,moa=moa,target_mask=tm,moa_mask=mm)
        r.update(target_frozen_vocabulary=ts,moa_frozen_vocabulary=ms,group=groups[r['broad_id']])
    output.mkdir(parents=True)
    summaries={}
    for condition,scope in source['conditions'].items():
        units=[u for u in source['units'] if u['condition']==condition]
        with np.load(raw_directory/(condition+'_module_dev_measurements.npz'),allow_pickle=False) as z:
            arrays={k:z[k].copy() for k in z.files}
        assert arrays['ids'].tolist()==[u['id'] for u in units]
        if arrays['feature_names'].tolist()!=source['feature_names']:
            raise ValueError('Measurement feature order changed')
        n=len(units);records=[resolved[u['broad_sample']] for u in units]
        parts=[encoded[u['broad_sample']] for u in units]
        folder=output/condition;folder.mkdir()
        values=dict(Y=arrays['Y'],ids=arrays['ids'],feature_names=arrays['feature_names'],well_ids=arrays['well_ids'],
            role_names=arrays['role_names'],groups=np.array([r['group'] for r in records]),
            chem=np.stack([v['chem'] for v in parts]),chem_mask=np.array([r['chemistry_available'] for r in records]),
            target=np.stack([v['target'] for v in parts]),moa=np.stack([v['moa'] for v in parts]),
            target_mask=np.array([v['target_mask'] for v in parts]),moa_mask=np.array([v['moa_mask'] for v in parts]),
            actual_dose_uM=np.full(n,np.nan),actual_dose_known=np.zeros(n,dtype=bool),
            recorded_concentrations=arrays['recorded_concentrations'],compound_ids=arrays['compound_ids'])
        assert values['chem'].shape==(n,513) and np.isfinite(values['Y']).all()
        assert np.all(values['chem'][~values['chem_mask']]==0)
        np.savez_compressed(folder/'data.npz',**values)
        enriched=[]
        for u,r in zip(units,records):
            enriched.append(dict(u,chemistry=r,layout_block=u['layout']+'::'+u['well'],
                cell_line='A549',exposure_hours_protocol_nominal=int(condition.split('_')[1][:-1]),
                actual_dose_uM=None,nominal_dose_uM=None,biological_condition_supported=False,
                biology_condition_reason='actual assay dose unverified; raw recorded concentration is not converted'))
        summary=dict(n=n,unique_compounds=len({u['compound_id'] for u in units}),
            chemistry_available_episodes=int(values['chem_mask'].sum()),
            chemistry_available_compounds=len({r['broad_id'] for r in records if r['chemistry_available']}),
            frozen_vocab_target_mask_episodes=int(values['target_mask'].sum()),
            frozen_vocab_moa_mask_episodes=int(values['moa_mask'].sum()),
            chemistry_unknown_episodes=int((~values['chem_mask']).sum()),role=scope['role'])
        save(folder/'metadata.json',dict(ids=arrays['ids'].tolist(),units=enriched,source=source,
            preprocessing=source['preprocessing'],target_names=pilot['target_names'],moa_names=pilot['moa_names'],
            chemical=dict(kind='Morgan binary fingerprint',radius=2,bits=512,
                final_coordinate='valid_SMILES_indicator',rdkit_version=rdBase.rdkitVersion),
            availability=summary,actual_dose_verified=False,formal_evaluation=False,
            cohort_filter='none: every1086 metadata-defined episode retained',
            group_definition='union of same Broad compound ID and known identical connectivity',
            biology_vocabulary='Frozen Pilot1 order; complete-vocabulary mask; full terms/OOV preserved in chemistry sidecars'))
        summaries[condition]=dict(summary,directory=str(folder.resolve()))
    pilot_ids=set(pilot['ids']);pilot_keys={u['chemistry']['inchikey14'] for u in pilot['units']}
    known=[r for r in resolved.values() if r['chemistry_available']]
    all_ids={r['broad_id'] for r in resolved.values()}
    summary=dict(conditions=summaries,n_compound_ids=len(all_ids),n_exact_samples=len(samples),
        exact_native_samples=sum(bool(r['exact_native_rows']) for r in resolved.values()),
        exact_2017_table_samples=sum(bool(r['exact_2017_sample_rows']) for r in resolved.values()),
        chemistry_available_samples=len(known),chemistry_available_compounds=len({r['broad_id'] for r in known}),
        identity_conflict_samples=[s for s,r in resolved.items() if r['identity_conflict']],
        unknown_structure_samples=[s for s,r in resolved.items() if not r['chemistry_available']],
        grouping=group_summary,overlap_old1188_compound_ids=len(all_ids&pilot_ids),
        known_lkcp_connectivity_groups=len({r['inchikey14'] for r in known}),
        known_connectivity_groups_overlapping_old1188=len({r['inchikey14'] for r in known}&pilot_keys),
        lkcp_compound_ids_with_known_connectivity_in_old1188=len({r['broad_id'] for r in known if r['inchikey14'] in pilot_keys}),
        unknown_chemistry_is_not_evidence_of_chemical_novelty=True,
        target_vocabulary_count=len(pilot['target_names']),moa_vocabulary_count=len(pilot['moa_names']),
        target_oov_samples=sum(bool(r['target_frozen_vocabulary']['oov_terms']) for r in resolved.values()),
        moa_oov_samples=sum(bool(r['moa_frozen_vocabulary']['oov_terms']) for r in resolved.values()),
        source_paths=[str(native_path),str(sample_path),str(BIO/'metadata_only_chemistry_available_map.json')],
        new_public_downloads=False,new_training=False,assay_dose_verified=False,
        no_compounds_or_episodes_discarded=True,no_broad_id_biology_imputation=True)
    save(output/'sample_annotations.json',resolved);save(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('unknown_structure_samples','source_paths')},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--raw-directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.raw_directory,args.output)
