"""Execute the authorized, one-shot R4 campaign without fitting new models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from opal2.r4_staged_data import prepare_stage_plan, export_stage

REPORT = ROOT/'reports/r4_execution_20260921_v1'
RUN = ROOT/'runs/r4_confirmation_20260921_v1'
QUAL = REPORT/'qualification'
USER_AUTH = ('User explicitly authorized excluding the two resolved overlapping objects, '
             'fixing the existing evaluation plan and one formal evaluation of the remaining '
             '1539 candidates on 21 September 2026.')
ADDED = {'EOS101686': 'MS023 / BRD-K20505391: resolved LKCP development overlap',
         'EOS101275': 'CGI-1746 / BRD-K62014504: resolved LKCP development overlap'}
OLD = {'EOS101092', 'EOS101189', 'EOS101383', 'EOS101386',
       'EOS101387', 'EOS101484', 'EOS101551', 'EOS101571'}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def status(stage, **extra):
    record = dict(state='RUNNING', stage=stage, updated_utc=now(), pid=os.getpid(),
        population_n=1539, device='cpu', cpu_threads_max=2, **extra)
    write_json(REPORT/'status.json', record)
    print(json.dumps(record), flush=True)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_rows_once(path, records):
    path = Path(path)
    if path.exists():
        if rows(path) != records:
            raise ValueError('Existing metadata differs: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare():
    """Only identity and plate metadata are read here."""
    old = ROOT/'reports/r4_preparation_20260919_v1/identity'
    original = rows(old/'EU_original1549_proposed_qualification.csv')
    if len(original) != 1549 or len({r['object_id'] for r in original}) != 1549:
        raise ValueError('Original protected cohort changed')
    excluded = OLD | set(ADDED)
    remaining = sorted((r for r in original if r['object_id'] not in excluded),
                       key=lambda r: r['object_id'])
    if len(remaining) != 1539 or len({r['connectivity'] for r in remaining}) != 1539:
        raise ValueError('Final population is not 1539 independent identities')
    proposed = {r['object_id'] for r in rows(old/'EU_remaining_candidate_proposal.csv')}
    if proposed-set(ADDED) != {r['object_id'] for r in remaining}:
        raise ValueError('Identity decision differs from previous candidate ledger')
    cohort = [dict(object_id=r['object_id'], connectivity=r['connectivity'], name=r['name'])
              for r in remaining]
    write_rows_once(QUAL/'cohort.csv', cohort)
    dispositions = [dict(object_id=r['object_id'], connectivity=r['connectivity'], name=r['name'],
        decision='EXCLUDED' if r['object_id'] in excluded else 'QUALIFIED',
        reason=ADDED.get(r['object_id'], 'Previously established development overlap' if
            r['object_id'] in OLD else 'No detected connection-layer overlap with development'))
        for r in sorted(original, key=lambda r: r['object_id'])]
    write_rows_once(QUAL/'original1549_disposition.csv', dispositions)
    write_json(QUAL/'qualification.json', dict(status='PASS', original_n=1549,
        excluded_n=10, final_n=1539, additional_exclusions=ADDED,
        independent_audit='Final cohort has no ID/connectivity overlap with DEV904, historical '
            'structure union or RxRx3 R2/R3 development groups.',
        unresolved_BRD_U='Retained in historical ledger; exposure audit found no use in reviewed model development.',
        identity_evidence=str(REPORT/'identity/resolved_LKCP_structure_evidence.json'),
        user_authorization=USER_AUTH, confirmation_measurements_read=False))
    meta = ROOT/'reports/new_data_qualification_20260917_v1/eu_openscreen'
    mapping = ROOT/'reports/new_data_assignment_20260917_v1/eu_metadata_resolution/filename_plate_mapping.csv'
    dev = ROOT/'reports/eu_core_development_20260917_v1'
    for stage in ('x', 'outcomes'):
        directory = REPORT/'stageplans'/('final_'+stage+'1539')
        if not directory.exists():
            prepare_stage_plan(QUAL/'cohort.csv', meta/'well_metadata.csv',
                meta/'archive_member_inventory.json', mapping, directory, site='FMP', stage=stage,
                control_space_path=dev/'prepared_data_cc904/control_space.json',
                identity_source=meta/'identity_raw.csv', member_headers_file=dev/'access_format/member_headers.json')
        plan = read_json(directory/'plan.json')
        if plan['n'] != 1539 or [r['object_id'] for r in plan['cohort']] != [r['object_id'] for r in cohort]:
            raise ValueError('Stage plan differs from final cohort')
    print('Final 1539 metadata cohort and distinct X/outcome plans prepared.', flush=True)


def freeze():
    """Release exact plans only after DEV anchors and model have completed."""
    anchor_dir = RUN/'external/anchor_space'
    anchor_file = anchor_dir/'ANCHORS_FROZEN.json'
    anchor = read_json(anchor_file)
    if anchor.get('status') != 'FROZEN':
        raise ValueError('External development anchor space is not frozen')
    if not read_json(RUN/'model/complete.json').get('complete'):
        raise ValueError('Final model not complete')
    meta = ROOT/'reports/new_data_qualification_20260917_v1/eu_openscreen'
    mapping = ROOT/'reports/new_data_assignment_20260917_v1/eu_metadata_resolution/filename_plate_mapping.csv'
    stages = {}
    for key, site, stage in [('x','FMP','x'), ('outcomes','FMP','outcomes'),
                             ('medina_outcomes','MEDINA','external_outcomes'),
                             ('usc_outcomes','USC','external_outcomes')]:
        directory = REPORT/'stageplans'/('final_'+key+'1539')
        if site != 'FMP' and not directory.exists():
            prepare_stage_plan(QUAL/'cohort.csv', meta/'well_metadata.csv', meta/'archive_member_inventory.json',
                mapping, directory, site=site, stage=stage,
                control_space_path=RUN/'external'/site/'anchors/control_space.json')
        stages[key] = dict(authorized=True, site=site, plan_file=str(directory/'plan.json'),
            freeze_file=str(REPORT/'freeze.json'), model_complete_file=str(RUN/'model/complete.json'))
        if stage != 'x':
            stages[key]['selections_file'] = str(RUN/'selections/SELECTIONS_FROZEN.json')
    config = read_json(REPORT/'algorithm_config.json')
    record = dict(status='FROZEN', created_utc=now(), user_authorization=USER_AUTH,
        population_n=1539, action_budget=384, selected_n_if_eligible=192,
        cohort_file=str(QUAL/'cohort.csv'), qualification_file=str(QUAL/'qualification.json'),
        model_dir=str(RUN/'model'), protocol_file=str(REPORT/'PROTOCOL.md'),
        protocol_text=(REPORT/'PROTOCOL.md').read_text(), algorithm=config,
        external_anchors_file=str(anchor_file), external_sites=['MEDINA','USC'],
        confirmation_X_opened=False, confirmation_outcomes_opened=False,
        biology_enabled=False, representation_extension_enabled=False)
    if (REPORT/'freeze.json').exists():
        saved = read_json(REPORT/'freeze.json')
        if saved['population_n'] != 1539 or saved['protocol_text'] != record['protocol_text']:
            raise ValueError('Do not change a released protocol')
    else:
        write_json(REPORT/'freeze.json', record)
    write_json(REPORT/'stage_authorizations.json', dict(schema='opal-r4-stage-authorizations-v1',
        user_authorization=USER_AUTH, stages=stages))


def stage_export(key, output):
    output = Path(output)
    if (output/'complete.json').exists():
        return
    export_stage(REPORT/'stageplans'/('final_'+key+'1539')/'plan.json',
        REPORT/'stage_authorizations.json', key, output)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def predictors_from_saved(query, predictions):
    from opal2.r4_final_model import CORE_ARM, GAUSSIAN_ARM, DIRECT_ARM
    lookup = {s:i for i,s in enumerate(query['ids'].astype(str))}
    out = {}
    for short, arm in [('CORE',CORE_ARM), ('GAUSSIAN',GAUSSIAN_ARM), ('HISTGB_CAL',DIRECT_ARM)]:
        data = npz(Path(predictions)/(arm+'.npz'))
        if set(data['ids'].astype(str)) != set(query['ids'][query['eligible']].astype(str)):
            raise ValueError('Prediction eligibility population differs')
        ind = np.array([lookup[s] for s in data['ids'].astype(str)])
        mu, probability = np.full(len(lookup),np.nan), np.full(len(lookup),np.nan)
        mu[ind], probability[ind] = data['predicted'], data['p_null']
        out[short] = dict(expected=mu, p_null=probability, **{'lambda':.2})
        out[short+'_LAMBDA0'] = dict(expected=mu, p_null=probability, **{'lambda':0.})
    return out


def run(wait_anchors=False):
    started = time.monotonic()
    prepare()
    anchor_file = RUN/'external/anchor_space/ANCHORS_FROZEN.json'
    if not anchor_file.exists():
        if not wait_anchors:
            raise FileNotFoundError('Complete DEV-only external anchors before opening X')
        status('WAITING_FOR_DEVELOPMENT_ANCHORS', confirmation_X_opened=False,
               confirmation_outcomes_opened=False)
        while not anchor_file.exists():
            time.sleep(10)
    freeze()
    status('EXPORT_CONFIRMATION_X', confirmation_X_access_started=True,
           confirmation_outcomes_opened=False)
    query_file = RUN/'ingest/x/query.npz'
    stage_export('x', query_file.parent)
    query = npz(query_file)
    expected_ids = np.array([r['object_id'] for r in rows(QUAL/'cohort.csv')])
    np.testing.assert_array_equal(query['ids'], expected_ids)
    from opal2.r4_final_model import score
    predictions = RUN/'predictions'
    manifest = predictions/'manifest.json'
    if not manifest.exists() or read_json(manifest).get('state') != 'COMPLETE':
        status('SCORING_X_ONLY', confirmation_X_opened=True, confirmation_outcomes_opened=False,
               eligible_n=int(query['eligible'].sum()))
        eligible_query = {k:query[k][query['eligible']] for k in ('ids','groups','layout','X','chem','chem_mask')}
        score(RUN/'model', eligible_query, predictions, population_size=len(query['ids']))
    from opal2.r4_evaluation import freeze_selections
    selections = RUN/'selections'
    if not (selections/'SELECTIONS_FROZEN.json').exists():
        freeze_selections(query['ids'],query['eligible'],predictors_from_saved(query,predictions),selections)
    status('EXPORT_FIXED_LIST_OUTCOMES', confirmation_X_opened=True,
           selections_frozen=True, confirmation_outcome_access_started=True)
    future_file = RUN/'ingest/outcomes/outcomes.npz'
    stage_export('outcomes', future_file.parent)
    status('PRIMARY_EVALUATION', confirmation_X_opened=True, confirmation_outcomes_opened=True,
           selections_frozen=True)
    from opal2.r4_primary_analysis import run_primary_analysis
    primary = REPORT/'primary'
    if not (primary/'complete.json').exists():
        run_primary_analysis(predictions,selections,query_file,future_file,primary,model_dir=RUN/'model')
        write_json(primary/'complete.json',dict(status='COMPLETE',completed_utc=now()))
    external_files = {}
    for site in ('MEDINA','USC'):
        status('EXTERNAL_OUTCOMES_'+site, confirmation_X_opened=True,
               confirmation_outcomes_opened=True, selections_frozen=True)
        file = RUN/'external'/site/'confirmation/outcomes.npz'
        stage_export(site.lower()+'_outcomes',file.parent)
        external_files[site] = file
    from opal2.r4_external_endpoint import evaluate_external_files
    secondary = REPORT/'external/evaluation'
    if not (secondary/'complete.json').exists():
        evaluate_external_files(RUN/'external/anchor_space',query_file,future_file,external_files,
            selections,REPORT/'freeze.json',secondary)
        write_json(secondary/'complete.json',dict(status='COMPLETE',completed_utc=now()))
    status('ALL_R4_COMPLETE', confirmation_X_opened=True, confirmation_outcomes_opened=True,
           selections_frozen=True, elapsed_seconds=time.monotonic()-started)
    saved = read_json(REPORT/'status.json')
    saved['state'] = 'COMPLETE'
    write_json(REPORT/'status.json',saved)
    write_json(RUN/'complete.json',saved)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run'])
    parser.add_argument('--wait-anchors', action='store_true')
    args = parser.parse_args()
    try:
        prepare() if args.command == 'prepare' else run(args.wait_anchors)
    except Exception as exc:
        previous = read_json(REPORT/'status.json') if (REPORT/'status.json').exists() else {}
        write_json(REPORT/'status.json',dict(**{k:v for k,v in previous.items() if k!='state'},
            state='FAILED',error_type=type(exc).__name__,error=str(exc),failed_utc=now()))
        raise
