"""Prepare authorized DEV anchors, then evaluate the frozen R4 external endpoint."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from opal2.r4_staged_data import export_stage, prepare_stage_plan

REPORT = ROOT/'reports/r4_execution_20260921_v1/external'
RUN = ROOT/'runs/r4_confirmation_20260921_v1/external'
DEV = ROOT/'reports/eu_core_development_20260917_v1/prepared_data_cc904'


def prepare():
    """Only read DEV identities/design metadata; never touch candidate profiles."""
    REPORT.mkdir(parents=True, exist_ok=True)
    cohort = REPORT/'development_cohort.csv'
    with np.load(DEV/'data.npz', allow_pickle=False) as z:
        ids, groups = z['ids'].astype(str), z['groups'].astype(str)
    if len(ids) != 904 or len(set(ids)) != 904 or len(set(groups)) != 904:
        raise ValueError('Expected exact DEV904 distinct identities')
    rows = [dict(object_id=ids[i], connectivity=groups[i]) for i in np.argsort(ids)]
    if cohort.exists():
        with cohort.open() as handle:
            if list(csv.DictReader(handle)) != rows:
                raise ValueError('Existing DEV cohort differs')
    else:
        with cohort.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['object_id', 'connectivity'])
            writer.writeheader()
            writer.writerows(rows)
    metadata = ROOT/'reports/new_data_qualification_20260917_v1/eu_openscreen'
    mapping = ROOT/'reports/new_data_assignment_20260917_v1/eu_metadata_resolution/filename_plate_mapping.csv'
    stages = {}
    for site in ('MEDINA', 'USC'):
        directory = REPORT/'stageplans'/f'{site.lower()}_anchors_v2'
        if directory.exists():
            plan = json.loads((directory/'plan.json').read_text())
            if (plan['site'] != site or plan['stage'] != 'anchors'
                    or [{k: r[k] for k in ('object_id', 'connectivity')} for r in plan['cohort']] != rows):
                raise ValueError('Existing anchor plan differs from DEV904')
        else:
            plan = prepare_stage_plan(cohort, metadata/'well_metadata.csv',
                metadata/'archive_member_inventory.json', mapping, directory,
                site=site, stage='anchors')
        stages[site.lower()+'_anchors'] = dict(authorized=True, site=site,
                                              plan_file=str(directory/'plan.json'))
        print(json.dumps(dict(site=site, n=plan['n'], counts=plan['resource_counts'],
                              members=len(plan['members']))), flush=True)
    authorization = dict(schema='opal-r4-stage-authorizations-v1',
        user_authorization='User authorized formal R4 on 2026-09-21 after excluding EOS101686 and '
            'EOS101275. Coordinator explicitly authorizes only DEV904 MEDINA/USC anchors and '
            'their declared DMSO controls before confirmation X. No confirmation access granted here.',
        stages=stages)
    auth = REPORT/'anchor_authorization_v2.json'
    if auth.exists() and json.loads(auth.read_text()) != authorization:
        raise ValueError('Existing anchor authorization differs')
    if not auth.exists():
        auth.write_text(json.dumps(authorization, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    export = commands.add_parser('export-anchors')
    export.add_argument('--site', required=True, choices=['MEDINA', 'USC'])
    freeze = commands.add_parser('freeze-anchors')
    freeze.add_argument('--confirmation-query', default=str(ROOT/'runs/r4_confirmation_20260921_v1/ingest/x/query.npz'))
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--query', required=True)
    evaluate.add_argument('--future', required=True)
    evaluate.add_argument('--medina', required=True)
    evaluate.add_argument('--usc', required=True)
    evaluate.add_argument('--selections', required=True)
    evaluate.add_argument('--protocol-freeze', required=True)
    evaluate.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'export-anchors':
        name = args.site.lower()+'_anchors'
        print(json.dumps(export_stage(REPORT/'stageplans'/(name+'_v2')/'plan.json',
            REPORT/'anchor_authorization_v2.json', name, RUN/args.site/'anchors'), indent=2))
    elif args.command == 'freeze-anchors':
        from opal2.r4_external_endpoint import freeze_anchor_space
        print(json.dumps(freeze_anchor_space(DEV/'data.npz',
            {site: RUN/site/'anchors' for site in ('MEDINA', 'USC')}, RUN/'anchor_space',
            confirmation_query=args.confirmation_query), indent=2))
    else:
        from opal2.r4_external_endpoint import evaluate_external_files
        print(json.dumps(evaluate_external_files(RUN/'anchor_space', args.query, args.future,
            {'MEDINA': args.medina, 'USC': args.usc}, args.selections,
            args.protocol_freeze, args.output), indent=2))


if __name__ == '__main__':
    main()
