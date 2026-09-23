"""Prepare metadata-only R4 plans or execute one explicitly authorized stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opal2.r4_staged_data import export_stage, prepare_stage_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Metadata only; no source measurement access")
    qualification = ROOT / "reports/new_data_qualification_20260917_v1/eu_openscreen"
    assignment = ROOT / "reports/new_data_assignment_20260917_v1/eu_metadata_resolution"
    dev = ROOT / "reports/eu_core_development_20260917_v1"
    plan.add_argument("--cohort", required=True)
    plan.add_argument("--site", required=True, choices=["FMP", "MEDINA", "USC"])
    plan.add_argument("--stage", required=True, choices=["x", "outcomes", "anchors", "external_outcomes"])
    plan.add_argument("--output", required=True)
    plan.add_argument("--well-metadata", default=str(qualification / "well_metadata.csv"))
    plan.add_argument("--archive-inventory", default=str(qualification / "archive_member_inventory.json"))
    plan.add_argument("--filename-mapping", default=str(assignment / "filename_plate_mapping.csv"))
    plan.add_argument("--identity-source", default=str(qualification / "identity_raw.csv"))
    plan.add_argument("--control-space")
    plan.add_argument("--member-headers")
    run = commands.add_parser("export", help="Stage authorization and completed file gates required")
    run.add_argument("--plan", required=True)
    run.add_argument("--authorization", required=True)
    run.add_argument("--authorization-key", required=True)
    run.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        space = args.control_space
        headers = args.member_headers
        if args.site == "FMP":
            space = space or str(dev / "prepared_data_cc904/control_space.json")
            headers = headers or str(dev / "access_format/member_headers.json")
        result = prepare_stage_plan(args.cohort, args.well_metadata, args.archive_inventory,
            args.filename_mapping, args.output, site=args.site, stage=args.stage,
            control_space_path=space, identity_source=args.identity_source, member_headers_file=headers)
        result = {k: v for k, v in result.items() if k not in ("cohort", "members")}
    else:
        result = export_stage(args.plan, args.authorization, args.authorization_key, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
