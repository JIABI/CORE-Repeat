"""Export only the explicitly authorized 911 FMP HepG2 objects + DMSO wells.

No protected outcome row is decoded or saved. The mandatory explicit flag is
for the user-authorized blind traversal of plate-member compression streams.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opal2.eu_profile_ingest import SelectionStats, iter_remote_member_rows

BASE = ROOT / "reports/eu_core_development_20260917_v1"
OUT = BASE / "ingest"


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    phase_path = BASE / "phase_manifest.json"
    phase = json.loads(phase_path.read_text())
    old = json.loads((BASE.parent / "new_data_assignment_20260917_v1/reservation_manifest.json").read_text())
    protected = set(old["reserved_connectivity_groups"])
    allow = rows(BASE / "measurement_allowlist.csv")
    dev = rows(BASE / "development_identities.csv")
    metadata = rows(BASE.parent / "new_data_qualification_20260917_v1/eu_openscreen/well_metadata.csv")
    plate_metadata = rows(BASE.parent / "new_data_qualification_20260917_v1/eu_openscreen/plate_metadata.csv")
    identity = rows(BASE.parent / "new_data_assignment_20260917_v1/identity_assignments.csv")
    ident = {r["object_id"]: r for r in identity if r["dataset"] == "EU_OPENSCREEN"}
    known = {(r["plate_uid"], r["well_position"]): r for r in metadata}
    assert phase["measurement_access_released"] is True
    assert phase["reserved_confirmation_measurements_released"] is False
    assert phase["site"] == "FMP" and phase["cell"] == "HepG2"
    assert len(allow) == 4428 and len(dev) == 911
    assert {r["object_id"] for r in dev} == set(phase["allowed_compound_ids"])
    assert not {r["connectivity"] for r in dev} & protected
    assert all(ident[r["object_id"]]["identity_role"] == "EU_TARGET_FIT_CANDIDATE" for r in dev)
    assert len({r["well_id"] for r in allow}) == len(allow)
    assert Counter(r["resource_kind"] for r in allow) == {"FIT_COMPOUND": 3644, "DMSO_CONTROL": 784}
    per_plate = defaultdict(dict)
    for r in allow:
        assert r["site"] == "FMP" and r["cell"] == "HepG2"
        src = known[(r["plate_uid"], r["well_position"])]
        assert src["object_id_raw"] == r["object_id"]
        if r["resource_kind"] == "FIT_COMPOUND":
            assert r["object_id"] in phase["allowed_compound_ids"]
            assert r["connectivity"] not in protected
        else:
            assert r["object_id"] == "DMSO"
        per_plate[(r["library_plate"], r["replicate"])][r["well_position"]] = r
    assert len(per_plate) == 28
    assert all(sum(r["resource_kind"] == "DMSO_CONTROL" for r in v.values()) == 28 for v in per_plate.values())
    member_list = json.loads((BASE / "access_format/member_headers.json").read_text())
    header = member_list[0]["columns"]
    assert len(header) == 2981
    assert header[:4] == ["Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_Object_Count"]
    assert len(member_list) == 28 and all(x["columns"] == header for x in member_list)
    pmap = {(p["library_plate"], p["replicate"]): p for p in plate_metadata
            if p["site"] == "FMP" and p["cell_line"] == "HepG2"}
    return allow, member_list, header, per_plate, pmap, {
        "phase_manifest_sha256": sha(phase_path),
        "allowlist_sha256": sha(BASE / "measurement_allowlist.csv"),
        "authorized_rows": len(allow), "fit_compounds": len(dev),
        "protected_connectivity_overlap": 0,
        "expected_fit_rows": 3644, "expected_dmso_rows": 784,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-stream-authorized", action="store_true")
    args = parser.parse_args()
    if not args.blind_stream_authorized:
        raise PermissionError("Explicit user permission for blind-stream filtering is required")
    allow, members, header, per_plate, pmap, preflight = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "allowed_members").mkdir(exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print("PREFLIGHT PASS: 4428 authorized wells; protected-group overlap=0", flush=True)

    def export_member(member):
        key = member["library_plate"], "R" + str(member["replicate"])
        allowed = per_plate[key]
        dest = OUT / "allowed_members" / Path(member["name"]).name
        audit_path = dest.with_suffix(".audit.json")
        if dest.exists() and audit_path.exists():
            report = json.loads(audit_path.read_text())
            if report["allowlist_sha256"] != preflight["allowlist_sha256"] or report["output_sha256"] != sha(dest):
                raise RuntimeError("Existing allowed export does not match pinned inputs")
            print("RESUME", key[0], key[1], "allowed", report["allowed_records"], flush=True)
            return report
        temp = dest.with_suffix(".allowed_only.partial")
        stats = SelectionStats()
        internal_plate = set()
        internal_batch = set()
        allowed_seen = set()
        with temp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in iter_remote_member_rows(member, allowed_wells=allowed,
                      expected_header=header, stats=stats, blind_stream_authorized=True,
                      require_all_allowed=False):
                assert r["Metadata_Well"] in allowed
                allowed_seen.add(r["Metadata_Well"])
                internal_plate.add(r["Metadata_Plate"])
                internal_batch.add(r["Metadata_Batch"])
                writer.writerow(r)
        assert allowed_seen <= set(allowed)
        assert stats.decoded_measurement_rows == len(allowed_seen)
        assert internal_plate == {key[0]} and internal_batch == {key[1]}
        report = {"member": member["name"], "library_plate": key[0], "replicate": key[1],
                  "physical_plate_uid": pmap[key]["plate_uid"],
                  "allowed_internal_plate_tokens": sorted(internal_plate),
                  "allowed_internal_batch_tokens": sorted(internal_batch),
                  "output": str(dest.relative_to(OUT)),
                  "output_sha256": sha(temp),
                  "allowlist_sha256": preflight["allowlist_sha256"],
                  "missing_authorized_wells": sorted(set(allowed) - allowed_seen),
                  "protected_numeric_rows_decoded": 0,
                  "unfiltered_profile_saved": False, **asdict(stats)}
        temp.replace(dest)
        audit_path.write_text(json.dumps(report, indent=2) + "\n")
        print("DONE", key[0], key[1], "allowed", stats.allowed_records,
              "missing", len(set(allowed) - allowed_seen), "excluded", stats.excluded_records,
              "transport_bytes", stats.transport_bytes, flush=True)
        return report

    # One network stream; respect server rate limits and keep CPU use minimal.
    with ThreadPoolExecutor(max_workers=1) as pool:
        reports = list(pool.map(export_member, members))
    temp = OUT / "raw_allowed_profiles.allowed_only.partial"
    meta_tmp = OUT / "row_metadata.allowed_only.partial"
    meta_fields = ["export_row_index", "source_member", "source_csv_batch", "source_csv_plate", "source_csv_well"] + list(allow[0])
    n = 0
    seen_well_ids = set()
    with temp.open("w", newline="") as f, meta_tmp.open("w", newline="") as g:
        writer = csv.DictWriter(f, fieldnames=header)
        meta_writer = csv.DictWriter(g, fieldnames=meta_fields)
        writer.writeheader()
        meta_writer.writeheader()
        for member in members:
            key = member["library_plate"], "R" + str(member["replicate"])
            permitted = per_plate[key]
            path = OUT / "allowed_members" / Path(member["name"]).name
            with path.open(newline="") as source:
                for r in csv.DictReader(source):
                    assert r["Metadata_Well"] in permitted
                    assert r["Metadata_Batch"] == key[1] and r["Metadata_Plate"] == key[0]
                    seen_well_ids.add(permitted[r["Metadata_Well"]]["well_id"])
                    writer.writerow(r)
                    meta_writer.writerow({"export_row_index": n, "source_member": member["name"],
                                          "source_csv_batch": r["Metadata_Batch"],
                                          "source_csv_plate": r["Metadata_Plate"],
                                          "source_csv_well": r["Metadata_Well"],
                                          **permitted[r["Metadata_Well"]]})
                    n += 1
    assert n <= 4428 and n == len(seen_well_ids)
    missing = [r for r in allow if r["well_id"] not in seen_well_ids]
    assert n + len(missing) == 4428
    with (OUT / "missing_allowed_wells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(allow[0]))
        writer.writeheader()
        writer.writerows(missing)
    temp.replace(OUT / "raw_allowed_profiles.csv")
    meta_tmp.replace(OUT / "row_metadata.csv")
    audit = {"status": "AUTHORIZED_INVENTORY_COMPLETE_DATASET_INCOMPLETE" if missing else "COMPLETE",
             "dataset_incomplete_training_blocked": bool(missing),
             "completed_utc": datetime.now(timezone.utc).isoformat(),
             **preflight, "output_rows": n, "numeric_columns": 2978,
             "missing_allowed_wells": len(missing),
             "missing_allowed_kind_counts": dict(Counter(r["resource_kind"] for r in missing)),
             "missing_unique_fit_objects": len({r["object_id"] for r in missing if r["resource_kind"] == "FIT_COMPOUND"}),
             "source_identity_mapping": "Metadata_Batch equals R1..R4; Metadata_Plate equals B1001..B1007; matched by verified member plus well to S3 dated plate_uid",
             "no_rows_imputed": True, "no_planned_objects_removed": True,
             "morphology_columns": 2977,
             "decoded_measurement_rows": sum(r["decoded_measurement_rows"] for r in reports),
             "excluded_records": sum(r["excluded_records"] for r in reports),
             "compressed_transport_bytes": sum(r["transport_bytes"] for r in reports),
             "raw_allowed_profiles_bytes": (OUT / "raw_allowed_profiles.csv").stat().st_size,
             "protected_numeric_rows_decoded": 0, "protected_outcome_rows_saved": 0,
             "full_archive_downloaded": False, "whole_member_or_unfiltered_csv_saved": False,
             "transport_contains_excluded_compressed_bytes": True,
             "bounded_inflater_buffers_pass_excluded_bytes": True,
             "excluded_values_text_or_numeric_decoded": False,
             "source_members": reports}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k:v for k,v in audit.items() if k != "source_members"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
