"""Staged EU confirmation access: metadata plan, X, then frozen-list outcomes.

Plans contain physical identities only. Compressed members are streamed through
the existing identity-first byte filter; unlisted values are never decoded.
FMP uses the DEV904 control transform unchanged. External DEV anchors can fit a
separate control-only space before any external confirmation profile is opened.
"""
from __future__ import annotations

import csv
import io
import json
import re
import struct
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .eu_assay_preprocessing import apply_control_space, fit_control_space
from .eu_fit_dataset import chemistry_for, read_rows
from .eu_profile_ingest import (
    ARCHIVE_URL, IDENTITY_PREFIX, SelectionStats, _range_get, _range_stream,
    iter_selected_csv_rows,
)

ROLES = ("X", "Z1", "Z2", "V")
STAGE_ROLES = {"x": ("X",), "outcomes": ROLES[1:], "anchors": ROLES,
               "external_outcomes": ROLES}
SITES = ("FMP", "MEDINA", "USC")
SCHEMA = "opal-r4-stage-plan-v1"


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _true(value):
    return value in (True, "true", "True")


def stage_allowlist(cohort, metadata, *, site, stage):
    """Resolve all designed roles before retaining the authorized stage only.

    Eligibility comes exclusively from the supplied frozen identity cohort and
    design metadata. Missing numerical rows never alter this population.
    """
    if site not in SITES or stage not in STAGE_ROLES:
        raise ValueError("Unsupported site or stage")
    if (stage in ("x", "outcomes") and site != "FMP"
            or stage in ("anchors", "external_outcomes") and site == "FMP"):
        raise ValueError("FMP main and external-site stages must be separate")
    identity = {str(r["object_id"]): r for r in cohort}
    if not identity or len(identity) != len(cohort):
        raise ValueError("A nonempty unique frozen identity cohort is required")
    if any(not r.get("connectivity") for r in cohort):
        raise ValueError("Every cohort identity requires its frozen connectivity")
    local = [r for r in metadata
             if r["site"] == site and r["cell_line_protocol"] == "HepG2"]
    by_id = defaultdict(list)
    for row in local:
        if row["object_id_raw"] in identity:
            by_id[row["object_id_raw"]].append(row)
    subjects, allowed = [], []
    for oid in sorted(identity):
        wells = by_id[oid]
        if Counter(r["replicate"] for r in wells) != {"R1": 1, "R2": 1, "R3": 1, "R4": 1}:
            raise ValueError("Four distinct designed replicate roles required: " + oid)
        if len({r["plate_uid"] for r in wells}) != 4 or len({r["library_plate"] for r in wells}) != 1:
            raise ValueError("Ambiguous plate/layout mapping: " + oid)
        for row in wells:
            if (float(row["concentration_metadata_value"]) != 10
                    or row["concentration_unit_protocol"] != "uM"
                    or float(row["exposure_hours_protocol"]) != 24
                    or _true(row.get("metadata_protocol_dose_conflict", False))):
                raise ValueError("Unmatched protocol condition: " + oid)
        subjects.append(dict(object_id=oid, connectivity=identity[oid]["connectivity"],
                             layout=wells[0]["library_plate"]))
        for row in wells:
            role = ROLES[int(row["replicate"][1:]) - 1]
            if role in STAGE_ROLES[stage]:
                allowed.append(_well_record(row, oid, identity[oid]["connectivity"], role,
                                           "DEV_ANCHOR" if stage == "anchors" else "CONFIRMATION"))
    plates = {r["plate_uid"] for r in allowed}
    for row in local:
        if row["plate_uid"] not in plates or row["object_id_raw"] != "DMSO":
            continue
        if _true(row["in_external_identity_table"]) or _true(row.get("metadata_protocol_dose_conflict", False)):
            raise ValueError("Ambiguous DMSO control identity")
        allowed.append(_well_record(row, "DMSO", "", "CONTROL", "DMSO_CONTROL"))
    keys = [r["well_id"] for r in allowed]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate physical well in stage allowlist")
    counts = Counter(r["plate_uid"] for r in allowed if r["resource_kind"] == "DMSO_CONTROL")
    if set(counts) != plates or any(n < 8 for n in counts.values()):
        raise ValueError("Every selected plate needs at least eight declared DMSO controls")
    return subjects, sorted(allowed, key=lambda r: r["well_id"])


def _well_record(row, oid, group, role, kind):
    return dict(object_id=oid, connectivity=group, resource_kind=kind,
        well_id=row["plate_uid"] + "|" + row["well_position"], plate_uid=row["plate_uid"],
        site=row["site"], cell="HepG2", batch_id=row["batch_id"],
        library_plate=row["library_plate"], replicate=row["replicate"],
        measurement_role=role, well_position=row["well_position"])


def prepare_stage_plan(cohort_file, well_metadata_file, archive_inventory_file,
                       filename_mapping_file, output, *, site, stage,
                       control_space_path=None, identity_source=None, member_headers_file=None):
    """Create a new metadata-only plan; this function has no measurement access."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    subjects, allowed = stage_allowlist(read_rows(cohort_file), read_rows(well_metadata_file),
                                       site=site, stage=stage)
    if stage != "anchors" and control_space_path is None:
        raise ValueError("Confirmation stages require a pre-frozen control space")
    space = _json(control_space_path) if control_space_path else None
    if space and set(r["plate_uid"] for r in allowed) - set(space["centers"]):
        raise ValueError("Frozen control space does not cover every planned plate")
    inventory = _json(archive_inventory_file)["Aggregated_Profiles.zip"]
    member_by_name = {r["name"]: r for r in inventory}
    headers = {_m["name"]: _m for _m in _json(member_headers_file)} if member_headers_file else {}
    mappings = read_rows(filename_mapping_file)
    plate_rows = {r["plate_uid"]: r for r in allowed}
    members = []
    for plate, row in sorted(plate_rows.items()):
        matches = [r for r in mappings if r["site"] == site and r["cell_line"] == "HepG2"
                   and r["logical_plate"] == row["library_plate"] + "_" + row["replicate"]
                   and r["metadata_batch"] == row["batch_id"]
                   and r["recommended_use_status"] == "UNAMBIGUOUS_METADATA_JOIN"]
        if len(matches) != 1 or matches[0]["archive_member"] not in member_by_name:
            raise ValueError("Exactly one audited archive member required for " + plate)
        mapping = matches[0]
        member = dict(member_by_name[mapping["archive_member"]])
        if (member["crc32"] != int(mapping["crc32_from_zip_directory"])
                or member["uncompressed_size"] != int(mapping["uncompressed_bytes_from_zip_directory"])):
            raise ValueError("Archive directory and plate mapping disagree")
        member.update(plate_uid=plate, site=site, library_plate=row["library_plate"],
                      replicate=row["replicate"], archive_url=ARCHIVE_URL)
        if site in ("MEDINA", "USC"):
            # These sites encode condition and full logical plate, unlike FMP.
            # Identity tokens were checked on the first DEV-allowed row before
            # parsing any numeric value; every later row must match exactly.
            member.update(expected_csv_batch="HepG2_10uM",
                          expected_csv_plate=row["library_plate"] + "_" + row["replicate"])
        if member["name"] in headers:
            member["columns"] = headers[member["name"]]["columns"]
        members.append(member)
    plan = dict(schema=SCHEMA, status="PROVISIONAL_METADATA_ONLY", site=site, stage=stage, n=len(subjects),
        roles=list(STAGE_ROLES[stage]), cohort=subjects,
        cohort_file=str(Path(cohort_file).resolve()),
        identity_source=str(Path(identity_source).resolve()) if identity_source else None,
        authorized_wells=len(allowed), resource_counts=dict(Counter(r["resource_kind"] for r in allowed)),
        members=members, frozen_control_space=space is not None,
        measurements_opened=False, full_population_preserved=True,
        source_archive=ARCHIVE_URL)
    output.mkdir(parents=True)
    _write_csv(output / "measurement_allowlist.csv", allowed)
    _write_json(output / "plan.json", plan)
    if space is not None:
        _write_json(output / "control_space.json", space)
    return plan


def check_stage_authorization(plan_file, authorization_file, authorization_key):
    """Check stage-specific file gates before any measurement or cache is read."""
    plan_file = Path(plan_file).resolve()
    plan, auth = _json(plan_file), _json(authorization_file)
    if plan.get("schema") != SCHEMA or auth.get("schema") != "opal-r4-stage-authorizations-v1":
        raise ValueError("Unknown stage/authorization schema")
    entry = auth.get("stages", {}).get(authorization_key, {})
    if entry.get("authorized") is not True or not auth.get("user_authorization"):
        raise PermissionError("This stage has not been authorized")
    if (Path(entry.get("plan_file", "")).resolve() != plan_file or entry.get("site") != plan["site"]):
        raise PermissionError("Authorization does not identify this exact stage plan")
    if plan["stage"] != "anchors":
        if _json(entry["freeze_file"]).get("status") != "FROZEN":
            raise PermissionError("Protocol has not been frozen")
        model = _json(entry["model_complete_file"])
        if not (model.get("complete") is True or model.get("status") == "COMPLETE"
                or model.get("state") == "COMPLETE"):
            raise PermissionError("Final DEV model has not completed")
    if plan["stage"] in ("outcomes", "external_outcomes"):
        if _json(entry["selections_file"]).get("status") != "FROZEN":
            raise PermissionError("Prediction/selection lists have not been frozen")
    return plan


def _archive_chunks(member, *, range_get=None):
    """Inflate one audited member in bounded buffers without saving its body."""
    if (member.get("site") not in SITES or member.get("archive_url") != ARCHIVE_URL
            or member.get("compression") != 8
            or not re.fullmatch(r"Aggregated_Profiles/aggregated_data/"
                                + member["site"] + r"_HepG2/[^/]+\.csv", member["name"])):
        raise ValueError("Unsupported archive member")
    get = range_get or _range_get
    offset = int(member["local_offset"])
    header = struct.unpack("<4s5H3L2H", get(offset, offset + 29))
    if header[0] != b"PK\x03\x04" or header[3] != 8 or header[2] & 1:
        raise ValueError("Unsupported local ZIP header")
    nlen, elen = header[-2:]
    if get(offset + 30, offset + 29 + nlen).decode("utf-8") != member["name"]:
        raise ValueError("ZIP member identity differs from the plan")
    start, size = offset + 30 + nlen + elen, int(member["compressed_size"])
    compressed = (_range_stream(start, start + size - 1) if range_get is None else
                  (get(start + i, start + min(i + 1_048_576, size) - 1)
                   for i in range(0, size, 1_048_576)))
    decoder, crc, count = zlib.decompressobj(-15), 0, 0
    for pending in compressed:
        while pending:
            raw = decoder.decompress(pending, max_length=65_536)
            pending = decoder.unconsumed_tail
            if raw:
                crc, count = zlib.crc32(raw, crc), count + len(raw)
                yield raw
    if not decoder.eof or decoder.unused_data or count != member["uncompressed_size"] or crc != member["crc32"]:
        raise ValueError("ZIP member length/CRC mismatch")


def _header_and_chunks(chunks):
    iterator, prefix = iter(chunks), bytearray()
    for chunk in iterator:
        end = chunk.find(b"\n")
        if end < 0:
            prefix.extend(chunk)
        else:
            prefix.extend(chunk[:end + 1])
            columns = next(csv.reader(io.StringIO(prefix.decode("utf-8-sig"))))
            if columns[:4] != list(IDENTITY_PREFIX) + ["Metadata_Object_Count"]:
                raise ValueError("Unexpected identity-first profile schema")
            def replay():
                yield bytes(prefix)
                yield chunk[end + 1:]
                yield from iterator
            return columns, replay()
        if len(prefix) > 1_000_000:
            raise ValueError("Profile header exceeds limit")
    raise ValueError("Missing profile header")


def read_member_values(member, allowed, *, expected_features=None, chunks=None):
    """Numerically parse only physical wells admitted by the byte-level filter."""
    if not allowed or {r["plate_uid"] for r in allowed} != {member["plate_uid"]}:
        raise ValueError("Member allowlist must identify its exact physical plate")
    remote = chunks is None
    columns, chunks = _header_and_chunks(_archive_chunks(member) if remote else chunks)
    if member.get("columns") is not None and columns != member["columns"]:
        raise ValueError("Profile schema differs from the saved audited header")
    if expected_features is not None and columns[4:] != list(expected_features):
        raise ValueError("Profile features differ from the frozen measurement space")
    wells = [r["well_position"] for r in allowed]
    if len(wells) != len(set(wells)):
        raise ValueError("Duplicate physical well")
    lookup = {well: i for i, well in enumerate(wells)}
    values = np.full((len(wells), len(columns) - 4), np.nan)
    counts, present = np.full(len(wells), np.nan), np.zeros(len(wells), bool)
    stats = SelectionStats()
    stats.transport_bytes = int(member["compressed_size"]) if remote else 0
    tokens = set()
    for row in iter_selected_csv_rows(chunks, allowed_wells=wells, expected_header=columns,
                                     stats=stats, require_all_allowed=False):
        tokens.add((row["Metadata_Batch"], row["Metadata_Plate"]))
        expected_tokens = (member.get("expected_csv_batch", member["replicate"]),
                           member.get("expected_csv_plate", member["library_plate"]))
        if (row["Metadata_Batch"], row["Metadata_Plate"]) != expected_tokens:
            raise ValueError("Allowed row's internal plate identity differs from audited mapping: "
                             + repr((row["Metadata_Batch"], row["Metadata_Plate"]))
                             + "; expected " + repr(expected_tokens))
        index = lookup[row["Metadata_Well"]]
        def number(value):
            return float(value) if value.strip() else np.nan
        values[index] = [number(row[name]) for name in columns[4:]]
        counts[index], present[index] = number(row["Metadata_Object_Count"]), True
    return dict(values=values, cell_count=counts, present=present,
                feature_names=np.asarray(columns[4:], str), well_ids=np.asarray([r["well_id"] for r in allowed], str),
                audit=dict(**asdict(stats), internal_identity_tokens=sorted(tokens),
                           missing_allowed_wells=[r["well_id"] for r, ok in zip(allowed, present) if not ok],
                           excluded_measurements_decoded=False, unfiltered_member_saved=False))


def transform_preserving_missing(values, plate_ids, present, space):
    """Apply a frozen space, preserving invalid/missing rows and population N."""
    values, plates, present = np.asarray(values, float), np.asarray(plate_ids, str), np.asarray(present, bool)
    if values.shape[0] != len(plates) or present.shape != plates.shape:
        raise ValueError("Values, physical plates and presence mask must align")
    transformed, audit = apply_control_space(values, plates, space,
                                            required_wells=np.zeros(len(plates), bool))
    fraction = np.asarray(audit["nonfinite_filled_count"]) / transformed.shape[1]
    valid = present & (fraction <= space["maximum_nonfinite_fraction"])
    valid &= np.linalg.norm(transformed, axis=1) > 0
    transformed[~valid] = np.nan
    return transformed, valid, audit


def assemble_stage(plan, allowed, values, cell_count, present, space, *, chemistry=None):
    """Create either an X-only predictor file or a separate future-outcome file."""
    ids = np.asarray([r["object_id"] for r in plan["cohort"]], str)
    if len(ids) != plan["n"] or len(ids) != len(set(ids)):
        raise ValueError("Frozen population is inconsistent")
    roles = STAGE_ROLES[plan["stage"]]
    transformed, valid_rows, audit = transform_preserving_missing(
        values, [r["plate_uid"] for r in allowed], present, space)
    shape = (len(ids), len(roles))
    profiles = np.full(shape + (transformed.shape[1],), np.nan)
    counts = np.full(shape, np.nan)
    observed, valid = np.zeros(shape, bool), np.zeros(shape, bool)
    well_ids = np.full(shape, "", dtype="<U200")
    lookup = {oid: i for i, oid in enumerate(ids)}
    for row_index, row in enumerate(allowed):
        if row["resource_kind"] == "DMSO_CONTROL":
            continue
        i, j = lookup[row["object_id"]], roles.index(row["measurement_role"])
        if well_ids[i, j]:
            raise ValueError("Duplicate object-role assignment")
        profiles[i, j], counts[i, j] = transformed[row_index], cell_count[row_index]
        observed[i, j], valid[i, j] = present[row_index], valid_rows[row_index]
        well_ids[i, j] = row["well_id"]
    if np.any(well_ids == ""):
        raise ValueError("Metadata plan does not contain every population role")
    result = dict(ids=ids, groups=np.asarray([r["connectivity"] for r in plan["cohort"]], str),
                  layout=np.asarray([r["layout"] for r in plan["cohort"]], str))
    if plan["stage"] == "x":
        if chemistry is None:
            raise ValueError("X export requires frozen-metadata chemistry")
        chem, mask = chemistry
        if np.asarray(chem).shape != (len(ids), 513) or np.asarray(mask).shape != (len(ids),):
            raise ValueError("Chemistry must align with every frozen candidate")
        result.update(X=profiles[:, 0], chem=chem, chem_mask=mask,
                      cell_count=counts[:, 0], well_ids=well_ids[:, 0],
                      x_present=observed[:, 0], x_valid=valid[:, 0], eligible=valid[:, 0])
    else:
        result.update(**({"future": profiles} if plan["stage"] == "outcomes" else {"Y": profiles}),
                      cell_count=counts, well_ids=well_ids, present=observed,
                      valid=valid, role_order=np.asarray(roles, str))
    return result, dict(n_population=len(ids), n_valid_objects=int(valid.all(axis=1).sum()),
        n_present_wells=int(observed.sum()), n_valid_wells=int(valid.sum()),
        missing_wells=int((~observed).sum()), invalid_present_wells=int((observed & ~valid).sum()),
        population_removed=0, features_selected_using_confirmation=False,
        feature_count=len(space["feature_names"]), assay_transform=audit)


def export_stage(plan_file, authorization_file, authorization_key, output, *, member_reader=None):
    """Execute one authorized stage with resumable allowed-only member caches."""
    plan_file, output = Path(plan_file).resolve(), Path(output)
    plan = check_stage_authorization(plan_file, authorization_file, authorization_key)
    if (output / "complete.json").exists():
        raise FileExistsError("Stage already completed: " + str(output))
    allowed = read_rows(plan_file.parent / "measurement_allowlist.csv")
    if len(allowed) != plan["authorized_wells"] or len({r["well_id"] for r in allowed}) != len(allowed):
        raise ValueError("Stage physical-well allowlist changed")
    expected_subjects = {(r["object_id"], role) for r in plan["cohort"] for role in plan["roles"]}
    actual_subjects = {(r["object_id"], r["measurement_role"]) for r in allowed
                       if r["resource_kind"] != "DMSO_CONTROL"}
    if actual_subjects != expected_subjects:
        raise ValueError("Stage object/role allowlist differs from frozen cohort")
    if any(r["site"] != plan["site"] or r["cell"] != "HepG2" for r in allowed):
        raise ValueError("Stage site or cell changed")
    space = _json(plan_file.parent / "control_space.json") if plan["frozen_control_space"] else None
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "allowed_members"
    cache.mkdir(exist_ok=True)
    source = member_reader or read_member_values
    records, ordered_rows, audits = [], [], []
    features = None if space is None else space["source_feature_names"]
    for member in plan["members"]:
        rows = [r for r in allowed if r["plate_uid"] == member["plate_uid"]]
        dest = cache / (member["library_plate"] + "_" + member["replicate"] + ".npz")
        report_path = dest.with_suffix(".json")
        if dest.exists() and report_path.exists():
            report = _json(report_path)
            if (report["source_member"] != member or report["plan_file"] != str(plan_file)
                    or report["site"] != plan["site"] or report["stage"] != plan["stage"]):
                raise ValueError("Saved allowed-only member belongs to another frozen plan")
            with np.load(dest, allow_pickle=False) as saved:
                record = {key: saved[key] for key in saved.files}
            record["audit"] = report["audit"]
        else:
            record = source(member, rows, expected_features=features)
            temporary = dest.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **{k: v for k, v in record.items() if k != "audit"})
            temporary.replace(dest)
            _write_json(report_path, dict(plan_file=str(plan_file), site=plan["site"],
                stage=plan["stage"], source_member=member, audit=record["audit"]))
        if list(record["well_ids"]) != [r["well_id"] for r in rows]:
            raise ValueError("Saved member physical-well order differs from stage")
        current_features = list(record["feature_names"])
        if features is None:
            features = current_features
        if current_features != features:
            raise ValueError("Feature schemas differ across selected plates")
        records.append(record)
        ordered_rows.extend(rows)
        audits.append(dict(plate_uid=member["plate_uid"], **record["audit"]))
        print(f"STAGE {plan['site']} {plan['stage']} {member['library_plate']} {member['replicate']}: "
              f"{int(record['present'].sum())}/{len(rows)} authorized wells present", flush=True)
    values = np.concatenate([r["values"] for r in records])
    counts = np.concatenate([r["cell_count"] for r in records])
    present = np.concatenate([r["present"] for r in records])
    if space is None:
        if plan["stage"] != "anchors":
            raise ValueError("Only external DEV anchors may fit a new control space")
        dmso = np.asarray([r["resource_kind"] == "DMSO_CONTROL" for r in ordered_rows])
        if not present[dmso].all():
            raise ValueError("Planned external DMSO controls are missing; freeze cannot be completed")
        space = fit_control_space(values, [r["plate_uid"] for r in ordered_rows], dmso, features)
    chemistry = (chemistry_for([r["object_id"] for r in plan["cohort"]], plan["identity_source"])
                 if plan["stage"] == "x" else None)
    data, report = assemble_stage(plan, ordered_rows, values, counts, present, space, chemistry=chemistry)
    name = {"x": "query.npz", "outcomes": "outcomes.npz", "anchors": "anchors.npz",
            "external_outcomes": "outcomes.npz"}[plan["stage"]]
    temporary = output / (name + ".partial.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(output / name)
    _write_json(output / "control_space.json", space)
    _write_json(output / "audit.json", dict(**report, site=plan["site"], stage=plan["stage"],
        source_members=audits, full_archive_saved=False, excluded_measurements_decoded=False,
        future_measurements_in_predictor_file=False, frozen_control_space_reused=plan["frozen_control_space"]))
    complete = dict(complete=True, status="COMPLETE", plan_file=str(plan_file), stage=plan["stage"],
                    site=plan["site"], n=plan["n"], data_file=name,
                    n_valid_objects=report["n_valid_objects"])
    _write_json(output / "complete.json", complete)
    return complete
