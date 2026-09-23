import io
import json
import zipfile
from collections import Counter

import numpy as np
import pytest

from opal2.eu_assay_preprocessing import fit_control_space
from opal2.r4_staged_data import (
    ARCHIVE_URL, ROLES, SCHEMA, _archive_chunks, assemble_stage, check_stage_authorization,
    export_stage, read_member_values, stage_allowlist, transform_preserving_missing,
)


def design(site="FMP"):
    cohort = [dict(object_id="a", connectivity="group-a"), dict(object_id="b", connectivity="group-b")]
    rows = []
    for replicate in range(1, 5):
        for oid, well in [("a", "A01"), ("b", "A02")] + [("DMSO", f"B{i:02}") for i in range(1, 9)]:
            rows.append(dict(site=site, cell_line_protocol="HepG2", object_id_raw=oid,
                replicate=f"R{replicate}", plate_uid=f"{site}|batch{replicate}|B1001_R{replicate}",
                library_plate="B1001", batch_id=f"batch{replicate}", well_position=well,
                concentration_metadata_value="10", concentration_unit_protocol="uM",
                exposure_hours_protocol="24", metadata_protocol_dose_conflict="False",
                in_external_identity_table=str(oid != "DMSO")))
    return cohort, rows


def test_identity_only_stage_plan_never_admits_future_roles_into_x():
    cohort, metadata = design()
    subjects, x = stage_allowlist(cohort, metadata, site="FMP", stage="x")
    assert len(subjects) == 2 and len(x) == 10
    assert {r["measurement_role"] for r in x} == {"X", "CONTROL"}
    assert {r["replicate"] for r in x} == {"R1"}
    _, future = stage_allowlist(cohort, metadata, site="FMP", stage="outcomes")
    assert len(future) == 30 and "X" not in {r["measurement_role"] for r in future}
    with pytest.raises(ValueError, match="distinct designed"):
        stage_allowlist(cohort, metadata[:-1] + [metadata[0]], site="FMP", stage="x")


def test_unauthorized_outcome_stage_fails_before_any_measurement_access(tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(dict(schema=SCHEMA, site="FMP", stage="outcomes")))
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(dict(schema="opal-r4-stage-authorizations-v1",
        user_authorization="test", stages={"outcomes": dict(authorized=False, site="FMP", plan_file=str(plan_file))})))
    with pytest.raises(PermissionError, match="not been authorized"):
        export_stage(plan_file, authorization, "outcomes", tmp_path / "output",
                     member_reader=lambda *a, **k: pytest.fail("accessed a source"))
    assert not (tmp_path / "output").exists()


def test_outcomes_require_frozen_selections_after_completed_model(tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in ("plan", "auth", "freeze", "model", "selections")}
    paths["plan"].write_text(json.dumps(dict(schema=SCHEMA, site="FMP", stage="outcomes")))
    paths["freeze"].write_text('{"status":"FROZEN"}')
    paths["model"].write_text('{"complete":true}')
    paths["selections"].write_text('{"status":"PENDING"}')
    entry = dict(authorized=True, site="FMP", plan_file=str(paths["plan"]),
        freeze_file=str(paths["freeze"]), model_complete_file=str(paths["model"]),
        selections_file=str(paths["selections"]))
    paths["auth"].write_text(json.dumps(dict(schema="opal-r4-stage-authorizations-v1",
        user_authorization="test", stages={"outcomes": entry})))
    with pytest.raises(PermissionError, match="selection lists"):
        check_stage_authorization(paths["plan"], paths["auth"], "outcomes")
    paths["selections"].write_text('{"status":"FROZEN"}')
    assert check_stage_authorization(paths["plan"], paths["auth"], "outcomes")["stage"] == "outcomes"


def test_excluded_non_utf8_values_not_decoded_and_missing_authorized_well_retained():
    _, metadata = design()
    _, allowed = stage_allowlist([dict(object_id="a", connectivity="g")], metadata, site="FMP", stage="x")
    member = dict(plate_uid=allowed[0]["plate_uid"], replicate="R1", library_plate="B1001")
    header = b"Metadata_Batch,Metadata_Plate,Metadata_Well,Metadata_Object_Count,Nuc_Value\n"
    raw = header + b"R1,B1001,A02,\xff,\x80\nR1,B1001,A01,200,3.5\n"
    record = read_member_values(member, allowed, chunks=(raw[i:i + 11] for i in range(0, len(raw), 11)))
    assert record["present"].sum() == 1
    assert record["values"][0, 0] == 3.5
    assert np.isnan(record["values"][1:]).all()
    assert record["audit"]["decoded_measurement_rows"] == 1
    assert record["audit"]["excluded_records"] == 1


def test_frozen_transform_keeps_feature_order_and_missing_population():
    rng = np.random.default_rng(5)
    names = [f"Nuc_Intensity_{i}" for i in range(10)]
    controls = rng.normal(size=(8, 10))
    space = fit_control_space(controls, ["p"] * 8, np.ones(8, bool), names)
    space_before = json.dumps(space, sort_keys=True)
    values = rng.normal(size=(3, 10))
    values[1] = np.nan
    values[2, :2] = np.nan
    actual, valid, _ = transform_preserving_missing(values, ["p"] * 3, [True, False, True], space)
    assert actual.shape == (3, 10) and valid.tolist() == [True, False, False]
    assert np.isnan(actual[1:]).all()
    assert json.dumps(space, sort_keys=True) == space_before


def test_x_artifact_contains_all_candidates_but_no_future_arrays():
    cohort, metadata = design()
    subjects, allowed = stage_allowlist(cohort, metadata, site="FMP", stage="x")
    rng = np.random.default_rng(9)
    values = rng.normal(size=(len(allowed), 10))
    present = np.ones(len(allowed), bool)
    present[1], values[1] = False, np.nan
    dmso = np.array([r["resource_kind"] == "DMSO_CONTROL" for r in allowed])
    space = fit_control_space(values, [r["plate_uid"] for r in allowed], dmso,
                              [f"Nuc_Intensity_{i}" for i in range(10)])
    plan = dict(stage="x", cohort=subjects, n=2)
    data, report = assemble_stage(plan, allowed, values, np.ones(len(allowed)) * 20, present, space,
                                 chemistry=(np.ones((2, 513)), np.ones(2, bool)))
    assert data["ids"].tolist() == ["a", "b"]
    assert data["eligible"].tolist() == [True, False]
    assert np.isnan(data["X"][1]).all()
    assert "Y" not in data and "future" not in data and "role_order" not in data
    assert report["n_population"] == 2 and report["population_removed"] == 0


def test_external_anchors_use_only_the_exact_dev_cohort_and_controls():
    cohort, metadata = design("MEDINA")
    _, allowed = stage_allowlist(cohort[:1], metadata, site="MEDINA", stage="anchors")
    assert Counter(r["resource_kind"] for r in allowed) == {"DEV_ANCHOR": 4, "DMSO_CONTROL": 32}
    assert all(r["object_id"] in ("a", "DMSO") for r in allowed)


def test_external_range_reader_checks_actual_member_and_crc_without_whole_archive():
    name = "Aggregated_Profiles/aggregated_data/MEDINA_HepG2/plate.csv"
    payload = b"IdentityHeader\n" + b"x" * 200_000
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    data = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        item = archive.getinfo(name)
    member = dict(name=name, site="MEDINA", archive_url=ARCHIVE_URL,
        compression=item.compress_type, local_offset=item.header_offset,
        compressed_size=item.compress_size, uncompressed_size=item.file_size, crc32=item.CRC)
    requests = []
    def get(start, end):
        requests.append((start, end))
        assert (start, end) != (0, len(data) - 1)
        return data[start:end + 1]
    chunks = list(_archive_chunks(member, range_get=get))
    assert b"".join(chunks) == payload and max(map(len, chunks)) <= 65_536
    member["crc32"] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        list(_archive_chunks(member, range_get=get))


def test_authorized_x_export_saves_only_predictor_arrays_and_keeps_missing_id(tmp_path):
    from opal2.r4_staged_data import _write_csv, _write_json
    cohort, metadata = design()
    subjects, allowed = stage_allowlist(cohort, metadata, site="FMP", stage="x")
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    names = [f"Nuc_Intensity_{i}" for i in range(10)]
    rng = np.random.default_rng(11)
    values = rng.normal(size=(len(allowed), len(names)))
    present = np.ones(len(allowed), bool)
    values[1], present[1] = np.nan, False
    dmso = np.array([r["resource_kind"] == "DMSO_CONTROL" for r in allowed])
    space = fit_control_space(values, [r["plate_uid"] for r in allowed], dmso, names)
    chemistry = tmp_path / "chemistry.csv"
    chemistry.write_text("object_id,smiles\na,CCO\nb,CCN\n")
    member = dict(plate_uid=allowed[0]["plate_uid"], library_plate="B1001", replicate="R1")
    plan = dict(schema=SCHEMA, stage="x", site="FMP", n=2, cohort=subjects,
                roles=["X"], authorized_wells=len(allowed), members=[member],
                identity_source=str(chemistry), frozen_control_space=True)
    _write_json(plan_dir / "plan.json", plan)
    _write_json(plan_dir / "control_space.json", space)
    _write_csv(plan_dir / "measurement_allowlist.csv", allowed)
    _write_json(tmp_path / "freeze.json", dict(status="FROZEN"))
    _write_json(tmp_path / "model.json", dict(complete=True))
    _write_json(tmp_path / "auth.json", dict(schema="opal-r4-stage-authorizations-v1",
        user_authorization="test", stages={"x": dict(authorized=True, site="FMP",
        plan_file=str(plan_dir / "plan.json"), freeze_file=str(tmp_path / "freeze.json"),
        model_complete_file=str(tmp_path / "model.json"))}))
    def source(member, rows, expected_features):
        assert expected_features == names
        return dict(values=values, present=present, cell_count=np.ones(len(rows)) * 100,
            feature_names=np.asarray(names), well_ids=np.asarray([r["well_id"] for r in rows]),
            audit=dict(decoded_measurement_rows=int(present.sum())))
    result = export_stage(plan_dir / "plan.json", tmp_path / "auth.json", "x", tmp_path / "out",
                          member_reader=source)
    assert result["n"] == 2 and result["n_valid_objects"] == 1
    with np.load(tmp_path / "out/query.npz", allow_pickle=False) as data:
        assert data["ids"].tolist() == ["a", "b"]
        assert data["eligible"].tolist() == [True, False]
        assert "Y" not in data.files and "future" not in data.files
    assert json.loads((tmp_path / "out/control_space.json").read_text()) == space
