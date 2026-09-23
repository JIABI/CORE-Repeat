"""Metadata audit unit fixtures are not biological evidence."""
import json
import gzip

import numpy as np
import pytest

from opal2.biology_coverage import (attach_annotations, attach_existing_jump_annotations, attach_protocol_duration,
                                  audit, count_schema_audit, coverage, export_typed_audit, field, present)


def exported_fixture(tmp_path):
    ids = np.array([f"id_{i}" for i in range(639)])
    np.savez(tmp_path / "measurements.npz", ids=ids,
             well_ids=np.array([[f"{i}_{j}" for j in range(4)] for i in range(639)]),
             chem_mask=np.ones(639, bool), n_cells_mask=np.zeros((639, 4), bool),
             # Invalid pickle object proves that the audit never accesses Y.
             Y=np.array([{"forbidden_outcome": True}], dtype=object))
    (tmp_path / "measurements.json").write_text(json.dumps({"metadata": {
        "scope": "639_DEV_FOUR_ROLES_ONLY", "smiles": ["CCO"] * 639,
        "nominal_concentration_uM": 10.0, "concentration_status": "protocol only"}}))


def test_reads_only_opened_metadata_not_outcomes(tmp_path):
    exported_fixture(tmp_path)
    report, records = audit(tmp_path)
    assert report["model_input_coverage"]["smiles"]["available"] == 639
    assert report["model_input_coverage"]["dose"]["status_counts"] == {"protocol_nominal": 639}
    assert report["model_input_coverage"]["target"]["unknown"] == 639
    assert "Y" not in report["provenance"]["npz_metadata_members_read"]
    assert records[0]["model_inputs"]["dose"]["unit"] == "uM"


def test_unknown_is_not_biological_absence():
    assert not present("unknown") and not present([])
    assert field([])["status"] == "unknown"
    with pytest.raises(ValueError, match="evidence source"):
        field(status="known_absent")
    annotation = field(status="known_absent", source="explicit validated report")
    values = coverage([{"model_inputs": {"target": annotation}}], "model_inputs")["target"]
    assert values["known_absent"] == 1 and values["available"] == 0


def test_annotation_join_retains_evidence_without_changing_model_inputs(tmp_path):
    exported_fixture(tmp_path)
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"source": "curated resource release X", "records": [
        {"compound_id": "id_0", "fields": {
            "target": {"value": ["GENE1"], "evidence": "experimental"},
            "action_direction": {"value": "inhibitor"}}}]}))
    report, records = audit(tmp_path, [path])
    assert report["joinable_annotation_coverage"]["target"]["available"] == 1
    assert records[0]["joinable_annotations"]["target"]["evidence"] == "experimental"
    assert records[0]["model_inputs"]["target"]["status"] == "unknown"


def test_units_and_duplicate_annotations_are_validated(tmp_path):
    records = [{"compound_id": "id", "joinable_annotations": {}}]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({"source": "release X", "records": [
        {"compound_id": "id", "fields": {"duration": {"value": 48}}}]}))
    with pytest.raises(ValueError, match="explicit unit"):
        attach_annotations(records, path)
    path.write_text(json.dumps({"source": "release X", "records": [
        {"compound_id": "id", "fields": {}}, {"compound_id": "id", "fields": {}}]}))
    with pytest.raises(ValueError, match="Duplicate"):
        attach_annotations(records, path)


def test_refuses_unscoped_allocation(tmp_path):
    (tmp_path / "measurements.json").write_text(json.dumps({"metadata": {"scope": "FINAL"}}))
    with pytest.raises(ValueError, match="opened four-role DEV"):
        audit(tmp_path)


def test_exact_identity_join_keeps_direction_unassigned(tmp_path):
    records = [{"compound_id": "id", "joinable_annotations": {}}]
    sources = [{"name": name, "url": "https://official.invalid/" + name,
                "local_file": name + ".tsv"}
               for name in ("hub_drug", "hub_sample", "jump_moa", "jump_target2")]
    (tmp_path / "resources.json").write_text(json.dumps(sources))
    compound_file = tmp_path / "compound.csv.gz"
    with gzip.open(compound_file, "wt") as stream:
        stream.write("Metadata_JCP2022,Metadata_InChIKey\nid,AAAAAAAAAAAAAA-BBBBBBBBBB-C\n")
    (tmp_path / "hub_drug.tsv").write_text("!comment\npert_iname\ttarget\tmoa\nnocodazole\tHPGDS\ttubulin polymerization inhibitor\n")
    (tmp_path / "hub_sample.tsv").write_text("pert_iname\tInChIKey\nnocodazole\tAAAAAAAAAAAAAA-BBBBBBBBBB-C\n")
    for name in ("jump_moa", "jump_target2"):
        (tmp_path / (name + ".tsv")).write_text("pert_iname\tInChIKey\ttarget_list\tmoa\nother\tAAAAAAAAAAAAAA-DDDDDDDDDD-C\tWRONG\tother agonist\n")
    result = attach_existing_jump_annotations(records, tmp_path, compound_file)
    assert result["exact_target_annotations"] == 1
    assert result["target_moa_alignment_requires_review"] == ["id"]
    assert result["explicit_target_direction_edges"] == 0
    assert records[0]["joinable_annotations"]["target"]["value"] == ["HPGDS"]
    assert "action_direction" not in records[0]["joinable_annotations"]
    assert records[0]["moa_direction_is_target_linked"] is False


def test_child_count_is_not_cell_count(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps({"columns": ["Cells_Children_Cytoplasm_Count", "Nuclei_Number_Object_Number"]}))
    report = count_schema_audit(path)
    assert report["recognized_cell_count_columns"] == []
    assert report["other_count_named_columns"] == ["Cells_Children_Cytoplasm_Count"]


def test_typed_export_is_audit_only_with_unknown_direction(tmp_path):
    from opal2.biology import load_biology
    record = {"compound_id": "JCP_ID", "well_ids": ["source_5::plate1::A01"],
              "model_inputs": {"smiles": {"value": "CCO"}, "dose": {"value": 10.0}},
              "joinable_annotations": {"target": {"value": ["GENE1"]}, "moa": {"value": ["inhibitor"]},
                  "cell_background": {"status": "protocol_nominal", "source": "https://official.invalid/protocol",
                                      "value": {"cell_line": "U2OS", "species": "Homo sapiens"}}},
              "public_annotation_evidence": [{"target": "GENE1", "mechanism_source": "https://official.invalid/annotation"}]}
    src = tmp_path / "objects.json"
    src.write_text(json.dumps([record]))
    plate = tmp_path / "plate.csv.gz"
    with gzip.open(plate, "wt") as stream:
        stream.write("Metadata_Source,Metadata_Plate,Metadata_Batch\nsource_5,plate1,batch1\n")
    out = tmp_path / "typed.json"
    summary = export_typed_audit(src, out, plate)
    assert summary["records"] == 1 and summary["active_input_relations"] == 0
    restored = load_biology(out)[0]
    assert restored.relations[0].role == "audit"
    assert restored.relations[0].confidence is None and restored.relations[0].direction == "unknown"
    assert restored.perturbation["target_entities"].status == "unknown"
    assert restored.perturbation["reported_moa_1"].role == "audit"
    assert restored.biological_context["cell_line"].interpretation == "nominal_protocol"
    assert restored.measurement_metadata["source_5::plate1::A01"]["cell_count"].status == "unknown"
    assert restored.measurement_metadata["source_5::plate1::A01"]["batch"].value == "batch1"


def test_nominal_duration_is_separate_from_execution(tmp_path):
    from opal2.biology import load_biology
    records = [{"compound_id": "id", "well_ids": ["source_5::plate1::A01"],
                "model_inputs": {"smiles": {"value": "CCO"}, "dose": {"value": 10.0}},
                "joinable_annotations": {}}]
    attach_protocol_duration(records, 48, "https://official.invalid/paper")
    source = tmp_path / "audit.json"
    source.write_text(json.dumps(records))
    output = tmp_path / "typed.json"
    export_typed_audit(source, output)
    restored = load_biology(output)[0]
    assert restored.perturbation["duration"].value == 48
    assert restored.perturbation["duration"].interpretation == "nominal_protocol"
    assert restored.perturbation["executed_duration"].status == "unknown"
