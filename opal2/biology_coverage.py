"""Metadata-only biological coverage audit for an explicitly supplied DEV export.

This reader never loads Y, profiles, latent representations, or acquisition labels.
External annotations are optional explicit JSON sidecars, not inferred from SMILES.
The output separates what the present model receives from joinable annotations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("smiles", "target", "moa", "action_direction", "pathway", "dose",
          "duration", "cell_background", "cell_count")
MISSING_TEXT = {"", "nan", "none", "null", "unknown", "n/a", "na"}


def present(value: Any) -> bool:
    """Empty/unknown annotations are not evidence of biological absence."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in MISSING_TEXT
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def field(value=None, *, source=None, unit=None, status=None, evidence=None):
    if status is None:
        status = "reported" if present(value) else "unknown"
    if status == "known_absent" and not source:
        raise ValueError("Known absence requires an explicit evidence source")
    if status in {"reported", "protocol_nominal"} and not present(value):
        raise ValueError("Available annotation needs a value")
    return {"value": value, "status": status, "unit": unit,
            "source": source, "evidence": evidence}


def read_metadata_export(directory: Path):
    directory = Path(directory)
    sidecar = directory / "measurements.json"
    document = json.loads(sidecar.read_text())
    meta = document["metadata"]
    if meta.get("scope") != "639_DEV_FOUR_ROLES_ONLY":
        raise ValueError("Expected the explicitly opened four-role DEV export")
    accessed = ["ids", "well_ids", "chem_mask", "n_cells_mask"]
    with np.load(directory / "measurements.npz", allow_pickle=False) as arrays:
        ids = arrays["ids"].astype(str).tolist()
        well_ids = arrays["well_ids"].astype(str)
        chemical_mask = arrays["chem_mask"].astype(bool)
        count_mask = arrays["n_cells_mask"].astype(bool)
        if count_mask.any():
            # These are only supplied metadata, never recovered from images.
            counts = arrays["n_cells"]
            accessed.append("n_cells")
        else:
            counts = None
    if len(ids) != 639 or len(set(ids)) != 639 or well_ids.shape != (639, 4):
        raise ValueError("Expected 639 distinct opened compounds and four assigned wells")
    if chemical_mask.shape != (639,) or count_mask.shape != (639, 4):
        raise ValueError("Metadata mask dimensions do not match the opened allocation")
    smiles = meta.get("smiles", [])
    if len(smiles) != len(ids):
        raise ValueError("SMILES are not aligned with the exported DEV objects")
    records = []
    for i, compound in enumerate(ids):
        available = {key: field() for key in FIELDS}
        if chemical_mask[i]:
            if not present(smiles[i]):
                raise ValueError("Valid-structure mask cannot accompany missing SMILES")
            available["smiles"] = field(smiles[i], source=str(sidecar) + "#metadata.smiles",
                                         evidence="RDKit parser validity recorded by the dataset adapter")
        if present(meta.get("nominal_concentration_uM")):
            available["dose"] = field(meta["nominal_concentration_uM"], unit="uM",
                                       status="protocol_nominal", source=str(sidecar),
                                       evidence=meta.get("concentration_status"))
        if count_mask[i].any():
            available["cell_count"] = field(
                [{"well_id": str(well_ids[i, j]), "value": float(counts[i, j])}
                 for j in range(4) if count_mask[i, j]],
                unit="cells_per_well", source=str(directory / "measurements.npz"))
        records.append({"compound_id": compound, "well_ids": well_ids[i].tolist(),
                        "model_inputs": available, "joinable_annotations": {}})
    return records, {"sidecar": str(sidecar), "npz_metadata_members_read": accessed,
                     "cell_count_wells": int(count_mask.sum()), "well_count": int(count_mask.size),
                     "reference_kind": meta.get("reference_kind"),
                     "target2_reference_available": bool(meta.get("target2_reference_available", False)),
                     "chemical_validation": meta.get("chemical", {})}


def attach_annotations(records, sidecar: Path):
    """Join explicit local annotations by exact compound ID, with provenance.

    Format: {source: str, records: [{compound_id, fields: {target: {...}}}]}.
    Each field carries value/status/unit/source/evidence. Multiple target/pathway
    values belong in a list; duplicate compound rows are rejected to prevent a
    silent many-to-many expansion. No annotation is made a model input here.
    """
    payload = json.loads(Path(sidecar).read_text())
    source = payload.get("source")
    if not source:
        raise ValueError("Annotation sidecar requires source provenance")
    index = {item["compound_id"]: item for item in records}
    seen = set()
    joined = 0
    for item in payload.get("records", []):
        identity = item["compound_id"]
        if identity in seen:
            raise ValueError("Duplicate compound annotation row")
        seen.add(identity)
        if identity not in index:
            continue
        fields = item.get("fields", {})
        for name, entry in fields.items():
            if name not in FIELDS:
                raise ValueError(f"Unknown biological coverage field: {name}")
            if name in index[identity]["joinable_annotations"]:
                raise ValueError(f"Conflicting annotation sources for {identity}/{name}; reconcile explicitly")
            if not isinstance(entry, dict):
                raise ValueError("Annotation fields need explicit value/status/evidence objects")
            annotation = field(entry.get("value"), source=entry.get("source") or source,
                               status=entry.get("status"), unit=entry.get("unit"),
                               evidence=entry.get("evidence"))
            if name in {"dose", "duration", "cell_count"} and annotation["status"] != "unknown" and not annotation["unit"]:
                raise ValueError(f"{name} requires an explicit unit")
            index[identity]["joinable_annotations"][name] = annotation
        joined += 1
    return {"path": str(sidecar), "source": source, "joined_compounds": joined,
            "unmatched_annotation_rows": len(seen) - joined}


def attach_existing_jump_annotations(records, annotation_directory: Path, compound_metadata: Path):
    """Reconcile exact-identity local public annotations against existing DEV IDs.

    The old coverage table is not used as the count source. The original public
    Hub/JUMP tables are joined again, with full InChIKey equality. Connectivity
    matches, inferred targets, and MoA-to-gene direction guesses are not accepted.
    """
    directory = Path(annotation_directory)
    compound_metadata = Path(compound_metadata)
    indexed = {row["compound_id"]: row for row in records}
    query_keys = {}
    with gzip.open(compound_metadata, "rt") as stream:
        for row in csv.DictReader(stream):
            identity = row["Metadata_JCP2022"]
            if identity in indexed:
                if identity in query_keys:
                    raise ValueError("Duplicate DEV identity in compound metadata")
                query_keys[identity] = row["Metadata_InChIKey"]
    if set(query_keys) != set(indexed):
        raise ValueError("Public identity table does not cover the exact DEV allocation")
    key_to_ids = {}
    for identity, key in query_keys.items():
        key_to_ids.setdefault(key, []).append(identity)
    resources = json.loads((directory / "resources.json").read_text())
    sources = {item["name"]: item for item in resources}

    def rows(name):
        path = directory / sources[name]["local_file"]
        with path.open() as stream:
            yield from csv.DictReader((line for line in stream if not line.startswith("!")), delimiter="\t")

    drug_names = {}
    for row in rows("hub_drug"):
        drug_names.setdefault(row["pert_iname"], []).append(row)
    hits = {identity: [] for identity in indexed}
    for resource in ("hub_sample", "jump_target2", "jump_moa"):
        for row in rows(resource):
            identities = key_to_ids.get(row["InChIKey"], ())
            if not identities:
                continue
            annotations = drug_names.get(row["pert_iname"], [{}]) if resource == "hub_sample" else [row]
            for annotation in annotations:
                entry = {"resource": resource, "source": sources[resource]["url"],
                         "mechanism_source": sources["hub_drug"]["url"] if resource == "hub_sample" else sources[resource]["url"],
                         "identity_key": row["InChIKey"], "name": row["pert_iname"],
                         "target": annotation.get("target_list", annotation.get("target", "")),
                         "moa": annotation.get("moa", ""), "match": "full_InChIKey"}
                for identity in identities:
                    if entry not in hits[identity]:
                        hits[identity].append(entry)
    for identity, annotations in hits.items():
        if not annotations:
            continue
        record = indexed[identity]
        record["public_identity_key"] = query_keys[identity]
        record["public_annotation_evidence"] = annotations
        for key in ("target", "moa"):
            values = sorted({value.strip() for annotation in annotations
                             for value in annotation[key].split("|") if present(value)})
            if values:
                record["joinable_annotations"][key] = field(
                    values, source=sorted({annotation["mechanism_source"] for annotation in annotations}),
                    evidence={"match": "full_InChIKey", "curation": "as reported; per-edge experimental/GO evidence codes absent",
                              "not_independent_corroboration": "JUMP labels may derive from Hub",
                              "original_rows": annotations})
        # A MoA label such as 'tubulin polymerization inhibitor' does not identify
        # the sign of a separately listed HPGDS target edge. Preserve text only.
        direction_words = {word for row in annotations for word in row["moa"].lower().split()
                           if word in {"agonist", "antagonist", "inhibitor", "activator"}}
        if direction_words:
            record["moa_direction_text"] = sorted(direction_words)
            record["moa_direction_is_target_linked"] = False
    mismatched = [identity for identity, entries in hits.items()
                  if any(item["name"].lower() == "nocodazole" and "HPGDS" in item["target"]
                         and "tubulin" in item["moa"].lower() for item in entries)]
    label_support = {}
    for kind in ("target", "moa"):
        counts = Counter(label for record in records
                         for label in record["joinable_annotations"].get(kind, {}).get("value", []))
        label_support[kind] = {"distinct_labels": len(counts),
                               "labels_shared_by_two_or_more_compounds": sum(count >= 2 for count in counts.values()),
                               "compound_counts_by_label": dict(sorted(counts.items()))}
    return {"path": str(directory), "source": "Local public Hub 2025-08-18 and JUMP Target/MOA source tables",
            "identity_source": str(compound_metadata),
            "joined_compounds": sum(bool(entries) for entries in hits.values()),
            "exact_target_annotations": sum(any(present(entry["target"]) for entry in entries) for entries in hits.values()),
            "exact_moa_annotations": sum(any(present(entry["moa"]) for entry in entries) for entries in hits.values()),
            "moa_direction_word_coverage": sum("moa_direction_text" in row for row in records),
            "explicit_target_direction_edges": 0,
            "per_edge_experimental_or_GO_evidence_codes": 0,
            "target_moa_alignment_requires_review": mismatched,
            "label_support": label_support,
            "unmatched_annotation_rows": None,
            "resource_scope": "These four local source tables only; not an exhaustive database search",
            "license": "Preserve source attribution; Hub file header terms require review before redistributing the underlying tables"}


def attach_protocol_background(records, source: str, *, cell_line="U2OS", species="Homo sapiens"):
    if not source:
        raise ValueError("Protocol background needs an explicit source")
    for row in records:
        row["joinable_annotations"]["cell_background"] = field(
            {"cell_line": cell_line, "species": species}, source=source,
            status="protocol_nominal", evidence="Dataset protocol, not executed-well authentication or molecular baseline")


def attach_protocol_duration(records, hours: float, source: str):
    if not source or not np.isfinite(hours) or hours <= 0:
        raise ValueError("Protocol duration requires a positive hours value and explicit source")
    for row in records:
        row["joinable_annotations"]["duration"] = field(
            float(hours), unit="hour", source=source, status="protocol_nominal",
            evidence="Production compound Cell Painting protocol default; not an executed-well exposure timestamp")


def count_schema_audit(path: Path):
    document = json.loads(Path(path).read_text())
    columns = document["columns"]
    actual_counts = [name for name in columns if name in {
        "Image_Count_Cells", "Image_Count_Nuclei", "Metadata_Cell_Count", "Metadata_CellCount",
        "Metadata_Nuclei_Count", "Count_Cells", "Count_Nuclei", "n_cells"}]
    return {"path": str(path), "recorded_plate": document.get("plate"), "columns": len(columns),
            "recognized_cell_count_columns": actual_counts,
            "other_count_named_columns": [name for name in columns if "count" in name.lower() and name not in actual_counts],
            "interpretation": "Per-object children counts are not measured cells per well; no count is inferred from object numbers or morphology."}


def coverage(records, layer):
    summary = {}
    for name in FIELDS:
        counts = Counter(record[layer].get(name, field())["status"] for record in records)
        summary[name] = {"available": sum(value for status, value in counts.items()
                                          if status not in {"unknown", "known_absent"}),
                         "known_absent": counts.get("known_absent", 0),
                         "unknown": counts.get("unknown", 0),
                         "status_counts": dict(sorted(counts.items()))}
    return summary


def audit(directory: Path, annotation_sidecars=(), *, legacy_annotation_directory=None,
          compound_metadata=None, protocol_background_source=None, profile_schema=None,
          protocol_duration_hours=None, protocol_duration_source=None):
    records, provenance = read_metadata_export(directory)
    joins = [attach_annotations(records, path) for path in annotation_sidecars]
    if legacy_annotation_directory:
        if not compound_metadata:
            raise ValueError("Exact-identity annotation audit requires compound_metadata")
        joins.append(attach_existing_jump_annotations(records, legacy_annotation_directory, compound_metadata))
    if protocol_background_source:
        attach_protocol_background(records, protocol_background_source)
    if protocol_duration_hours is not None:
        attach_protocol_duration(records, protocol_duration_hours, protocol_duration_source)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "639 already-open source_5 DEV compounds, four original assigned roles",
              "n_compounds": len(records), "provenance": provenance,
              "external_annotation_joins": joins,
              "model_input_coverage": coverage(records, "model_inputs"),
              "joinable_annotation_coverage": coverage(records, "joinable_annotations"),
              "upstream_cell_count_schema": count_schema_audit(profile_schema) if profile_schema else None,
              "interpretation": [
                  "Unknown means not established by the audited sources, not no biological target or pathway.",
                  "Protocol nominal dose is not a verified executed-well dose or a dose-response series.",
                  "Joinable annotations are not currently consumed by the model and were not used to retrain it.",
                  "Absence of duration or cell background in the export does not prove those metadata do not exist externally.",
                  "Only named metadata NPZ members were read; Y and other profile arrays were not read."]}
    return report, records


def markdown(report):
    n = report["n_compounds"]
    labels = {"smiles": "SMILES", "target": "靶点", "moa": "MoA 注释", "action_direction": "明确靶点边的作用方向",
              "pathway": "通路", "dose": "剂量", "duration": "处理时间",
              "cell_background": "细胞背景", "cell_count": "细胞数"}
    rows = ["# 已开放 DEV 生物信息覆盖审计", "", f"范围：{n} 个 source_5 DEV 化合物、四个既定角色。",
            "", "| 信息 | 现有模型可用 | 另有明确来源、可接入 | 现有输入未知 |",
            "| --- | ---: | ---: | ---: |"]
    for key, label in labels.items():
        a, b = report["model_input_coverage"][key], report["joinable_annotation_coverage"][key]
        rows.append(f"| {label} | {a['available']}/{n} | {b['available']}/{n} | {a['unknown']}/{n} |")
    rows += ["", "现有剂量为协议名义 10 µM，并非逐孔核验的执行值。SMILES 的有效性来自已保存的 RDKit 解析标记。",
             "", "现有数据缺少靶点、作用方向和通路输入；未知靶点不能当作无靶点。公共注释即使可关联，也不等于已经加入模型。",
             "", f"细胞数覆盖：{report['provenance']['cell_count_wells']}/{report['provenance']['well_count']} 个已分配孔。",
             "", "## 来源与边界", "", f"- 导出元数据：{report['provenance']['sidecar']}",
             "- 读取的数值包成员仅为：" + ", ".join(report["provenance"]["npz_metadata_members_read"]) + "。",
             "- 未加载测量谱 Y、收益标签、旧 FINAL、第五重复或新增开发对象。",
             "- 数据集、旧终点、训练配置和旧结果均未修改。"]
    for join in report["external_annotation_joins"]:
        rows += [f"- 注释来源：{join['source']}；精确 ID 关联 {join['joined_compounds']} 个对象。"]
        if "moa_direction_word_coverage" in join:
            rows += [f"- MoA 文本含 inhibitor/antagonist 等方向词：{join['moa_direction_word_coverage']}/{n}；没有显式靶点边方向字段，不能直接转成带符号的药物—靶点边。",
                     "- 需人工核查的靶点/MoA 配对：" + ", ".join(join["target_moa_alignment_requires_review"]) + "。这些记录未自动输入机制先验。",
                     "- JUMP 注释可能来自 Hub，不将重复来源计为独立佐证；当前文件未提供逐边实验/计算证据代码。"]
    if report.get("upstream_cell_count_schema"):
        rows += ["- 上游已保存的原始列名清单也没有识别到细胞总数列；Children_Cytoplasm_Count 为对象子计数，不能代替每孔细胞数。"]
    rows += ["", "细胞背景若列为可接入，仅代表官方 JUMP 协议的 U2OS/Homo sapiens；不代表已拥有逐孔细胞鉴定、基因型或基线转录组。"]
    if report["joinable_annotation_coverage"]["duration"]["available"]:
        rows += ["本轮另核对原始生产论文的实验条件，补充协议名义暴露时间 48 小时。逐孔实际执行时长仍未知；没有用板名日期差推断暴露时长。"]
    else:
        rows += ["处理时间仍未知；不能把板名日期差当作暴露时长。"]
    return "\n".join(rows) + "\n"


def typed_records_from_audit(annotation_path: Path, plate_metadata: Path | None = None):
    """Convert the metadata audit into typed records without enabling mechanisms.

    All source target/MoA statements are retained as audit annotations. No action
    sign or confidence is inferred, and none become model-input relations. The
    current chemical input remains available when biological knowledge is absent.
    """
    from .biology import BiologyRecord, MechanismRelation, Provenance, TypedValue, validate_records

    annotation_path = Path(annotation_path).resolve()
    objects = json.loads(annotation_path.read_text())
    plate_lookup = {}
    if plate_metadata:
        with gzip.open(plate_metadata, "rt") as stream:
            for entry in csv.DictReader(stream):
                key = (entry["Metadata_Source"], entry["Metadata_Plate"])
                if key in plate_lookup:
                    raise ValueError("Duplicate source/plate identity in public plate metadata")
                plate_lookup[key] = entry["Metadata_Batch"]
    records = []
    for index, row in enumerate(objects):
        identity = row["compound_id"]
        existing = row["model_inputs"]
        joined = row["joinable_annotations"]
        exported = Provenance(
            source="Previously opened JUMP source_5 DEV metadata export",
            reference=str(annotation_path) + f"#/{index}",
            source_version="source5 four-role DEV allocation r1; metadata audit 2026-09-12",
            available_at="2026-09-12")

        def known(kind, value, *, unit=None, role="input", interpretation="reported", provenance=exported):
            return TypedValue(kind=kind, status="known", value=value, unit=unit,
                              provenance=provenance, role=role, availability="decision",
                              interpretation=interpretation)

        def unknown(kind, *, unit=None):
            return TypedValue(kind=kind, status="unknown", unit=unit,
                              role="input", availability="unknown")

        perturbation = {"entities": known("entities", ["JUMP:" + identity]),
                        "smiles": known("category", existing["smiles"]["value"]),
                        "dose": known("quantity", existing["dose"]["value"], unit="uM",
                                      interpretation="nominal_protocol"),
                        "duration": unknown("quantity", unit="hour"),
                        "target_entities": unknown("entities"),
                        "target_action_direction": unknown("category"),
                        "pathway_entities": unknown("entities")}
        duration = joined.get("duration", {})
        if duration.get("status") == "protocol_nominal":
            duration_provenance = Provenance(
                source="Original JUMP production dataset paper, Experimental conditions",
                reference=duration["source"], source_version="bioRxiv 2023.03.23.534023v2",
                available_at="2026-09-12")
            perturbation["duration"] = known("quantity", duration["value"], unit=duration["unit"],
                                             interpretation="nominal_protocol", provenance=duration_provenance)
        perturbation["executed_duration"] = unknown("quantity", unit="hour")
        context = {"cell_line": unknown("category"), "species": unknown("category"),
                   "genotype": unknown("category"), "baseline_transcriptome": unknown("entities")}
        background = joined.get("cell_background", {})
        if background.get("status") == "protocol_nominal":
            source = background["source"]
            protocol = Provenance(source="Official JUMP dataset description",
                                  reference=source, source_version="live documentation checked 2026-09-12",
                                  available_at="2026-09-12")
            for name, value in background["value"].items():
                context[name] = known("category", value, interpretation="nominal_protocol", provenance=protocol)
        measurements = {}
        for well in row["well_ids"]:
            source, plate, position = well.split("::")
            measurements[well] = {
                "source": known("category", source, interpretation="planned"),
                "plate": known("category", plate, interpretation="planned"),
                "well": known("category", position, interpretation="planned"),
                "row": known("quantity", ord(position[0].upper()) - ord("A") + 1,
                             unit="one_based_row_index", interpretation="planned"),
                "column": known("quantity", int(position[1:]), unit="one_based_column_index", interpretation="planned"),
                "cell_count": unknown("quantity", unit="cells_per_well")}
            if plate_metadata:
                if (source, plate) not in plate_lookup:
                    raise ValueError("An opened physical well's plate is absent from public metadata")
                batch_provenance = Provenance(
                    source="Official JUMP plate metadata",
                    reference=str(Path(plate_metadata).resolve()),
                    source_version="datasets 016e865fa0691244e0860943e41c7d6a88ed2580; local retrieval 2026-09-10",
                    available_at="2026-09-10")
                measurements[well]["batch"] = known("category", plate_lookup[(source, plate)],
                                                    interpretation="planned", provenance=batch_provenance)
            else:
                measurements[well]["batch"] = unknown("category")
        relations, deduplicate = [], set()
        annotations = row.get("public_annotation_evidence", [])
        for evidence in annotations:
            provenance = Provenance(source="Drug Repurposing Hub / JUMP public source annotation",
                                    reference=evidence["mechanism_source"],
                                    source_version="2025-08-18 Hub release; locally retrieved 2026-09-10",
                                    available_at="2026-09-10")
            for target in evidence["target"].split("|"):
                target = target.strip()
                key = (target, evidence["mechanism_source"])
                if not present(target) or key in deduplicate:
                    continue
                deduplicate.add(key)
                relations.append(MechanismRelation(
                    subject="JUMP:" + identity, predicate="reported_target_association",
                    object="GENE_SYMBOL:" + target, direction="unknown",
                    evidence_family="curatorial", evidence_code="not_supplied", confidence=None,
                    provenance=provenance, role="audit", availability="unknown"))
        if "moa" in joined:
            for j, moa in enumerate(joined["moa"]["value"]):
                perturbation[f"reported_moa_{j + 1}"] = known("category", moa, role="audit")
        if "target" in joined:
            perturbation["reported_target_entities"] = known(
                "entities", ["GENE_SYMBOL:" + target for target in joined["target"]["value"]], role="audit")
        records.append(BiologyRecord(
            unit_id=identity, perturbation_type="small_molecule", perturbation=perturbation,
            biological_context=context, measurement_metadata=measurements,
            relation_coverage="unknown", relations=tuple(relations)))
    return validate_records(records, [row["compound_id"] for row in objects],
                            [row["well_ids"] for row in objects])


def export_typed_audit(annotation_path: Path, output: Path, plate_metadata: Path | None = None):
    from .biology import load_biology, save_biology, validate_records

    records = typed_records_from_audit(annotation_path, plate_metadata)
    save_biology(records, output)
    restored = load_biology(output)
    validate_records(restored, [record.unit_id for record in records],
                     [list(record.measurement_metadata) for record in records])
    return {"records": len(restored), "physical_wells": sum(len(row.measurement_metadata) for row in restored),
            "audit_relations": sum(len(row.relations) for row in restored),
            "active_input_relations": sum(relation.usable for row in restored for relation in row.relations),
            "mechanism_prior_activated": False, "output": str(Path(output).resolve())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--annotation", action="append", default=[], type=Path)
    parser.add_argument("--legacy-annotation-directory", type=Path)
    parser.add_argument("--compound-metadata", type=Path)
    parser.add_argument("--protocol-background-source")
    parser.add_argument("--profile-schema", type=Path)
    parser.add_argument("--protocol-duration-hours", type=float)
    parser.add_argument("--protocol-duration-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--export-typed-from-audit", type=Path)
    parser.add_argument("--typed-output", type=Path)
    parser.add_argument("--plate-metadata", type=Path)
    parser.add_argument("--replace-output", action="store_true", help="Regenerate this audit report only; never modifies input data")
    args = parser.parse_args(argv)
    if args.export_typed_from_audit:
        if not args.typed_output:
            parser.error("--export-typed-from-audit requires --typed-output")
        print(json.dumps(export_typed_audit(args.export_typed_from_audit, args.typed_output, args.plate_metadata)))
        return
    if not args.data or not args.output:
        parser.error("Coverage audit requires --data and --output")
    report, records = audit(args.data, args.annotation,
                           legacy_annotation_directory=args.legacy_annotation_directory,
                           compound_metadata=args.compound_metadata,
                           protocol_background_source=args.protocol_background_source,
                           profile_schema=args.profile_schema,
                           protocol_duration_hours=args.protocol_duration_hours,
                           protocol_duration_source=args.protocol_duration_source)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("coverage.json", "object_annotations.json", "coverage.md"):
        if (args.output / name).exists() and not args.replace_output:
            raise FileExistsError(f"Audit output already exists: {args.output / name}")
    for name, payload in (("coverage.json", report), ("object_annotations.json", records)):
        path = args.output / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (args.output / "coverage.md").write_text(markdown(report))
    print(json.dumps({"compounds": report["n_compounds"],
                      "model_input_coverage": report["model_input_coverage"],
                      "joinable_annotation_coverage": report["joinable_annotation_coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
