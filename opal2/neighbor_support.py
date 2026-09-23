"""Decision-time-only neighbor support audit; no response or noise prediction.

The library is always the fixed TRAIN set. The public numerical API receives
only initial profiles, fingerprints, identities and partition membership. It
cannot inspect future wells, gains, fitted-model predictions or mechanism
labels. Distances indicate available neighbors, not biological equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import numpy as np


METRICS = ("morphology_cosine", "morphology_rms", "chemical_tanimoto")
THRESHOLDS = (.3, .5, .7)


def pairwise_profile_distances(query_x, library_x, metric="cosine"):
    """Cosine distance or coordinate-RMS Euclidean distance in fixed X space."""
    query, library = np.asarray(query_x, dtype=np.float64), np.asarray(library_x, dtype=np.float64)
    if (query.ndim != 2 or library.ndim != 2 or query.shape[1] != library.shape[1]
            or not query.shape[1] or not np.isfinite(query).all() or not np.isfinite(library).all()):
        raise ValueError("Finite query and library profile matrices with matching coordinates required")
    q2, l2 = np.einsum("ij,ij->i", query, query), np.einsum("ij,ij->i", library, library)
    cross = query @ library.T
    if metric == "cosine":
        if np.any(q2 <= 0) or np.any(l2 <= 0):
            raise ValueError("Zero-norm standardized profiles have undefined cosine support")
        similarity = cross / np.sqrt(q2[:, None] * l2[None])
        return 1 - np.clip(similarity, -1., 1.)
    if metric == "rms":
        squared = (q2[:, None] + l2[None] - 2 * cross) / query.shape[1]
        return np.sqrt(np.maximum(squared, 0.))
    raise ValueError("metric must be cosine or rms")


def tanimoto_distances(query_bits, library_bits, query_valid=None, library_valid=None):
    """Distance on original binary fingerprint bits, excluding any validity bit.

    Invalid structures and empty fingerprints yield NaN, never fabricated
    all-zero molecules with similarity one. The caller explicitly removes the
    separately recorded valid-SMILES indicator before invoking this function.
    """
    query, library = np.asarray(query_bits, dtype=np.float64), np.asarray(library_bits, dtype=np.float64)
    if query.ndim != 2 or library.ndim != 2 or query.shape[1] != library.shape[1] or not query.shape[1]:
        raise ValueError("Matching binary fingerprint matrices required")
    if not np.isin(query, [0., 1.]).all() or not np.isin(library, [0., 1.]).all():
        raise ValueError("Use raw binary fingerprint bits, not standardized chemistry")
    qv = np.ones(len(query), bool) if query_valid is None else np.asarray(query_valid, bool)
    lv = np.ones(len(library), bool) if library_valid is None else np.asarray(library_valid, bool)
    if qv.shape != (len(query),) or lv.shape != (len(library),):
        raise ValueError("Validity masks must have one entry per molecule")
    qv = qv & (query.sum(1) > 0)
    lv = lv & (library.sum(1) > 0)
    intersection = query @ library.T
    union = query.sum(1)[:, None] + library.sum(1)[None] - intersection
    out = np.full(intersection.shape, np.nan)
    valid = qv[:, None] & lv[None] & (union > 0)
    out[valid] = 1 - intersection[valid] / union[valid]
    return out


def exponential_ess(distances, bandwidth):
    """Top-set ESS plus absolute affinity; high ESS alone is not good support."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("ESS needs nonempty finite nonnegative distances")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive and finite")
    # A common exponent shift preserves normalized weights and avoids an
    # artificial 0/0 ESS when all twenty neighbors are absolutely distant.
    shifted = np.exp(-(distances - distances.min()) / bandwidth)
    weights = shifted / shifted.sum()
    return {"ess": float(1 / np.square(weights).sum()),
            "absolute_affinity_sum": float(np.exp(-distances / bandwidth).sum()),
            "nearest_distance_over_bandwidth": float(distances.min() / bandwidth)}


def _quantiles(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    quantiles = np.quantile(values, [0., .05, .25, .5, .75, .95, 1.])
    return {"n": int(len(values)), "mean": float(values.mean()),
            **{key: float(value) for key, value in zip(("min", "q05", "q25", "median", "q75", "q95", "max"), quantiles)}}


def neighbor_statistics(distances, query_ids: Sequence[str], library_ids: Sequence[str],
                        train_query_indices, *, chemical=False, bandwidth_k=5, ess_top_k=20):
    """Describe each query against TRAIN, excluding every matching identity.

    Distances must be [all query objects, TRAIN library objects]. The bandwidth
    is the median TRAIN leave-one-out fifth-neighbor distance. Percentiles are
    descriptive midrank positions within TRAIN LOO distances, not OOD p-values.
    """
    distance = np.asarray(distances, float).copy()
    query_ids, library_ids = np.asarray(query_ids, str), np.asarray(library_ids, str)
    train = np.asarray(train_query_indices)
    if distance.shape != (len(query_ids), len(library_ids)) or distance.ndim != 2:
        raise ValueError("Distance matrix must match query and TRAIN library IDs")
    if (train.ndim != 1 or not np.issubdtype(train.dtype, np.integer)
            or not len(train) or len(set(train.tolist())) != len(train)
            or np.any(train < 0) or np.any(train >= len(query_ids))):
        raise ValueError("TRAIN query indices must be unique and in range")
    if set(query_ids[train]) != set(library_ids):
        raise ValueError("Only the declared TRAIN identities may enter the neighbor library")
    if np.any(np.isinf(distance)) or np.any(distance[np.isfinite(distance)] < -1e-12):
        raise ValueError("Input distances must be nonnegative finite values or explicit NaN")
    if bandwidth_k < 1 or ess_top_k < 1:
        raise ValueError("Positive neighbor counts required")
    distance = np.maximum(distance, 0.)
    distance[query_ids[:, None] == library_ids[None]] = np.nan
    # Lexicographic ID tie-breaking is fixed and independent of input row order.
    lex = np.argsort(library_ids, kind="stable")
    order = lex[np.argsort(np.where(np.isfinite(distance[:, lex]), distance[:, lex], np.inf),
                           axis=1, kind="stable")]
    sorted_distance = np.take_along_axis(distance, order, axis=1)
    if bandwidth_k > len(library_ids):
        raise ValueError("Too few TRAIN library entries for the declared bandwidth")
    train_k = sorted_distance[train, bandwidth_k - 1]
    available = train_k[np.isfinite(train_k)]
    if not len(available):
        raise ValueError("TRAIN LOO kth-neighbor bandwidth is unavailable")
    bandwidth = float(np.median(available))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Degenerate TRAIN LOO bandwidth; no unreported bandwidth floor is used")
    reference = {k: sorted_distance[train, k - 1] for k in (1, 5, 10) if k <= len(library_ids)}
    reference_summary = {f"top{k}_distance": _quantiles(v) for k, v in reference.items()}
    rows = []
    for i, identity in enumerate(query_ids):
        valid = np.isfinite(sorted_distance[i])
        ranked = sorted_distance[i, valid]
        neighbors = library_ids[order[i, valid]]
        row = {"compound_id": str(identity), "valid_neighbor_count": int(len(ranked)),
               "top10_neighbor_ids": neighbors[:10].tolist(),
               "top10_distances": ranked[:10].tolist(),
               "absolute_neighbor_support_not_established_by_ess": True}
        for k in (1, 5, 10):
            value = float(ranked[k - 1]) if len(ranked) >= k else None
            row[f"top{k}_distance"] = value
            row[f"top{k}_mean_distance"] = float(ranked[:k].mean()) if len(ranked) >= k else None
            if chemical:
                row[f"top{k}_similarity"] = 1 - value if value is not None else None
                row[f"top{k}_mean_similarity"] = 1 - row[f"top{k}_mean_distance"] if value is not None else None
            ref = reference.get(k, np.array([]))
            ref = np.sort(ref[np.isfinite(ref)])
            if value is None or not len(ref):
                row[f"top{k}_train_loo_midrank_percentile"] = None
                row[f"top{k}_above_train_loo_q95"] = None
                row[f"top{k}_train_loo_iqr_z"] = None
            else:
                row[f"top{k}_train_loo_midrank_percentile"] = float(
                    (np.searchsorted(ref, value, "left") + np.searchsorted(ref, value, "right")) / (2 * len(ref)))
                row[f"top{k}_above_train_loo_q95"] = bool(value > np.quantile(ref, .95))
                q25, median, q75 = np.quantile(ref, [.25, .5, .75])
                row[f"top{k}_train_loo_iqr_z"] = float((value - median) / (q75 - q25)) if q75 > q25 else None
        count = min(ess_top_k, len(ranked))
        if count:
            weight = exponential_ess(ranked[:count], bandwidth)
            row.update(ess_top20=weight["ess"], top20_absolute_affinity_sum=weight["absolute_affinity_sum"],
                       top20_used_neighbors=count, nearest_distance_over_bandwidth=weight["nearest_distance_over_bandwidth"])
        else:
            row.update(ess_top20=None, top20_absolute_affinity_sum=None,
                       top20_used_neighbors=0, nearest_distance_over_bandwidth=None)
        if chemical:
            for threshold in THRESHOLDS:
                # No valid structure is unknown, rather than evidence of no
                # chemically similar molecules in an otherwise valid query.
                row[f"support_tanimoto_ge_{threshold:.1f}"] = int((ranked <= 1 - threshold + 1e-12).sum()) if count else None
        rows.append(row)
    info = {"bandwidth": bandwidth, "bandwidth_rule": f"median TRAIN leave-one-out neighbor {bandwidth_k} distance",
            "bandwidth_valid_train_queries": int(len(available)), "train_loo_reference": reference_summary,
            "ess_scope": f"fixed nearest {ess_top_k}; read together with absolute distance and affinity",
            "self_exclusion": "all equal compound IDs excluded", "tie_break": "lexicographic TRAIN compound ID"}
    return rows, info


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_inputs(data, scaler_path):
    """Read the authorized export, retaining only X and decision-time chemistry.

    NPZ stores all four roles in one compressed Y member. Decompressing that
    member is unavoidable; only Y[:,0,:] is retained, and no future value is
    used in a statistic, neighbor, threshold, validation or protocol choice.
    """
    data, scaler_path = Path(data).resolve(), Path(scaler_path).resolve()
    if data.name != "source5_primary_fullcontrols":
        raise ValueError("Only the named already-opened DEV export is allowed")
    manifest = json.loads((data / "manifest.json").read_text())
    document = json.loads((data / "measurements.json").read_text())
    metadata = document["metadata"]
    if (manifest.get("old_final_opened") is not False or manifest.get("fifth_repeat_read") is not False
            or metadata.get("scope") != "639_DEV_FOUR_ROLES_ONLY"):
        raise ValueError("Original FINAL/fifth-repeat boundary must be preserved")
    with np.load(data / "measurements.npz", allow_pickle=False) as arrays:
        ids = arrays["ids"].astype(str)
        names = arrays["feature_names"].astype(str)
        chemistry = arrays["chem"].copy()
        chemical_mask = arrays["chem_mask"].astype(bool)
        all_roles = arrays["Y"]
        if all_roles.shape != (639, 4, 3617):
            raise ValueError("Only the existing 639 by four by 3617 export is permitted")
        initial = all_roles[:, 0, :].copy()
        del all_roles
    if len(ids) != 639 or len(set(ids)) != 639 or names.shape != (3617,):
        raise ValueError("Existing unique identities and complete coordinate order are required")
    split = json.loads((data / "splits.json").read_text())["compound_ids"]
    expected = {"train": 383, "validation": 96, "calibration": 64, "evaluation": 96}
    if {k: len(v) for k, v in split.items()} != expected:
        raise ValueError("The original four-way allocation changed")
    all_ids = [item for values in split.values() for item in values]
    if len(set(all_ids)) != 639 or set(all_ids) != set(ids):
        raise ValueError("Declared allocation must partition the existing identities")
    lookup = {value: i for i, value in enumerate(ids)}
    indices = {k: np.asarray([lookup[value] for value in values], dtype=int) for k, values in split.items()}
    scaler = json.loads(scaler_path.read_text())
    if scaler["train_ids"] != split["train"] or scaler["feature_names"] != names.tolist():
        raise ValueError("Reuse the existing G TRAIN scaler with identical coordinate and ID order")
    center, scale = np.asarray(scaler["y_center"]), np.asarray(scaler["y_scale"])
    if center.shape != (3617,) or scale.shape != (3617,) or np.any(scale <= 0):
        raise ValueError("Complete positive input scaling required")
    x = (initial - center) / scale
    chemical = metadata["chemical"]
    bits = int(chemical["bits"])
    if (chemical.get("kind") != "RDKit Morgan fingerprint" or chemical.get("radius") != 2
            or bits != 512 or chemical.get("final_coordinate") != "valid_SMILES_indicator"
            or chemistry.shape != (639, 513) or chemical_mask.shape != (639,)):
        raise ValueError("Expected the original 512-bit radius-2 Morgan fingerprint plus validity indicator")
    if not np.array_equal(chemistry[:, -1], chemical_mask.astype(chemistry.dtype)):
        raise ValueError("Fingerprint validity indicator and exported mask disagree")
    smiles = metadata.get("smiles", [])
    if len(smiles) != 639:
        raise ValueError("SMILES metadata must align exactly with the original object order")
    known = np.array([bool(str(v).strip()) and str(v).strip().lower() not in {"nan", "none", "null", "unknown"}
                      for v in smiles])
    if np.any(chemical_mask & ~known):
        raise ValueError("A valid molecule cannot have missing source SMILES")
    provenance = dict(data_directory=str(data), input_scaler=str(scaler_path),
        selected_npz_fields=["ids", "feature_names", "chem", "chem_mask", "Y[:,0,:]"],
        storage_caveat="Y is one compressed four-role NPZ member; only its initial X slice is retained and used",
        fingerprint_definition=chemical, fingerprint_bits_used=512, valid_indicator_used_in_tanimoto=False,
        smiles_source=str(data / "measurements.json") + "#metadata.smiles",
        smiles_adapter_provenance="source5.py:morgan_fingerprints consumes the original initial_metadata.SMILES in ID order",
        smiles_nonmissing=int(known.sum()), valid_structure_count=int(chemical_mask.sum()),
        unique_smiles_strings=len(set(str(v).strip() for v in smiles if str(v).strip())),
        source=metadata.get("group_names", {}).get("source"),
        initial_X_batches=metadata.get("legacy_info", {}).get("initial_X_batches", []),
        dose=dict(value=metadata.get("nominal_concentration_uM"), unit="uM", status="protocol_nominal",
                  evidence=metadata.get("concentration_status"), same_declared_dose_across_export=True),
        cell_background=dict(value=metadata.get("cell_line"), status="not_provided_in_this_export"),
        exposure_duration=dict(value=metadata.get("duration_hours"), unit="h", status="not_provided_in_this_export"),
        actual_cell_counts_available=bool(metadata.get("cell_counts_available", False)),
        biological_same_condition_not_independently_verified=True)
    return ids, x, chemistry[:, :512], chemical_mask, indices, provenance


def audit(data, scaler, output):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Neighbor audit output must be absent or empty")
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "decision-time neighbor support, not a predictive validation of route B",
        "query_scope": "639 already-opened DEV objects; fixed original partitions",
        "library_scope": "383 TRAIN objects only; all equal-ID neighbors excluded",
        "morphology": "complete 3617 X coordinates under the existing G TRAIN scaler; cosine and RMS Euclidean distances",
        "chemistry": "raw 512-bit Morgan radius2 Tanimoto; validity indicator excluded",
        "summary_neighbors": [1, 5, 10], "ess_neighbors": 20,
        "bandwidth": "per metric median TRAIN leave-one-out fifth-neighbor distance",
        "chemical_support_thresholds": list(THRESHOLDS),
        "distance_outlier_reference": "TRAIN LOO fifth/first/tenth distances; descriptive percentile/IQR, not an OOD certificate",
        "future_outcomes_used": False, "model_predictions_used": False,
        "neighbor_response_or_noise_relevance_tested": False,
        "kernel_fitted": False, "new_partitions": False, "final_opened": False, "fifth_repeat_opened": False,
        "threads": 2, "interpretation": "ESS can be high even if all neighbors are far; read absolute similarity/distance alongside ESS",
    }
    _write_json(root / "protocol.json", protocol)
    ids, x, bits, valid, splits, provenance = load_inputs(data, scaler)
    train = splits["train"]
    library_ids = ids[train]
    distances = {
        "morphology_cosine": pairwise_profile_distances(x, x[train], "cosine"),
        "morphology_rms": pairwise_profile_distances(x, x[train], "rms"),
        "chemical_tanimoto": tanimoto_distances(bits, bits[train], valid, valid[train]),
    }
    per_metric, fitting = {}, {}
    for name in METRICS:
        per_metric[name], fitting[name] = neighbor_statistics(distances[name], ids, library_ids,
            train, chemical=name == "chemical_tanimoto")
    partition = {int(index): name for name, indices in splits.items() for index in indices}
    records = []
    for i, identity in enumerate(ids):
        record = {"compound_id": str(identity), "partition": partition[i],
                  "valid_structure": bool(valid[i]),
                  **{name: per_metric[name][i] for name in METRICS}}
        chemical_neighbors = set(per_metric["chemical_tanimoto"][i]["top10_neighbor_ids"])
        overlap = {}
        for name in METRICS[:2]:
            morphological = set(per_metric[name][i]["top10_neighbor_ids"])
            union = morphological | chemical_neighbors
            overlap[name] = dict(count=len(morphological & chemical_neighbors),
                jaccard=len(morphological & chemical_neighbors) / len(union) if union and chemical_neighbors else None,
                defined=bool(morphological and chemical_neighbors))
        record["top10_morphology_chemical_overlap"] = overlap
        records.append(record)
    with (root / "per_object.jsonl").open("w") as stream:
        for record in records:
            stream.write(json.dumps(_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"protocol": protocol, "provenance": provenance, "train_library_count": len(train),
               "queries": len(ids), "distance_fitting": fitting, "partitions": {}}
    for name, indices in splits.items():
        partition_metrics = {}
        for metric in METRICS:
            rows = [per_metric[metric][i] for i in indices]
            fields = ["top1_distance", "top5_distance", "top10_distance", "top5_mean_distance", "top10_mean_distance",
                      "ess_top20", "top20_absolute_affinity_sum", "nearest_distance_over_bandwidth"]
            if metric == "chemical_tanimoto":
                fields += ["top1_similarity", "top5_similarity", "top10_similarity", "top5_mean_similarity", "top10_mean_similarity"]
            report = {field: _quantiles([r[field] if r[field] is not None else np.nan for r in rows]) for field in fields}
            report["no_valid_neighbors"] = sum(r["valid_neighbor_count"] == 0 for r in rows)
            for k in (1, 5, 10):
                key = f"top{k}_above_train_loo_q95"
                flags = [r[key] for r in rows if r[key] is not None]
                report[f"top{k}_distance_above_train_loo_q95"] = dict(n=sum(flags), denominator=len(flags))
            if metric == "chemical_tanimoto":
                report["threshold_support"] = {}
                for threshold in THRESHOLDS:
                    counts = [r[f"support_tanimoto_ge_{threshold:.1f}"] for r in rows]
                    available = [c for c in counts if c is not None]
                    report["threshold_support"][f"{threshold:.1f}"] = dict(
                        counts=_quantiles(available), query_with_at_least_one=sum(c >= 1 for c in available),
                        query_with_at_least_five=sum(c >= 5 for c in available),
                        no_support=sum(c == 0 for c in available), denominator=len(available), unknown=len(counts)-len(available))
            partition_metrics[metric] = report
        partition_metrics["top10_chemical_overlap"] = {
            metric: _quantiles([records[i]["top10_morphology_chemical_overlap"][metric]["count"]
                               for i in indices if records[i]["top10_morphology_chemical_overlap"][metric]["defined"]])
            for metric in METRICS[:2]}
        summary["partitions"][name] = dict(n=len(indices), metrics=partition_metrics)
    _write_json(root / "summary.json", summary)
    lines = ["# Route B: decision-time neighbor support", "",
             "Only the existing 639 DEV X profiles and raw chemistry were used. TRAIN 383 is the only neighbor library; matching IDs are excluded.", "",
             "This tests availability of similar objects, not whether similarity predicts response, repeat noise, Gamma, or useful acquisition.", "",
             "| Partition | n | Chemical nearest Tanimoto median | At least one ≥0.5 | At least one ≥0.7 | Chemical top20 ESS median | Cosine/chemical top10 overlap median |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in summary["partitions"].items():
        chemical = values["metrics"]["chemical_tanimoto"]
        support = chemical["threshold_support"]
        overlap = values["metrics"]["top10_chemical_overlap"]["morphology_cosine"]
        lines.append(f"| {name} | {values['n']} | {chemical['top1_similarity'].get('median', float('nan')):.3f} | "
            f"{support['0.5']['query_with_at_least_one']}/{support['0.5']['denominator']} | "
            f"{support['0.7']['query_with_at_least_one']}/{support['0.7']['denominator']} | "
            f"{chemical['ess_top20'].get('median', float('nan')):.2f} | {overlap.get('median', float('nan')):.1f}/10 |")
    lines += ["", "## Reading the support measures", "",
        "Top5/top10 distance means the fifth/tenth neighbor; corresponding mean-distance fields are separate. All nearest IDs are retained in per_object.jsonl.",
        "RMS is sqrt(mean((scaled X_query - scaled X_train)^2)); cosine measures angle and ignores amplitude. Both use the complete fixed input coordinate space.",
        "Top20 ESS measures concentration among the selected twenty weights, not the number of genuinely close or independent biological examples. Absolute distances, affinity sums and chemical threshold coverage must be read with it.",
        "Morphology/chemistry neighbor overlap is descriptive; neither high nor low overlap proves mechanism or failure of local borrowing.",
        "The export declares the same nominal 10 uM concentration and initial X batch. These are recorded conditions, not independent per-well verification. Cell background and duration are not supplied in this specific export; no outside metadata were downloaded.",
        "Raw hashed fingerprints can collide and chemical similarity is not drug-target or action-direction knowledge. The validity indicator is excluded from Tanimoto.",
        "No future profiles, gains, fitted predictions or outcome-based thresholds entered the neighbor calculation. NPZ decompression reads the stored Y member, but retains and uses only X.",
        "No kernel was trained and no original endpoint, contract, split, FINAL or fifth repeat was changed.", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        report = audit(args.data, args.scaler, args.output)
    print(json.dumps({"state": "COMPLETE", "queries": report["queries"],
                      "train_library_count": report["train_library_count"],
                      "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
