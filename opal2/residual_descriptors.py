"""Fold-fitted, decision-time descriptors for conditional CORE error models.

Inference reads X, declared conditions, and X-plate controls only. Repeated
profiles are used exclusively inside ``fit_indices`` to estimate an empirical
cross-repeat coordinate system. This is not a physical noise decomposition:
fixed well positions and shared plates can also contribute to reproducibility.

CellProfiler MADIntensity/RadialCV/Texture_Variance summaries describe
within-object pixel/spatial statistics aggregated over cells. They are NOT
estimates of between-cell heterogeneity. DMSO inputs are already normalized
profiles, so their descriptors measure residual shape/tails, not raw noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


EPS = 1e-12
FAMILIES = {
    "pixel_mad": "_Intensity_MADIntensity_",
    "radial_cv": "_RadialDistribution_RadialCV_",
    "texture_variance": "_Texture_Variance_",
}
CONTROL_NAMES = (
    "log_control_count", "log_normalized_rms_median", "log_rms_q90_over_median",
    "normalized_abs_gt3", "normalized_abs_gt5", "normalized_abs_gt10",
    "log_feature_mad_median", "spectral_participation_fraction",
    "first_spectral_energy_fraction", "mean_control_cosine",
    "mean_abs_control_cosine", "log_pair_energy_median", "log_pair_q90_over_median",
)


def _finite_number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _numeric_field(units, key, *, first_role=False):
    return np.asarray([_finite_number(
        u.get("roles", {}).get("X", {}).get(key) if first_role else u.get(key))
        for u in units], dtype=float)


def _first_well(data, metadata):
    ids = np.asarray(data["ids"], str)
    x = np.asarray(data["Y"][:, 0], dtype=float)
    features = np.asarray(data["feature_names"], str)
    if (x.ndim != 2 or x.shape != (len(ids), len(features)) or
            not np.isfinite(x).all() or len(metadata["units"]) != len(ids)):
        raise ValueError("First-well inputs must be finite and aligned")
    if any(str(u.get("id", u.get("compound_id"))) != i for u, i in zip(metadata["units"], ids)):
        raise ValueError("Object metadata is not aligned with X")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate object identifiers")
    return x, ids, features


def normalized_control_summary(profiles):
    """Summarize one plate's normalized DMSO wells without query outcomes."""
    c = np.asarray(profiles, dtype=float)
    if c.ndim != 2 or len(c) < 3 or not np.isfinite(c).all():
        raise ValueError("Control profiles require >=3 finite wells")
    rms = np.sqrt(np.mean(c*c, axis=1))
    med = np.median(c, axis=0)
    mad = np.median(np.abs(c-med), axis=0)
    centered = c-c.mean(0)
    gram = centered@centered.T/c.shape[1]
    ev = np.maximum(np.linalg.eigvalsh(gram), 0.)
    participation = ev.sum()**2/max(float(ev@ev), EPS)/max(min(c.shape)-1, 1)
    unit = c/np.maximum(np.linalg.norm(c, axis=1, keepdims=True), EPS)
    cosine = (unit@unit.T)[np.triu_indices(len(c), 1)]
    pair = cdist(c, c, metric="sqeuclidean")[np.triu_indices(len(c), 1)]/c.shape[1]
    return np.asarray([
        np.log1p(len(c)), np.log(max(np.median(rms), EPS)),
        np.log(max(np.quantile(rms, .9), EPS)/max(np.median(rms), EPS)),
        np.mean(np.abs(c)>3), np.mean(np.abs(c)>5), np.mean(np.abs(c)>10),
        np.log(max(np.median(mad), EPS)), participation,
        ev[-1]/max(ev.sum(), EPS), np.mean(cosine), np.mean(np.abs(cosine)),
        np.log(max(np.median(pair), EPS)),
        np.log(max(np.quantile(pair, .9), EPS)/max(np.median(pair), EPS)),
    ], dtype=float)


def load_normalized_plate_controls(metadata, feature_names, cache_directory):
    """Read only named X-plate DMSO rows from the already downloaded cache.

    The returned map contains small summary vectors rather than all control
    coordinates, and can be reused across outer/inner fits. No future-role
    plate is accessed unless it is also an actual X plate for another object.
    Missing files are explicitly omitted and become missing flags downstream.
    """
    import pandas as pd

    directory = Path(cache_directory)
    names = list(np.asarray(feature_names, str))
    plates = sorted({str(u.get("roles", {}).get("X", {}).get("plate", ""))
                     for u in metadata["units"]})
    result = {}
    for plate in plates:
        path = directory/f"{plate}_normalized_dmso.csv.gz"
        if not plate or not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=["Metadata_broad_sample", *names])
        controls = frame.loc[frame["Metadata_broad_sample"].astype(str).str.upper().eq("DMSO"), names]
        if len(controls) < 3:
            raise ValueError(f"Not enough verified DMSO rows on X plate {plate}")
        result[plate] = normalized_control_summary(controls.to_numpy(float))
    return result


def information_availability(data, metadata, plate_controls=None):
    """Metadata/feature feasibility audit; does not inspect future values."""
    x, ids, features = _first_well(data, metadata)
    units = metadata["units"]
    counts = _numeric_field(units, "cell_count", first_role=True)
    dose = _numeric_field(units, "actual_dose_uM")
    hours = _numeric_field(units, "exposure_hours_protocol_nominal")
    cache = {} if plate_controls is None else plate_controls
    plates = [str(u.get("roles", {}).get("X", {}).get("plate", "")) for u in units]
    # Mere target/MoA or compound names never count as a measured potency.
    potency_keys = {"ec50", "ic50", "pec50", "pic50", "ec50_um", "ic50_um",
                    "pec50_matched", "ec50_matched_um"}
    potency = []
    for unit in units:
        found = {}
        for source in (unit, unit.get("chemistry", {})):
            for key, value in source.items():
                if key.lower() in potency_keys and np.isfinite(_finite_number(value)):
                    found[key] = float(value)
        potency.append(found)
    quality = [f for f in features if any(k in f.lower() for k in ("focus", "saturat", "imagequality"))]
    def span(v):
        finite = v[np.isfinite(v)]
        return None if not len(finite) else [float(finite.min()), float(finite.max())]
    return dict(
        objects=len(ids), first_well_features=x.shape[1],
        cell_count_available=int(np.sum(np.isfinite(counts)&(counts>0))),
        cell_count_source="metadata.units.roles.X.cell_count only",
        actual_dose_available=int(np.isfinite(dose).sum()), actual_dose_uM_range=span(dose),
        exposure_hours_available=int(np.isfinite(hours).sum()), exposure_hours_range=span(hours),
        exposure_provenance="protocol nominal; not per-well execution verification",
        cell_backgrounds=sorted({str(u.get("cell_line", "UNKNOWN")) for u in units}),
        feature_family_counts={name: int(sum(token in f for f in features))
                               for name, token in FAMILIES.items()},
        heterogeneity_interpretation="Within-object pixel/texture summaries of median-aggregated profiles; not cell-to-cell dispersion",
        single_cell_heterogeneity_available=False,
        measured_focus_saturation_feature_names=quality,
        normalized_dmso_query_coverage=sum(p in cache for p in plates),
        normalized_dmso_plates_available=len(set(plates)&set(cache)),
        normalized_dmso_interpretation="Residual shape and tails after existing plate normalization; not original measurement variance",
        raw_plate_noise_scale_available=False,
        potency_annotated_objects=sum(bool(p) for p in potency),
        matched_morphology_potency_verified=False,
        potency_status="No matched EC50/IC50 assay records available in current prepared metadata" if not any(potency)
            else "Potential numeric potency fields found; endpoint/context/units require separate verification before use",
        raw_batch_ids_used=False, future_plate_context_used=False,
    )


@dataclass
class DescriptorMatrix:
    values: np.ndarray
    names: list[str]
    blocks: dict[str, list[int]]
    latent_input: np.ndarray
    raw_values: np.ndarray
    report: dict

    def select_blocks(self, *blocks):
        if len(blocks) == 1 and not isinstance(blocks[0], str):
            blocks = tuple(blocks[0])
        columns = [i for block in blocks for i in self.blocks[block]]
        return self.values[:, columns]


class ResidualDescriptorTransformer:
    """All learned coordinates/imputation/scaling fitted on declared fit rows."""

    @classmethod
    def fit(cls, data, metadata, fit_indices, *, plate_controls: Mapping | None = None,
            pca_dim=64, latent_pca_dim=32, seed=20260916):
        x, ids, features = _first_well(data, metadata)
        rows = np.asarray(fit_indices, dtype=int)
        if (rows.ndim != 1 or len(rows) < 5 or len(np.unique(rows)) != len(rows)
                or np.any(rows<0) or np.any(rows>=len(x))):
            raise ValueError("At least five distinct valid fit rows are required")
        obj = cls()
        obj.fit_indices = rows.copy()
        obj.fit_ids = ids[rows].tolist()
        obj.fit_groups = np.asarray(data.get("groups", ids), str)[rows].copy()
        obj.feature_names = features.copy()
        obj.plate_controls = {} if plate_controls is None else dict(plate_controls)
        obj.x_center = x[rows].mean(0)
        obj.x_scale = x[rows].std(0, ddof=1)
        positive = obj.x_scale[obj.x_scale>EPS]
        scale_floor = max(float(np.median(positive))*.01, 1e-6) if len(positive) else 1.
        obj.x_scale = np.maximum(obj.x_scale, scale_floor)
        standardized = (x[rows]-obj.x_center)/obj.x_scale
        rank = min(int(pca_dim), x.shape[1], len(rows)-1)
        obj.pca = PCA(rank, svd_solver="randomized", random_state=seed).fit(standardized)
        obj.pc_scale = np.sqrt(np.maximum(obj.pca.explained_variance_, EPS))
        obj.fit_pc = obj.pca.transform(standardized)/obj.pc_scale
        uncentered_pc = (x[rows]/obj.x_scale)@obj.pca.components_.T
        obj.fit_direction = uncentered_pc/np.maximum(np.linalg.norm(uncentered_pc, axis=1, keepdims=True), EPS)
        obj.latent_dim = min(int(latent_pca_dim), x.shape[1], len(rows)-1)
        # The latent path has separate direction coordinates. Reusing the
        # descriptor PCA would mix amplitude into the proposed direction input.
        direction = x[rows]/np.maximum(np.linalg.norm(x[rows], axis=1, keepdims=True), EPS)
        obj.latent_pca = PCA(obj.latent_dim, svd_solver="randomized", random_state=seed+1).fit(direction)
        obj.latent_pc_scale = np.sqrt(np.maximum(obj.latent_pca.explained_variance_, EPS))
        # Only fit rows enter cross-repeat statistics, never held-out repeats.
        repeats = np.asarray(data["Y"][rows], dtype=float)
        if repeats.shape[1] < 2 or not np.isfinite(repeats).all():
            raise ValueError("Fit objects need >=2 finite repeats for reliability coordinates")
        scores = ((repeats-obj.x_center)/obj.x_scale)@obj.pca.components_.T
        scores -= scores.mean(0, keepdims=True)
        r = repeats.shape[1]
        marginal = np.einsum("nri,nrj->ij", scores, scores)/((len(rows)-1)*r)
        cross = sum(scores[:, a].T@scores[:, b]+scores[:, b].T@scores[:, a]
                    for a in range(r) for b in range(a+1, r))/((len(rows)-1)*r*(r-1))
        ridge = max(float(np.trace(marginal)/rank)*1e-3, 1e-8)
        values, vectors = eigh((cross+cross.T)/2, (marginal+marginal.T)/2+ridge*np.eye(rank))
        order = np.argsort(values)[::-1]
        obj.reliability_values = values[order]
        obj.reliability_vectors = vectors[:, order]
        obj.reliable_mask = obj.reliability_values >= .3
        obj.cell_lines = sorted({str(metadata["units"][i].get("cell_line") or "UNKNOWN") for i in rows})
        raw, names, blocks = obj._raw(data, metadata)
        obj.names, obj.blocks = names, blocks
        # All missing fields retain explicit indicator columns. Imputation is fit-only.
        obj.raw_center = np.asarray([np.nanmedian(raw[rows, j]) if np.isfinite(raw[rows, j]).any() else 0.
                                     for j in range(raw.shape[1])])
        filled = np.where(np.isfinite(raw), raw, obj.raw_center)
        obj.raw_scale = np.std(filled[rows], axis=0, ddof=1)
        obj.raw_scale[obj.raw_scale<1e-8] = 1.
        obj.latent_amp_center = float(np.log(np.maximum(np.sqrt(np.mean(x[rows]**2, axis=1)), EPS)).mean())
        obj.latent_amp_scale = max(float(np.log(np.maximum(np.sqrt(np.mean(x[rows]**2, axis=1)), EPS)).std(ddof=1)), 1e-6)
        obj.report = dict(
            fit_ids=obj.fit_ids, fit_indices=rows.tolist(), fit_groups=sorted(set(obj.fit_groups)),
            pca_dimension=rank, latent_pca_dimension=obj.latent_dim, descriptor_names=names,
            latent_input_interpretation="Separate fit-only PCA of unit-norm X direction, plus explicit standardized log RMS",
            descriptor_blocks=blocks, descriptor_count=len(names),
            reliability_values=obj.reliability_values.tolist(), reliable_threshold=.3,
            reliable_direction_count=int(obj.reliable_mask.sum()),
            reliability_interpretation="Fit-only empirical cross-repeat coordinates, not physical independent/shared components; fixed position may contribute",
            ratio_interpretation="Homogeneous scale-invariant ratios conditional on fitted coordinates, not independence from amplitude",
            learned_scaling_scope="fit_indices only; same-chemical-group references excluded from distance features",
            context_source="actual dose in uM; nominal protocol time; declared cell background; X-well position only",
            availability=information_availability(data, metadata, plate_controls),
        )
        return obj

    def _raw(self, data, metadata):
        x, ids, features = _first_well(data, metadata)
        if not np.array_equal(features, self.feature_names):
            raise ValueError("Feature order differs from fitted descriptor transform")
        units = metadata["units"]
        columns, names, blocks = [], [], {}
        def add(block, name, values):
            v = np.asarray(values, float)
            if v.shape != (len(x),):
                raise ValueError("Descriptor length mismatch: "+name)
            blocks.setdefault(block, []).append(len(names))
            names.append(name); columns.append(v)
        rms = np.sqrt(np.mean(x*x, axis=1))
        add("amplitude", "log_rms_X", np.log(np.maximum(rms, EPS)))
        add("amplitude", "zero_norm_X", rms<=EPS)
        for key, field in (("dose", "actual_dose_uM"), ("time", "exposure_hours_protocol_nominal")):
            value = _numeric_field(units, field)
            valid = np.isfinite(value)&(value>0)
            add("context", f"log_{key}", np.where(valid, np.log(np.maximum(value, EPS)), np.nan))
            add("context", f"{key}_missing", ~valid)
        lines = np.asarray([str(u.get("cell_line") or "UNKNOWN") for u in units])
        for line in self.cell_lines:
            add("context", "cell_background_"+line, lines==line)
        add("context", "cell_background_unseen", ~np.isin(lines, self.cell_lines))
        count = _numeric_field(units, "cell_count", first_role=True)
        valid_count = np.isfinite(count)&(count>0)
        add("cell_count", "log_cell_count_X", np.where(valid_count, np.log(np.maximum(count, 1)), np.nan))
        add("cell_count", "inverse_sqrt_cell_count_X", np.where(valid_count, 1/np.sqrt(np.maximum(count, 1)), np.nan))
        add("cell_count", "cell_count_missing", ~valid_count)
        total = np.sum(x*x, axis=1)
        for family, token in FAMILIES.items():
            allmask = np.asarray([token in f for f in features])
            for compartment in ("Cells", "Cytoplasm", "Nuclei"):
                mask = allmask&np.char.startswith(features, compartment+"_")
                prefix = family+"_"+compartment
                if mask.any():
                    part = x[:, mask]
                    summaries = (np.median(part, axis=1), np.sqrt(np.mean(part*part, axis=1)),
                                 np.quantile(np.abs(part), .9, axis=1))
                else:
                    summaries = (np.full(len(x), np.nan),)*3
                for suffix, values in zip(("signed_median", "rms", "abs_q90"), summaries):
                    add("within_object_texture", prefix+"_"+suffix, values)
                add("within_object_texture", prefix+"_unavailable", np.full(len(x), not mask.any()))
            add("within_object_texture", family+"_energy_fraction",
                np.sum(x[:, allmask]**2, axis=1)/np.maximum(total, EPS))
        centered = (x-self.x_center)/self.x_scale
        pc = self.pca.transform(centered)/self.pc_scale
        distance = cdist(pc, self.fit_pc)/np.sqrt(pc.shape[1])
        groups = np.asarray(data.get("groups", ids), str)
        excluded = groups[:, None] == self.fit_groups[None, :]
        distance[excluded] = np.inf
        available = (~excluded).sum(1)
        sorted_distance = np.sort(distance, axis=1)
        nearest = sorted_distance[:, 0]
        k = np.minimum(available, 5)
        top = np.where(np.arange(sorted_distance.shape[1])[None, :]<k[:, None], sorted_distance, 0.)
        knn = top.sum(1)/np.maximum(k, 1)
        nearest[available==0] = np.nan; knn[available==0] = np.nan
        add("train_distance", "log1p_mahalanobis_rms_X", np.log1p(np.sqrt(np.mean(pc*pc, axis=1))))
        add("train_distance", "log1p_nearest_fit_distance", np.log1p(nearest))
        add("train_distance", "log1p_knn5_fit_distance", np.log1p(knn))
        add("train_distance", "distance_no_eligible_reference", available==0)
        uncentered_pc = (x/self.x_scale)@self.pca.components_.T
        direction = uncentered_pc/np.maximum(np.linalg.norm(uncentered_pc, axis=1, keepdims=True), EPS)
        cosine = direction@self.fit_direction.T
        cosine[excluded] = -np.inf
        best = np.max(cosine, axis=1); best[available==0] = np.nan
        add("train_distance", "nearest_fit_projected_direction_cosine", best)
        reconstruction = self.pca.inverse_transform(pc*self.pc_scale)
        add("train_distance", "log1p_pca_residual_rms", np.log1p(np.sqrt(np.mean((centered-reconstruction)**2, axis=1))))
        rel = uncentered_pc@self.reliability_vectors
        energy = rel*rel
        denom = np.maximum(energy.sum(1), EPS)
        add("reliability", "reliable_direction_energy_fraction", energy[:, self.reliable_mask].sum(1)/denom)
        add("reliability", "positive_reliability_weighted_energy", energy@np.clip(self.reliability_values, 0., 1.)/denom)
        add("reliability", "first_reliability_axis_energy_fraction", energy[:, 0]/denom)
        scaled = x/self.x_scale
        add("reliability", "pca_retained_uncentered_energy_fraction",
            (uncentered_pc*uncentered_pc).sum(1)/np.maximum((scaled*scaled).sum(1), EPS))
        plate = [str(u.get("roles", {}).get("X", {}).get("plate", "")) for u in units]
        control = np.asarray([self.plate_controls.get(p, np.full(len(CONTROL_NAMES), np.nan)) for p in plate], float)
        if control.shape != (len(x), len(CONTROL_NAMES)):
            raise ValueError("Plate control map must contain normalized_control_summary vectors")
        for j, name in enumerate(CONTROL_NAMES):
            add("plate_controls", name, control[:, j])
        add("plate_controls", "normalized_dmso_unavailable", ~np.isfinite(control).all(1))
        # Well location is observable, but fixed layout can confound biology.
        # Do not include plate/batch identifiers or other compound profiles.
        positions = [str(u.get("roles", {}).get("X", {}).get("well", "")) for u in units]
        edge, known = [], []
        for well in positions:
            match = re.fullmatch(r"([A-P])0?([0-9]{1,2})", well.upper())
            valid = match is not None and 1<=int(match.group(2))<=24
            known.append(valid)
            edge.append(float(match.group(1) in ("A", "P") or int(match.group(2)) in (1, 24)) if valid else np.nan)
        add("position_qc", "edge_well_384", edge)
        add("position_qc", "well_position_unavailable", ~np.asarray(known))
        add("position_qc", "measured_focus_saturation_unavailable", np.ones(len(x)))
        return np.column_stack(columns), names, blocks

    def transform(self, data, metadata):
        raw, names, blocks = self._raw(data, metadata)
        if names != self.names or blocks != self.blocks:
            raise ValueError("Descriptor schema changed")
        filled = np.where(np.isfinite(raw), raw, self.raw_center)
        values = (filled-self.raw_center)/self.raw_scale
        x = np.asarray(data["Y"][:, 0], float)
        direction = x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
        pc = self.latent_pca.transform(direction)/self.latent_pc_scale
        amplitude = np.log(np.maximum(np.sqrt(np.mean(x*x, axis=1)), EPS))
        latent = np.column_stack([pc,
            (amplitude-self.latent_amp_center)/self.latent_amp_scale])
        if not np.isfinite(values).all() or not np.isfinite(latent).all():
            raise ValueError("Nonfinite fitted descriptors")
        return DescriptorMatrix(values, names, blocks, latent, raw, self.report)

    def fit_transform(self, data, metadata):
        """Compatibility convenience: apply this already-fitted transformer."""
        return self.transform(data, metadata)
