"""Portable measurement records, train-fit transforms and leakage-safe episodes.

Measurements are fixed observed profile coordinates, not model embeddings.  A
record's availability is distinct from whether it has been revealed as context.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from .biology import BiologyRecord, validate_records, fit_vocabulary, validate_vocabulary, encode_relations


SCHEMA_VERSION = 2
REFERENCE_LEVELS = ("source", "batch", "plate")


def _boolean_array(values, label):
    array = np.asarray(values)
    if array.dtype.kind not in "bifu" or not np.isfinite(array).all() or not np.isin(array,[0,1]).all():
        raise ValueError(f"{label} must be finite binary masks, not scores")
    return array.astype(bool,copy=False)


def cellprofiler_groups(feature_names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Group named coordinates by observed compartment and measurement family.

    A group is not asserted to be a biological pathway. Channel tokens are
    preserved in the feature names; families without channels remain valid.
    """
    keys = []
    for name in feature_names:
        parts = str(name).split("_")
        if len(parts) < 2 or parts[0] not in {"Cells", "Cytoplasm", "Nuclei"}:
            keys.append("Other::" + (parts[0] if parts else "unknown"))
        else:
            channels = [p for p in parts[2:] if p.upper() in {"DNA", "RNA", "ER", "AGP", "MITO", "MITOCHONDRIA", "GOLGI", "ACTIN", "DAPI"}]
            keys.append("::".join(parts[:2] + list(dict.fromkeys(channels))))
    names = sorted(set(keys))
    mapping = {name: i for i, name in enumerate(names)}
    return np.asarray([mapping[key] for key in keys], dtype=np.int64), names


@dataclass
class MeasurementDataset:
    Y: np.ndarray
    ids: np.ndarray
    feature_names: np.ndarray
    feature_group_index: np.ndarray
    feature_group_names: list[str]
    cond: np.ndarray
    reference: np.ndarray
    reference_mask: np.ndarray
    groups: np.ndarray
    chem: np.ndarray
    observed_mask: np.ndarray | None = None
    well_mask: np.ndarray | None = None
    well_ids: np.ndarray | None = None
    chem_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Catalog rows are identity-matched reference profiles, not fictitious wells.
    # A row may be a control-identity mean; its actual constituent wells must be
    # listed in panel_members. Repeated hierarchy membership is not replication.
    panel_y: np.ndarray | None = None
    panel_ids: np.ndarray | None = None
    panel_identity: np.ndarray | None = None
    panel_groups: np.ndarray | None = None
    panel_members: np.ndarray | None = None
    panel_index: np.ndarray | None = None
    panel_template: np.ndarray | None = None
    panel_template_mask: np.ndarray | None = None
    n_cells: np.ndarray | None = None
    n_cells_mask: np.ndarray | None = None
    library_bank: Any = field(default=None, repr=False, compare=False)
    biology_records: tuple[BiologyRecord, ...] | None = None
    biology_vocabulary: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.Y = np.asarray(self.Y, dtype=np.float64)
        if self.Y.ndim != 3:
            raise ValueError("Y must have dimensions [compound, well, feature]")
        n, w, d = self.Y.shape
        self.ids = np.asarray(self.ids, dtype=str)
        self.feature_names = np.asarray(self.feature_names, dtype=str)
        self.feature_group_index = _integer_array(self.feature_group_index,"Feature group indices")
        self.feature_group_names = list(map(str, self.feature_group_names))
        self.cond = np.asarray(self.cond, dtype=np.float64)
        self.reference = np.asarray(self.reference, dtype=np.float64)
        self.reference_mask = _boolean_array(self.reference_mask,"Reference mask")
        self.groups = _integer_array(self.groups,"Environment groups")
        self.chem = np.asarray(self.chem, dtype=np.float64)
        self.well_mask = np.ones((n, w), dtype=bool) if self.well_mask is None else _boolean_array(self.well_mask,"Well mask")
        self.observed_mask = (np.isfinite(self.Y).all(-1) & self.well_mask
                              if self.observed_mask is None else _boolean_array(self.observed_mask,"Observed mask"))
        self.chem_mask = np.ones(n, dtype=bool) if self.chem_mask is None else _boolean_array(self.chem_mask,"Chemical mask")
        if self.well_ids is None:
            # Synthetic IDs permit explicit software fixtures, never real-data
            # physical-identity evidence. Portable experiments must supply IDs.
            if not self.metadata.get("fixture", False):
                raise ValueError("Real measurements require explicit physical well_ids")
            self.well_ids = np.asarray([[f"{self.ids[i]}::well_{j}" for j in range(w)] for i in range(n)], dtype=str)
            self.metadata = {**self.metadata, "physical_identity_verified": False, "synthetic_fixture_ids": True}
        else:
            self.well_ids = np.asarray(self.well_ids, dtype=str)
        if self.ids.shape != (n,) or len(set(self.ids)) != n:
            raise ValueError("Each compound needs one unique identifier")
        if self.feature_names.shape != (d,) or len(set(self.feature_names)) != d:
            raise ValueError("Feature names must be unique and match Y")
        if self.feature_group_index.shape != (d,) or not self.feature_group_names:
            raise ValueError("Feature grouping must cover each fixed coordinate")
        if np.any(self.feature_group_index < 0) or np.any(self.feature_group_index >= len(self.feature_group_names)):
            raise ValueError("Invalid feature group index")
        if self.cond.ndim != 3 or self.cond.shape[:2] != (n, w):
            raise ValueError("cond must be [N,W,K]")
        if self.reference.ndim != 4 or self.reference.shape[:3] != (n, w, 3):
            raise ValueError("reference must be [N,W,3,R]")
        if self.reference_mask.shape != (n, w, 3) or self.groups.shape != (n, w, 3):
            raise ValueError("reference mask/groups must have source,batch,plate levels")
        if self.chem.ndim != 2 or self.chem.shape[0] != n or self.chem_mask.shape != (n,):
            raise ValueError("Chemical descriptors must align by compound")
        if self.observed_mask.shape != (n, w) or self.well_mask.shape != (n, w) or self.well_ids.shape != (n, w):
            raise ValueError("Well identifiers and masks must match Y")
        if np.any(self.observed_mask & ~self.well_mask):
            raise ValueError("An observed outcome cannot occupy a padded well")
        physical = self.well_ids[self.well_mask].tolist()
        if any(not x.strip() for x in physical) or len(set(physical)) != len(physical):
            raise ValueError("Physical well_ids must be nonempty and globally unique; fields of view are not repeats")
        if not np.isfinite(self.Y[self.observed_mask]).all():
            raise ValueError("Present measurements must be finite; mark absent outcomes missing")
        if not np.isfinite(self.cond[self.well_mask]).all() or not np.isfinite(self.chem[self.chem_mask]).all():
            raise ValueError("Decision covariates must be finite (encode missingness explicitly)")
        self.chem = np.where(self.chem_mask[:, None], self.chem, 0.0)
        if not np.isfinite(self.reference[self.reference_mask]).all():
            raise ValueError("Available reference summaries must be finite")
        if np.any(self.reference_mask & ~self.well_mask[..., None]):
            raise ValueError("Padded wells cannot expose reference panels")
        self.n_cells = np.zeros((n, w)) if self.n_cells is None else np.asarray(self.n_cells, float)
        self.n_cells_mask = np.zeros((n, w), bool) if self.n_cells_mask is None else _boolean_array(self.n_cells_mask,"Cell count mask")
        if self.n_cells.shape != (n,w) or self.n_cells_mask.shape != (n,w):
            raise ValueError("Cell counts and masks must align with wells")
        if np.any(self.n_cells_mask & ~self.observed_mask) or np.any(~np.isfinite(self.n_cells[self.n_cells_mask])) or np.any(self.n_cells[self.n_cells_mask] <= 0):
            raise ValueError("Executed cell counts are positive observed-well QC, not future information")
        self.n_cells = np.where(self.n_cells_mask, self.n_cells, 0.0)
        self.panel_y = np.empty((0,d)) if self.panel_y is None else np.asarray(self.panel_y, float)
        m = len(self.panel_y)
        self.panel_ids = np.empty(0,str) if self.panel_ids is None else np.asarray(self.panel_ids,str)
        self.panel_identity = np.empty(0,str) if self.panel_identity is None else np.asarray(self.panel_identity,str)
        self.panel_groups = np.empty((0,3),int) if self.panel_groups is None else _integer_array(self.panel_groups,"panel groups")
        self.panel_members = np.empty((0,0),str) if self.panel_members is None else np.asarray(self.panel_members,str)
        self.panel_index = np.full((n,w,3,0),-1,int) if self.panel_index is None else _integer_array(self.panel_index,"panel index")
        self.panel_template = np.zeros((m,d)) if self.panel_template is None else np.asarray(self.panel_template,float)
        self.panel_template_mask = np.zeros(m,bool) if self.panel_template_mask is None else _boolean_array(self.panel_template_mask,"Reference template mask")
        if self.panel_y.shape != (m,d) or self.panel_ids.shape != (m,) or self.panel_identity.shape != (m,) or self.panel_groups.shape != (m,3):
            raise ValueError("Reference catalog dimensions disagree")
        if self.panel_members.ndim != 2 or self.panel_members.shape[0] != m or self.panel_template.shape != (m,d) or self.panel_template_mask.shape != (m,):
            raise ValueError("Reference catalog members/templates disagree")
        if self.panel_index.ndim != 4 or self.panel_index.shape[:3] != (n,w,3) or np.any(self.panel_index < -1) or np.any(self.panel_index >= m):
            raise ValueError("Reference panel lookup must be [N,W,3,P], padded by -1")
        if len(set(self.panel_ids)) != m or any(not x for x in self.panel_ids) or any(not x for x in self.panel_identity):
            raise ValueError("Reference catalog needs unique declared IDs and chemical/control identities")
        if not np.isfinite(self.panel_y).all() or not np.isfinite(self.panel_template[self.panel_template_mask]).all():
            raise ValueError("Reference profiles and available train templates must be finite")
        treatment_ids = set(physical)
        catalog_members_seen = set()
        for member in self.panel_members:
            actual = [x for x in member if x]
            if not actual or len(actual) != len(set(actual)) or treatment_ids.intersection(actual):
                raise ValueError("Every reference profile needs distinct real control wells, disjoint from treatment outcomes")
            if catalog_members_seen.intersection(actual):
                raise ValueError("Reference catalog rows cannot duplicate a physical control; share its catalog index across hierarchy levels")
            catalog_members_seen.update(actual)
        if self.biology_records is not None:
            self.biology_records = validate_records(self.biology_records, self.ids, self.well_ids)
        if self.biology_vocabulary is not None:
            validate_vocabulary(self.biology_vocabulary)

    def __len__(self):
        return len(self.ids)

    @property
    def feature_groups(self) -> dict[str, list[int]]:
        return {name: np.flatnonzero(self.feature_group_index == i).tolist()
                for i, name in enumerate(self.feature_group_names)}

    @property
    def dimensions(self) -> dict[str, int]:
        return {"D": self.Y.shape[-1], "K": self.cond.shape[-1],
                "R": self.reference.shape[-1], "H": self.chem.shape[-1],
                "G": len(self.feature_group_names)}

    def subset(self, indices: Sequence[int]) -> "MeasurementDataset":
        ix = _indices(indices, len(self))
        return replace(self, **{name: getattr(self, name)[ix].copy()
                                for name in ("Y", "ids", "cond", "reference", "reference_mask", "groups", "chem", "observed_mask", "well_mask", "well_ids", "chem_mask", "panel_index", "n_cells", "n_cells_mask")},
                       biology_records=(None if self.biology_records is None else tuple(self.biology_records[i] for i in ix)),
                       metadata={**self.metadata, "subset_compounds": len(ix)})


def _integer_array(values, label):
    a = np.asarray(values)
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        # Empty Python lists have float dtype; the empty integer set is legal.
        if a.size == 0:
            return a.astype(np.int64)
        raise ValueError(f"{label} must contain exact integers, not truncated values")
    return a.astype(np.int64,copy=False)


def _indices(indices: Sequence[int], n: int) -> np.ndarray:
    ix = _integer_array(indices,"Compound indices")
    if ix.ndim != 1 or not len(ix) or np.any(ix < 0) or np.any(ix >= n) or len(set(ix.tolist())) != len(ix):
        raise ValueError("Expected nonempty, unique, in-range compound indices")
    return ix


def split_compounds(dataset: MeasurementDataset, fractions=(0.7, 0.15, 0.15), seed=0,
                    groups: Sequence[str] | None = None) -> dict[str, np.ndarray]:
    """Split compounds (or externally supplied coarser groups) before pairing."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Three positive train/calibration/test fractions must sum to one")
    units = dataset.ids if groups is None else np.asarray(groups, dtype=str)
    if units.shape != dataset.ids.shape:
        raise ValueError("One split group is needed per compound")
    unique = np.unique(units)
    if len(unique) < 3:
        raise ValueError("At least three independent allocation groups are needed")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_train = max(1, min(len(unique) - 2, int(fractions[0] * len(unique))))
    n_cal = max(1, min(len(unique) - n_train - 1, int(fractions[1] * len(unique))))
    parts = (shuffled[:n_train], shuffled[n_train:n_train + n_cal], shuffled[n_train + n_cal:])
    return {key: np.flatnonzero(np.isin(units, part))
            for key, part in zip(("train", "calibration", "test"), parts)}


def save_dataset(dataset: MeasurementDataset, path: str | Path, *, overwrite=False) -> tuple[Path, Path]:
    path = Path(path).with_suffix(".npz")
    sidecar = path.with_suffix(".json")
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError("Refusing to overwrite an existing dataset export")
    path.parent.mkdir(parents=True, exist_ok=True)
    array_names = ("Y", "ids", "feature_names", "feature_group_index", "cond", "reference", "reference_mask", "groups", "chem", "observed_mask", "well_mask", "well_ids", "chem_mask", "panel_y", "panel_ids", "panel_identity", "panel_groups", "panel_members", "panel_index", "panel_template", "panel_template_mask", "n_cells", "n_cells_mask")
    np.savez_compressed(path, **{key: getattr(dataset, key) for key in array_names})
    sidecar.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                                  "feature_group_names": dataset.feature_group_names,
                                  "metadata": dataset.metadata,
                                  "biology_records": (None if dataset.biology_records is None else [r.to_dict() for r in dataset.biology_records]),
                                  "biology_vocabulary": dataset.biology_vocabulary}, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path, sidecar


def load_dataset(path: str | Path) -> MeasurementDataset:
    path = Path(path).with_suffix(".npz")
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["schema_version"] not in {1, SCHEMA_VERSION}:
        raise ValueError("Unsupported portable measurement schema")
    with np.load(path, allow_pickle=False) as arrays:
        return MeasurementDataset(**{key: arrays[key].copy() for key in arrays.files},
                                  feature_group_names=meta["feature_group_names"], metadata=meta["metadata"],
                                  biology_records=meta.get("biology_records"), biology_vocabulary=meta.get("biology_vocabulary"))


@dataclass
class TrainScaler:
    """One invertible affine transform, fitted only to specified training groups."""
    y_center: np.ndarray
    y_scale: np.ndarray
    cond_center: np.ndarray
    cond_scale: np.ndarray
    reference_center: np.ndarray
    reference_scale: np.ndarray
    train_ids: list[str]
    feature_names: list[str]
    minimum_scale: float = 1e-6
    template_identities: list[str] = field(default_factory=list)
    template_values: list[list[float]] = field(default_factory=list)
    template_panel_ids: list[str] = field(default_factory=list)
    template_training_groups: list[list[int]] = field(default_factory=list)
    template_panel_profiles: list[list[float]] = field(default_factory=list)
    template_panel_identity: list[str] = field(default_factory=list)
    template_panel_members: list[list[str]] = field(default_factory=list)
    template_counts: list[int] = field(default_factory=list)
    biology_vocabulary: dict[str, Any] | None = None

    @staticmethod
    def _moments(values, minimum_scale):
        if not len(values):
            raise ValueError("Cannot fit a measurement transform without observations")
        center = np.mean(values, axis=0, dtype=np.float64)
        scale = np.std(values, axis=0, dtype=np.float64)
        scale = np.where(scale >= minimum_scale, scale, 1.0)
        return center, scale

    @classmethod
    def fit(cls, dataset: MeasurementDataset, train_indices: Sequence[int], minimum_scale=1e-6, *, biology_enabled=False,
            biology_evidence_weight_policy="confidence_or_unit_support") -> "TrainScaler":
        ix = _indices(train_indices, len(dataset))
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        ym = dataset.observed_mask[ix]
        yc, ys = cls._moments(dataset.Y[ix][ym], minimum_scale)
        cc, cs = cls._moments(dataset.cond[ix][dataset.well_mask[ix]], minimum_scale)
        r = dataset.reference.shape[-1]
        rc, rs = np.zeros((3, r)), np.ones((3, r))
        for level in range(3):
            available = dataset.reference_mask[ix, :, level]
            # Repeated sharing within a compound is allowed; no held-out group is read.
            if available.any():
                rc[level], rs[level] = cls._moments(dataset.reference[ix, :, level][available], minimum_scale)
        # Reference-identity templates are fitted to training-owned panels only.
        # Selection comes from the well mask, not every panel in the catalog.
        indices = np.where(dataset.reference_mask[ix][...,None],dataset.panel_index[ix],-1)[dataset.well_mask[ix]]
        used = np.unique(indices[indices >= 0])
        identities, templates, counts = [], [], []
        for identity in sorted(set(dataset.panel_identity[used])):
            selected = used[dataset.panel_identity[used] == identity]
            identities.append(str(identity))
            templates.append(dataset.panel_y[selected].mean(0).tolist())
            counts.append(len(selected))
        biology = None
        if biology_enabled:
            records = dataset.biology_records or tuple(BiologyRecord(str(unit)) for unit in dataset.ids)
            biology = fit_vocabulary(records, dataset.ids[ix].tolist(), biology_evidence_weight_policy)
        return cls(yc, ys, cc, cs, rc, rs, dataset.ids[ix].tolist(), dataset.feature_names.tolist(), minimum_scale,
                   identities, templates, dataset.panel_ids[used].tolist(), dataset.panel_groups[used].tolist(),
                   [], dataset.panel_identity[used].tolist(), dataset.panel_members[used].tolist(),counts,biology)

    def transform_y(self, values):
        if torch.is_tensor(values):
            center = torch.as_tensor(self.y_center, dtype=values.dtype, device=values.device)
            scale = torch.as_tensor(self.y_scale, dtype=values.dtype, device=values.device)
        else:
            center, scale = self.y_center, self.y_scale
        return (values - center) / scale

    def inverse_y(self, values):
        """Invert any [*,D] ndarray/tensor, including joint Monte Carlo samples."""
        if torch.is_tensor(values):
            center = torch.as_tensor(self.y_center, dtype=values.dtype, device=values.device)
            scale = torch.as_tensor(self.y_scale, dtype=values.dtype, device=values.device)
        else:
            center, scale = self.y_center, self.y_scale
        return values * scale + center

    def transform(self, dataset: MeasurementDataset) -> MeasurementDataset:
        if dataset.feature_names.tolist() != self.feature_names:
            raise ValueError("Fixed measurement-coordinate order changed")
        if dataset.metadata.get("train_scaler_applied"):
            raise ValueError("Refusing to apply a training scaler twice")
        y = self.transform_y(dataset.Y)
        cond = (dataset.cond - self.cond_center) / self.cond_scale
        ref = (dataset.reference - self.reference_center) / self.reference_scale
        ref = np.where(dataset.reference_mask[..., None], ref, 0.0)
        lookup = dict(zip(self.template_identities, self.template_values))
        template_mask = np.array([identity in lookup for identity in dataset.panel_identity], bool)
        template = np.array([lookup.get(identity, np.zeros(dataset.Y.shape[-1])) for identity in dataset.panel_identity],float).reshape(len(dataset.panel_y),dataset.Y.shape[-1])
        if self.template_counts:
            counts = dict(zip(self.template_identities,self.template_counts))
            trained_ids = set(self.template_panel_ids)
            for row,identity in enumerate(dataset.panel_identity):
                if dataset.panel_ids[row] in trained_ids:
                    count = counts.get(identity,0)
                    template_mask[row] = count > 1
                    template[row] = ((np.asarray(lookup[identity])*count - dataset.panel_y[row])/(count-1)
                                     if count > 1 else 0.)
        elif self.template_panel_profiles:
            profiles = np.asarray(self.template_panel_profiles)
            for row, identity in enumerate(dataset.panel_identity):
                members = set(dataset.panel_members[row]) - {""}
                legal = [i for i, value in enumerate(self.template_panel_identity)
                         if value == identity and not members.intersection(self.template_panel_members[i])]
                template_mask[row] = bool(legal)
                template[row] = profiles[legal].mean(0) if legal else 0.
        # Existing fitted templates are never trusted on a newly supplied data
        # file: this scaler's checkpoint-bound train templates are authoritative.
        template = np.where(template_mask[:,None], self.transform_y(template), 0.0)
        return replace(dataset, Y=y, cond=cond, reference=ref, panel_y=self.transform_y(dataset.panel_y),
                       panel_template=template, panel_template_mask=template_mask,
                       biology_vocabulary=self.biology_vocabulary,
                       metadata={**dataset.metadata, "train_scaler_applied": True,
                                 "scaler_training_compounds": len(self.train_ids),
                                 "likelihood_coordinate_transform": "training-only affine; invert before fixed utility"})

    def to_dict(self):
        return {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("y_center", "y_scale", "cond_center", "cond_scale", "reference_center", "reference_scale"):
            value[key] = np.asarray(value[key], dtype=np.float64)
        return cls(**value)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def _well_indices(indices, w, label, allow_empty=False):
    value = _integer_array(indices, f"{label} indices")
    if value.ndim != 1 or (not len(value) and not allow_empty) or np.any(value < 0) or np.any(value >= w) or len(set(value.tolist())) != len(value):
        raise ValueError(f"{label} indices must be unique and in range")
    return value


def _reference_access(dataset, compounds, targets, contexts, reference_access):
    """Availability policy, not normalization access, governs predictor inputs."""
    if reference_access not in {"observed_only", "all_declared", "none"}:
        raise ValueError("reference_access must be observed_only, all_declared, or none")
    cmask = dataset.reference_mask[compounds][:, contexts].copy()
    tmask = dataset.reference_mask[compounds][:, targets].copy()
    if reference_access == "none":
        cmask[:] = False
        tmask[:] = False
    elif reference_access == "observed_only":
        # A target's own source/batch summary may use controls not yet revealed.
        # Do not copy it merely because a categorical source ID matches context.
        tmask[:] = False
    return cmask, tmask


@dataclass
class LibraryBank:
    """Checkpoint-bound initial-well training context, never target profiles."""
    Y: np.ndarray
    cond: np.ndarray
    groups: np.ndarray
    ids: np.ndarray
    well_ids: np.ndarray
    max_neighbors: int = 32
    feature_names: list[str] = field(default_factory=list)
    scaled: bool = False

    def __post_init__(self):
        self.Y, self.cond = np.asarray(self.Y,float), np.asarray(self.cond,float)
        self.groups, self.ids, self.well_ids = np.asarray(self.groups,int),np.asarray(self.ids,str),np.asarray(self.well_ids,str)
        if self.Y.ndim != 2 or self.cond.ndim != 2 or len(self.Y) != len(self.ids) or self.groups.shape != (len(self.ids),3):
            raise ValueError("Library bank arrays disagree")
        if self.well_ids.shape != self.ids.shape or len(set(self.ids)) != len(self.ids) or len(set(self.well_ids)) != len(self.well_ids):
            raise ValueError("Library bank requires one unique physical initial well per compound")
        if not np.isfinite(self.Y).all() or not np.isfinite(self.cond).all() or self.max_neighbors < 1:
            raise ValueError("Invalid library values or neighbor limit")

    def save(self,path):
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,Y=self.Y,cond=self.cond,groups=self.groups,ids=self.ids,well_ids=self.well_ids,
                            max_neighbors=np.array(self.max_neighbors),feature_names=np.asarray(self.feature_names,str),scaled=np.array(self.scaled))
        return path

    @classmethod
    def load(cls,path):
        with np.load(Path(path).with_suffix(".npz"),allow_pickle=False) as a:
            return cls(a["Y"],a["cond"],a["groups"],a["ids"],a["well_ids"],int(a["max_neighbors"]),
                       a["feature_names"].tolist(),bool(a["scaled"]))


def fit_library_context(dataset: MeasurementDataset, train_indices: Sequence[int], *, context_index=0,
                        max_neighbors=32) -> LibraryBank:
    ix = _indices(train_indices,len(dataset))
    wi = int(_well_indices([context_index],dataset.Y.shape[1],"Library initial well")[0])
    available = dataset.observed_mask[ix,wi] & dataset.well_mask[ix,wi]
    ix = ix[available]
    if not len(ix):
        raise ValueError("No authorized initial training wells for library context")
    return LibraryBank(dataset.Y[ix,wi].copy(),dataset.cond[ix,wi].copy(),dataset.groups[ix,wi].copy(),
                       dataset.ids[ix].copy(),dataset.well_ids[ix,wi].copy(),max_neighbors,
                       dataset.feature_names.tolist(),bool(dataset.metadata.get("train_scaler_applied")))


def attach_library_context(dataset: MeasurementDataset, bank: LibraryBank) -> MeasurementDataset:
    if bank.feature_names != dataset.feature_names.tolist() or bank.scaled != bool(dataset.metadata.get("train_scaler_applied")):
        raise ValueError("Library bank and measurement coordinate transform disagree")
    return replace(dataset,library_bank=bank)


def _library_inputs(dataset,ix,ci):
    b,d,k = len(ix),dataset.Y.shape[-1],dataset.cond.shape[-1]
    bank = dataset.library_bank
    length = 0 if bank is None else min(bank.max_neighbors,len(bank.ids))
    y,c,g,m = np.zeros((b,length,d)),np.zeros((b,length,k)),np.full((b,length,3),-1,int),np.zeros((b,length),bool)
    bank_index = np.full((b,length),-1,np.int64)
    mean,var,count,density = np.zeros((b,d)),np.zeros((b,d)),np.zeros(b),np.zeros((b,2))
    if bank is not None:
        for row,compound in enumerate(ix):
            legal = np.flatnonzero(bank.ids != dataset.ids[compound])
            if not len(legal):
                continue
            profiles = bank.Y[legal]
            mean[row],var[row],count[row] = profiles.mean(0),profiles.var(0),len(legal)
            visible = ci[dataset.observed_mask[compound,ci] & dataset.well_mask[compound,ci]]
            if len(visible):
                initial = int(visible[0])
                query = dataset.Y[compound,initial]
                distance = np.mean(np.square(profiles-query),axis=1)
                query_group = dataset.groups[compound,initial]
                tier = np.where(bank.groups[legal,2] == query_group[2],0,
                       np.where(bank.groups[legal,0] == query_group[0],1,2))
            else:
                # Zero-well state: condition matching is legal, profile querying
                # is not. No hidden X read is performed in this branch.
                planned = np.flatnonzero(dataset.well_mask[compound])
                query_cond = dataset.cond[compound,planned[0]] if len(planned) else np.zeros(k)
                distance = np.mean(np.square(bank.cond[legal]-query_cond),axis=1)
                tier = np.ones(len(legal),int)
            order = np.lexsort((bank.ids[legal],distance,tier))[:length]
            chosen = legal[order]
            y[row,:len(chosen)],c[row,:len(chosen)],g[row,:len(chosen)],m[row,:len(chosen)] = bank.Y[chosen],bank.cond[chosen],bank.groups[chosen],True
            bank_index[row,:len(chosen)] = chosen
            density[row] = (float(np.mean(distance[order])),float(np.mean(np.exp(-np.minimum(distance,700.)))))
    return {"library_y":y,"library_cond":c,"library_group":g,"library_mask":m,"library_index":bank_index,
            "library_global_mean":mean,"library_global_variance":var,"library_count":count,
            "library_density":density}


def make_inference_batch(dataset: MeasurementDataset, indices: Sequence[int],
                         context_indices=(0,), target_indices=(1, 2, 3), *,
                         reference_access="observed_only", device=None, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """Build only legal inputs. Target Y is never accessed by this function.

    ``all_declared`` is an explicit assumption that each available stored panel
    was obtained before the decision; it is not established by file existence.
    All target references are hidden under the conservative observed_only mode.
    """
    ix = _indices(indices, len(dataset))
    ci = _well_indices(context_indices, dataset.Y.shape[1], "Context", allow_empty=True)
    ti = _well_indices(target_indices, dataset.Y.shape[1], "Target")
    if set(ci) & set(ti):
        raise ValueError("A candidate future well cannot already be in context")
    cm = dataset.observed_mask[ix][:, ci] & dataset.well_mask[ix][:, ci]
    physical = dataset.well_ids[dataset.well_mask].tolist()
    if len(set(physical)) != len(physical):
        raise ValueError("Duplicate physical well identity at inference")
    tm = dataset.well_mask[ix][:, ti]
    crmask, trmask = _reference_access(dataset, ix, ti, ci, reference_access)
    crmask &= cm[..., None]
    trmask &= tm[..., None]
    # np.ix_ avoids materializing target outcomes when extracting context.
    cy = dataset.Y[np.ix_(ix, ci, np.arange(dataset.Y.shape[-1]))]
    arrays = {
        "context_y": np.where(cm[..., None], cy, 0.0),
        "context_cond": np.where(cm[..., None], dataset.cond[ix][:, ci], 0.0),
        "context_mask": cm,
        "context_reference": np.where(crmask[..., None], dataset.reference[ix][:, ci], 0.0),
        "context_reference_mask": crmask,
        "context_group": dataset.groups[ix][:, ci],
        "target_cond": np.where(tm[..., None], dataset.cond[ix][:, ti], 0.0),
        "target_reference": np.where(trmask[..., None], dataset.reference[ix][:, ti], 0.0),
        "target_reference_mask": trmask,
        "target_group": dataset.groups[ix][:, ti],
        "target_mask": tm,
        "chem": np.where(dataset.chem_mask[ix,None],dataset.chem[ix],0.0),
        "chem_mask": dataset.chem_mask[ix],
        "context_n_cells": np.where(cm & dataset.n_cells_mask[ix][:,ci],dataset.n_cells[ix][:,ci],0.),
        "context_n_cells_mask": cm & dataset.n_cells_mask[ix][:,ci],
        "target_n_cells": np.zeros(tm.shape),
        "target_n_cells_mask": np.zeros(tm.shape,bool),
    }
    # Only accessible catalog entries are materialized. No B*C*P*D replication:
    # a shared reference is encoded once and referenced by integer catalog index.
    raw_panel_indices = {}
    for prefix,wi,rmask in (("context",ci,crmask),("target",ti,trmask)):
        pi = dataset.panel_index[ix][:,wi]
        raw_panel_indices[prefix] = np.where((pi>=0)&rmask[...,None],pi,-1)
    union = np.unique(np.concatenate([v.ravel() for v in raw_panel_indices.values()]))
    union = union[union>=0]
    remap = {old:new for new,old in enumerate(union)}
    arrays.update(panel_catalog_y=dataset.panel_y[union],panel_catalog_template=dataset.panel_template[union],
                  panel_catalog_template_mask=dataset.panel_template_mask[union],panel_catalog_ids=union)
    for prefix, wi, rmask in (("context",ci,crmask),("target",ti,trmask)):
        pi = dataset.panel_index[ix][:,wi]
        valid = (pi >= 0) & rmask[...,None]
        ptmask = np.zeros(pi.shape,bool)
        identity = np.full(pi.shape,-1,np.int64)
        mapped = np.full(pi.shape,-1,np.int64)
        if valid.any():
            rows = pi[valid]
            ptmask[valid] = dataset.panel_template_mask[rows]
            mapped[valid] = [remap[int(x)] for x in rows]
            identities = {v:i for i,v in enumerate(sorted(set(dataset.panel_identity)))}
            identity[valid] = [identities[x] for x in dataset.panel_identity[rows]]
        arrays.update({prefix+"_panel_index":mapped,
                       prefix+"_panel_mask":valid,prefix+"_panel_template_mask":ptmask,
                       prefix+"_panel_identity":identity})
    arrays.update(_library_inputs(dataset,ix,ci))
    if dataset.biology_vocabulary is not None:
        records = (tuple(BiologyRecord(str(dataset.ids[i])) for i in ix) if dataset.biology_records is None
                   else tuple(dataset.biology_records[i] for i in ix))
        arrays.update(encode_relations(records, dataset.biology_vocabulary))
    return {key: torch.as_tensor(np.array(value, copy=True), device=device,
                                 dtype=torch.bool if value.dtype == bool else torch.long if np.issubdtype(value.dtype, np.integer) else dtype)
            for key, value in arrays.items()}


def make_episode(dataset: MeasurementDataset, compound_index: int, context_indices=(0,),
                 target_indices=(1, 2, 3), *, reference_access="observed_only", dtype=torch.float32):
    """Training object with target values outside the inference dictionary."""
    inputs = make_inference_batch(dataset, [compound_index], context_indices, target_indices,
                                  reference_access=reference_access, dtype=dtype)
    inputs = {key: value if key.startswith("panel_catalog_") else value[0] for key, value in inputs.items()}
    ti = np.asarray(target_indices, dtype=np.int64)
    mask = dataset.observed_mask[compound_index, ti] & dataset.well_mask[compound_index, ti]
    target_y = np.where(mask[:, None], dataset.Y[compound_index, ti], 0.0)
    return {"inputs": inputs, "target_y": torch.as_tensor(target_y.copy(), dtype=dtype),
            "target_mask": torch.as_tensor(mask.copy()), "compound_index": int(compound_index),
            "context_indices": tuple(map(int, context_indices)), "target_indices": tuple(map(int, target_indices))}


def make_training_batch(dataset: MeasurementDataset, indices: Sequence[int],
                        context_indices=(0,), target_indices=(1, 2, 3), *,
                        reference_access="observed_only", device=None, dtype=torch.float32):
    """Fixed-role batch with outcomes explicitly outside the model inputs."""
    inputs = make_inference_batch(dataset, indices, context_indices, target_indices,
                                  reference_access=reference_access, device=device, dtype=dtype)
    ix = _indices(indices, len(dataset))
    ti = _well_indices(target_indices, dataset.Y.shape[1], "Target")
    mask = dataset.observed_mask[ix][:, ti] & dataset.well_mask[ix][:, ti]
    y = dataset.Y[np.ix_(ix, ti, np.arange(dataset.Y.shape[-1]))]
    return {"inputs": inputs,
            "target_y": torch.as_tensor(np.where(mask[..., None], y, 0.0), dtype=dtype, device=device),
            "target_mask": torch.as_tensor(mask.copy(), dtype=torch.bool, device=device),
            "compound_index": torch.as_tensor(ix, dtype=torch.long, device=device)}


class EpisodeDataset(Dataset):
    """Enumerate context subsets only within an already selected compound split.

    All candidate targets are the complement unless target_size is specified.
    Example expansion creates learning tasks, not additional independent units.
    """
    def __init__(self, dataset: MeasurementDataset, compound_indices: Sequence[int],
                 context_sizes=(1, 2, 3, 4), target_size: int | None = None,
                 reference_access="observed_only", max_episodes_per_compound: int | None = None, seed=0):
        self.dataset = dataset
        self.compound_indices = _indices(compound_indices, len(dataset))
        self.reference_access = reference_access
        self.episodes = []
        if target_size is not None and target_size < 1:
            raise ValueError("target_size must be positive or None")
        context_sizes = sorted(set(_integer_array(context_sizes,"context sizes").tolist()))
        if not context_sizes or min(context_sizes) < 0:
            raise ValueError("Context sizes must be nonnegative")
        rng = np.random.default_rng(seed)
        for compound in self.compound_indices:
            available = np.flatnonzero(dataset.observed_mask[compound] & dataset.well_mask[compound]).tolist()
            local = []
            for c in context_sizes:
                if c >= len(available):
                    continue
                for context in combinations(available, c):
                    remaining = tuple(j for j in available if j not in context)
                    choices = [remaining] if target_size is None else combinations(remaining, target_size)
                    local.extend((int(compound), context, tuple(target)) for target in choices)
            if max_episodes_per_compound is not None:
                if max_episodes_per_compound < 1:
                    raise ValueError("max_episodes_per_compound must be positive")
                if len(local) > max_episodes_per_compound:
                    selected = np.sort(rng.choice(len(local), max_episodes_per_compound, replace=False))
                    local = [local[i] for i in selected]
            self.episodes.extend(local)
        if not self.episodes:
            raise ValueError("No observed context/target episodes within this allocation")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        compound, context, target = self.episodes[index]
        return make_episode(self.dataset, compound, context, target, reference_access=self.reference_access)


def collate_episodes(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable context/target lengths. Padding never becomes an observation."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    max_c = max(item["inputs"]["context_y"].shape[0] for item in items)
    max_t = max(item["target_y"].shape[0] for item in items)
    inputs = {}
    catalog_ids = sorted({int(x) for item in items for x in item["inputs"]["panel_catalog_ids"].tolist()})
    catalog_lookup = {old:new for new,old in enumerate(catalog_ids)}
    catalogs = {}
    for item in items:
        value = item["inputs"]
        for i,raw_id in enumerate(value["panel_catalog_ids"].tolist()):
            catalogs[int(raw_id)] = {key:tensor[i] for key,tensor in value.items() if key.startswith("panel_catalog_")}
    for key,value in items[0]["inputs"].items():
        if key.startswith("panel_catalog_"):
            inputs[key] = torch.stack([catalogs[i][key] for i in catalog_ids]) if catalog_ids else value[:0]
    for key in items[0]["inputs"]:
        if key.startswith("panel_catalog_"):
            continue
        values = [item["inputs"][key] for item in items]
        if key.startswith("biology_"):
            if values[0].ndim == 0:
                inputs[key] = torch.stack(values)
            else:
                max_r = max(len(value) for value in values)
                result = values[0].new_zeros((len(values), max_r, *values[0].shape[1:]))
                for i, value in enumerate(values):
                    result[i, :len(value)] = value
                inputs[key] = result
            continue
        if key in {"context_panel_index","target_panel_index"}:
            values = [value.clone() for value in values]
            for value,item in zip(values,items):
                local_ids = item["inputs"]["panel_catalog_ids"].tolist()
                valid = value>=0
                value[valid] = torch.tensor([catalog_lookup[local_ids[x]] for x in value[valid].tolist()],dtype=value.dtype)
        if key in {"chem", "chem_mask"} or key.startswith("library_"):
            inputs[key] = torch.stack(values)
            continue
        length = max_c if key.startswith("context_") else max_t
        fill = -1 if key.endswith("group") or key.endswith("panel_index") or key.endswith("panel_identity") else 0
        result = values[0].new_full((len(items), length, *values[0].shape[1:]), fill)
        for i, value in enumerate(values):
            result[i, :len(value)] = value
        inputs[key] = result
    y = items[0]["target_y"].new_zeros((len(items), max_t, items[0]["target_y"].shape[-1]))
    mask = torch.zeros((len(items), max_t), dtype=torch.bool)
    for i, item in enumerate(items):
        y[i, :len(item["target_y"])] = item["target_y"]
        mask[i, :len(item["target_mask"])] = item["target_mask"]
    return {"inputs": inputs, "target_y": y, "target_mask": mask,
            "compound_index": torch.tensor([item["compound_index"] for item in items]),
            "context_indices": [item["context_indices"] for item in items],
            "target_indices": [item["target_indices"] for item in items]}
