"""Compound-disjoint development splits and explicit source-held-out splits."""
import json
from pathlib import Path
import numpy as np
from dataclasses import replace
from .data import _integer_array


def assert_disjoint_splits(ids, splits, *, complete=True):
    ids = np.asarray(ids, str)
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset compound IDs are not unique")
    seen = set()
    for name, indices in splits.items():
        ix = _integer_array(indices,"Split indices")
        if ix.ndim != 1 or len(ix) == 0 or len(set(ix.tolist())) != len(ix):
            raise ValueError(f"Empty, repeated or invalid split: {name}")
        if (ix < 0).any() or (ix >= len(ids)).any():
            raise ValueError("Split index out of range")
        current = set(ids[ix])
        if seen & current:
            raise ValueError("Compound leakage across partitions")
        seen |= current
    if complete and seen != set(ids):
        raise ValueError("Split does not cover dataset")


def development_split(ids, seed=20260911, fractions=(.60, .15, .10, .15)):
    if len(fractions) != 4 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Four positive fractions must sum to one")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    edges = np.rint(np.cumsum(fractions)[:-1] * len(ids)).astype(int)
    result = dict(zip(("train", "validation", "calibration", "evaluation"),
                      [np.sort(x) for x in np.split(perm, edges)]))
    assert_disjoint_splits(ids, result)
    return result


def source_split(well_sources, observed_mask, *, train_sources, validation_sources,
                 calibration_sources, evaluation_sources):
    """Return per-well partition masks before generating any training pairs.

    Compounds may occur in several sources: this tests new-source, known-or-new
    compound generalization. A simultaneous new-compound claim additionally
    requires intersecting these masks with a disjoint compound partition.
    """
    sources = np.asarray(well_sources, str)
    mask = np.asarray(observed_mask, bool)
    if sources.shape != mask.shape:
        raise ValueError("Source labels and observed mask disagree")
    sets = dict(train=set(train_sources), validation=set(validation_sources),
                calibration=set(calibration_sources), evaluation=set(evaluation_sources))
    seen = set()
    for group in sets.values():
        if not group or group & seen:
            raise ValueError("Source partitions must be nonempty and disjoint")
        seen |= group
    return {k: mask & np.isin(sources, sorted(v)) for k, v in sets.items()}


def materialize_source_partitions(dataset, well_sources, *, train_sources, validation_sources,
                                  calibration_sources, evaluation_sources, minimum_wells=2,
                                  allow_known_compounds=False):
    """Export truly source-restricted datasets before fitting or episode pairing.

    Unallocated treatment coordinates, covariates and reference catalog rows are
    physically absent from each returned object. Only declared same-source wells
    are compacted, in their predeclared original order (not outcome rank). This
    supports a train-fitted scaler/template/bank applied later to new-source
    objects. Known-compound new-source evaluation requires explicit opt-in.
    """
    if not isinstance(minimum_wells,(int,np.integer)) or minimum_wells < 2:
        raise ValueError("Source partitions require at least two real wells per unit")
    masks = source_split(well_sources,dataset.observed_mask & dataset.well_mask,
                         train_sources=train_sources,validation_sources=validation_sources,
                         calibration_sources=calibration_sources,evaluation_sources=evaluation_sources)
    result, seen = {},set()
    for name,mask in masks.items():
        compounds = np.flatnonzero(mask.sum(1) >= minimum_wells)
        if not len(compounds):
            raise ValueError(f"No compounds with {minimum_wells} same-source observations in {name}")
        ids = set(dataset.ids[compounds])
        if not allow_known_compounds and ids.intersection(seen):
            raise ValueError("Compound overlap across source partitions; explicitly declare known-compound transfer or supply disjoint units")
        seen |= ids
        counts = mask[compounds].sum(1)
        nw = int(counts.max())
        n,d = len(compounds),dataset.Y.shape[-1]
        fields = {}
        for field in ("Y","cond","reference","reference_mask","groups","observed_mask","well_mask","well_ids","panel_index","n_cells","n_cells_mask"):
            original = getattr(dataset,field)
            fill = "" if original.dtype.kind in "US" else -1 if field in {"groups","panel_index"} else np.nan if field == "Y" else 0
            output = np.full((n,nw,*original.shape[2:]),fill,dtype=original.dtype)
            for i,compound in enumerate(compounds):
                wi = np.flatnonzero(mask[compound])
                output[i,:len(wi)] = original[compound,wi]
            fields[field] = output
        pi = fields["panel_index"]
        used = np.unique(pi[pi >= 0])
        allowed_source_groups = set(dataset.groups[...,0][mask].tolist())
        if any(int(x) not in allowed_source_groups for x in dataset.panel_groups[used,0]):
            raise ValueError("A reference panel from an excluded source was attached to this source partition")
        lookup = {old:new for new,old in enumerate(used)}
        if len(used):
            fields["panel_index"] = np.where(pi>=0,np.asarray([lookup.get(int(x),-1) for x in pi.ravel()]).reshape(pi.shape),-1)
        for field in ("panel_y","panel_ids","panel_identity","panel_groups","panel_members"):
            fields[field] = getattr(dataset,field)[used].copy()
        fields.update(panel_template=np.zeros((len(used),d)),panel_template_mask=np.zeros(len(used),bool))
        result[name] = replace(dataset,ids=dataset.ids[compounds].copy(),chem=dataset.chem[compounds].copy(),
                               chem_mask=dataset.chem_mask[compounds].copy(),library_bank=None,
                               metadata={**dataset.metadata,"source_partition":name,
                                         "source_partition_values":sorted(set(np.asarray(well_sources)[mask])),
                                         "known_compound_transfer":bool(allow_known_compounds),
                                         "source_masks_materialized_before_pairing":True,
                                         "source_partition_original_well_indices":[np.flatnonzero(mask[c]).tolist() for c in compounds]},**fields)
    return result


def merge_compound_disjoint_partitions(partitions):
    """Merge prepared source-restricted partitions for the standard trainer.

    Compound identity is never rewritten to simulate independence. The resulting
    split indices preserve both new-source and new-compound evaluation. The
    scaler, library and templates must still fit only ``splits['train']``.
    """
    expected = ("train","validation","calibration","evaluation")
    if set(partitions) != set(expected):
        raise ValueError("Four named source partitions are required")
    base = partitions["train"]
    seen_ids,seen_sources = set(),set()
    max_w = max(part.Y.shape[1] for part in partitions.values())
    max_p = max(part.panel_index.shape[-1] for part in partitions.values())
    max_members = max(part.panel_members.shape[-1] for part in partitions.values())
    lists = {key:[] for key in ("Y","ids","cond","reference","reference_mask","groups","chem","chem_mask",
                               "observed_mask","well_mask","well_ids","panel_index","n_cells","n_cells_mask")}
    catalogs = {key:[] for key in ("panel_y","panel_ids","panel_identity","panel_groups","panel_members")}
    panel_seen = set()
    offset, panel_offset, splits = 0,0,{}
    for name in expected:
        part = partitions[name]
        if part.feature_names.tolist() != base.feature_names.tolist() or part.dimensions != base.dimensions or part.feature_groups != base.feature_groups:
            raise ValueError("Source partitions need identical frozen measurement/condition schemas")
        if part.metadata.get("train_scaler_applied"):
            raise ValueError("Merge unscaled source partitions; fit a single training-only transform afterward")
        ids = set(part.ids)
        sources = set(part.metadata.get("source_partition_values",[]))
        if not sources or sources.intersection(seen_sources):
            raise ValueError("Source partition provenance missing or overlapping")
        if seen_ids.intersection(ids):
            raise ValueError("Known compounds cannot be merged into the compound-disjoint trainer; identities must not be renamed")
        if panel_seen.intersection(part.panel_ids):
            raise ValueError("A physical reference catalog row is shared across claimed disjoint sources")
        seen_ids.update(ids); seen_sources.update(sources);panel_seen.update(part.panel_ids)
        n,w = part.Y.shape[:2]
        splits[name] = np.arange(offset,offset+n,dtype=int)
        offset += n
        for key in lists:
            value = getattr(part,key)
            if key in {"ids","chem","chem_mask"}:
                lists[key].append(value.copy());continue
            fill = "" if value.dtype.kind in "US" else -1 if key in {"groups","panel_index"} else np.nan if key == "Y" else 0
            shape = (n,max_w,3,max_p) if key == "panel_index" else (n,max_w,*value.shape[2:])
            target = np.full(shape,fill,dtype=value.dtype)
            if key == "panel_index":
                target[:,:w,:,:value.shape[-1]] = np.where(value>=0,value+panel_offset,-1)
            else:
                target[:,:w] = value
            lists[key].append(target)
        for key in catalogs:
            value = getattr(part,key)
            if key == "panel_members":
                padded = np.full((len(value),max_members),"",dtype=object)
                padded[:,:value.shape[1]] = value
                value = padded.astype(str)
            catalogs[key].append(value.copy())
        panel_offset += len(part.panel_y)
    fields = {key:np.concatenate(value,axis=0) for key,value in {**lists,**catalogs}.items()}
    merged = replace(base,**fields,panel_template=None,panel_template_mask=None,library_bank=None,
                     metadata={**base.metadata,"source_partition":"merged_compound_disjoint",
                               "source_partitions":{name:partitions[name].metadata["source_partition_values"] for name in expected},
                               "evidence_scope":"SOURCE_AND_COMPOUND_DISJOINT_PORTABLE_DATA",
                               "source_masks_materialized_before_pairing":True})
    assert_disjoint_splits(merged.ids,splits)
    return merged,splits


def save_split(path, ids, splits, *, evidence_scope):
    assert_disjoint_splits(ids, splits)
    payload = {"evidence_scope": evidence_scope,
               "compound_ids": {k: np.asarray(ids, str)[ix].tolist() for k, ix in splits.items()}}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_split(path, ids):
    payload = json.loads(Path(path).read_text())
    lookup = {str(x): i for i, x in enumerate(ids)}
    result = {k: np.array([lookup[x] for x in v], dtype=int)
              for k, v in payload["compound_ids"].items()}
    assert_disjoint_splits(ids, result)
    return result, payload["evidence_scope"]
