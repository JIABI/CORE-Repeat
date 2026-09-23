"""Batch-pair conditional affine shrinkage plus compound-joint residual bootstrap.

The group key is (context source, context batch, target source, target batch).
Plate identity is supplied for an explicit three-level schema but is NOT fitted:
small plate groups would otherwise confound chemical composition with offsets.
This is a within-known-batch comparator, not zero-shot biological transfer.

One shared per-feature affine mapping is fitted to all observed target wells.
Each batch pair receives a ridge-shrunk intercept and slope deviation. Residuals
are recentered within their fitted batch pair, so bootstrap mean offsets do not
silently undo shrinkage, then retained as whole compound vectors, preserving feature and
between-well dependence. Exact matching schedules can reorder their residual
columns without changing predictions when the query listing is permuted.

For an unseen pair, the mean uses the TRAINING pooled mapping. If the full
schedule has too few joint donors, the noise uses one training compound's
distinct randomly permuted wells; this explicit pooled-exchangeability fallback
does not claim that unseen-batch covariance was learned. In-sample residual
bootstrap spread is not a calibrated predictive interval or a likelihood model.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def _groups(context_groups, target_groups, n=None):
    c, t = np.asarray(context_groups), np.asarray(target_groups)
    if c.ndim != 2 or c.shape[1] != 3 or t.ndim != 3 or t.shape[0] != len(c) or t.shape[2] != 3:
        raise ValueError("Groups must have source/batch/plate columns: context[N,3], target[N,T,3]")
    if not len(c) or not t.shape[1] or (n is not None and len(c) != n):
        raise ValueError("Nonempty group rows must align with measurement rows")
    c, t = c.astype(str), t.astype(str)
    if np.any(c == "") or np.any(t == ""):
        raise ValueError("Use explicit unknown identity tokens, not empty group fields")
    keys = np.empty(t.shape[:2], dtype=object)
    for i in range(len(c)):
        for j in range(t.shape[1]):
            keys[i, j] = json.dumps([c[i, 0], c[i, 1], t[i, j, 0], t[i, j, 1]], separators=(",", ":"))
    return keys.astype(str)


class BatchConditionalResidualBootstrap:
    def __init__(self, *, global_ridge=1., offset_shrinkage=20., slope_shrinkage=100.,
                 min_schedule_donors=3, scale_floor=1e-6):
        if (not np.isfinite([global_ridge, offset_shrinkage, slope_shrinkage, scale_floor]).all()
                or min(global_ridge, offset_shrinkage, slope_shrinkage, scale_floor) <= 0):
            raise ValueError("Ridge strengths and scale floor must be strictly positive")
        if not isinstance(min_schedule_donors, (int, np.integer)) or min_schedule_donors < 1:
            raise ValueError("min_schedule_donors must be a positive integer")
        self.global_ridge = float(global_ridge)
        self.offset_shrinkage = float(offset_shrinkage)
        self.slope_shrinkage = float(slope_shrinkage)
        self.min_schedule_donors = int(min_schedule_donors)
        self.scale_floor = float(scale_floor)

    def fit(self, train_X, targets, context_groups, target_groups, *, train_ids=None, target_mask=None):
        x, y = np.asarray(train_X, float), np.asarray(targets, float)
        if x.ndim != 2 or y.ndim != 3 or x.shape != (len(y), y.shape[-1]) or len(x) < 2 or not x.shape[1]:
            raise ValueError("Expected at least two training compounds: X[N,D], targets[N,T,D]")
        keys = _groups(context_groups, target_groups, len(x))
        if keys.shape != y.shape[:2]:
            raise ValueError("Target group count does not match the target wells")
        mask = np.ones(y.shape[:2], bool) if target_mask is None else np.asarray(target_mask, bool)
        if mask.shape != y.shape[:2] or not mask.any() or not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
            raise ValueError("Finite context/observed target values and an aligned target mask are required")
        ids = tuple(map(str, train_ids)) if train_ids is not None else tuple(f"training_row_{i}" for i in range(len(x)))
        if len(ids) != len(x) or len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError("Training IDs must identify distinct compounds")
        self.train_ids = ids
        self.feature_count, self.target_count = y.shape[-1], y.shape[1]
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.maximum(x.std(axis=0), self.scale_floor)
        z = (x - self.x_mean) / self.x_scale
        row, col = np.nonzero(mask)
        observed_x, observed_y = z[row], y[row, col]
        # Repeated context rows receive one contribution per observed future well.
        mean_z, mean_y = observed_x.mean(0), observed_y.mean(0)
        dx, dy = observed_x - mean_z, observed_y - mean_y
        self.global_slope = (dx * dy).sum(0) / ((dx * dx).sum(0) + self.global_ridge)
        self.global_intercept = mean_y - mean_z * self.global_slope
        self.group_keys = tuple(sorted(set(keys[mask])))
        self.group_lookup = {k: i for i, k in enumerate(self.group_keys)}
        self.group_intercept = np.zeros((len(self.group_keys), self.feature_count))
        self.group_slope = np.zeros_like(self.group_intercept)
        self.group_counts = np.zeros(len(self.group_keys), int)
        pooled_residual = observed_y - self.global_intercept - observed_x * self.global_slope
        observed_keys = keys[row, col]
        for key, index in self.group_lookup.items():
            use = observed_keys == key
            zg, residual = observed_x[use], pooled_residual[use]
            count = int(use.sum())
            sx, sxx = zg.sum(0), (zg * zg).sum(0)
            a, b, cross = count + self.offset_shrinkage, sxx + self.slope_shrinkage, sx
            rhs_a, rhs_b = residual.sum(0), (zg * residual).sum(0)
            determinant = a * b - cross * cross
            self.group_intercept[index] = (b * rhs_a - cross * rhs_b) / determinant
            self.group_slope[index] = (a * rhs_b - cross * rhs_a) / determinant
            self.group_counts[index] = count
        self.training_keys, self.training_mask = keys.copy(), mask.copy()
        fitted = self._predict_from_keys(x, keys)
        self.residuals = np.where(mask[..., None], y - fitted, 0.)
        self.residual_center = np.zeros_like(self.group_intercept)
        for key, index in self.group_lookup.items():
            use = (keys == key) & mask
            self.residual_center[index] = self.residuals[use].mean(0)
            self.residuals[use] -= self.residual_center[index]
        self._schedule_cache = {}
        return self

    def _predict_from_keys(self, x, keys):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape != (len(keys), self.feature_count) or not np.isfinite(x).all():
            raise ValueError("Finite query X[N,D] must use the training feature schema")
        z = (x - self.x_mean) / self.x_scale
        mean = np.broadcast_to(self.global_intercept + z[:, None, :] * self.global_slope,
                               (len(x), keys.shape[1], self.feature_count)).copy()
        for key in np.unique(keys):
            index = self.group_lookup.get(key)
            if index is not None:
                rows, cols = np.nonzero(keys == key)
                mean[rows, cols] += self.group_intercept[index] + z[rows] * self.group_slope[index]
        return mean

    def predict_mean(self, X, context_groups, target_groups):
        return self._predict_from_keys(X, _groups(context_groups, target_groups, len(X)))

    predict = predict_mean

    def _matching_donors(self, plan):
        plan = tuple(plan)
        if plan not in self._schedule_cache:
            donors, mappings = [], []
            for i, row in enumerate(self.training_keys):
                available = defaultdict(list)
                for j, key in enumerate(row):
                    if self.training_mask[i, j]:
                        available[key].append(j)
                used, mapping = defaultdict(int), []
                for key in plan:
                    count = used[key]
                    if count >= len(available[key]):
                        break
                    mapping.append(available[key][count])
                    used[key] += 1
                if len(mapping) == len(plan):
                    donors.append(i); mappings.append(mapping)
            self._schedule_cache[plan] = (np.asarray(donors, int), np.asarray(mappings, int).reshape(-1, len(plan)))
        return self._schedule_cache[plan]

    def prediction_diagnostics(self, context_groups, target_groups):
        keys = _groups(context_groups, target_groups)
        seen = np.isin(keys, self.group_keys)
        counts = np.array([len(self._matching_donors(row)[0]) for row in keys])
        fallback_available = int((self.training_mask.sum(axis=1) >= keys.shape[1]).sum())
        return {"batch_pair_seen": seen.tolist(), "unseen_pair_count": int((~seen).sum()),
                "matched_schedule_donors": counts.tolist(),
                "pooled_noise_fallback": (counts < self.min_schedule_donors).tolist(),
                "available_pooled_joint_donors": fallback_available,
                "grouping": "context source/batch -> planned target source/batch; plate not fitted",
                "unseen_mean_fallback": "training-only shared per-feature affine mapping",
                "unmatched_noise_fallback": "one training compound, distinct randomly permuted observed target wells",
                "scope": "known-batch conditional empirical residual baseline; unseen batches are pooled fallback, not zero-shot transfer"}

    def sample_joint(self, X, context_groups, target_groups, n_samples=128, *, seed=0, dtype=np.float32):
        """Return [draw,compound,target,feature]; never receives future target values.

        Joint donors are independent across query compounds. They preserve a
        donor's cross-well residual structure but do not simulate an additional
        campaign-wide latent batch shock shared by all query compounds.
        """
        if not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer")
        keys = _groups(context_groups, target_groups, len(X))
        mean = self._predict_from_keys(X, keys)
        t = keys.shape[1]
        pooled = np.flatnonzero(self.training_mask.sum(1) >= t)
        rng = np.random.default_rng(seed)
        samples = np.broadcast_to(mean[None], (n_samples,) + mean.shape).astype(dtype, copy=True)
        for i, row in enumerate(keys):
            donors, mappings = self._matching_donors(row)
            if len(donors) >= self.min_schedule_donors:
                choices = rng.integers(len(donors), size=n_samples)
                selected_donors, selected_columns = donors[choices], mappings[choices]
            else:
                if not len(pooled):
                    raise ValueError("No training compound has enough jointly observed wells for this plan's pooled fallback")
                selected_donors = rng.choice(pooled, size=n_samples)
                selected_columns = np.array([rng.choice(np.flatnonzero(self.training_mask[d]), size=t, replace=False)
                                              for d in selected_donors])
            samples[:, i] += self.residuals[selected_donors[:, None], selected_columns].astype(dtype)
        return samples

    def metadata(self):
        return {"model": type(self).__name__, "version": 1,
                "parameters": {name: getattr(self, name) for name in
                               ("global_ridge", "offset_shrinkage", "slope_shrinkage", "min_schedule_donors", "scale_floor")},
                "train_ids": list(self.train_ids), "feature_count": self.feature_count,
                "target_count": self.target_count, "group_keys": list(self.group_keys),
                "group_counts": self.group_counts.tolist(),
                "training_compounds": len(self.train_ids), "training_observed_wells": int(self.training_mask.sum()),
                "grouping": "source/batch pair, not role or plate", "residual_centering": "training batch-pair empirical mean",
                "predictive_calibration_certified": False}

    def save(self, path):
        arrays = {name: getattr(self, name) for name in ("x_mean", "x_scale", "global_intercept", "global_slope",
                  "group_intercept", "group_slope", "group_counts", "training_keys", "training_mask", "residuals", "residual_center")}
        with Path(path).open("wb") as handle:
            np.savez_compressed(handle, metadata=np.array(json.dumps(self.metadata())), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            if meta.get("version") != 1 or meta.get("model") != cls.__name__:
                raise ValueError("Unsupported batch-baseline checkpoint schema")
            model = cls(**meta["parameters"])
            for name in archive.files:
                if name != "metadata":
                    setattr(model, name, archive[name].copy())
        model.train_ids = tuple(meta["train_ids"])
        model.feature_count, model.target_count = meta["feature_count"], meta["target_count"]
        model.group_keys = tuple(meta["group_keys"])
        model.group_lookup = {key: index for index, key in enumerate(model.group_keys)}
        model._schedule_cache = {}
        return model
