"""Full frozen STATE50 inference on compatible, ordered profile features.

Loads the actual epoch-50 state-conditioned model, including its complete frozen
old-information A/HR/ridge backbone and reference bank. No component is refitted
and no ridge-only approximation is used. The original RIDGE error second moment
is returned as a *base scatter*, before new reference/distribution calibration.

The caller owns measurement preprocessing compatibility, reference-role isolation,
and transfer evaluation. A new acquisition context is not automatically a new
chemical entity: overlap with the original Pilot1 cohort is reported explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains
from .gram_oof_ridge import transform_input, transform_target
from .gram_simple_models import GramSimpleGaussian
from .state_biology_kernel import StateBiologyKernelMean


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_RUN = PROJECT / "runs/lincs_state_biology_20260916_v1"


class FrozenState50Transfer:
    """Read-only complete predictor with a separate calibration-target API."""

    def __init__(self, model, stats, covariance, manifest, fold, provenance):
        self.model = model.eval().requires_grad_(False)
        self.stats = stats
        self.base_scatter = np.asarray(covariance, dtype=np.float64).copy()
        if self.base_scatter.shape != (9, 9) or not np.isfinite(self.base_scatter).all():
            raise ValueError("Original RIDGE scatter must be finite 9 by 9")
        np.linalg.cholesky(self.base_scatter)
        self.feature_names = tuple(manifest["feature_names"])
        self.fold = int(fold)
        self.provenance = provenance
        self._manifest = manifest
        self._record = next(record for record in manifest["folds"] if record["fold"] == fold)
        self._scope = next(scope for scope in manifest["scopes"] if scope["fold"] == fold)
        self.target_names = tuple(model.bank.target_names)
        self.moa_names = tuple(model.bank.moa_names)
        self.chemical_dimension = int(model.bank.chemical_dim)
        if len(self.feature_names)+1 != model.bank.input_dim:
            raise ValueError("Saved feature names and complete model inputs disagree")

    @classmethod
    def load(cls, state_run=DEFAULT_STATE_RUN, fold=0):
        state_run = Path(state_run).resolve()
        manifest = json.loads((state_run/"run_manifest.json").read_text())
        matching = [r for r in manifest["folds"] if r["fold"] == fold]
        if len(matching) != 1:
            raise ValueError("Choose one existing predeclared saved fold")
        folder = state_run/"folds"/f"fold_{fold}"
        path = folder/"arms/STATE50/epoch50.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["epoch"] != 50 or payload.get("actual_checkpoint_epoch", 50) != 50:
            raise ValueError("Expected the saved actual epoch50, not a best/last substitute")
        if payload["model_config"]["mode"] != "state_only":
            raise ValueError("Expected the complete STATE50 state-only augmentation")
        # Constructors initialize modules before loading their saved parameters;
        # isolate that initialization from the caller's new experiment RNG.
        with torch.random.fork_rng(devices=[]):
            model = StateBiologyKernelMean.from_config(payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval().requires_grad_(False)
        original_a = torch.load(folder/"frozen_A.pt", map_location="cpu", weights_only=True)
        if original_a["model_config"] != model.base_a.config:
            raise ValueError("Frozen A configuration does not match STATE50")
        if original_a["state_dict"].keys() != model.base_a.state_dict().keys() or any(
                not torch.equal(value, model.base_a.state_dict()[name])
                for name, value in original_a["state_dict"].items()):
            raise ValueError("STATE50 no longer contains the original full frozen A")
        base_folder = Path(manifest["frozen_base_reference_run"])/"folds"/f"fold_{fold}"
        ridge_path = base_folder/"ridge.npz"
        ridge = GramSimpleGaussian.load(ridge_path)
        stats = json.loads((folder/"preprocessing.json").read_text())
        if stats != json.loads((base_folder/"preprocessing.json").read_text()):
            raise ValueError("Saved STATE50 and original RIDGE coordinate frames differ")
        scope = next(s for s in manifest["scopes"] if s["fold"] == fold)
        if (payload["fit_ids"] != scope["commonbranchfit_ids"]
                or payload["validation_ids"] != matching[0]["inner_validation_ids"]):
            raise ValueError("Saved checkpoint fitting/validation identity mismatch")
        provenance = dict(state_run=str(state_run), checkpoint=str(path),
            actual_epoch=50, saved_fold=int(fold), original_ridge=str(ridge_path),
            preprocessing=str(folder/"preprocessing.json"), complete_frozen_A_verified=True,
            model_refitted=False, reference_bank_refitted=False,
            biology_active=False, jepa_active=False,
            base_scatter="original RIDGE nested-OOF prediction-error second moment; not physical noise",
            scope="frozen source predictor; new-domain distribution calibration is separate",
            new_chemical_identity_claim=False)
        return cls(model, stats, ridge.covariance, manifest, fold, provenance)

    def _validate_features(self, X, feature_names):
        X = np.asarray(X, dtype=np.float64)
        if (X.ndim != 2 or len(X) < 1 or X.shape[1] != len(self.feature_names)
                or not np.isfinite(X).all()):
            raise ValueError("X must be finite [objects, saved feature dimension]")
        if feature_names is not None and tuple(map(str, feature_names)) != self.feature_names:
            raise ValueError("Feature names/order must exactly match the frozen source model")
        return X

    def unknown_biology(self, n):
        """Exact saved vocab dimensions, with UNKNOWN represented by false masks."""
        return dict(target=np.zeros((n, len(self.target_names)), dtype=np.float64),
                    moa=np.zeros((n, len(self.moa_names)), dtype=np.float64),
                    target_mask=np.zeros(n, dtype=bool), moa_mask=np.zeros(n, dtype=bool))

    def predict(self, X, chem, chem_mask=None, *, biology=None, feature_names=None):
        """X/chemistry-only forward; no future profiles or target arguments accepted.

        ``feature_names=None`` means the caller explicitly owns ordered-feature
        compatibility. Supply names to enforce it programmatically. Annotation
        arrays, if supplied, must follow the saved target/MoA vocabularies;
        omitting them supplies unknowns, not known negatives.
        """
        X = self._validate_features(X, feature_names)
        chemistry = np.asarray(chem, dtype=np.float64)
        if chemistry.shape != (len(X), self.chemical_dimension):
            raise ValueError("Chemistry must match the full saved chemical schema")
        mask = np.ones(len(X), dtype=bool) if chem_mask is None else np.asarray(chem_mask)
        if mask.shape != (len(X),) or mask.dtype != bool:
            raise ValueError("Chemical availability requires an aligned boolean mask")
        if not np.isfinite(chemistry[mask]).all():
            raise ValueError("Available chemistry must be finite")
        # Unavailable chemistry may be missing, but never enters a descriptor.
        chemistry = np.where(mask[:, None], chemistry, 0.)
        biology = self.unknown_biology(len(X)) if biology is None else biology
        if set(biology) != {"target", "moa", "target_mask", "moa_mask"}:
            raise ValueError("Biology needs the two profiles and separate availability masks")
        bio = {}
        for name, values in biology.items():
            values = np.asarray(values)
            if name.endswith("_mask") and values.dtype != bool:
                raise ValueError("Biology masks must be boolean, not inferred from zero profiles")
            bio[name] = torch.as_tensor(values, dtype=torch.bool if name.endswith("_mask") else torch.float64)
        x = transform_input(X, self.stats)
        with torch.no_grad():
            packed = self.model.bank.pack_information(torch.as_tensor(chemistry), bio)
            mean = self.model(torch.as_tensor(x), packed, torch.as_tensor(mask)).cpu().numpy()
        if mean.shape != (len(X), 9) or not np.isfinite(mean).all():
            raise ValueError("Frozen full-model prediction failed")
        return dict(mean_u=mean, base_scatter_u=np.broadcast_to(self.base_scatter, (len(X), 9, 9)).copy(),
                    log_amplitude=np.log(np.linalg.norm(X, axis=1)), input_x=x,
                    norm2_per_feature=np.square(X).mean(1),
                    feature_order_verified=feature_names is not None)

    def target_coordinates(self, Y, *, feature_names=None):
        """Actual target for declared calibration/evaluation, never inference."""
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 3 or Y.shape[1] != 4:
            raise ValueError("Target measurements must be [objects,4,features]")
        self._validate_features(Y.reshape(-1, Y.shape[-1]), feature_names)
        grams = profiles_to_gram(torch.as_tensor(Y))
        raw = gram_to_coordinates(grams).cpu().numpy()
        return transform_target(raw, self.stats)

    def calibration_records(self, Y, chem, chem_mask=None, *, biology=None,
                            feature_names=None, ids=None, groups=None):
        """Compute frozen residuals without fitting anything.

        Supply only outcomes belonging to the externally declared calibration
        role. This function cannot infer role permission from arrays alone.
        Query-outcome exclusion must be enforced by the experiment runner.
        """
        Y = np.asarray(Y, dtype=np.float64)
        actual_u = self.target_coordinates(Y, feature_names=feature_names)
        out = self.predict(Y[:, 0], chem, chem_mask, biology=biology, feature_names=feature_names)
        out.update(actual_u=actual_u, residual_u=actual_u-out["mean_u"],
                   actual_gamma=gram_gains(profiles_to_gram(torch.as_tensor(Y))).cpu().numpy()[:, 2])
        for name, values in (("ids", ids), ("groups", groups)):
            if values is not None:
                values = np.asarray(values).astype(str)
                if values.shape != (len(Y),) or (name == "ids" and len(set(values)) != len(values)):
                    raise ValueError("Calibration IDs/groups must align; IDs must be unique")
                out[name] = values.copy()
        return out

    def overlap_report(self, *, ids=None, groups=None):
        """Source overlap only; repeated compounds can be valid context transfer."""
        if ids is None and groups is None:
            raise ValueError("Supply query IDs and/or chemistry groups")
        source_ids = np.asarray(self._manifest["ids"], dtype=str)
        source_groups = np.asarray(self._manifest["groups"], dtype=str)
        role_indices = dict(original_model_fit=np.asarray(self._record["fit"], int),
                            original_model_validation=np.asarray(self._record["inner_validation"], int),
                            original_source_cohort=np.arange(len(source_ids)))
        answer = dict(new_chemical_identity_claim=False, saved_fold=self.fold, roles={})
        for name, rows in role_indices.items():
            item = {}
            for key, supplied, original in (("ids", ids, source_ids), ("groups", groups, source_groups)):
                if supplied is not None:
                    values = np.asarray(supplied).astype(str)
                    if values.ndim != 1:
                        raise ValueError("Overlap identifiers must be vectors")
                    overlap = np.isin(values, original[rows])
                    item[key+"_overlap_count"] = int(overlap.sum())
                    item[key+"_overlap_mask"] = overlap.tolist()
            answer["roles"][name] = item
        return answer
