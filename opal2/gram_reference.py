"""Frozen full-space predictive references exported as four-well Gram draws.

This adapter performs inference only. Actual future profiles are never supplied
to the predictor. High-dimensional draws live only for one object/draw block;
the retained Monte Carlo result is [sample, object, 4, 4] in float64. The four
roles are always X, Z1, Z2, V, and the common scale is the observed X norm.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import torch

from .closed_form_baseline import ClosedFormBaseline
from .gram_geometry import profiles_to_gram


@dataclass
class GramReferenceSamples:
    grams: np.ndarray
    conditional_mean_gram: np.ndarray
    metadata: dict

    @property
    def mean_profile_gram(self):
        """Gram of [X, E(Z1|X), E(Z2|X), E(V|X)], NOT E(Gram)."""
        return self.conditional_mean_gram


def _validate_request(x, n_samples, seed, object_chunk_size, draw_chunk_size):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("X must be a finite nonempty [object, feature] array")
    if np.any(np.linalg.norm(x, axis=-1) == 0):
        raise ValueError("Zero-norm X has no declared normalized-Gram interpretation")
    for value in (n_samples, object_chunk_size, draw_chunk_size):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Sample and chunk sizes must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("An explicit nonnegative integer seed is required")
    return x


def _gram(x, future):
    """Combine observed X with future draws without changing physical units."""
    future = torch.as_tensor(future).detach().to(device="cpu", dtype=torch.float64)
    x = torch.as_tensor(x, dtype=torch.float64)
    if future.ndim == 3:
        context = x[:, None, :]
    elif future.ndim == 4:
        context = x[None, :, None, :].expand(future.shape[0], -1, -1, -1)
    else:
        raise ValueError("Future profiles must be [N,3,D] or [S,N,3,D]")
    if future.shape[-2] != 3 or future.shape[-1] != x.shape[-1] or future.shape[-3] != len(x):
        raise ValueError("Future profiles must match X and exactly three future roles")
    return profiles_to_gram(torch.cat((context, future), dim=-2), normalize_x=True).cpu().numpy()


def _metadata(n_samples, seed, object_chunk_size, draw_chunk_size):
    return dict(n_samples=int(n_samples), seed=int(seed),
        object_chunk_size=int(object_chunk_size), draw_chunk_size=int(draw_chunk_size),
        dtype="float64", roles=["X", "Z1", "Z2", "V"],
        coordinate_space="original exported endpoint coordinates, not raw image pixels",
        normalization="all four profiles divided by the observed X norm",
        mean_point="Gram of conditional mean profiles; not Monte Carlo mean Gram",
        future_outcomes_used_as_inputs=False, new_training_steps=0,
        reproducibility="seed, object order, object chunks and draw chunks must all match")


class ClosedFormGramReference:
    """The existing complete Gaussian L baseline, with no refit or truncation."""

    def __init__(self, baseline: ClosedFormBaseline, *, provenance=None):
        if baseline.slot_mean.shape[0] != 4:
            raise ValueError("This adapter is restricted to the original four-role baseline")
        self.baseline = baseline
        self.provenance = {} if provenance is None else dict(provenance)

    @classmethod
    def from_checkpoint(cls, path):
        path = Path(path).resolve()
        return cls(ClosedFormBaseline.load(path), provenance=dict(
            checkpoint=str(path), model="frozen closed-form Gaussian L",
            parameter_source="existing saved moment fit; not refitted"))

    def sample(self, X, n_samples: int, *, seed: int, object_chunk_size=4,
               draw_chunk_size=32) -> GramReferenceSamples:
        x = _validate_request(X, n_samples, seed, object_chunk_size, draw_chunk_size)
        if x.shape[-1] != len(self.baseline.center):
            raise ValueError("X does not match the baseline's complete feature coordinates")
        grams = np.empty((n_samples, len(x), 4, 4), dtype=np.float64)
        mean_gram = np.empty((len(x), 4, 4), dtype=np.float64)
        rng = np.random.default_rng(seed)
        for start in range(0, len(x), object_chunk_size):
            stop = min(start + object_chunk_size, len(x))
            bx = x[start:stop]
            distribution = self.baseline.conditional(bx[:, None, :], [0], [1, 2, 3])
            mean_gram[start:stop] = _gram(bx, distribution.mean)
            for first in range(0, n_samples, draw_chunk_size):
                last = min(first + draw_chunk_size, n_samples)
                draws = distribution.sample_joint(last - first, rng=rng)
                grams[first:last, start:stop] = _gram(bx, draws)
                del draws
        metadata = dict(self.provenance, **_metadata(n_samples, seed, object_chunk_size, draw_chunk_size))
        metadata.update(independent_objects=True,
                        joint_future_noise="shared conditional compound factor plus independent well noise")
        return GramReferenceSamples(grams, mean_gram, metadata)


@contextmanager
def _no_bytecode_writes():
    """Reading frozen source must not create new __pycache__ files in its run."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous


def _snapshot_package(snapshot: Path):
    """Import frozen relative imports under an isolated name, not live opal2."""
    snapshot = Path(snapshot).resolve()
    package = snapshot / "opal2"
    init = package / "__init__.py"
    if not init.is_file():
        raise FileNotFoundError(init)
    alias = "_opal2_gram_snapshot_" + hashlib.sha256(str(snapshot).encode()).hexdigest()[:16]
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, init,
            submodule_search_locations=[str(package)])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the frozen source package {package}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        try:
            with _no_bytecode_writes():
                spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[alias]
            raise
    return alias


def sample_neural_grams(X, distribution_for_indices: Callable, inverse_y: Callable,
                       environment_cache_factory: Callable, n_samples: int, *, seed: int,
                       object_chunk_size=4, draw_chunk_size=32, provenance=None):
    """Sample a frozen neural distribution whose callback sees local indices only.

    A cache is shared across all object chunks for each Monte Carlo block. This
    preserves global source/batch/plate draws as well as within-object sharing.
    The native checkpoint sampling/inverse-affine precision is retained, then
    all profile inner products are calculated in float64.
    """
    x = _validate_request(X, n_samples, seed, object_chunk_size, draw_chunk_size)
    grams = np.empty((n_samples, len(x), 4, 4), dtype=np.float64)
    mean_gram = np.empty((len(x), 4, 4), dtype=np.float64)
    offsets = list(range(0, n_samples, draw_chunk_size))
    caches = [environment_cache_factory() for _ in offsets]
    # This is the historical evaluator's per-block stream layout. The caches
    # and streams persist while processing subsequent chunks of compounds.
    generators = [torch.Generator(device="cpu").manual_seed(int(seed) + 501 + block * 100003)
                  for block in range(len(offsets))]
    with torch.no_grad():
        for start in range(0, len(x), object_chunk_size):
            stop = min(start + object_chunk_size, len(x))
            distribution = distribution_for_indices(np.arange(start, stop))
            if distribution.mean.device.type != "cpu":
                raise ValueError("Frozen reference sampling currently preserves the original CPU inference route")
            mean_gram[start:stop] = _gram(x[start:stop], inverse_y(distribution.mean))
            for block, first in enumerate(offsets):
                last = min(first + draw_chunk_size, n_samples)
                draws = distribution.sample_joint(last - first, generator=generators[block],
                                                   environment_noise_cache=caches[block])
                grams[first:last, start:stop] = _gram(x[start:stop], inverse_y(draws))
                del draws
            del distribution
    metadata = dict({} if provenance is None else provenance,
                    **_metadata(n_samples, seed, object_chunk_size, draw_chunk_size))
    metadata.update(independent_objects=False, environment_cache_reused_across_object_chunks=True,
                    native_prediction_and_inverse_affine_precision_preserved=True,
                    mc_stream_seed_offsets="seed + 501 + 100003 * draw_block")
    return GramReferenceSamples(grams, mean_gram, metadata)


class FrozenFaithfulGramReference:
    """Original J/F checkpoint plus its frozen legal inference pipeline.

    ``from_run`` only accepts the known four-role 639-object DEV export. It
    loads preprocessing and library parameters, never fits them. Future Y is
    zeroed and marked unobserved in the retained inference dataset; only X and
    declared decision-time conditions/reference panels can reach the model.
    """

    @classmethod
    def from_run(cls, run_directory, arm="J_JOINT", *, expected_epoch=10):
        root = Path(run_directory).resolve()
        if arm not in ("J_JOINT", "F_MEAN_PROTECTED"):
            raise ValueError("Only the frozen probabilistic J and F arms are reference distributions")
        manifest = json.loads((root / "run_manifest.json").read_text())
        if manifest.get("data_shape") != [639, 4, 3617]:
            raise ValueError("Only the already-opened 639-object, four-role DEV cohort is permitted")
        data = (root.parents[1] / "data/source5_primary_fullcontrols").resolve()
        if Path(manifest["data_directory"]).resolve() != data:
            raise ValueError("Refusing to load a different measurement export")
        snapshot = (root / "source_snapshot").resolve()
        if Path(manifest["source_snapshot"]).resolve() != snapshot:
            raise ValueError("The run must use its own frozen source snapshot")
        for key in ("final_opened", "fifth_repeat_opened", "original_endpoint_changed",
                    "original_contract_changed", "original_split_changed"):
            if manifest.get(key) is not False:
                raise ValueError(f"The run does not preserve {key}")
        with _no_bytecode_writes():
            alias = _snapshot_package(snapshot)
            runner = importlib.import_module(alias + ".faithful_experiment")
            frozen_data = importlib.import_module(alias + ".data")
            frozen_training = importlib.import_module(alias + ".training")
            frozen_model = importlib.import_module(alias + ".model")
            frozen_faithful = importlib.import_module(alias + ".faithful")
            _, loaded_manifest, dataset, splits, normalized, scaler, bank, config = runner._load_run(root)
        if loaded_manifest != manifest or config.device != "cpu":
            raise ValueError("The recorded CPU evaluation configuration changed")
        checkpoint = root / "arms" / arm / "best_validation_mean.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("epoch") != expected_epoch:
            raise ValueError("Checkpoint is not the declared fixed epoch")
        if (payload.get("train_config") != manifest["config"]
                or payload.get("train_ids") != manifest["compound_ids"]["train"]
                or payload.get("validation_ids") != manifest["compound_ids"]["validation"]
                or payload.get("feature_names") != manifest["feature_names"]
                or payload.get("optimizer_steps") != expected_epoch * manifest["steps_per_epoch"]
                or payload.get("selection") != "validation physical mean MSE"
                or payload.get("historical_dev") is not True
                or payload.get("formal_certificate") is not False):
            raise ValueError("The frozen checkpoint provenance does not match its original run")
        with torch.random.fork_rng(devices=[]):
            model = frozen_faithful.FaithfulMeasurementModel.from_config(payload["model_config"])
        expected_kwargs = frozen_training.model_kwargs(normalized, config)
        if any(model.mean_model.config.get(key) != value for key, value in expected_kwargs.items()):
            raise ValueError("The saved model architecture differs from the frozen run configuration")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()
        model.requires_grad_(False)
        if bank is not None:
            model.library_bank = bank
        instance = cls()
        instance.model, instance.scaler, instance.config = model, scaler, config
        instance.X, instance.ids = dataset.Y[:, 0].copy(), dataset.ids.copy()
        instance.splits = {key: np.asarray(value, dtype=np.int64).copy() for key, value in splits.items()}
        context_y = np.zeros_like(normalized.Y)
        context_y[:, 0] = normalized.Y[:, 0]
        context_mask = np.zeros_like(normalized.observed_mask)
        context_mask[:, 0] = normalized.observed_mask[:, 0]
        instance.normalized = replace(normalized, Y=context_y, observed_mask=context_mask)
        instance._make_inputs = frozen_data.make_inference_batch
        instance._ablate_inputs = frozen_training.apply_information_ablation
        instance._cache_factory = frozen_model.EnvironmentNoiseCache
        instance.provenance = dict(checkpoint=str(checkpoint), source_snapshot=str(snapshot),
            source_package=alias, epoch=int(expected_epoch), arm=arm,
            checkpoint_selection=payload["selection"],
            model="frozen full faithful conditional Gaussian", historical_dev=True,
            formal_certificate=False, scaler=str(root / "scaler.json"),
            library_context=str(root / "library_context.npz") if bank is not None else None,
            reference_access=config.reference_access, final_opened=False, fifth_repeat_opened=False)
        return instance

    def sample(self, indices, n_samples: int, *, seed: int, object_chunk_size=4,
               draw_chunk_size=32):
        ix = np.asarray(indices)
        if (ix.ndim != 1 or not len(ix) or not np.issubdtype(ix.dtype, np.integer)
                or np.any(ix < 0) or np.any(ix >= len(self.X)) or len(set(ix.tolist())) != len(ix)):
            raise ValueError("Unique in-range opened-DEV indices are required")

        def distribution_for_indices(local):
            inputs = self._make_inputs(self.normalized, ix[local], (0,), (1, 2, 3),
                reference_access=self.config.reference_access, device=self.config.device)
            return self.model(self._ablate_inputs(inputs, self.config))

        provenance = dict(self.provenance, compound_ids=self.ids[ix].tolist())
        return sample_neural_grams(self.X[ix], distribution_for_indices, self.scaler.inverse_y,
            self._cache_factory, n_samples, seed=seed, object_chunk_size=object_chunk_size,
            draw_chunk_size=draw_chunk_size, provenance=provenance)
