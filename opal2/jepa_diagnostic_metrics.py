"""Fixed-role JEPA diagnostics, separate from the original training objective.

These describe held-out embedding prediction and geometry, not biological or
downstream utility. Representation samples are target wells; repeats belonging
to one compound are not independent observations.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from typing import Any

import numpy as np
import torch
from torch import nn

from .training import fixed_batch


@contextmanager
def _evaluation_mode(*roots):
    modules = {id(module): module for root in roots if root is not None
               for module in root.modules()}
    states = [(module, module.training) for module in modules.values()]
    try:
        for root in roots:
            if root is not None:
                root.eval()
        yield
    finally:
        # Restore exact child states, including a frozen teacher in a training
        # learner. Recursive train() calls would overwrite mixed child states.
        for module, training in states:
            module.training = training


def _valid_targets(target_y, target_mask):
    if target_mask is not None:
        if target_mask.dtype != torch.bool:
            raise ValueError("Target mask must be boolean")
        if target_mask.shape == target_y.shape:
            target_y = torch.where(target_mask, target_y,
                                   torch.full_like(target_y, float("nan")))
            target_mask = target_mask.any(-1)
        if target_mask.shape != target_y.shape[:2]:
            raise ValueError("Target mask must index target wells or coordinates")
    valid = torch.isfinite(target_y).any(-1)
    if target_mask is not None:
        valid = valid & target_mask
    return target_y, valid


def _matrix(values):
    values = values.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("Representation samples must have shape [N, D]")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite diagnostic representation")
    return values


def _representation_statistics(values, *, near_zero_std=1e-4):
    values = _matrix(values)
    n, dimension = values.shape
    result: dict[str, Any] = {
        "n_samples": n, "dimension": dimension, "sample_unit": "target_well",
        "std_ddof": 1, "near_zero_std_threshold": near_zero_std,
        "rank_upper_bound": min(dimension, max(0, n - 1)),
        "rank_limited_by_sample_count": n - 1 < dimension,
        "rank_bound_note": "Centered sample covariance rank <= min(dimension, n_samples - 1).",
    }
    if n < 2:
        result.update(std_min=None, std_mean=None, near_zero_dimension_fraction=None,
                      covariance_participation_rank=None, covariance_entropy_rank=None,
                      zero_covariance=None, insufficient_samples=True)
        return result
    centered = values - values.mean(0, keepdim=True)
    std = centered.square().sum(0).div(n - 1).sqrt()
    # Singular values avoid constructing a full covariance matrix and preserve
    # the sample-rank restriction when N is smaller than the embedding dimension.
    eigenvalues = torch.linalg.svdvals(centered).square().div(n - 1)
    trace = eigenvalues.sum()
    if trace.item() == 0:
        participation, entropy = 0.0, 0.0
    else:
        probabilities = eigenvalues / trace
        positive = probabilities[probabilities > 0]
        participation = float(1 / probabilities.square().sum())
        entropy = float(torch.exp(-(positive * positive.log()).sum()))
    result.update(
        std_min=float(std.min()), std_mean=float(std.mean()),
        near_zero_dimension_fraction=float((std <= near_zero_std).double().mean()),
        covariance_participation_rank=participation,
        covariance_entropy_rank=entropy, zero_covariance=trace.item() == 0,
        insufficient_samples=False,
    )
    return result


def _prediction_metrics(predicted, reference):
    predicted, reference = _matrix(predicted), _matrix(reference)
    if predicted.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ")
    if not len(reference):
        return {"n_samples": 0, "mse": None, "reference_variance": None,
                "reference_variance_ddof": 0, "nmse": None,
                "zero_reference_variance": None, "nmse_undefined": True,
                "cosine_mean": None, "cosine_valid_samples": 0,
                "cosine_zero_norm_samples": 0}
    mse = (predicted - reference).square().mean()
    variance = (reference - reference.mean(0, keepdim=True)).square().mean()
    pnorm, rnorm = predicted.norm(dim=1), reference.norm(dim=1)
    nonzero = (pnorm > 0) & (rnorm > 0)
    cosine = ((predicted[nonzero] / pnorm[nonzero, None]) *
              (reference[nonzero] / rnorm[nonzero, None])).sum(1)
    zero_variance = variance.item() == 0
    return {
        "n_samples": len(reference), "mse": float(mse),
        "reference_variance": float(variance), "reference_variance_ddof": 0,
        "nmse": None if zero_variance else float(mse / variance),
        "zero_reference_variance": zero_variance, "nmse_undefined": zero_variance,
        "cosine_mean": float(cosine.mean()) if len(cosine) else None,
        "cosine_valid_samples": int(nonzero.sum()),
        "cosine_zero_norm_samples": int((~nonzero).sum()),
    }


def _anchor_modules(anchor):
    if anchor is None:
        return None, None
    if isinstance(anchor, (tuple, list)) and len(anchor) == 2:
        encoder, projector = anchor
    else:
        encoder, projector = anchor.teacher_encoder, anchor.teacher_projector
    if not isinstance(encoder, nn.Module) or not isinstance(projector, nn.Module):
        raise TypeError("anchor must provide an encoder and projector module")
    return encoder, projector


def evaluate_jepa(learner, normalized_dataset, indices, config, *, chunk_size=16, anchor=None):
    """Evaluate fixed X0 -> X1,X2,X3 and pool all valid target wells.

    ``anchor`` is an initial frozen ``(teacher_encoder, teacher_projector)``
    tuple, or an initial learner exposing those attributes, on the same device.
    Only the two teacher modules are used. All returned values are JSON safe.
    No dataset is loaded here and neither parameters nor teacher EMA are updated.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    anchor_encoder, anchor_projector = _anchor_modules(anchor)
    names = ("teacher_projected", "teacher_encoder", "student_projected",
             "student_encoder", "predictor")
    samples = {name: [] for name in names}
    anchor_samples = {"encoder": [], "projected": []}
    compounds_with_targets, target_slots = 0, 0
    with _evaluation_mode(learner, anchor_encoder, anchor_projector), torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            batch, target_y, target_mask = fixed_batch(
                normalized_dataset, indices[start:start + chunk_size], config,
                contexts=(0,), targets=(1, 2, 3))
            target_y, valid = _valid_targets(target_y, target_mask)
            target_slots += valid.numel()
            compounds_with_targets += int(valid.any(-1).sum())
            if not valid.any():
                continue
            # Encode observed targets once per encoder. The learner's forward
            # separately encodes context and never receives target measurements.
            observed = target_y[valid]
            teacher_raw = learner.teacher_encoder(observed)
            student_raw = learner.student_encoder(observed)
            outputs = {
                "teacher_encoder": teacher_raw,
                "teacher_projected": learner.teacher_projector(teacher_raw),
                "student_encoder": student_raw,
                "student_projected": learner.student_projector(student_raw),
                "predictor": learner(batch)[valid],
            }
            for name, value in outputs.items():
                samples[name].append(_matrix(value))
            if anchor_encoder is not None:
                raw = anchor_encoder(observed)
                anchor_samples["encoder"].append(_matrix(raw))
                anchor_samples["projected"].append(_matrix(anchor_projector(raw)))
    dimension = learner.hidden_dim
    merged = {name: torch.cat(parts, 0) if parts else torch.empty((0, dimension), dtype=torch.float64)
              for name, parts in samples.items()}
    alignment = _prediction_metrics(merged["predictor"], merged["teacher_projected"])
    result = {
        "schema_version": 1, "contexts": [0], "targets": [1, 2, 3],
        "n_compounds": len(indices), "n_compounds_with_valid_targets": compounds_with_targets,
        "n_target_slots": target_slots, "n_valid_target_wells": alignment["n_samples"],
        "sample_dependence_note": "Target wells from the same compound are not independent samples.",
        "alignment_mse": alignment["mse"],
        "teacher_target_variance": alignment["reference_variance"],
        "teacher_target_variance_ddof": 0, "alignment_nmse": alignment["nmse"],
        "zero_teacher_target_variance": alignment["zero_reference_variance"],
        "alignment_nmse_undefined": alignment["nmse_undefined"],
        "alignment_cosine": alignment["cosine_mean"],
        "alignment_cosine_valid_samples": alignment["cosine_valid_samples"],
        "alignment_cosine_zero_norm_samples": alignment["cosine_zero_norm_samples"],
        "representations": {name: _representation_statistics(values) for name, values in merged.items()},
        "teacher_student_same_well_drift": {
            "projected": _prediction_metrics(merged["student_projected"], merged["teacher_projected"]),
            "encoder": _prediction_metrics(merged["student_encoder"], merged["teacher_encoder"]),
        },
    }
    if anchor_encoder is not None:
        reference = {name: torch.cat(parts, 0) if parts else torch.empty((0, dimension), dtype=torch.float64)
                     for name, parts in anchor_samples.items()}
        result["anchor_target_drift"] = {
            "reference": "initial_teacher_on_same_target_wells",
            "projected": _prediction_metrics(merged["teacher_projected"], reference["projected"]),
            "encoder": _prediction_metrics(merged["teacher_encoder"], reference["encoder"]),
        }
    return result


def component_gradient_diagnostics(learner, item):
    """Weighted-loss gradients on student encoder parameters, leaving .grad intact.

    ``item`` contains ``inputs``, ``target_y`` and optional ``target_mask``.
    This performs one additional forward and three autograd.grad evaluations;
    the caller chooses frequency and preserves RNG if its modules use dropout.
    The current module modes are used unchanged. No optimizer/EMA step occurs.
    """
    parameters = [p for p in learner.student_encoder.parameters() if p.requires_grad]
    weights = {name: float(getattr(learner, name + "_weight"))
               for name in ("alignment", "variance", "covariance")}
    gradients, components = {}, {}
    with torch.enable_grad():
        losses = learner.loss(item["inputs"], item["target_y"], item.get("target_mask"))
        for name, weight in weights.items():
            weighted = losses[name] * weight
            if not torch.isfinite(weighted):
                raise FloatingPointError("Nonfinite weighted diagnostic loss")
            grads = torch.autograd.grad(weighted, parameters, retain_graph=True, allow_unused=True) \
                if parameters and weighted.requires_grad else (None,) * len(parameters)
            # CPU double accumulation keeps diagnostics stable without retaining
            # a second parameter-sized flat vector for each component.
            detached = tuple(None if g is None else g.detach().to(device="cpu", dtype=torch.float64)
                             for g in grads)
            norm_sq = sum(float(g.square().sum()) for g in detached if g is not None)
            norm = norm_sq ** .5
            if not np.isfinite(norm):
                raise FloatingPointError("Nonfinite component gradient norm")
            gradients[name] = detached
            components[name] = {
                "weight": weight, "weighted_loss": float(weighted.detach()),
                "grad_norm": norm, "zero_norm": norm == 0,
                "unused_parameter_tensors": sum(g is None for g in grads),
                "parameter_tensors": len(parameters),
                "all_gradients_missing": all(g is None for g in grads),
            }
    cosines, undefined = {}, {}
    for left, right in combinations(weights, 2):
        key = left + "__" + right
        denominator = components[left]["grad_norm"] * components[right]["grad_norm"]
        undefined[key] = denominator == 0
        if denominator == 0:
            cosines[key] = None
        else:
            dot = sum(float((a * b).sum()) for a, b in zip(gradients[left], gradients[right])
                      if a is not None and b is not None)
            cosines[key] = max(-1.0, min(1.0, dot / denominator))
    return {"parameter_scope": "student_encoder", "components": components,
            "pairwise_cosines": cosines, "pairwise_cosine_undefined": undefined}
