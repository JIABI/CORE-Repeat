"""EU R3 wrapper: empirical correction for all rows, biology for supported rows.

CORE stays external and frozen. The two outputs are bounded log-scatter
increments for the existing rank-3/rank-6 geometric blocks, not estimates of
physical shared/independent noise. Historical LINCS adapter semantics are not
changed. Training reuses its full 60-epoch optimizer and branch definitions.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from .dual_branch_biology import (
    DualBranchBiologyAdapter, _fit, _matrix, _scaling, _support,
    fit_gelu_branch,
)


SUPPORT_SEMANTICS = "empirical_all_rows_biology_supported_only"


class EUResidualAdapter(DualBranchBiologyAdapter):
    """Use biological eligibility only on the right branch.

    Disabled adapter yields exact zero increments. Disabling only the right
    branch yields the same frozen left predictions, including unsupported rows.
    """

    def components(self, empirical, biological=None, support=None, *, enabled=True,
                   left_enabled=True, right_enabled=True):
        if empirical.ndim != 2 or empirical.shape[1] != self.config["empirical_dim"]:
            raise ValueError("Empirical descriptor shape mismatch")
        zeros = empirical.new_zeros((len(empirical), 2))
        if not enabled:
            return dict(left=zeros, right=zeros, total=zeros)
        mask = (torch.ones(len(empirical), dtype=torch.bool, device=empirical.device)
                if support is None else support)
        if mask.dtype != torch.bool or mask.shape != (len(empirical),):
            raise ValueError("Support must be a boolean vector")
        left = zeros
        if left_enabled:
            left = self.left((empirical-self.empirical_center)/self.empirical_scale)
        right = zeros
        rows = torch.nonzero(mask, as_tuple=True)[0]
        if self.right is not None and right_enabled and len(rows):
            if biological is None or biological.shape != (len(empirical), len(self.config["biological_names"])):
                raise ValueError("Biological descriptor shape mismatch")
            values = self.right(biological[rows],
                (biological[rows]-self.biological_center)/self.biological_scale)
            right = zeros.index_copy(0, rows, values)
        return dict(left=left, right=right, total=left+right)

    @classmethod
    def load(cls, path):
        model = super().load(path)
        if model.report.get("support_semantics") != SUPPORT_SEMANTICS:
            raise ValueError("Artifact does not declare EU R3 independent support semantics")
        return model


def fit_left(empirical, energies, ids, seed=20260917):
    """Fit the unchanged GELU branch on every supplied honest REF row.

    Energies must be computed from CORE-standardized residuals in the fixed
    geometric projection frame. Therefore the original two scale offsets are
    exactly one. No query/evaluation targets are accepted by this interface.
    """
    x = _matrix(empirical, "empirical")
    e = _matrix(energies, "energies")
    fitted = fit_gelu_branch(x, e, np.ones_like(e), ids, support=None, seed=seed)
    with torch.random.fork_rng():
        model = EUResidualAdapter(**fitted.config)
    model.load_state_dict(fitted.state_dict())
    model.report = dict(deepcopy(fitted.report), support_semantics=SUPPORT_SEMANTICS,
                        empirical_fitted_on_all_supplied_rows=True,
                        base_scale="ones; energies already standardized against CORE")
    return model.eval().requires_grad_(False)


def fit_right(left, empirical, biological, names, support, energies, ids,
              mode, seed=20260917):
    """Keep the full-REF left branch fixed; fit only supported biological rows.

    Generic and structured modes receive identical named inputs and have
    identical parameter counts. The right branch's readout starts at zero.
    Unlike the historical fitter, left fitting IDs need not equal the supported
    subset: they must equal the full supplied REF identity sequence.
    """
    if not isinstance(left, EUResidualAdapter) or left.config["mode"] != "none":
        raise TypeError("Supply the common fitted EU R3 left-only adapter")
    x = _matrix(empirical, "empirical")
    b = _matrix(biological, "biological")
    e = _matrix(energies, "energies")
    mask = _support(support, len(x))
    names = list(names)
    identifiers = np.asarray(ids, str)
    if (identifiers.shape != (len(x),) or len(set(identifiers)) != len(identifiers)
            or identifiers.tolist() != left.report.get("fitting_ids")):
        raise ValueError("Right fitting IDs must equal the full left REF identity sequence")
    if x.shape[1] != left.config["empirical_dim"]:
        raise ValueError("Empirical descriptor shape mismatch")
    if b.shape != (len(x), len(names)) or len(set(names)) != len(names):
        raise ValueError("Biological feature names must uniquely align")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = EUResidualAdapter(x.shape[1], names, mode=mode,
                                  hidden_dim=left.config["hidden_dim"])
    if model.right is None:
        raise ValueError("The right branch requires generic or structured mode")
    model.left.load_state_dict(left.left.state_dict())
    model.empirical_center.copy_(left.empirical_center)
    model.empirical_scale.copy_(left.empirical_scale)
    center, scale = _scaling(b[mask])
    model.biological_center.copy_(torch.as_tensor(center))
    model.biological_scale.copy_(torch.as_tensor(scale))
    model.report = dict(left_report=deepcopy(left.report),
        biological_feature_names=names, biological_feature_kinds=list(model.right.kinds),
        basis_interpretation="Support/error response bases, not established pharmacological laws",
        right_trainable_parameters=sum(p.numel() for p in model.right.parameters()),
        support_semantics=SUPPORT_SEMANTICS,
        empirical_fitted_on_all_supplied_rows=True,
        base_scale="ones; energies already standardized against CORE")
    return _fit(model, x, b, mask, e, np.ones_like(e), identifiers,
                branch="right", seed=seed, max_epochs=60)


def predict(model, empirical, biological=None, support=None, *, enabled=True,
            right_enabled=True):
    """Return left/right/total [N, 2] increments; receives no outcome targets."""
    if not isinstance(model, EUResidualAdapter):
        raise TypeError("EU R3 prediction requires the independent-support adapter")
    return model.predict_components(empirical, biological, support,
                                    enabled=enabled, right_enabled=right_enabled)
