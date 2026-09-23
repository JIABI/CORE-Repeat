"""Optional, exactly removable two-branch residual-distribution adapter.

CORE is external and never evaluated or modified here. Both branches produce
two bounded log-scatter increments, in the already defined geometric error
blocks. They do not identify physical shared/independent noise. The structured
branch encodes statistical support/shrinkage functions of observed biological
reference summaries, not unobserved EC50 values or proven biological laws.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEGREES = (3., 6.)
BOUND = math.log(2.)


def _matrix(value, name):
    out = np.asarray(value, dtype=float)
    if out.ndim != 2 or not np.isfinite(out).all():
        raise ValueError(name+" must be a finite matrix")
    return out


def _support(value, n):
    if value is None:
        return np.ones(n, bool)
    mask = np.asarray(value)
    if mask.shape != (n,) or mask.dtype != bool:
        raise ValueError("Support must be an aligned boolean vector")
    return mask


def _scaling(x):
    if not len(x):
        return np.zeros(x.shape[1]), np.ones(x.shape[1])
    center, scale = x.mean(0), x.std(0)
    return center, np.where(scale > 1e-8, scale, 1.)


def feature_kind(name):
    """Classify documented reference fields; never infer potency from names."""
    n = name.lower()
    if any(token in n for token in ("ec50", "ic50", "potency")):
        raise ValueError("Potency needs a separately verified assay model, not this adapter")
    if any(token in n for token in ("missing", "available", "known", "context", "match", "support_flag")):
        return "availability"
    if any(token in n for token in ("ess", "effective_n", "neighbor_count", "neighbour_count", "donor_count", "log_count")):
        return "effective_support"
    if any(token in n for token in ("standard_error", "stderr", "uncertainty", "_se", "dispersion", "amp_gap", "angle_gap")):
        return "uncertainty"
    if any(token in n for token in ("mass", "weight_sum", "strength")):
        return "relation_strength"
    if any(token in n for token in ("cell_count", "n_cells")):
        return "cell_count"
    if any(token in n for token in ("reliability", "confidence", "similarity")):
        return "reliability"
    if any(token in n for token in ("excess", "residual", "error", "energy", "mean", "effect")):
        return "error_summary"
    return "other_numeric"


class GELUBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return BOUND*torch.tanh(self.network(x)/BOUND)


class BiologicalBasisBranch(nn.Module):
    """Independent local basis coefficients and readout, matched across modes."""
    def __init__(self, names, mode, structured_support_constant=8.):
        super().__init__()
        if mode not in ("generic", "structured") or not names:
            raise ValueError("A named generic/structured biological input is required")
        self.names = list(names)
        self.mode = mode
        self.kinds = [feature_kind(n) for n in names]
        self.support_constant = float(structured_support_constant)
        self.local_coefficients = nn.Parameter(torch.empty(len(names), 4))
        nn.init.normal_(self.local_coefficients, std=.1)
        self.readout = nn.Linear(len(names), 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def basis(self, raw, standardized):
        if self.mode == "generic":
            x = standardized
            return torch.stack((x,
                torch.exp(-.5*(x+1).square())-math.exp(-.5),
                torch.exp(-.5*x.square())-1.,
                torch.exp(-.5*(x-1).square())-math.exp(-.5)), dim=-1)
        result = []
        for j, kind in enumerate(self.kinds):
            x, value = standardized[:, j], raw[:, j]
            # These specific reference fields are supplied as log1p quantities.
            # In contrast, log_amp_gap is already an absolute log-amplitude
            # difference and must not be inverted. Energy summaries stay in
            # their supplied log coordinates in the signed response bases.
            if self.names[j].endswith(("_log_count", "_log_mass", "_log_ess", "_log_se")):
                value = torch.expm1(value)
            nonnegative = torch.clamp_min(value, 0.)
            if kind in ("effective_support", "cell_count"):
                scale = self.support_constant if kind == "effective_support" else 100.
                t = nonnegative/scale
                # Alternative support/finite-sample precision response bases.
                block = (t/(1+t), 1-torch.exp(-t), torch.log1p(t), 1-torch.rsqrt(1+t))
            elif kind == "uncertainty":
                # Larger donor uncertainty need not force a contribution; the
                # learned coefficients can select, ignore or reverse a basis.
                t = nonnegative
                block = (1/(1+t), 1/(1+t.square()), torch.exp(-t), torch.log1p(t))
            elif kind == "relation_strength":
                t = nonnegative
                block = (t/(1+t), 1-torch.exp(-t), torch.log1p(t), torch.sqrt(t+1)-1)
            elif kind in ("availability", "reliability"):
                block = (value, value.square(), torch.tanh(value), value/(1+value.abs()))
            else:
                # Signed error-excess estimates retain their sign and size;
                # these are robust response bases, not a pharmacological law.
                block = (x, torch.tanh(x), x/(1+x.abs()), torch.sign(x)*torch.log1p(x.abs()))
            result.append(torch.stack(block, -1))
        return torch.stack(result, 1)

    def forward(self, raw, standardized):
        local = (self.basis(raw, standardized)*self.local_coefficients).sum(-1)
        return BOUND*torch.tanh(self.readout(local)/BOUND)


class DualBranchBiologyAdapter(nn.Module):
    def __init__(self, empirical_dim, biological_names=(), *, mode="none", hidden_dim=16):
        super().__init__()
        if mode not in ("none", "generic", "structured"):
            raise ValueError("Unknown right branch")
        self.config = dict(empirical_dim=int(empirical_dim), biological_names=list(biological_names),
                           mode=mode, hidden_dim=int(hidden_dim))
        self.left = GELUBranch(empirical_dim, hidden_dim)
        self.right = None if mode == "none" else BiologicalBasisBranch(biological_names, mode)
        self.register_buffer("empirical_center", torch.zeros(empirical_dim))
        self.register_buffer("empirical_scale", torch.ones(empirical_dim))
        self.register_buffer("biological_center", torch.zeros(len(biological_names)))
        self.register_buffer("biological_scale", torch.ones(len(biological_names)))
        self.report = {}
        self.double()

    def components(self, empirical, biological=None, support=None, *, enabled=True,
                   left_enabled=True, right_enabled=True):
        if empirical.ndim != 2 or empirical.shape[1] != self.config["empirical_dim"]:
            raise ValueError("Empirical descriptor shape mismatch")
        zeros = empirical.new_zeros((len(empirical), 2))
        if not enabled:
            return dict(left=zeros, right=zeros, total=zeros)
        mask = torch.ones(len(empirical), dtype=torch.bool, device=empirical.device) if support is None else support
        if mask.dtype != torch.bool or mask.shape != (len(empirical),):
            raise ValueError("Support must be a boolean vector")
        rows = torch.nonzero(mask, as_tuple=True)[0]
        if not len(rows):
            return dict(left=zeros, right=zeros, total=zeros)
        left = zeros
        if left_enabled:
            values = self.left((empirical[rows]-self.empirical_center)/self.empirical_scale)
            left = zeros.index_copy(0, rows, values)
        right = zeros
        if self.right is not None and right_enabled:
            if biological is None or biological.shape != (len(empirical), len(self.config["biological_names"])):
                raise ValueError("Biological descriptor shape mismatch")
            values = self.right(biological[rows], (biological[rows]-self.biological_center)/self.biological_scale)
            right = zeros.index_copy(0, rows, values)
        # torch.where guarantees exact zero for unsupported objects, not a tiny
        # sigmoid value. It does not imply a statistically certified gate.
        left = torch.where(mask[:, None], left, zeros)
        right = torch.where(mask[:, None], right, zeros)
        return dict(left=left, right=right, total=left+right)

    def forward(self, empirical, biological=None, support=None, **kwargs):
        return self.components(empirical, biological, support, **kwargs)["total"]

    def predict_components(self, empirical, biological=None, support=None, **kwargs):
        x = _matrix(empirical, "empirical")
        b = None if biological is None else _matrix(biological, "biological")
        mask = _support(support, len(x))
        self.eval()
        with torch.no_grad():
            out = self.components(torch.as_tensor(x, dtype=torch.float64),
                None if b is None else torch.as_tensor(b, dtype=torch.float64), torch.as_tensor(mask), **kwargs)
        return {k:v.numpy() for k,v in out.items()}

    def predict_increment(self, empirical, biological=None, support=None, **kwargs):
        return self.predict_components(empirical, biological, support, **kwargs)["total"]

    def apply_scale(self, base_scale, empirical, biological=None, support=None, *, enabled=True, **kwargs):
        base = _matrix(base_scale, "base_scale")
        if base.shape != (len(empirical), 2) or np.any(base<=0):
            raise ValueError("Two positive original scale offsets are required")
        if not enabled:
            return base.copy()
        delta = self.predict_increment(empirical, biological, support, **kwargs)
        result = base*np.exp(delta)
        result[np.all(delta==0, axis=1)] = base[np.all(delta==0, axis=1)]
        return result

    def save(self, path):
        torch.save(dict(config=self.config, state_dict=self.state_dict(), report=self.report), Path(path))

    @classmethod
    def load(cls, path):
        saved = torch.load(Path(path), map_location="cpu", weights_only=True)
        with torch.random.fork_rng():
            obj = cls(**saved["config"])
        obj.load_state_dict(saved["state_dict"])
        obj.report = saved["report"]
        return obj.eval().requires_grad_(False)


def projection_loss(log_scale, energies):
    degrees = log_scale.new_tensor(DEGREES)
    return .5*(degrees*log_scale+energies*torch.exp(-log_scale)).sum(-1)


def _fit(model, empirical, biological, support, energies, base_scale, ids, *, branch, seed,
         max_epochs=60, callback=None):
    x = _matrix(empirical, "empirical")
    e, base = _matrix(energies, "energies"), _matrix(base_scale, "base_scale")
    b = None if biological is None else _matrix(biological, "biological")
    mask = _support(support, len(x))
    ids = np.asarray(ids, str)
    if (e.shape != (len(x), 2) or base.shape != e.shape or np.any(e<=0) or np.any(base<=0)
            or ids.shape != (len(x),) or len(set(ids)) != len(ids)
            or (b is not None and len(b)!=len(x))):
        raise ValueError("Honest positive energy labels, scale offsets and IDs must align")
    if max_epochs != 60:
        raise ValueError("The declared adapter experiment uses 60 full epochs")
    rows = np.flatnonzero(mask)
    fixed_left = deepcopy(model.left.state_dict()) if branch == "right" else None
    parameter_module = model.left if branch == "left" else model.right
    if parameter_module is None:
        raise ValueError("Missing branch to fit")
    model.requires_grad_(False)
    parameter_module.requires_grad_(True)
    params = [p for p in parameter_module.parameters() if p.requires_grad]
    tensors = [torch.as_tensor(v, dtype=torch.float64) for v in (x, e, np.log(base))]
    bt = None if b is None else torch.as_tensor(b, dtype=torch.float64)
    mt = torch.as_tensor(mask)
    history = []
    if len(rows) >= 2:
        opt = torch.optim.AdamW(params, lr=.0003, weight_decay=.001)
        rng = np.random.default_rng(seed)
        for epoch in range(1, 61):
            lr = .0003*(epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-5)/55)))
            for group in opt.param_groups: group["lr"] = lr
            model.train()
            order = rng.permutation(rows)
            preclip_gradient_norms = []
            for start in range(0, len(order), 64):
                ii = order[start:start+64]
                comp = model.components(tensors[0][ii], None if bt is None else bt[ii], mt[ii])
                increment = comp[branch]
                loss = projection_loss(tensors[2][ii]+comp["total"], tensors[1][ii]).mean()+.05*increment.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite full adapter objective")
                opt.zero_grad(set_to_none=True); loss.backward()
                norm = nn.utils.clip_grad_norm_(params, 5., error_if_nonfinite=True)
                preclip_gradient_norms.append(float(norm))
                opt.step()
            if epoch == 1 or epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    comp = model.components(tensors[0][rows], None if bt is None else bt[rows], mt[rows])
                    row = dict(epoch=epoch, learning_rate=lr,
                        preclip_gradient_norm_mean=float(np.mean(preclip_gradient_norms)),
                        preclip_gradient_norm_max=float(np.max(preclip_gradient_norms)),
                        clipped_step_fraction=float(np.mean(np.asarray(preclip_gradient_norms)>5.)),
                        training_projection_nll=float(projection_loss(tensors[2][rows]+comp["total"], tensors[1][rows]).mean()),
                        core_projection_nll=float(projection_loss(tensors[2][rows], tensors[1][rows]).mean()),
                        left_increment_rms=float(comp["left"].square().mean().sqrt()),
                        right_increment_rms=float(comp["right"].square().mean().sqrt()))
                history.append(row)
                if callback is not None: callback(row)
    if fixed_left is not None and any(not torch.equal(v, model.left.state_dict()[k]) for k,v in fixed_left.items()):
        raise RuntimeError("The identical frozen empirical branch changed")
    model.report = dict(model.report, fitted_branch=branch, fitting_ids=ids[rows].tolist(),
        supplied_ids=ids.tolist(), fitting_supported_n=len(rows), epochs=60 if len(rows)>=2 else 0,
        status="fitted" if len(rows)>=2 else "insufficient_support_fitted_branch_zero",
        history=history, optimizer="AdamW", initial_lr=.0003, warmup_epochs=5,
        schedule="cosine to zero", weight_decay=.001, increment_penalty=.05,
        batch_size=64, gradient_clip=5., checkpoint_selection=False,
        per_branch_logscale_bound=BOUND, active_parameters=sum(p.numel() for p in params),
        target="honest geometric projection energy, Gaussian projection quasi-likelihood; not realized Gamma",
        core_external_frozen=True, biological_mechanistic_law_claim=False,
        left_unchanged=branch=="right", right_mode=model.config["mode"])
    return model.eval().requires_grad_(False)


def fit_gelu_branch(empirical, energies, base_scale, ids, *, support=None, seed=20260917,
                    hidden_dim=16, max_epochs=60, callback=None):
    x = _matrix(empirical, "empirical")
    mask = _support(support, len(x))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = DualBranchBiologyAdapter(x.shape[1], hidden_dim=hidden_dim)
    center, scale = _scaling(x[mask])
    model.empirical_center.copy_(torch.as_tensor(center))
    model.empirical_scale.copy_(torch.as_tensor(scale))
    return _fit(model, x, None, mask, energies, base_scale, ids, branch="left", seed=seed,
                max_epochs=max_epochs, callback=callback)


def fit_biology_branch(left, empirical, biological, biological_names, support, energies,
                       base_scale, ids, *, mode, seed=20260917, max_epochs=60, callback=None):
    if not isinstance(left, DualBranchBiologyAdapter) or left.config["mode"] != "none":
        raise TypeError("Supply the common fitted left-only adapter, not a different carrier")
    x, b = _matrix(empirical, "empirical"), _matrix(biological, "biological")
    mask = _support(support, len(x))
    if b.shape != (len(x), len(biological_names)) or len(set(biological_names)) != len(biological_names):
        raise ValueError("Biological feature names must uniquely align")
    if left.report.get("fitting_ids") is not None:
        if np.asarray(ids, str)[mask].tolist() != left.report["fitting_ids"]:
            raise ValueError("Right branch must use the identical supported fitting objects as the frozen left")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = DualBranchBiologyAdapter(x.shape[1], biological_names, mode=mode,
                                         hidden_dim=left.config["hidden_dim"])
    model.left.load_state_dict(left.left.state_dict())
    model.empirical_center.copy_(left.empirical_center)
    model.empirical_scale.copy_(left.empirical_scale)
    center, scale = _scaling(b[mask])
    model.biological_center.copy_(torch.as_tensor(center))
    model.biological_scale.copy_(torch.as_tensor(scale))
    model.report = dict(left_report=deepcopy(left.report), biological_feature_names=list(biological_names),
        biological_feature_kinds=list(model.right.kinds),
        basis_interpretation="Support/error-excess response functions of biological references, not proven pharmacological measurement laws",
        right_trainable_parameters=sum(p.numel() for p in model.right.parameters()))
    return _fit(model, x, b, mask, energies, base_scale, ids, branch="right", seed=seed,
                max_epochs=max_epochs, callback=callback)
