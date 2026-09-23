"""Cell-local, nested group-OOF prediction of biological retrieval log-score gain.

The input population is a cell's permitted DIST_CAL records, not pooled historic
OOF predictions. The supplied amplitude bandwidth is a frozen REF_FIT nuisance.
Every inner donor set refits the empirical radial law and its radial bandwidth.
Only independently generated residual radii are accepted; this module does not
fit the world model, read future query outcomes, or provide certification.

Each relation channel is tested separately. Alpha directly mixes normalized
relation weights with amplitude weights, without a second ESS/strength gate.
ESS and absolute similarity enter the learned gate as decision-time features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .empirical_radial import fit_radial, radial_nll, reference_weights


ALPHAS = np.array([0., .25, .5, 1.])
STRATEGIES = ("FIXED_SELECTED", "DELTA_RIDGE", "DELTA_BOOST")
FEATURE_NAMES = (
    "amplitude_centered_ref_sd", "log_relation_ess", "relation_ess_fraction",
    "relation_strength", "maximum_similarity", "supported_reference_fraction",
    "relation_amp_difference", "relation_amp_rms_distance", "core_amp_rms_distance",
    "relation_minus_core_logradius_mean", "relation_logradius_sd",
    "core_logradius_sd", "relation_logradius_mean",
)
MIN_SUPPORTED_GROUPS = 4


def _vector(value, count, name, *, positive=False):
    result = np.asarray(value, float)
    if (result.shape != (count,) or not np.isfinite(result).all()
            or (positive and np.any(result <= 0))):
        raise ValueError(f"{name} must be a finite aligned vector" + (" > 0" if positive else ""))
    return result


def _identifiers(value, count, name, *, unique=False):
    result = np.asarray(value).astype(str)
    if result.shape != (count,) or (unique and len(np.unique(result)) != count):
        raise ValueError(f"{name} must be aligned" + (" and unique" if unique else ""))
    return result


def _similarity(value, shape):
    result = np.asarray(value, float)
    if (result.shape != shape or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError("Similarity must be an aligned finite matrix in [0,1]")
    return result


def _base_weights(reference_amp, query_amp, bandwidth, allowed):
    base = reference_weights(reference_amp, query_amp, bandwidth, conditional=True)["weights"]
    if not allowed.all():
        base = np.where(allowed, base, 0.)
        if np.any(base.sum(1) <= 0):
            raise ValueError("Every query requires an eligible amplitude reference")
        base /= base.sum(1, keepdims=True)
    return base


def _context(reference_radii, reference_amp, reference_groups, query_amp, query_groups,
             similarity, bandwidth, *, base_weights=None):
    allowed = query_groups[:, None] != reference_groups[None, :]
    if np.any(~allowed.any(1)):
        raise ValueError("Every query requires a different-group reference")
    base = _base_weights(reference_amp, query_amp, bandwidth, allowed)
    if base_weights is not None:
        supplied = np.asarray(base_weights, float)
        if (supplied.shape != base.shape or not np.isfinite(supplied).all()
                or np.any(supplied < 0) or np.any(supplied[~allowed] != 0)
                or not np.allclose(supplied.sum(1), 1., atol=1e-12, rtol=0)):
            raise ValueError("Invalid supplied CORE weights")
        # The optional bit-exact replay must be the same declared amplitude law,
        # not an unrelated set of weights with a different fitting convention.
        if not np.allclose(supplied, base, atol=1e-12, rtol=1e-12):
            raise ValueError("Supplied CORE weights differ from the declared amplitude baseline")
        base = supplied.copy()
    sim = np.where(allowed, similarity, 0.)
    maximum = sim.max(1)
    supported = maximum > 0
    scaled = np.divide(sim, maximum[:, None], out=np.zeros_like(sim), where=maximum[:, None] > 0)
    relation = np.divide(scaled, scaled.sum(1, keepdims=True), out=np.zeros_like(scaled),
                         where=scaled.sum(1, keepdims=True) > 0)
    # Unsupported rows have no biological intervention, and every downstream
    # feature remains finite without declaring unknown relationships absent.
    relation[~supported] = base[~supported]
    ess = 1 / np.square(relation).sum(1)
    strength = (relation * sim).sum(1)
    delta_amp = (query_amp[:, None] - reference_amp[None, :]) / bandwidth
    log_r = np.log(reference_radii)
    core_mean = base @ log_r
    rel_mean = relation @ log_r
    core_sd = np.sqrt(np.maximum(base @ np.square(log_r) - core_mean**2, 0.))
    rel_sd = np.sqrt(np.maximum(relation @ np.square(log_r) - rel_mean**2, 0.))
    features = np.column_stack((
        (query_amp - reference_amp.mean()) / bandwidth,
        np.log1p(np.where(supported, ess, 0.)),
        np.where(supported, ess / allowed.sum(1), 0.),
        strength, maximum, (sim > 0).sum(1) / allowed.sum(1),
        (relation * delta_amp).sum(1), np.sqrt((relation * delta_amp**2).sum(1)),
        np.sqrt((base * delta_amp**2).sum(1)), rel_mean - core_mean, rel_sd, core_sd, rel_mean,
    ))
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite decision-time gate features")
    return dict(base=base, relation=relation, features=features, supported=supported,
                ess=np.where(supported, ess, 0.), strength=strength,
                eligible_reference_count=allowed.sum(1),
                positive_reference_count=(sim > 0).sum(1))


def _mix(context, alpha):
    alpha = np.asarray(alpha, float)
    if alpha.shape != context["supported"].shape or np.any(~np.isin(alpha, ALPHAS)):
        raise ValueError("Alpha must be one declared coefficient per query")
    coefficient = np.where(context["supported"], alpha, 0.)
    weights = context["base"].copy()
    active = coefficient > 0
    weights[active] = ((1 - coefficient[active, None]) * weights[active]
                       + coefficient[active, None] * context["relation"][active])
    return weights, coefficient


def _logp(radii, law, weights):
    dimension = law["dimension"]
    residual = np.zeros((len(radii), dimension))
    residual[:, 0] = radii
    return -radial_nll(residual, np.broadcast_to(np.eye(dimension),
                       (len(radii), dimension, dimension)), law, weights)


def _candidate_deltas(query_radii, law, context):
    baseline = _logp(query_radii, law, context["base"])
    delta = np.zeros((len(query_radii), len(ALPHAS)))
    for column, alpha in enumerate(ALPHAS[1:], 1):
        weights, _ = _mix(context, np.full(len(query_radii), alpha))
        delta[:, column] = _logp(query_radii, law, weights) - baseline
    delta[~context["supported"]] = 0.
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite heldout log-score gain")
    return delta


def _group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return weights / weights.mean()


def _group_mean(values, groups):
    return np.mean([values[groups == group].mean(0) for group in np.unique(groups)], axis=0)


def _group_splits(groups, splits):
    count = min(int(splits), len(np.unique(groups)))
    if count < 2:
        raise ValueError("Cross-fitting requires at least two chemical groups")
    return GroupKFold(count).split(np.zeros(len(groups)), groups=groups)


def _make_oof(radii, amp, groups, ids, similarity, bandwidth, *, splits=3, dimension=9):
    n = len(radii)
    features = np.full((n, len(FEATURE_NAMES)), np.nan)
    delta = np.full((n, len(ALPHAS)), np.nan)
    supported = np.zeros(n, bool)
    ledger = []
    for fold, (fit, query) in enumerate(_group_splits(groups, splits)):
        law = fit_radial(radii[fit], dimension)
        context = _context(radii[fit], amp[fit], groups[fit], amp[query], groups[query],
                           similarity[np.ix_(query, fit)], bandwidth)
        features[query] = context["features"]
        supported[query] = context["supported"]
        delta[query] = _candidate_deltas(radii[query], law, context)
        ledger.append(dict(fold=fold, fit_ids=ids[fit].tolist(), query_ids=ids[query].tolist(),
                           fit_groups=np.unique(groups[fit]).tolist(),
                           query_groups=np.unique(groups[query]).tolist(),
                           radial_bandwidth=float(law["bandwidth"]),
                           frozen_amplitude_bandwidth=float(bandwidth)))
    if not np.isfinite(features).all() or not np.isfinite(delta).all():
        raise ValueError("Incomplete OOF pseudo-outcomes")
    return dict(features=features, delta=delta, supported=supported, ledger=ledger)


@dataclass
class _DeltaPredictor:
    scaler: object
    estimators: list
    kind: str
    diagnostics: dict

    def predict(self, features, supported):
        predictions = np.zeros((len(features), len(ALPHAS)))
        if self.estimators:
            x = self.scaler.transform(features) if self.scaler is not None else features
            if self.kind == "ridge":
                predictions[:, 1:] = self.estimators[0].predict(x)
            else:
                with threadpool_limits(limits=1):
                    predictions[:, 1:] = np.column_stack([model.predict(x) for model in self.estimators])
        predictions[~supported] = 0.
        return predictions


def _fit_predictor(oof, groups, kind, seed):
    supported = oof["supported"]
    n_groups = len(np.unique(groups[supported]))
    diagnostics = dict(kind=kind, supported_rows=int(supported.sum()), supported_groups=n_groups,
                       minimum_supported_groups=MIN_SUPPORTED_GROUPS, fallback_reason=None)
    if n_groups < MIN_SUPPORTED_GROUPS:
        diagnostics["fallback_reason"] = "fewer_than_four_supported_training_groups"
        return _DeltaPredictor(None, [], kind, diagnostics)
    x, y = oof["features"][supported], oof["delta"][supported, 1:]
    sample_weight = _group_weights(groups[supported])
    if kind == "ridge":
        scaler = StandardScaler().fit(x, sample_weight=sample_weight)
        estimator = Ridge(alpha=20.).fit(scaler.transform(x), y, sample_weight=sample_weight)
        diagnostics["configuration"] = dict(alpha=20., standardized=True, group_equal_weights=True)
        return _DeltaPredictor(scaler, [estimator], kind, diagnostics)
    estimators = []
    with threadpool_limits(limits=1):
        for column in range(y.shape[1]):
            model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=.05,
                max_iter=100, max_leaf_nodes=4, min_samples_leaf=3, l2_regularization=1.,
                early_stopping=False, random_state=seed + column)
            model.fit(x, y[:, column], sample_weight=sample_weight)
            estimators.append(model)
    diagnostics["configuration"] = dict(loss="squared_error", learning_rate=.05, max_iter=100,
        max_leaf_nodes=4, min_samples_leaf=3, l2_regularization=1., early_stopping=False,
        group_equal_weights=True)
    diagnostics["total_actual_splits"] = [int(sum(np.count_nonzero(tree[0].nodes["is_leaf"] == 0)
        for tree in model._predictors)) for model in estimators]
    return _DeltaPredictor(None, estimators, kind, diagnostics)


def _fit_gate_set(oof, groups, seed):
    candidate_mean = _group_mean(oof["delta"], groups)
    # Alpha zero is first, so exact ties and nonpositive gains choose off.
    fixed_index = int(np.argmax(candidate_mean))
    return dict(fixed_alpha=float(ALPHAS[fixed_index]),
                fixed_predicted_delta=candidate_mean,
                ridge=_fit_predictor(oof, groups, "ridge", seed),
                boost=_fit_predictor(oof, groups, "boost", seed))


def _predict_gate_set(gates, context):
    n = len(context["features"])
    predictions = {"FIXED_SELECTED": np.broadcast_to(gates["fixed_predicted_delta"], (n, len(ALPHAS))).copy(),
                   "DELTA_RIDGE": gates["ridge"].predict(context["features"], context["supported"]),
                   "DELTA_BOOST": gates["boost"].predict(context["features"], context["supported"])}
    outputs = {}
    for strategy, delta in predictions.items():
        delta[~context["supported"]] = 0.
        alpha = ALPHAS[np.argmax(delta, axis=1)]
        weights, alpha = _mix(context, alpha)
        outputs[strategy] = dict(weights=weights, alpha=alpha, predicted_delta=delta,
            predicted_selected_delta=delta[np.arange(n), np.argmax(delta, axis=1)],
            supported=context["supported"].copy(), features=context["features"].copy())
    return outputs


def _diagnose(values, groups):
    v = np.asarray(values, float)
    unique = np.unique(groups)
    group_values = np.array([v[groups == group].mean() for group in unique])
    return dict(mean=float(v.mean()), group_equal_mean=float(group_values.mean()),
                group_se=float(group_values.std(ddof=1) / np.sqrt(len(unique))) if len(unique) > 1 else None,
                n=len(v), groups=len(unique))


def _prediction_diagnostics(truth, prediction, supported, groups):
    result = {}
    for scope, mask in (("all", np.ones(len(supported), bool)), ("supported", supported)):
        if not mask.any():
            result[scope] = dict(n=0, groups=0, mse=None, r2=None)
            continue
        y, p = truth[mask, 1:], prediction[mask, 1:]
        denominator = np.square(y - y.mean(0)).sum()
        result[scope] = dict(n=int(mask.sum()), groups=len(np.unique(groups[mask])),
            mse=float(np.square(y-p).mean()),
            r2=float(1-np.square(y-p).sum()/denominator) if denominator > 0 else None)
    return result


@dataclass
class BiologyDeltaGates:
    radii: np.ndarray
    logamp: np.ndarray
    groups: np.ndarray
    ids: np.ndarray
    fit_amp_sd: float
    dimension: int
    channels: dict
    report: dict

    @property
    def law(self):
        return fit_radial(self.radii, self.dimension)

    def apply(self, query_logamp, query_groups, query_similarity, query_ids, *, base_weights=None):
        """Return six separate plans; accepts no query outcome or future profile."""
        query_amp = np.asarray(query_logamp, float)
        if query_amp.ndim != 1 or len(query_amp) == 0:
            raise ValueError("Nonempty query log amplitudes required")
        query_amp = _vector(query_amp, len(query_amp), "query_logamp")
        groups = _identifiers(query_groups, len(query_amp), "query_groups")
        ids = _identifiers(query_ids, len(query_amp), "query_ids", unique=True)
        if set(groups) & set(self.groups) or set(ids) & set(self.ids):
            raise ValueError("Query IDs/groups must be isolated from gate/reference fitting")
        if set(query_similarity) != set(self.channels):
            raise ValueError("Query relation channels must match fitted channels")
        plans = {}
        base = None
        for name, channel in self.channels.items():
            similarity = _similarity(query_similarity[name], (len(query_amp), len(self.radii)))
            context = _context(self.radii, self.logamp, self.groups, query_amp, groups,
                               similarity, self.fit_amp_sd, base_weights=base_weights)
            base = context["base"]
            for strategy, plan in _predict_gate_set(channel["gates"], context).items():
                plan["channel"] = name
                plan["strategy"] = strategy
                plans[name.upper()+"_"+strategy] = plan
        return dict(plans=plans, law=self.law, base_weights=base,
                    query_ids=ids.copy(), reference_ids=self.ids.copy(),
                    feature_names=FEATURE_NAMES, alphas=ALPHAS.copy())

    def save(self, path):
        """Serialize fitted estimators and training ledger to a NEW local artifact."""
        with Path(path).open("xb") as stream:
            joblib.dump(self, stream)

    @classmethod
    def load(cls, path):
        """Load only trusted local artifacts produced by this implementation."""
        result = joblib.load(Path(path))
        if not isinstance(result, cls):
            raise ValueError("Not a biological delta-gate artifact")
        return result


def fit_biology_delta_gates(radii, logamp, groups, cal_similarity, ids, *, fit_amp_sd,
                            seed=20260916, splits=3, dimension=9):
    """Fit nested development gates within one permitted reference cell.

    cal_similarity must already enforce annotation availability and biological
    context matching. Frozen mean/scatter and logamp-bandwidth are external,
    independently fitted nuisances. Threefold outer evaluation repeats the WHOLE
    inner OOF-delta/gate construction, including fixed-alpha selection. All arms
    are returned; no arm is adopted based on these developmental outcomes.
    """
    r = np.asarray(radii, float)
    if r.ndim != 1 or len(r) < 8:
        raise ValueError("At least eight reference records required")
    r = _vector(r, len(r), "radii", positive=True)
    amp = _vector(logamp, len(r), "logamp")
    groups = _identifiers(groups, len(r), "groups")
    ids = _identifiers(ids, len(r), "ids", unique=True)
    if len(np.unique(groups)) < 6:
        raise ValueError("At least six groups required for nested gate evaluation")
    if not np.isfinite(fit_amp_sd) or fit_amp_sd <= 0:
        raise ValueError("Frozen REF_FIT amplitude bandwidth must be positive")
    if not isinstance(splits, int) or isinstance(splits, bool) or splits < 2:
        raise ValueError("splits must be an integer at least two")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if not isinstance(cal_similarity, dict) or not cal_similarity or not set(cal_similarity) <= {"target", "moa"}:
        raise ValueError("Separate target and/or moa similarity channels required")
    matrices = {key: _similarity(value, (len(r), len(r))) for key, value in cal_similarity.items()}
    channels, channel_reports = {}, {}
    for number, (name, similarity) in enumerate(matrices.items()):
        channel_seed = seed + number * 1000
        nested = {strategy: dict(alpha=np.zeros(len(r)), predicted_delta=np.zeros((len(r), len(ALPHAS))),
                    realized_delta=np.zeros(len(r))) for strategy in STRATEGIES}
        actual_deltas = np.zeros((len(r), len(ALPHAS)))
        supported = np.zeros(len(r), bool)
        outer_ledger = []
        for fold, (fit, query) in enumerate(_group_splits(groups, splits)):
            inner = _make_oof(r[fit], amp[fit], groups[fit], ids[fit], similarity[np.ix_(fit, fit)],
                             fit_amp_sd, splits=splits, dimension=dimension)
            gates = _fit_gate_set(inner, groups[fit], channel_seed + fold * 10)
            law = fit_radial(r[fit], dimension)
            context = _context(r[fit], amp[fit], groups[fit], amp[query], groups[query],
                               similarity[np.ix_(query, fit)], fit_amp_sd)
            truth = _candidate_deltas(r[query], law, context)
            actual_deltas[query] = truth
            supported[query] = context["supported"]
            plans = _predict_gate_set(gates, context)
            for strategy, plan in plans.items():
                nested[strategy]["alpha"][query] = plan["alpha"]
                nested[strategy]["predicted_delta"][query] = plan["predicted_delta"]
                chosen = np.searchsorted(ALPHAS, plan["alpha"])
                nested[strategy]["realized_delta"][query] = truth[np.arange(len(query)), chosen]
            outer_ledger.append(dict(fold=fold, fit_ids=ids[fit].tolist(), query_ids=ids[query].tolist(),
                fit_groups=np.unique(groups[fit]).tolist(), query_groups=np.unique(groups[query]).tolist(),
                inner_oof=inner["ledger"], radial_bandwidth=float(law["bandwidth"]),
                fixed_alpha=gates["fixed_alpha"], ridge=gates["ridge"].diagnostics,
                boost=gates["boost"].diagnostics))
        final_oof = _make_oof(r, amp, groups, ids, similarity, fit_amp_sd, splits=splits, dimension=dimension)
        final_gates = _fit_gate_set(final_oof, groups, channel_seed + 100)
        diagnostics = {}
        for strategy, values in nested.items():
            diagnostics[strategy] = dict(logscore_gain=_diagnose(values["realized_delta"], groups),
                prediction=_prediction_diagnostics(actual_deltas, values["predicted_delta"], supported, groups),
                active_rows=int((values["alpha"] > 0).sum()),
                full_bio_paired_gain=_diagnose(values["realized_delta"]-actual_deltas[:, -1], groups))
        channel_reports[name] = dict(
            nested_diagnostics=diagnostics, nested_supported_rows=int(supported.sum()),
            nested_supported_groups=len(np.unique(groups[supported])),
            nested_fixed_alpha_gains={str(alpha): _diagnose(actual_deltas[:, j], groups)
                                     for j, alpha in enumerate(ALPHAS)},
            outer_ledger=outer_ledger, final_oof_ledger=final_oof["ledger"],
            final_fixed_alpha=final_gates["fixed_alpha"],
            final_fixed_candidate_group_mean=final_gates["fixed_predicted_delta"].tolist(),
            final_ridge=final_gates["ridge"].diagnostics, final_boost=final_gates["boost"].diagnostics,
            nested_records=dict(ids=ids.tolist(), groups=groups.tolist(), supported=supported.tolist(),
                actual_candidate_deltas=actual_deltas.tolist(),
                strategies={s:{k:v.tolist() for k,v in values.items()} for s,values in nested.items()}),
            final_training_records=dict(ids=ids.tolist(), groups=groups.tolist(),
                supported=final_oof["supported"].tolist(), features=final_oof["features"].tolist(),
                candidate_deltas=final_oof["delta"].tolist()))
        channels[name] = dict(gates=final_gates)
    report = dict(seed=int(seed), calibration_ids=ids.tolist(), calibration_groups=np.unique(groups).tolist(),
        alphas=ALPHAS.tolist(), feature_names=list(FEATURE_NAMES), channels=channel_reports,
        frozen_amplitude_bandwidth=float(fit_amp_sd), dimension=dimension,
        target="log p_fixed_bio_alpha(r | legal references) - log p_amplitude_CORE(r | same references)",
        reference_roles="one cell's independently generated DIST_CAL records only; no pooled historic OOF",
        radial_law="refitted inside every inner and outer donor fit; full cell law only for final deployment",
        gate_selection="argmax predicted gain over 0/.25/.5/1; zero and smaller alpha win exact ties",
        biological_support="raw normalized relation weights; no second ESS/strength multiplier; unsupported exact CORE",
        uncertainty="group standard errors describe nested development results, not certification",
        query_outcomes_used=False, world_model_retrained=False)
    return BiologyDeltaGates(r.copy(), amp.copy(), groups.copy(), ids.copy(), float(fit_amp_sd),
                            dimension, channels, report)
