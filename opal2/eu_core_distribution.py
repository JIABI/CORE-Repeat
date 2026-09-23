"""Complete CORE error law with separate REF, DIST_CAL and X-only QUERY APIs.

All means, residuals and scatter matrices use the same MODEL_TRAIN-standardized
nine-dimensional geometry coordinates. X is the observed first-well profile,
not a future profile or a learned target. This module fits no conditional mean.

Recipe: LOCAL_SCALE donor leave-group-out selection, AMPLITUDE_TOTAL, then
AMP_EMP_LOCAL calibration. The empirical radial law is not moment-normalized:
the predictive covariance equals scatter times the reported radial multiplier.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy

import numpy as np

from .conditional_joint_error import fit_covariance_family
from .empirical_radial import fit_radial, reference_weights as radial_weights
from .empirical_radial import variance_multiplier
from .joint_contrast_scale import fit_scale, predict_scale
from .joint_tail_calibration import mahalanobis_scores
from .reference_information_memory import (
    chemistry_similarity, morphology_similarity, normalized_topk_weights,
)


DIMENSION = 9
COORDINATE_SPACE = 'MODEL_TRAIN-standardized nine-dimensional geometry u'
RECIPE = 'LOCAL_SCALE + AMPLITUDE_TOTAL + AMP_EMP_LOCAL'
INPUT_KEYS = frozenset(('ids', 'groups', 'X', 'chem'))


def _names(values, name, count=None, unique=False):
    original = np.asarray(values)
    if original.ndim != 1 or not len(original) or any(v is None for v in original):
        raise ValueError(name + ' must contain nonempty identities')
    result = np.asarray(values, dtype=str)
    if ((count is not None and len(result) != count) or np.any(result == '')
            or (unique and len(np.unique(result)) != len(result))):
        raise ValueError(name + ' must contain aligned ' + ('unique ' if unique else '')
                         + 'nonempty identities')
    return result.copy()


def _inputs(values, role, *, require_mean=False):
    """Whitelist decision-time inputs, rejecting future fields rather than ignoring them."""
    if not isinstance(values, Mapping):
        raise ValueError(role + ' inputs must be a mapping')
    allowed = INPUT_KEYS | ({'mean_u'} if require_mean else set())
    if set(values) != allowed:
        missing, unexpected = sorted(allowed - set(values)), sorted(set(values) - allowed)
        raise ValueError(f'{role} decision inputs: missing={missing}, unexpected={unexpected}; '
                         'future measurements, targets, residuals and Gamma are not accepted')
    ids = _names(values['ids'], role + ' ids', unique=True)
    groups = _names(values['groups'], role + ' groups', len(ids))
    x, chem = np.asarray(values['X'], float), np.asarray(values['chem'], float)
    if (x.ndim != 2 or x.shape[0] != len(ids) or x.shape[1] == 0
            or not np.isfinite(x).all()):
        raise ValueError(role + ' X must be a finite first-well matrix')
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError(role + ' X must have finite strictly positive norms')
    logamp = np.log(norms)
    if (chem.ndim != 2 or chem.shape[0] != len(ids) or chem.shape[1] < 512
            or not np.isfinite(chem).all()
            or np.any((chem[:, :512] != 0) & (chem[:, :512] != 1))):
        raise ValueError(role + ' chem must contain 512 binary Morgan bits, then optional flags')
    result = dict(ids=ids, groups=groups, X=x.copy(), chem=chem.copy(), log_amplitude=logamp)
    if require_mean:
        result['mean_u'] = _residual(values['mean_u'], len(ids), role + ' mean_u')
    return result


def _residual(values, count, name):
    result = np.asarray(values, float)
    if result.shape != (count, DIMENSION) or not np.isfinite(result).all():
        raise ValueError(name + ' must be finite [N,9] in standardized-u coordinates')
    return result.copy()


def _disjoint(left, right, left_name, right_name):
    for key in ('ids', 'groups'):
        if set(left[key]) & set(right[key]):
            raise ValueError(f'{left_name}/{right_name} {key} overlap; whole-group isolation required')


def _covariance_reference_weights(query, donor, bandwidth):
    """Exactly the prior LOCAL_SCALE similarity, without a four-well data argument."""
    if query['X'].shape[1] != donor['X'].shape[1]:
        raise ValueError('Query and reference first-well dimensions differ')
    allowed = query['groups'][:, None] != donor['groups'][None, :]
    direction = morphology_similarity(query['X'], donor['X'])
    chemical = chemistry_similarity(query['chem'][:, :512], donor['chem'][:, :512])
    amplitude = np.exp(-.5 * ((query['log_amplitude'][:, None]
                              - donor['log_amplitude'][None, :]) / bandwidth) ** 2)
    similarity = ((direction + chemical) / 2 + amplitude) / 2
    result = normalized_topk_weights(similarity, donor_ids=donor['ids'], top_k=16,
                                    eligible=allowed)
    if np.any(result['weights'][~allowed] != 0):
        raise ValueError('A complete query group entered its reference weights')
    return result


def decision_reference_weights(query_inputs, donor_inputs, bandwidth):
    """Public X-only reference weighting, also permitting group-excluded REF LOO."""
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Positive finite reference bandwidth required')
    return _covariance_reference_weights(_inputs(query_inputs, 'query'),
                                         _inputs(donor_inputs, 'donor'), float(bandwidth))


def fit_eu_distribution(ref_inputs, ref_residual, cal_inputs, cal_residual,
                        base_scatter, training_log_amplitude_sd, *,
                        model_training_ids=None, model_training_groups=None):
    """Fit the full frozen-mean CORE law without QUERY data or outcomes.

    Residuals are ``transform_target(raw, stats) - mean_u``. ``base_scatter``
    must use those same standardized coordinates. Caller must supply means
    fitted without REF/DIST_CAL/QUERY future observations. Optional training
    IDs/groups enforce that declared provenance against REF and DIST_CAL and
    subsequently QUERY. Calibration uses ID-min representatives per group.
    """
    ref, cal = _inputs(ref_inputs, 'REF'), _inputs(cal_inputs, 'DIST_CAL')
    _disjoint(ref, cal, 'REF', 'DIST_CAL')
    rr = _residual(ref_residual, len(ref['ids']), 'REF residual')
    cr = _residual(cal_residual, len(cal['ids']), 'DIST_CAL residual')
    base = np.asarray(base_scatter, float)
    if (base.shape != (DIMENSION, DIMENSION) or not np.isfinite(base).all()
            or not np.allclose(base, base.T, atol=1e-12, rtol=1e-12)):
        raise ValueError('Base scatter must be a finite symmetric [9,9] standardized-u matrix')
    try:
        factor = np.linalg.cholesky(base)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Base scatter must be positive definite') from exc
    if (np.ndim(training_log_amplitude_sd) != 0
            or not np.isfinite(training_log_amplitude_sd) or training_log_amplitude_sd < 0):
        raise ValueError('MODEL_TRAIN log-amplitude sd must be a finite nonnegative scalar')
    train = None
    if (model_training_ids is None) != (model_training_groups is None):
        raise ValueError('Supply both MODEL_TRAIN identities and groups or neither')
    if model_training_ids is not None:
        train_ids = _names(model_training_ids, 'MODEL_TRAIN ids', unique=True)
        train = dict(ids=train_ids, groups=_names(model_training_groups,
                     'MODEL_TRAIN groups', len(train_ids)))
        _disjoint(train, ref, 'MODEL_TRAIN', 'REF')
        _disjoint(train, cal, 'MODEL_TRAIN', 'DIST_CAL')

    bandwidth = max(float(training_log_amplitude_sd), .1)
    loo = _covariance_reference_weights(ref, ref, bandwidth)
    cal_weight = _covariance_reference_weights(cal, ref, bandwidth)
    covariance = fit_covariance_family(rr, base, loo['weights'], cal_weight['weights'],
                                       'LOCAL_SCALE')
    energy = mahalanobis_scores(rr, covariance['loo_covariance'])
    amplitude = fit_scale(energy, DIMENSION, ref['log_amplitude'],
                          conditional=True, penalty=1.)
    ref_amp = predict_scale(amplitude, ref['log_amplitude'])
    cal_amp = predict_scale(amplitude, cal['log_amplitude'])
    ref_scatter = covariance['loo_covariance'] * ref_amp[:, None, None]
    cal_scatter = covariance['query_covariance'] * cal_amp[:, None, None]
    representatives = np.array([min(np.flatnonzero(cal['groups'] == group),
        key=lambda i: cal['ids'][i]) for group in np.unique(cal['groups'])], dtype=int)
    radii = np.sqrt(mahalanobis_scores(cr[representatives], cal_scatter[representatives]))
    law = fit_radial(radii, dimension=DIMENSION)
    radial_sd = float(ref['log_amplitude'].std())
    # Validate the original REF amplitude bandwidth, without replacing it by
    # MODEL_TRAIN sd or by a query/calibration-fitted value.
    reference_local = radial_weights(cal['log_amplitude'][representatives],
        ref['log_amplitude'], radial_sd, conditional=True)
    reference_multiplier = variance_multiplier(law, reference_local['weights'])
    relative_ref_energy = np.square(np.linalg.solve(factor, rr.T)).sum(0) / DIMENSION
    report = dict(recipe=RECIPE, coordinate_space=COORDINATE_SPACE,
        covariance_family='LOCAL_SCALE', local_covariance_choice=covariance['choice'],
        amplitude_fit=amplitude, covariance_reference_bandwidth=bandwidth,
        covariance_reference_bandwidth_source='max(MODEL_TRAIN log||X|| sd, .1)',
        radial_reference_bandwidth=radial_sd, radial_reference_bandwidth_source='REF log||X|| sd',
        ref_count=len(ref['ids']), ref_group_count=len(np.unique(ref['groups'])),
        calibration_count=len(cal['ids']), calibration_group_count=len(representatives),
        calibration_representative_rule='lexicographically smallest ID in each DIST_CAL group',
        calibration_representative_ids=cal['ids'][representatives].tolist(),
        radial_ess_shrinkage_constant=20., radial_gaussian_guard=law['epsilon'],
        mean_fitted_or_changed=False, query_outcomes_used=False,
        model_training_provenance_checked=train is not None,
        model_training_provenance_requirement='mean and preprocessing fitted without REF, DIST_CAL or QUERY outcomes',
        residual_interpretation='prediction error including bias, not identified physical shared/independent noise',
        covariance_interpretation='scatter_u * radial_variance_multiplier; radius law is not moment-normalized')
    return dict(recipe=RECIPE, coordinate_space=COORDINATE_SPACE, ref_inputs=ref,
        calibration_ids=cal['ids'], calibration_groups=cal['groups'],
        calibration_representative_indices=representatives,
        representative_ids=cal['ids'][representatives],
        representative_log_amplitude=cal['log_amplitude'][representatives],
        model_training_identities=train, base_scatter=base.copy(),
        reference_relative_base_energy=relative_ref_energy,
        covariance_choice=copy.deepcopy(covariance['choice']), amplitude_fit=amplitude,
        covariance_reference_bandwidth=bandwidth, radial_reference_bandwidth=radial_sd,
        law=law, reference_loo_weights=loo['weights'],
        reference_loo_covariance=covariance['loo_covariance'], reference_loo_energy=energy,
        reference_scatter_u=ref_scatter, reference_radial_weights=reference_local['weights'],
        reference_radial_variance_multiplier=reference_multiplier,
        reference_covariance_u=ref_scatter * reference_multiplier[:, None, None],
        calibration_reference_weights=cal_weight['weights'],
        calibration_scatter_u=cal_scatter, calibration_radii=radii, report=report)


def predict_eu_distribution(fitted, query_inputs):
    """Apply frozen CORE distribution to legal QUERY X and unchanged ``mean_u``."""
    if fitted.get('recipe') != RECIPE or fitted.get('coordinate_space') != COORDINATE_SPACE:
        raise ValueError('Expected a fitted standardized-u EU CORE distribution')
    query = _inputs(query_inputs, 'QUERY', require_mean=True)
    ref = fitted['ref_inputs']
    cal = dict(ids=fitted['calibration_ids'], groups=fitted['calibration_groups'])
    _disjoint(query, ref, 'QUERY', 'REF')
    _disjoint(query, cal, 'QUERY', 'DIST_CAL')
    if fitted['model_training_identities'] is not None:
        _disjoint(query, fitted['model_training_identities'], 'QUERY', 'MODEL_TRAIN')
    selected = _covariance_reference_weights(query, ref, fitted['covariance_reference_bandwidth'])
    beta = fitted['covariance_choice']['beta']
    tau = selected['weights'] @ fitted['reference_relative_base_energy']
    if beta == 0:
        local_scatter = np.broadcast_to(fitted['base_scatter'],
                                        (len(query['ids']), DIMENSION, DIMENSION)).copy()
    else:
        local_scatter = (1 - beta + beta * tau)[:, None, None] * fitted['base_scatter']
    amplitude = predict_scale(fitted['amplitude_fit'], query['log_amplitude'])
    scatter = local_scatter * amplitude[:, None, None]
    np.linalg.cholesky(scatter)
    local = radial_weights(fitted['representative_log_amplitude'], query['log_amplitude'],
                            fitted['radial_reference_bandwidth'], conditional=True)
    multiplier = variance_multiplier(fitted['law'], local['weights'])
    covariance = scatter * multiplier[:, None, None]
    if not np.isfinite(covariance).all():
        raise ValueError('Predictive covariance overflowed')
    np.linalg.cholesky(covariance)
    return dict(ids=query['ids'], groups=query['groups'], mean_u=query['mean_u'],
        base_scatter_u=local_scatter, scatter_u=scatter, covariance_u=covariance,
        law=copy.deepcopy(fitted['law']),
        radial_weights=local['weights'], radial_variance_multiplier=multiplier,
        radial_ess=local['ess'], radial_local_ess=local['local_ess'],
        radial_shrinkage=local['shrinkage'], reference_weights=selected['weights'],
        reference_ess=selected['neff'], amplitude_factor=amplitude,
        local_covariance_factor=1 - beta + beta * tau,
        report=dict(recipe=RECIPE, coordinate_space=COORDINATE_SPACE,
            query_count=len(query['ids']), means_changed=False, query_outcomes_used=False,
            reference_count=len(ref['ids']), calibration_groups=len(fitted['representative_ids']),
            covariance_interpretation=fitted['report']['covariance_interpretation']))
