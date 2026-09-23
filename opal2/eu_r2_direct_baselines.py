"""Complete direct Gamma/NULL baselines with disjoint fit/select/calibrate APIs.

Inputs have already been transformed using the caller's training-only recipe.
This module retains every supplied coordinate. TRAIN fits estimators, VALIDATION
selects a fixed finite grid, and DIST_CAL fits a scalar residual distribution
and a fixed L2 Platt calibrator. QUERY outcomes are not accepted during fitting.

The caller can supply access-matched TRAIN+REF_FIT or TRAIN-only inputs without
changing this recipe. The returned metadata records the actual fitting count.
"""
from __future__ import annotations

import warnings
import time

import numpy as np
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits


BOUNDS = (-1.02, .98)
LEVELS = (.5, .8, .9, .95, .99)
ARMS = ('RIDGE', 'EXTRATREES', 'HISTGB')
RIDGE_PENALTIES = (10., 1., .1, .01, .001, .0001)
LOGISTIC_C = (.01, .1, 1., 10.)
TREE_LEAVES = (20, 5, 2)
HIST_GRID = ((7, 20), (15, 10), (31, 10))
TREE_COUNT = 256
HIST_ITERATIONS = 200
TIE_ATOL = 1e-12
TIE_RTOL = 1e-12
LOGIT_EPS = 1e-6


class ConstantNullClassifier:
    """Exact observed-class fallback when the fitting labels have one class."""
    def __init__(self, probability):
        self.probability = float(probability)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack((1-p, p))


class FixedPlattCalibrator:
    """One CAL-only L2 sigmoid; no identity-vs-calibrator selection on queries."""
    def __init__(self, classifier=None, constant=None):
        self.classifier = classifier
        self.constant = constant

    @staticmethod
    def features(probability):
        p = np.clip(np.asarray(probability, float), LOGIT_EPS, 1-LOGIT_EPS)
        return (np.log(p)-np.log1p(-p))[:, None]

    def predict(self, probability):
        if self.constant is not None:
            return np.full(len(probability), float(self.constant))
        return self.classifier.predict_proba(self.features(probability))[:, 1]


def _matrix(value, name, columns=None):
    out = np.asarray(value, np.float64)
    if (out.ndim != 2 or min(out.shape) < 1 or not np.isfinite(out).all()
            or (columns is not None and out.shape[1] != columns)):
        raise ValueError(name+' must be a finite nonempty matrix with the common full input dimension')
    return out


def _gamma(value, count, name):
    out = np.asarray(value, float)
    if out.shape != (count,) or not np.isfinite(out).all():
        raise ValueError(name+' must be a finite aligned Gamma vector')
    if np.any(out < BOUNDS[0]-1e-12) or np.any(out > BOUNDS[1]+1e-12):
        raise ValueError(name+' lies outside the declared ADD_TWO Gamma range')
    return out


def _bounded_prediction(model, x):
    prediction = np.asarray(model.predict(x), float)
    if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite or misaligned regression prediction')
    return np.clip(prediction, *BOUNDS)


def _probability(model, x):
    proba = np.asarray(model.predict_proba(x), float)
    columns = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(columns) != 1:
        raise ValueError('NULL class must appear exactly once in classifier output')
    p = proba[:, columns[0]]
    if p.shape != (len(x),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Classifier returned an invalid NULL probability')
    return p


def _models(arm, classifier, n_train, seed):
    if arm == 'RIDGE':
        if classifier:
            return [(dict(C=c, regularization='L2', solver='lbfgs', max_iter=5000, tol=1e-6),
                     LogisticRegression(C=c, solver='lbfgs', max_iter=5000, tol=1e-6,
                                        random_state=seed)) for c in LOGISTIC_C]
        return [(dict(lambda_value=p, alpha=n_train*p, fit_intercept=True, solver='cholesky'),
                 Ridge(alpha=n_train*p, fit_intercept=True, solver='cholesky'))
                for p in RIDGE_PENALTIES]
    if arm == 'EXTRATREES':
        constructor = ExtraTreesClassifier if classifier else ExtraTreesRegressor
        return [(dict(n_estimators=TREE_COUNT, min_samples_leaf=leaf, max_features=1.,
                      max_depth=None, bootstrap=False, n_jobs=1),
                 constructor(n_estimators=TREE_COUNT, min_samples_leaf=leaf,
                     max_features=1., max_depth=None, bootstrap=False, n_jobs=1,
                     random_state=seed)) for leaf in TREE_LEAVES]
    if arm == 'HISTGB':
        constructor = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        return [(dict(max_leaf_nodes=nodes, min_samples_leaf=leaf, max_iter=HIST_ITERATIONS,
                      learning_rate=.05, l2_regularization=1., max_bins=255, early_stopping=False),
                 constructor(max_leaf_nodes=nodes, min_samples_leaf=leaf,
                     max_iter=HIST_ITERATIONS, learning_rate=.05, l2_regularization=1.,
                     max_bins=255, early_stopping=False, random_state=seed))
                for nodes, leaf in HIST_GRID]
    raise ValueError('Unknown baseline arm')


def _select(arm, x_train, y_train, x_valid, y_valid, seed, *, classifier):
    if classifier and len(np.unique(y_train)) == 1:
        model = ConstantNullClassifier(float(y_train[0]))
        loss = float(np.square(_probability(model, x_valid)-y_valid).mean())
        return model, dict(selection='single-class fitting labels: constant observed class',
            selected_index=None, candidates=[], validation_loss=loss,
            constant_probability=float(y_train[0]), criterion='VALIDATION Brier')
    records, best, best_score, best_index = [], None, np.inf, None
    for index, (parameters, model) in enumerate(_models(arm, classifier, len(x_train), seed)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(x_train, y_train)
        convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        estimate = _probability(model, x_valid) if classifier else _bounded_prediction(model, x_valid)
        loss = float(np.square(estimate-y_valid).mean())
        if not np.isfinite(loss):
            raise ValueError('Validation score is nonfinite')
        records.append(dict(index=index, parameters=parameters, validation_loss=loss,
                            convergence_warnings=convergence))
        if best is None or (loss < best_score and not np.isclose(
                loss, best_score, atol=TIE_ATOL, rtol=TIE_RTOL)):
            best, best_score, best_index = model, loss, index
    if records[best_index]['convergence_warnings']:
        raise RuntimeError('Selected classifier did not converge within the declared full fitting budget')
    return best, dict(selection='fixed grid on VALIDATION; retained TRAIN-only fitted model',
        selected_index=best_index, selected_parameters=records[best_index]['parameters'],
        candidates=records, validation_loss=best_score,
        criterion='VALIDATION Brier' if classifier else 'VALIDATION bounded-Gamma MSE',
        tie_rule='first in simplicity-ordered grid within atol=rtol=1e-12')


def _fit_platt(probability, label, seed):
    if len(np.unique(label)) < 2:
        p = float((label.sum()+1)/(len(label)+2))
        return FixedPlattCalibrator(constant=p), dict(
            method='single-class CAL Laplace-smoothed constant', C=None,
            smoothing_successes=1, smoothing_failures=1, probability=p,
            fit_scope='DIST_CAL only', identity_candidate_selected=False)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=5000, tol=1e-8,
                               random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(FixedPlattCalibrator.features(probability), label)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Fixed CAL Platt fit did not converge')
    return FixedPlattCalibrator(classifier=model), dict(method='fixed L2 Platt on logit(raw probability)',
        C=1., coefficient=float(model.coef_[0, 0]), intercept=float(model.intercept_[0]),
        input_probability_clip_for_logit=LOGIT_EPS, fit_scope='DIST_CAL only',
        identity_candidate_selected=False, query_performance_used=False)


def empirical_gamma_support(predicted, residuals):
    """Equal-weight CAL residual law, shifted per query and clipped to Gamma bounds."""
    mu, error = np.asarray(predicted, float), np.asarray(residuals, float)
    if (mu.ndim != 1 or error.ndim != 1 or min(len(mu), len(error)) < 1
            or not np.isfinite(mu).all() or not np.isfinite(error).all()):
        raise ValueError('Finite nonempty point predictions and CAL residuals required')
    return np.clip(mu[:, None]+np.sort(error)[None, :], *BOUNDS)


def exact_empirical_crps(support, actual):
    """Exact CRPS of the finite equal-weight distribution, using the m² pair term.

    This is not the m(m-1) unbiased/fair Monte Carlo estimator of a different
    underlying continuous law: the empirical CAL mixture itself is the model.
    """
    values = _matrix(support, 'Predictive support')
    y = np.asarray(actual, float)
    if y.shape != (len(values),) or not np.isfinite(y).all():
        raise ValueError('Actual query values must align with predictive support')
    ordered = np.sort(values, axis=1)
    m = values.shape[1]
    coefficient = 2*np.arange(1, m+1)-m-1
    return np.abs(values-y[:, None]).mean(1)-(ordered@coefficient)/m**2


def fit_direct_baselines(x_train, gamma_train, x_valid, gamma_valid,
                         x_cal, gamma_cal, x_query, *, seed):
    """Fit three complete scalar baselines; no QUERY outcome argument exists.

    ``predicted`` is the bounded direct regression estimate. For a coherent
    distribution policy use ``gamma_distribution_mean`` and
    ``p_null_from_gamma`` together. ``p_null`` and ``p_null_calibrated`` come
    from a separately trained classifier, not from the Gamma residual law.
    The caller must supply group-disjoint fitting/validation/calibration/query
    populations; this array-only interface cannot infer their identities.
    """
    train = _matrix(x_train, 'TRAIN inputs')
    valid = _matrix(x_valid, 'VALIDATION inputs', train.shape[1])
    cal = _matrix(x_cal, 'DIST_CAL inputs', train.shape[1])
    query = _matrix(x_query, 'QUERY inputs', train.shape[1])
    yt = _gamma(gamma_train, len(train), 'TRAIN Gamma')
    yv = _gamma(gamma_valid, len(valid), 'VALIDATION Gamma')
    yc = _gamma(gamma_cal, len(cal), 'DIST_CAL Gamma')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('A declared nonnegative integer seed is required')
    labels = [(y <= 0).astype(int) for y in (yt, yv, yc)]
    result = {}
    with threadpool_limits(limits=1):
        for arm in ARMS:
            stage_started = time.perf_counter()
            regression, regression_meta = _select(arm, train, yt, valid, yv, int(seed), classifier=False)
            regression_seconds = time.perf_counter()-stage_started
            stage_started = time.perf_counter()
            classifier, classifier_meta = _select(arm, train, labels[0], valid, labels[1],
                                                   int(seed), classifier=True)
            classification_seconds = time.perf_counter()-stage_started
            stage_started = time.perf_counter()
            point = _bounded_prediction(regression, query)
            cal_point = _bounded_prediction(regression, cal)
            residual = np.sort(yc-cal_point)
            probability = _probability(classifier, query)
            cal_probability = _probability(classifier, cal)
            prediction_seconds = time.perf_counter()-stage_started
            stage_started = time.perf_counter()
            platt, platt_meta = _fit_platt(cal_probability, labels[2], int(seed))
            calibrated = platt.predict(probability)
            support = empirical_gamma_support(point, residual)
            calibration_seconds = time.perf_counter()-stage_started
            result[arm] = dict(predicted=point, p_null=probability,
                p_null_calibrated=calibrated, gamma_residuals=residual,
                gamma_distribution_mean=support.mean(1), p_null_from_gamma=(support <= 0).mean(1),
                model=dict(regression=regression, classifier=classifier, platt=platt),
                metadata=dict(arm=arm, sklearn_version=sklearn.__version__, seed=int(seed),
                    timing=dict(regression_grid_fit_and_validation_seconds=regression_seconds,
                        classification_grid_fit_and_validation_seconds=classification_seconds,
                        query_and_calibration_prediction_seconds=prediction_seconds,
                        calibration_and_scalar_law_seconds=calibration_seconds,
                        query_rows=len(query), calibration_rows=len(cal),
                        interpretation='wall time on this actual fit; fixed-grid validation is included; no history inferred'),
                    n_train=len(train), n_validation=len(valid), n_calibration=len(cal), n_query=len(query),
                    full_input_dimension=train.shape[1], feature_truncation=False, pca_used=False,
                    input_transform='caller TRAIN-only transformed full X/log-norm plus chemistry',
                    fitting_scope='supplied TRAIN only; caller declares TRAIN or TRAIN+REF access arm',
                    validation_refit=False, regression=regression_meta, classification=classifier_meta,
                    null_definition='Gamma <= 0', platt=platt_meta,
                    residual_definition='DIST_CAL actual Gamma minus bounded regression prediction; no centering',
                    residual_mean=float(residual.mean()), residual_sd=float(residual.std()),
                    distribution='uniform finite CAL residual mixture shifted by query regression point',
                    gamma_bounds=list(BOUNDS), bounded_regression_predictions=True,
                    distribution_support_clipped=True, actual_outcomes_clipped=False,
                    gamma_crps_definition='exact finite distribution: mean absolute error minus half m^-2 all-pair distance',
                    gamma_crps_is_fair_monte_carlo_estimator=False,
                    classifier_probability_is_separate_from_gamma_distribution=True,
                    calibrated_classifier_selected_on_query=False, query_outcomes_accepted=False,
                    threads=1, reference_group_isolation='caller-validated before array API'))
    return result


def evaluate_direct_distribution(arm, actual_query):
    """Read-only scoring after predictions freeze; never selects any fitted model."""
    point = np.asarray(arm['predicted'], float)
    actual = _gamma(actual_query, len(point), 'QUERY evaluation Gamma')
    support = empirical_gamma_support(point, arm['gamma_residuals'])
    levels = np.asarray(LEVELS)
    lower = np.quantile(support, (1-levels)/2, axis=1, method='inverted_cdf').T
    upper = np.quantile(support, (1+levels)/2, axis=1, method='inverted_cdf').T
    null = actual <= 0
    return dict(crps=exact_empirical_crps(support, actual),
        gamma_point_mse=np.square(point-actual),
        gamma_distribution_mean_mse=np.square(support.mean(1)-actual),
        brier_raw=np.square(arm['p_null']-null),
        brier_calibrated=np.square(arm['p_null_calibrated']-null),
        brier_from_gamma=np.square((support <= 0).mean(1)-null),
        gamma_coverage_by_level=((actual[:, None] >= lower)&(actual[:, None] <= upper)),
        gamma_width_by_level=upper-lower, gamma_lower_by_level=lower, gamma_upper_by_level=upper,
        gamma_levels=levels, predictive_distribution_sample_count=support.shape[1])
