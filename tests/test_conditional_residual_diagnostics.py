"""Analytic moment identities and information-boundary checks."""
import numpy as np
import pytest

from opal2.conditional_residual_diagnostics import (
    projected_error_diagnostics, amplitude_strata, summarize_subset, _moment_summary,
)
from opal2.joint_contrast_scale import contrast_projector


def fixture(n=7):
    rng = np.random.default_rng(326)
    raw = rng.normal(scale=.15, size=(n, 9))
    scale = np.linspace(.5, 1.4, 9)
    matrix = rng.normal(scale=.04, size=(n, 9, 9))+np.eye(9)[None]
    scatter = matrix@matrix.swapaxes(-1, -2)
    mean = rng.normal(scale=.2, size=(n, 9))
    actual = mean+rng.normal(size=(n, 9))
    return mean, actual, raw, scale, scatter


def test_isotropic_core_and_anisotropic_expected_projection_energies():
    mean, actual, raw, scale, scatter = fixture()
    decomposition = contrast_projector(raw, scale, scatter)
    P, L = decomposition['projector'], decomposition['factor']
    alternate = L@(2*P+.5*(np.eye(9)-P))@L.swapaxes(-1, -2)
    out = projected_error_diagnostics(mean, actual, raw, scale, scatter,
                                      dict(CORE=scatter*1.7, DESCRIPTORS=alternate))
    np.testing.assert_allclose(out['expected']['CORE']['energies'], np.tile([5.1,10.2], (len(mean),1)), atol=1e-12)
    np.testing.assert_allclose(out['expected']['DESCRIPTORS']['energies'], np.tile([6,3], (len(mean),1)), atol=1e-12)
    np.testing.assert_allclose(out['expected']['CORE']['expected_energy_fraction'], 1/3)
    np.testing.assert_allclose(out['expected']['DESCRIPTORS']['expected_energy_fraction'], 2/3)
    np.testing.assert_allclose(out['energies'].sum(1), np.square(out['whitened_residual']).sum(1), atol=1e-12)


def test_zero_realized_error_is_explicit_missing_angle_not_zero_angle():
    mean, _, raw, scale, scatter = fixture()
    out = projected_error_diagnostics(mean, mean, raw, scale, scatter, dict(CORE=scatter))
    assert np.isnan(out['angular_fraction']).all()
    summary = summarize_subset(out, np.ones(len(mean), bool))
    assert summary['observed_zero_error_rows'] == len(mean)
    assert summary['observed_mean_angular_fraction'] is None
    assert summary['observed_ratio_of_pooled_energies'] is None


def test_centering_preserves_bias_and_exact_finite_cohort_identity():
    rng = np.random.default_rng(17)
    w = rng.normal(size=(40,9))+np.arange(9)[None]*.2
    summary = _moment_summary(w)
    np.testing.assert_allclose(summary['whitened_residual_mean'], w.mean(0))
    assert abs(summary['decomposition_roundoff']) < 1e-12
    assert summary['mean_vector_squared_norm'] > 0
    assert summary['mean_uncentered_total_energy'] > summary['mean_centered_total_energy']


def test_amplitude_strata_fit_only_and_ties_explicit():
    train = np.arange(20, dtype=float)
    labels, rule = amplitude_strata(train, [-100, 4.75, 9.5, 14.25, 1000])
    assert rule['edges'] == [4.75, 9.5, 14.25]
    assert labels.tolist() == [0,1,2,3,3]
    _, changed_query_rule = amplitude_strata(train, [1e8]*5)
    assert changed_query_rule == rule
    tied, tied_rule = amplitude_strata(np.ones(20), [0,1,2])
    assert tied.tolist() == [0,3,3]
    assert tied_rule['edges'] == [1.,1.,1.]


def test_bad_covariance_and_wrong_shapes_rejected():
    mean, actual, raw, scale, scatter = fixture()
    invalid = scatter.copy(); invalid[:,0,0] = -10
    with pytest.raises(np.linalg.LinAlgError):
        projected_error_diagnostics(mean, actual, raw, scale, scatter, dict(bad=invalid))
    asymmetric = scatter.copy(); asymmetric[:,0,1] += .1
    with pytest.raises(ValueError, match='symmetric'):
        projected_error_diagnostics(mean, actual, raw, scale, scatter, dict(bad=asymmetric))
    with pytest.raises(ValueError, match='nine-coordinate'):
        projected_error_diagnostics(mean[:,:8], actual, raw, scale, scatter, dict(CORE=scatter))
