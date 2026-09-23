"""Synthetic numerical cases only; no real fitted-model outcomes."""
import numpy as np
import pytest
import torch

from opal2.gram_geometry import coordinates_to_gram, gram_gains
from opal2.gram_forward_diagnostic import coordinates_factor_forward, factor_forward_consistency


def test_normal_forward_is_identical_to_strict_decoder_and_vector_formulas():
    u = .3*torch.randn((9, 5, 9), generator=torch.Generator().manual_seed(71), dtype=torch.float64)
    gram, diagnostics = coordinates_factor_forward(u)
    assert torch.equal(gram, coordinates_to_gram(u))
    assert diagnostics["draw_object_count"] == 45
    assert diagnostics["recovered_schur_failure_count"] == 0
    assert diagnostics["recovered_schur_failed_indices"] == []
    assert diagnostics["direct_H_cholesky_passed"]
    check = factor_forward_consistency(u, gram)
    assert check["gains_max_absolute_error"] < 5e-15
    assert check["observables_max_absolute_error"] < 2e-14
    assert check["zero_virtual_acquired_average_count"] == 0
    assert check["reference_uses_profiles_to_gram"] is False
    assert check["gain_absolute_error_per_draw_object"].shape == (9, 5)


def test_extreme_projection_keeps_forward_gains_and_exposes_lost_schur_energy():
    u = torch.zeros((2, 3, 9), dtype=torch.float64)
    u[..., :3] = 1e10
    with pytest.raises(ValueError, match="floating-point singular boundary"):
        coordinates_to_gram(u)
    gram, diagnostics = coordinates_factor_forward(u)
    assert diagnostics["recovered_schur_failure_count"] == 6
    assert diagnostics["recovered_schur_failed_mask"].all()
    assert diagnostics["recovered_vs_direct_H_max_absolute_difference"] == 1
    np.testing.assert_array_equal(diagnostics["recovered_schur_failed_indices"],
                                  [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    check = factor_forward_consistency(u, gram)
    assert check["gains_max_absolute_error"] < 1e-14
    assert torch.isfinite(gram_gains(gram)).all()
    # A common p=1e10 makes future Gram entries round together: forward gains
    # remain accurate here, but their difference-vector energy (2) is lost.
    assert check["observables_max_absolute_error_by_name"]["norm2_Z1_minus_Z2_over_norm2_X"] == 2
    assert not check["automatic_acceptance_threshold_applied"]
    assert not diagnostics["previous_failure_reclassified"]


def test_recovered_schur_mask_counts_each_case_without_removal():
    u = torch.zeros((2, 3, 9), dtype=torch.float64)
    u[0, 1, :3] = 1e10
    u[1, 2, :3] = 1e10
    gram, diagnostics = coordinates_factor_forward(u)
    assert gram.shape == (2, 3, 4, 4)
    assert diagnostics["recovered_schur_failure_count"] == 2
    assert diagnostics["recovered_schur_failed_indices"] == [[0, 1], [1, 2]]
    expected = np.zeros((2, 3), dtype=bool)
    expected[0, 1] = expected[1, 2] = True
    np.testing.assert_array_equal(diagnostics["recovered_schur_failed_mask"], expected)
    assert diagnostics["rows_dropped"] == diagnostics["draws_resampled"] == 0


@pytest.mark.parametrize("value", [-1000., 1000., -400., 400.])
def test_exponential_or_squared_underflow_and_overflow_still_raise(value):
    u = torch.zeros(9, dtype=torch.float64)
    u[3] = value
    with pytest.raises(ValueError, match="overflow|underflow|numerically SPD"):
        coordinates_factor_forward(u)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates_are_not_completed_or_resampled(value):
    u = torch.zeros(9, dtype=torch.float64)
    u[2] = value
    with pytest.raises(ValueError, match="finite"):
        coordinates_factor_forward(u)


def test_provided_gram_cannot_be_silently_modified():
    u = torch.zeros(9, dtype=torch.float64)
    gram, diagnostics = coordinates_factor_forward(u)
    changed = gram.clone()
    changed[1, 1] += .1
    with pytest.raises(ValueError, match="unchanged factor-forward"):
        factor_forward_consistency(u, changed)
    assert diagnostics["draw_object_count"] == 1
    assert diagnostics["recovered_schur_failed_mask"].shape == ()


def test_near_cancelled_acquired_mean_cannot_be_repaired_by_forward_diagnostic():
    u = torch.zeros(9, dtype=torch.float64)
    u[0] = -1
    u[3] = -25
    gram, diagnostics = coordinates_factor_forward(u)
    # The virtual (X+Z1)/2 has a tiny positive orthogonal norm, but the rounded
    # Gram has lost it. Calling original Gram gains must still fail explicitly.
    assert diagnostics["recovered_schur_failure_count"] == 1
    with pytest.raises(ValueError, match="acquired average"):
        factor_forward_consistency(u, gram)
