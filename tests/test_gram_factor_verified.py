import numpy as np
import pytest
import torch

from opal2.gram_factor_verified import decode_draws
from opal2.gram_oof_experiment import decode_draws as strict_decode


def test_normal_draws_are_bitwise_unchanged_and_all_observables_checked():
    u = np.random.default_rng(153).normal(scale=.4, size=(12, 7, 9))
    old, _ = strict_decode(u)
    new, audit = decode_draws(u)
    np.testing.assert_array_equal(old, new)
    assert audit['draw_object_count'] == 84
    assert audit['functional_consistency']['every_draw_checked']
    assert len(audit['functional_consistency']['observables_max_absolute_error_by_name']) == 20
    assert audit['rows_dropped'] == audit['draws_resampled'] == 0


def test_ill_conditioned_factor_preserves_forward_draw_not_rounded_inverse():
    u = np.array([.2, -.1, .3, np.log(1e-5), 1., np.log(1e-8), 2., 1., np.log(1e-5)], dtype=np.float64)
    with pytest.raises(ValueError, match='numerically SPD'):
        strict_decode(u[None, None])
    gram, audit = decode_draws(u[None, None])
    assert audit['direct_H_refactorization_failure_count'] == 1
    assert audit['failed_factor_draws_high_precision_verified'] == 1
    assert audit['high_precision_used_to_modify_predictions'] is False
    assert audit['high_precision_audits'][0]['exact_float64_lift']
    assert np.isfinite(gram).all()


@pytest.mark.parametrize('value', [1000., -1000., -400.])
def test_overflow_or_underflow_still_fails(value):
    u = np.zeros((1, 1, 9), dtype=np.float64)
    u[..., 3] = value
    with pytest.raises(ValueError, match='underflow|overflow'):
        decode_draws(u)


def test_dtype_nonfinite_and_disabled_verification_rejected():
    with pytest.raises(ValueError, match='float64'):
        decode_draws(np.zeros((2, 9), dtype=np.float32))
    with pytest.raises(ValueError, match='finite'):
        decode_draws(np.full((2, 9), np.nan))
    with pytest.raises(ValueError, match='mandatory'):
        decode_draws(np.zeros((2, 9)), verify=False)
