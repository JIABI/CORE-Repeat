import numpy as np
import pytest

from opal2.gram_high_precision import verify_factor_draw


def _fixture(factor=None):
    p = np.array([.3, .5, .7], dtype=np.float64)
    if factor is None:
        factor = np.array([[1., 0., 0.], [.2, .8, 0.], [-.1, .3, .9]])
    rows = np.vstack((np.array([1., 0., 0., 0.]), np.column_stack((p, factor))))
    h = factor @ factor.T
    gram = np.empty((4, 4), dtype=np.float64)
    gram[0, 0], gram[0, 1:], gram[1:, 0] = 1., p, p
    gram[1:, 1:] = np.outer(p, p) + h
    norm2 = np.square(rows).sum(-1)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    cosines = [rows[i] @ rows[j] / np.sqrt(norm2[i] * norm2[j]) for i, j in pairs]
    x, z1, z2, v = rows
    vectors = (z1-z2, z1-v, z2-v, (z1+z2)/2, (z1+v)/2, (z2+v)/2,
               (x+z1)/2, (x+z2)/2, (x+z1+z2)/3, (z1+z2+v)/3)
    obs = np.array(cosines + list(np.sqrt(norm2/norm2[0]))
                   + [row @ row / norm2[0] for row in vectors])
    before = x @ v / np.sqrt(norm2[0]*norm2[3])
    gain = np.array([.5*(row @ v/np.sqrt((row @ row)*norm2[3])-before)-cost
                     for row, cost in zip(((x+z1)/2, (x+z2)/2, (x+z1+z2)/3), (.01,.01,.02))])
    return p, factor, gram, gain, obs


def test_regular_draw_passes_without_mutation_or_replacement():
    values = _fixture()
    before = [value.copy() for value in values]
    result = verify_factor_draw(*values)
    assert result["dps_used"] == 80
    assert result["high_precision_cholesky_passed"]
    assert result["gains_max_absolute_error"] < 1e-12
    assert not result["replacement_values_returned"]
    assert len(result["sylvester_pivots"]) == 3
    for old, current in zip(before, values):
        assert np.array_equal(old, current)


def test_positive_factor_with_rounded_singular_h_passes_forward_audit():
    factor = np.array([[1e-3, 0., 0.], [1000., 1e-6, 0.], [300., 400., 1e-2]])
    values = _fixture(factor)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(factor @ factor.T)
    result = verify_factor_draw(*values)
    assert result["exact_float64_lift"]
    assert result["observables_max_tolerance_ratio"] < 1
    assert result["gram_max_tolerance_ratio"] < 1


@pytest.mark.parametrize("which", [2, 3, 4])
def test_corrupted_outputs_are_rejected(which):
    values = list(_fixture())
    values[which] = values[which].copy()
    values[which].flat[0] += .01
    with pytest.raises(ValueError, match="mismatch"):
        verify_factor_draw(*values)


def test_precision_fallback_only_on_cholesky(monkeypatch):
    import opal2.gram_high_precision as module
    original = module.mp.cholesky
    attempts = []
    def checked(matrix):
        attempts.append(module.mp.mp.dps)
        if len(attempts) == 1:
            raise ValueError("Simulated insufficient working precision")
        return original(matrix)
    monkeypatch.setattr(module.mp, "cholesky", checked)
    result = verify_factor_draw(*_fixture())
    assert attempts == [80, 160]
    assert result["precision_attempts"] == [80, 160]
    assert result["dps_used"] == 160


def test_invalid_inputs_stop_before_audit():
    values = list(_fixture())
    values[1] = values[1].copy()
    values[1][1,1] = 0.
    with pytest.raises(ValueError, match="strictly positive"):
        verify_factor_draw(*values)
    values = list(_fixture())
    values[0][0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        verify_factor_draw(*values)
    with pytest.raises(ValueError, match="float64"):
        verify_factor_draw(*[value.astype(np.float32) for value in _fixture()])
