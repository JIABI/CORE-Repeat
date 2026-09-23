import numpy as np

from scripts.diagnose_r3_relation_variance_amplitude_20260921 import amplitude_features


def test_amplitude_kernels_use_saved_train_edges_and_are_psd():
    edges = np.array([2., 3., 4., 5.])
    bins, z, center, scale = amplitude_features(np.array([1., 2., 3., 4., 5., 6.]), edges)
    np.testing.assert_array_equal(bins, [0, 1, 2, 3, 4, 4])
    assert center == 3.5 and scale == 3.
    k_bin = np.equal.outer(bins, bins).astype(float)
    k_amp = np.outer(z, z)
    assert np.linalg.eigvalsh(k_bin).min() > -1e-12
    assert np.linalg.eigvalsh(k_amp).min() > -1e-12
    assert k_amp[0, -1] < 0
    _, z_extra, same_center, same_scale = amplitude_features(np.array([1., 100.]), edges)
    assert same_center == center and same_scale == scale
    assert z_extra[0] == z[0]  # Other query values do not fit the scale.
