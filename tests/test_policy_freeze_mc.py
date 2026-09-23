import numpy as np

from opal2.policy_freeze_mc import moment_scores,mc_score_se


def test_score_se_matches_joint_sample_not_independent_error_approximation():
    rng=np.random.default_rng(27);gamma=rng.normal(size=(1000,3))
    r=moment_scores(gamma);indicator=gamma<=0
    assert np.all(r['mc_mean_null_covariance']<=0)
    for lam in (0.,.2):
        expected=(gamma-lam*indicator).std(0,ddof=1)/np.sqrt(len(gamma))
        np.testing.assert_allclose(mc_score_se(r['gamma_mc_se'],r['null_mc_se'],r['mc_mean_null_covariance'],lam),expected,rtol=1e-13)
