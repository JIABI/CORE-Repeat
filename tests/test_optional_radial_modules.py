import numpy as np
import pytest

from opal2.optional_radial_modules import (supported_retrieval, blend_retrieval,
    fit_radial_switch, state_similarity, BIO_COEFFICIENTS)


def test_exact_off_and_unsupported():
    base = np.array([[.1,.2,.7],[.3,.4,.3]])
    channel = supported_retrieval([[0,0,0],[1,0,.1]])
    out, gate = blend_retrieval(base,[channel],[.5])
    np.testing.assert_array_equal(out[0],base[0])
    assert gate[0] == 0 and gate[1] > 0
    np.testing.assert_array_equal(blend_retrieval(base,[],[],enabled=False)[0],base)
    np.testing.assert_allclose(out.sum(1),1)


def test_ess_cannot_amplify_tiny_similarity():
    c = supported_retrieval(np.full((1,100),1e-29))
    assert c['ess'][0] > 99
    assert 0 < c['gate'][0] < 1e-28


def test_relation_channels_are_independent_and_exclude_pairs():
    base = np.full((2,3),1/3)
    allowed = np.array([[False,True,True],[True,True,True]])
    t = supported_retrieval(np.array([[1,0,0],[1,0,0]]),allowed=allowed)
    m = supported_retrieval(np.array([[0,1,0],[0,0,1]]),allowed=allowed)
    a,_ = blend_retrieval(base,[t,m],[.5,0])
    b,_ = blend_retrieval(base,[t,m],[0,.5])
    np.testing.assert_array_equal(a[0],base[0])
    assert not np.array_equal(b[0],base[0])


def test_calibration_no_support_selects_off():
    r = np.linspace(.8,4,12); a = np.linspace(-2,2,12); g = np.arange(12).astype(str)
    fitted = fit_radial_switch(r,a,g,[np.zeros((12,12)),np.zeros((12,12))],
        fit_amp_sd=1.,coefficient_grid=BIO_COEFFICIENTS)
    assert fitted.coefficients == (0.,0.)
    assert not fitted.enabled


def test_admission_prefers_smaller_mixing_not_lowest_eligible_loss(monkeypatch):
    # Isolate the declared selection policy from the radial density estimator:
    # both nonzero settings clear the margin, but the stronger setting has the
    # lower loss. The protocol still requires the smaller intervention.
    calls = 0

    def controlled_nll(residual, scatter, law, weights):
        nonlocal calls
        candidate = calls % 3
        calls += 1
        return np.full(len(residual), (10., 8., 6.)[candidate])

    monkeypatch.setattr('opal2.optional_radial_modules.radial_nll', controlled_nll)
    r = np.linspace(.8,4,12)
    amp = np.linspace(-2,2,12)
    groups = np.arange(12).astype(str)
    fitted = fit_radial_switch(r,amp,groups,[np.ones((12,12))],
        fit_amp_sd=1.,coefficient_grid=((0.,),(.25,),(.5,)))
    assert calls == 9
    assert fitted.coefficients == (.25,)
    candidates = fitted.selection['candidates']
    assert candidates[1]['eligible'] and candidates[2]['eligible']
    assert candidates[2]['mean_nll'] < candidates[1]['mean_nll']
    assert fitted.selection['selected_index'] == 1


def test_group_exclusion_and_state_kernel():
    r = np.linspace(1,4,12); a = np.linspace(-1,1,12); g = np.repeat(np.arange(6),2).astype(str)
    # A self-group-only channel has no heldout donor and cannot be selected.
    s = (g[:,None] == g[None,:]).astype(float)
    fitted = fit_radial_switch(r,a,g,[s],fit_amp_sd=1.)
    assert not fitted.enabled
    sim = state_similarity([[0,0],[1,1]],[[0,0]],scale=[1,1])
    assert sim.shape == (1,2) and sim[0,0] == 1 and 0 < sim[0,1] < 1


def test_bad_mixture_is_rejected():
    base = np.full((1,2),.5); c = supported_retrieval([[1,0]])
    with pytest.raises(ValueError): blend_retrieval(base,[c],[1.1])
