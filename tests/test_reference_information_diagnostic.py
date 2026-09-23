import numpy as np
import torch

from opal2.gram_geometry import profiles_to_gram, gram_to_coordinates, gram_gains, gram_observables
from opal2.reference_information_diagnostic import gamma_forward, mean_select, covariance_select


def test_original_gamma_and_repeat_contrasts_are_preserved():
    y=np.random.default_rng(1).normal(size=(25,4,64))
    g=profiles_to_gram(torch.tensor(y))
    u=gram_to_coordinates(g).numpy()
    gamma,contrast=gamma_forward(u,True)
    np.testing.assert_allclose(gamma,gram_gains(g).numpy()[:,2],atol=1e-12)
    obs=gram_observables(g).numpy()
    np.testing.assert_allclose(contrast[:,:3],np.log1p(obs[:,10:13]),atol=1e-12)
    np.testing.assert_allclose(contrast[:,3],np.log1p(obs[:,19]),atol=1e-12)


def test_mean_selector_can_reject_borrowing():
    r=np.array([[1.],[-1.]])
    w=np.array([[0.,1.],[1.,0.]])
    choice=mean_select([(0.,w),(1.,w)],r)
    assert choice['alpha']==0 and choice['lam']==0


def test_covariance_selector_remains_positive_definite():
    rng=np.random.default_rng(3)
    r=rng.normal(size=(12,9))
    w=(1-np.eye(12))/11
    choice=covariance_select([(0.,w)],r,np.broadcast_to(np.eye(9),(12,9,9)))
    assert choice['beta'] in (0.,.25,.5,.75)
