import pytest
import torch

from opal2.kernels import MeasurementBasis, MeasurementKernelOperator, SplineKANLinear


def test_cubic_spline_partition_and_trainable_gradients():
    torch.manual_seed(1)
    layer = SplineKANLinear(3, 5).double()
    x = torch.randn(11, 3, dtype=torch.float64, requires_grad=True)
    assert torch.allclose(layer.basis(x).sum(-1), torch.ones_like(x), atol=1e-12)
    layer(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.spline_weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["measurement", "generic", "mlp"])
def test_actual_operator_modes_shapes_and_gradients(mode):
    torch.manual_seed(2)
    operator = MeasurementKernelOperator(12, mode)
    context, query = torch.randn(2, 3, 4, 12), torch.randn(2, 3, 4, 12)
    descriptors = torch.rand(2, 3, 4, 6)
    value = operator(context, query, descriptors)
    assert value.shape == context.shape
    value.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in operator.parameters() if p.grad is not None)
    if mode != "mlp":
        assert operator.coefficients[0].spline_weight.grad.abs().sum() > 0


def test_measurement_basis_responds_to_declared_context_count():
    d = torch.zeros(2, 6)
    d[:, 4] = torch.tensor([1 / 2, 3 / 4])
    basis = MeasurementBasis()(d)
    assert basis.shape == (2, 16)
    assert basis[0, 9] != basis[1, 9]


def test_operator_rejects_unknown_mode():
    with pytest.raises(ValueError):
        MeasurementKernelOperator(8, "unknown")
