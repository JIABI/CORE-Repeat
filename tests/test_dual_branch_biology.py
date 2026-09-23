"""Adapter invariants and real optimizer paths; synthetic data do not test efficacy."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from opal2.dual_branch_biology import (
    BOUND, BiologicalBasisBranch, DualBranchBiologyAdapter, feature_kind,
    fit_biology_branch, fit_gelu_branch, projection_loss,
)


FIELDS = ("available", "log_count", "log_mass", "log_ess", "mean_similarity",
          "pair_log_energy", "remainder_log_energy", "pair_log_se", "remainder_log_se",
          "log_amp_gap", "angle_gap", "confidence")
NAMES = [f"{relation}_{name}" for relation in ("target", "moa") for name in FIELDS]


def fixture_arrays(n=80):
    rng = np.random.default_rng(8106)
    x = rng.normal(size=(n, 7))
    b = np.zeros((n, 24))
    support = np.arange(n) % 5 != 0
    for offset in (0, 12):
        count = rng.integers(2, 20, n)
        mass = rng.uniform(.1, 5., n)
        ess = rng.uniform(1, 10., n)
        b[:, offset:offset+12] = np.column_stack((
            np.ones(n), np.log1p(count), np.log1p(mass), np.log1p(ess),
            rng.uniform(0, 1, n), np.log1p(np.exp(.7*x[:, 1]+.5)),
            np.log1p(np.exp(-.5*x[:, 1]+.5)), np.log1p(rng.uniform(.01, 1, n)),
            np.log1p(rng.uniform(.01, 1, n)), rng.uniform(0, 2, n),
            rng.uniform(0, 2, n), ess/(ess+5)*(1-np.exp(-mass))))
    b[~support] = 0
    base = np.ones((n, 2))
    energy = np.asarray((3., 6.))*np.exp(np.column_stack((.5+.5*x[:, 0]+.3*x[:, 1],
                                                       -.3-.4*x[:, 0]+.2*x[:, 1])))
    ids = [f"compound_{j}" for j in range(n)]
    return x, b, support, energy, base, ids


@pytest.fixture(scope="module")
def fitted():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x, b, mask, e, base, ids = fixture_arrays()
    left = fit_gelu_branch(x, e, base, ids, support=mask, seed=91)
    generic = fit_biology_branch(left, x, b, NAMES, mask, e, base, ids, mode="generic", seed=42)
    structured = fit_biology_branch(left, x, b, NAMES, mask, e, base, ids, mode="structured", seed=42)
    yield (x, b, mask, e, base, ids), left, generic, structured
    torch.set_num_threads(original_threads)


def test_initialization_is_exact_off_but_not_all_parameters_zero():
    x, b, mask, _, base, _ = fixture_arrays(8)
    model = DualBranchBiologyAdapter(7, NAMES, mode="structured")
    assert np.array_equal(model.predict_increment(x, b, mask), np.zeros((8, 2)))
    assert np.array_equal(model.apply_scale(base, x, b, mask), base)
    assert torch.count_nonzero(model.left.network[0].weight) > 0
    assert torch.count_nonzero(model.right.local_coefficients) > 0
    assert torch.count_nonzero(model.right.readout.weight) == 0


def test_exact_core_off_and_unsupported_rows_skip_both_branches(fitted):
    (x, b, mask, _, base, _), _, _, model = fitted
    assert np.array_equal(model.apply_scale(base, x, b, mask, enabled=False), base)
    out = model.predict_components(x, b, mask)
    for value in out.values():
        assert np.array_equal(value[~mask], np.zeros((sum(~mask), 2)))
    assert np.array_equal(model.apply_scale(base, x, b, mask)[~mask], base[~mask])
    # If the entire cohort lacks support, biological summaries are unnecessary.
    empty_mask = np.zeros(len(x), dtype=bool)
    assert np.array_equal(model.predict_increment(x, support=empty_mask), np.zeros((len(x), 2)))
    assert np.array_equal(model.predict_increment(x, enabled=False), np.zeros((len(x), 2)))


def test_full_training_changes_both_real_paths_and_reports_all_epochs(fitted):
    (x, b, mask, _, _, ids), left, generic, structured = fitted
    for model in (left, generic, structured):
        comp = model.predict_components(x, None if model is left else b, mask)
        assert np.any(comp["left"][mask] != 0)
        assert model.report["epochs"] == 60
        assert [h["epoch"] for h in model.report["history"]] == [1, 10, 20, 30, 40, 50, 60]
        assert model.report["fitting_ids"] == np.asarray(ids)[mask].tolist()
        assert model.report["history"][-1]["training_projection_nll"] < model.report["history"][0]["training_projection_nll"]
        assert not model.report["checkpoint_selection"]
        for row in model.report["history"]:
            assert 0 < row["preclip_gradient_norm_mean"] <= row["preclip_gradient_norm_max"]
            assert 0 <= row["clipped_step_fraction"] <= 1
        assert not any(p.requires_grad for p in model.parameters())
        if model is not left:
            assert np.any(comp["right"][mask] != 0)
            assert np.max(np.abs(comp["right"])) <= BOUND
            assert model.report["left_unchanged"]


def test_identical_frozen_left_and_branch_ablation(fitted):
    (x, b, mask, _, _, _), left, generic, structured = fitted
    expected = left.predict_increment(x, support=mask)
    for model in (generic, structured):
        for name, value in left.left.state_dict().items():
            assert torch.equal(value, model.left.state_dict()[name])
        assert np.array_equal(expected, model.predict_increment(x, b, mask, right_enabled=False))
        parts = model.predict_components(x, b, mask)
        assert np.array_equal(parts["total"], parts["left"]+parts["right"])
        assert np.array_equal(parts["right"], model.predict_increment(x, b, mask, left_enabled=False))
    assert generic.report["right_trainable_parameters"] == structured.report["right_trainable_parameters"] == 146
    assert generic.right.readout.weight.data_ptr() != structured.right.readout.weight.data_ptr()
    assert not np.array_equal(generic.predict_increment(x, b, mask), structured.predict_increment(x, b, mask))


def test_gradients_reach_trained_local_coefficients_and_empirical_hidden_layer(fitted):
    (x, b, mask, e, _, _), _, _, trained = fitted
    model = deepcopy(trained).requires_grad_(True)
    out = model(torch.tensor(x), torch.tensor(b), torch.tensor(mask))
    loss = projection_loss(out[mask], torch.tensor(e)[mask]).mean()
    loss.backward()
    for p in (model.left.network[0].weight, model.left.network[-1].weight,
              model.right.local_coefficients, model.right.readout.weight):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.linalg.vector_norm(p.grad) > 0


def test_log_support_fields_are_decoded_once_and_gap_not_decoded():
    names = ["target_log_ess", "target_log_mass", "target_pair_log_se", "target_log_amp_gap"]
    branch = BiologicalBasisBranch(names, "structured").double()
    raw = torch.tensor([[np.log1p(8.), np.log1p(3.), np.log1p(2.), 2.]], dtype=torch.float64)
    bases = branch.basis(raw, torch.ones_like(raw))
    assert bases[0, 0, 0] == pytest.approx(.5)
    assert bases[0, 1, 0] == pytest.approx(.75)
    assert bases[0, 2, 0] == pytest.approx(1/3)
    assert bases[0, 3, 0] == pytest.approx(1/3)
    assert feature_kind("target_log_count") == "effective_support"
    assert feature_kind("target_confidence") == "reliability"
    assert feature_kind("target_mean_similarity") == "reliability"


def test_no_unverified_potency_claim():
    with pytest.raises(ValueError, match="separately verified"):
        BiologicalBasisBranch(["target_ec50"], "structured")


def test_serialization_preserves_predictions_bitwise(fitted, tmp_path):
    (x, b, mask, _, _, _), _, generic, _ = fitted
    path = tmp_path/"adapter.pt"
    generic.save(path)
    restored = DualBranchBiologyAdapter.load(path)
    assert restored.report == generic.report
    assert np.array_equal(generic.predict_increment(x, b, mask), restored.predict_increment(x, b, mask))
    assert not any(p.requires_grad for p in restored.parameters())


def test_scalers_fit_only_supplied_supported_training_objects(fitted):
    (x, b, mask, _, _, _), left, generic, _ = fitted
    np.testing.assert_allclose(left.empirical_center.numpy(), x[mask].mean(0))
    np.testing.assert_allclose(generic.biological_center.numpy(), b[mask].mean(0))
    before = deepcopy(generic.state_dict())
    generic.predict_increment(x*100, b, mask)
    for name, value in before.items():
        assert torch.equal(value, generic.state_dict()[name])


def test_invalid_offsets_ids_and_support_are_rejected(fitted):
    (x, b, mask, e, base, ids), left, _, _ = fitted
    with pytest.raises(ValueError, match="identical supported"):
        fit_biology_branch(left, x, b, NAMES, np.ones(len(mask), bool), e, base, ids, mode="generic")
    with pytest.raises(ValueError, match="positive"):
        fit_gelu_branch(x, e, np.zeros_like(base), ids, support=mask)
    with pytest.raises(ValueError, match="IDs"):
        fit_gelu_branch(x, e, base, ["duplicate"]*len(ids), support=mask)
    with pytest.raises(ValueError, match="boolean"):
        left.predict_increment(x, support=mask.astype(float))


def test_no_supported_training_objects_returns_zero():
    x, _, _, e, base, ids = fixture_arrays(5)
    mask = np.zeros(len(x), bool)
    model = fit_gelu_branch(x, e, base, ids, support=mask)
    assert model.report["epochs"] == 0
    assert model.report["status"] == "insufficient_support_fitted_branch_zero"
    assert np.array_equal(model.predict_increment(x), np.zeros((len(x), 2)))


def test_projection_loss_matches_gaussian_block_likelihood():
    log_scale = torch.tensor([[np.log(2.), np.log(.5)]])
    energies = torch.tensor([[6., 3.]])
    expected = .5*(3*np.log(2.)+6/2+6*np.log(.5)+3/.5)
    assert projection_loss(log_scale, energies).item() == pytest.approx(expected)
