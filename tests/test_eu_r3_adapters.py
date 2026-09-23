"""EU support semantics and full optimizer paths, not synthetic efficacy claims."""
from copy import deepcopy
import inspect

import numpy as np
import pytest
import torch

from opal2.dual_branch_biology import BOUND, DualBranchBiologyAdapter
from opal2.eu_r3_adapters import EUResidualAdapter, fit_left, fit_right, predict


NAMES = ["target_available", "target_log_ess", "target_log_mass",
         "target_pair_log_energy", "target_remainder_log_energy", "target_log_amp_gap"]


def increment(*args, **kwargs):
    return predict(*args, **kwargs)["total"]


def arrays():
    rng = np.random.default_rng(6317)
    x = rng.normal(size=(72, 5))
    support = np.arange(len(x)) % 3 != 0
    b = np.column_stack((np.ones(len(x)), np.log1p(rng.uniform(2, 15, len(x))),
                         np.log1p(rng.uniform(.1, 3, len(x))),
                         np.log1p(np.exp(x[:, 1])), np.log1p(np.exp(-x[:, 1])),
                         rng.uniform(0, 2, len(x))))
    b[~support] = 0
    energies = np.array([3., 6.])*np.exp(np.column_stack((.6+.4*x[:, 0]+.4*x[:, 1],
                                                        -.4-.3*x[:, 0]+.4*x[:, 1])))
    return x, b, support, energies, [f"ref_{i}" for i in range(len(x))]


@pytest.fixture(scope="module")
def fitted():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = arrays()
    x, b, mask, energy, ids = data
    left = fit_left(x, energy, ids, seed=87)
    generic = fit_right(left, x, b, NAMES, mask, energy, ids, mode="generic", seed=23)
    structured = fit_right(left, x, b, NAMES, mask, energy, ids, mode="structured", seed=23)
    yield data, left, generic, structured
    torch.set_num_threads(threads)


def test_zero_initialized_readouts_and_bounds():
    x, b, mask, _, _ = arrays()
    model = EUResidualAdapter(x.shape[1], NAMES, mode="structured")
    assert np.array_equal(increment(model, x, b, mask), np.zeros((len(x), 2)))
    assert torch.count_nonzero(model.left.network[-1].weight) == 0
    assert torch.count_nonzero(model.right.readout.weight) == 0
    assert torch.count_nonzero(model.right.local_coefficients) > 0


def test_full_off_exact_core_and_right_off_exact_left(fitted):
    (x, b, mask, _, _), left, generic, structured = fitted
    baseline = increment(left, x)
    for model in (left, generic, structured):
        assert np.array_equal(increment(model, x, enabled=False), np.zeros_like(baseline))
        assert np.array_equal(increment(model, x, right_enabled=False), baseline)
        original = np.exp(np.column_stack((x[:, 0], x[:, 1])))
        assert np.array_equal(model.apply_scale(original, x, enabled=False), original)


def test_no_biology_support_keeps_empirical_branch_active(fitted):
    (x, b, mask, _, _), left, generic, structured = fitted
    expected = increment(left, x)
    assert np.any(expected[~mask] != 0)
    no_support = np.zeros(len(x), bool)
    for model in (generic, structured):
        comp = model.predict_components(x, b, mask)
        assert np.array_equal(comp["left"], expected)
        assert np.array_equal(comp["right"][~mask], np.zeros((sum(~mask), 2)))
        assert np.array_equal(comp["total"][~mask], expected[~mask])
        assert np.array_equal(increment(model, x, support=no_support), expected)


def test_training_uses_all_left_rows_only_supported_right_rows(fitted):
    (x, b, mask, _, ids), left, generic, structured = fitted
    assert left.report["fitting_ids"] == ids
    np.testing.assert_allclose(left.empirical_center.numpy(), x.mean(0))
    for model in (generic, structured):
        assert model.report["supplied_ids"] == ids
        assert model.report["fitting_ids"] == np.asarray(ids)[mask].tolist()
        np.testing.assert_allclose(model.biological_center.numpy(), b[mask].mean(0))
        for key, value in left.left.state_dict().items():
            assert torch.equal(value, model.left.state_dict()[key])
        assert torch.equal(model.empirical_center, left.empirical_center)
        assert torch.equal(model.empirical_scale, left.empirical_scale)
        assert model.report["left_unchanged"]
        assert np.any(model.predict_components(x, b, mask)["right"][mask] != 0)


def test_full_60_epoch_recipe_is_preserved_and_parameters_matched(fitted):
    (x, b, mask, _, _), left, generic, structured = fitted
    for model in (left, generic, structured):
        report = model.report
        assert report["epochs"] == 60
        assert report["optimizer"] == "AdamW"
        assert report["initial_lr"] == .0003
        assert report["warmup_epochs"] == 5
        assert report["schedule"] == "cosine to zero"
        assert report["increment_penalty"] == .05
        assert report["batch_size"] == 64
        assert report["gradient_clip"] == 5.
        assert not report["checkpoint_selection"]
        assert [r["epoch"] for r in report["history"]] == [1, 10, 20, 30, 40, 50, 60]
        assert report["history"][-1]["learning_rate"] == 0.
        assert not any(p.requires_grad for p in model.parameters())
        parts = model.predict_components(x, b, mask)
        assert np.max(np.abs(parts["left"])) <= BOUND
        assert np.max(np.abs(parts["right"])) <= BOUND
    assert generic.report["right_trainable_parameters"] == structured.report["right_trainable_parameters"]
    assert generic.report["biological_feature_names"] == structured.report["biological_feature_names"] == NAMES


def test_empty_right_training_support_is_exact_left_not_exact_core(fitted):
    (x, b, mask, energy, ids), left, _, _ = fitted
    model = fit_right(left, x, b, NAMES, np.zeros(len(x), bool), energy, ids,
                      mode="structured", seed=45)
    assert model.report["epochs"] == 0
    assert model.report["fitting_ids"] == []
    assert np.array_equal(increment(model, x, b, mask), increment(left, x))


def test_query_prediction_has_no_target_input_and_cannot_fit_state(fitted):
    (x, b, mask, _, _), _, generic, _ = fitted
    assert set(inspect.signature(predict).parameters) == {
        "model", "empirical", "biological", "support", "enabled", "right_enabled"}
    assert set(inspect.signature(fit_left).parameters) == {"empirical", "energies", "ids", "seed"}
    before = deepcopy(generic.state_dict())
    predict(generic, x*3, b, mask)
    for key, value in before.items():
        assert torch.equal(value, generic.state_dict()[key])
    with pytest.raises(TypeError):
        predict(generic, x, b, mask, query_gamma=np.ones(len(x)))


def test_new_or_reordered_right_label_ids_are_rejected(fitted):
    (x, b, mask, energy, ids), left, _, _ = fitted
    changed = list(ids)
    changed[0] = "future_query"
    for bad in (changed, ids[::-1], [ids[0]]*len(ids)):
        with pytest.raises(ValueError, match="full left REF"):
            fit_right(left, x, b, NAMES, mask, energy, bad, mode="generic")


def test_serialization_retains_new_semantics_and_rejects_old_artifact(fitted, tmp_path):
    (x, b, mask, _, _), _, generic, _ = fitted
    path = tmp_path/"eu_adapter.pt"
    generic.save(path)
    restored = EUResidualAdapter.load(path)
    assert np.array_equal(increment(generic, x, b, mask), increment(restored, x, b, mask))
    assert restored.report == generic.report
    old = DualBranchBiologyAdapter(x.shape[1])
    old.save(tmp_path/"old_adapter.pt")
    with pytest.raises(ValueError, match="independent support semantics"):
        EUResidualAdapter.load(tmp_path/"old_adapter.pt")
