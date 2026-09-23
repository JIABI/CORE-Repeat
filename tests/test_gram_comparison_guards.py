import numpy as np
import pytest
import torch

from opal2.gram_geometry import profiles_to_gram, gram_gains
from opal2.gram_evaluation import score_grams, fit_score_scale, paired_score_comparison


def test_energy_comparison_rejects_unscaled_gram():
    y = torch.tensor(np.random.default_rng(7).normal(size=(6, 4, 20)))
    g = profiles_to_gram(y).numpy()
    with pytest.raises(ValueError, match="G00=1"):
        score_grams(np.repeat((g*2)[None], 4, 0), g, [str(k) for k in range(6)],
            train_actual_gains=gram_gains(torch.tensor(g)).numpy(),
            score_scale=fit_score_scale(g), n_bootstrap=2, n_random=2)


@pytest.mark.parametrize("key", ["actual", "actual_grams", "score_scale"])
def test_paired_comparison_rejects_mismatched_target_or_metric(tmp_path, key):
    common = dict(ids=np.array(["a", "b"]), actual=np.zeros((2, 3)),
                  actual_grams=np.zeros((2, 4, 4)), score_scale=np.ones(9))
    np.savez(tmp_path/"a.npz", **common)
    changed = dict(common)
    changed[key] = common[key] + 1
    np.savez(tmp_path/"b.npz", **changed)
    with pytest.raises(ValueError, match="mismatched endpoint or score scale"):
        paired_score_comparison(tmp_path/"a.npz", tmp_path/"b.npz")
