import numpy as np

from scripts.summarize_r3_crossdose_response_20260920 import (
    group_mean_scores, bootstrap_group_risks, contrast, compact_scores,
)


def test_group_equal_not_row_equal_and_paired_interaction():
    # Three rows in group A and one in B. Every metric repeats the same values.
    scores = np.zeros((2, 4, 2, 3))
    scores[:, :, 0] = np.array([2., 2., 2., 10.])[None, :, None]
    scores[0, :, 1] = .9*scores[0, :, 0]
    scores[1, :, 1] = .7*scores[1, :, 0]
    labels, grouped, counts = group_mean_scores(scores, np.array(['A','A','A','B']), np.ones(4, bool))
    np.testing.assert_array_equal(counts, [3, 1])
    assert grouped.mean(0)[0, 0, 0] == 6.
    assert scores.mean(1)[0, 0, 0] == 4.
    draws = bootstrap_group_risks(grouped, replicates=80, seed=7)
    out = contrast(grouped.mean(0), draws, 1, 0)['profile_mse']
    np.testing.assert_allclose(out['SAME']['relative_improvement']['estimate'], .1)
    np.testing.assert_allclose(out['CROSS']['relative_improvement']['estimate'], .3)
    np.testing.assert_allclose(out['CROSS_minus_SAME_relative_improvement']['ci95'], [.2,.2])


def test_support_mask_applied_before_group_average():
    s = np.ones((2, 3, 1, 3))
    s[:, 1] = 1000.
    labels, grouped, count = group_mean_scores(s, np.array(['A','A','B']), np.array([True,False,True]))
    np.testing.assert_array_equal(count, [1,1])
    np.testing.assert_array_equal(grouped, np.ones((2,2,1,3)))


def test_compact_random_means_average_errors_not_predictions():
    from scripts.run_r3_crossdose_response_20260920 import arm_names
    names = arm_names()
    scores = np.zeros((2, 2, len(names), 3))
    for rep in range(1,21):
        scores[:, :, names.index(f'TARGET_R{rep:02d}_TRANSPORT_CAL')] = float(rep**2)
    compact, cn = compact_scores(scores, names)
    val = compact[:, :, cn.index('TARGET_RANDOM_MEAN_TRANSPORT_CAL')]
    np.testing.assert_array_equal(val, np.full((2,2,3), np.mean(np.arange(1,21)**2)))
    assert val[0,0,0] != np.mean(np.arange(1,21))**2
