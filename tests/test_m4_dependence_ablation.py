import unittest
import numpy as np

from opal2.m4_dependence_ablation import diagonal_scatter, select


class FrozenAblationTests(unittest.TestCase):
    def test_diagonal_preserves_coordinate_scales_without_mutating_input(self):
        generator = np.random.default_rng(3)
        matrix = generator.normal(size=(3, 9, 9))
        scatter = matrix @ matrix.transpose(0, 2, 1) + np.eye(9)
        before = scatter.copy()
        diagonal = diagonal_scatter(scatter)
        np.testing.assert_array_equal(scatter, before)
        np.testing.assert_array_equal(np.diagonal(scatter, axis1=-2, axis2=-1),
                                      np.diagonal(diagonal, axis1=-2, axis2=-1))
        self.assertTrue(np.all(np.linalg.eigvalsh(diagonal) > 0))
        self.assertEqual(np.count_nonzero(diagonal), 27)

    def test_quota_and_ties(self):
        ids = np.array(["c", "a", "b"])
        np.testing.assert_array_equal(select(ids, np.ones(3), np.zeros(3), 2, .2),
                                      [False, True, True])

    def test_two_average_does_not_include_triple(self):
        scores = np.zeros((2, 10))
        scores[:, 9] = 100
        np.testing.assert_array_equal(scores[:, 6:9].mean(1), [0, 0])


if __name__ == "__main__":
    unittest.main()
