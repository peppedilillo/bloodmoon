import unittest
import numpy as np
from typing import Dict, Tuple

from iros.images import rbilinear


class TestAntibilinear(unittest.TestCase):
    def assertWeightsEqual(self, actual: Dict[Tuple[int, int], float], expected: np.ndarray):
        result = np.zeros(expected.shape)
        for (i, j), weight in actual.items():
            result[i, j] = weight
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_center_point(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.5, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_top(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., .5, 0., 0.],
            [0., 0., .5, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.5, 3., bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_bottom(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., .5, 0., 0.],
            [0., 0., .5, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.5, 4., bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_right(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0.5, 0.5, 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(3.0, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_left(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0.5, 0.5, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.0, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_grid_corner(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0.25, 0.25, 0., 0.],
            [0., 0.25, 0.25, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.0, 2.0, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_topleft(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(.1, .1, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_topright(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(4.99, 0.01, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_botright(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.]
        ])
        weights = rbilinear(4.99, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_botleft(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.]
        ])
        weights = rbilinear(0.01, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_bottom_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.]
        ])
        weights = rbilinear(1.5, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_right_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(4.99, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_top_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(2.5, 0.01, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_left_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]
        ])
        weights = rbilinear(0.01, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            rbilinear(1, 1, np.array([1]), np.array([1, 2]))

        with self.assertRaises(ValueError):
            rbilinear(1, 1, np.array([2, 1]), np.array([1, 2]))

    def test_point_outside_grid(self):
        bins_x = np.array([0, 1, 2])
        bins_y = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            rbilinear(-1, 1, bins_x, bins_y)
        with self.assertRaises(ValueError):
            rbilinear(1, 2.1, bins_x, bins_y)


if __name__ == '__main__':
    unittest.main()