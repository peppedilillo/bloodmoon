from typing import Dict, Tuple
import unittest

import numpy as np

from bloodmoon.images import _rbilinear
from bloodmoon.images import _rbilinear_relative


class TestAntibilinear(unittest.TestCase):
    def assertWeightsEqual(self, actual: Dict[Tuple[int, int], float], expected: np.ndarray):
        result = np.zeros(expected.shape)
        for (i, j), weight in actual.items():
            result[i, j] = weight
        try:
            np.testing.assert_array_almost_equal(result, expected, decimal=2)
        except AssertionError:
            raise AssertionError(f"Weights do not match expected matrix:\n\n{result}")

    def test_invalid_grid(self):
        with self.assertRaises(ValueError):
            _rbilinear(1, 1, np.array([1]), np.array([1, 2]))

        with self.assertRaises(ValueError):
            _rbilinear(1, 1, np.array([2, 1]), np.array([1, 2]))

    def test_point_outside_grid(self):
        bins_x = np.array([0, 1, 2])
        bins_y = np.array([0, 1, 2])
        with self.assertRaises(ValueError):
            _rbilinear(-1, 1, bins_x, bins_y)
        with self.assertRaises(ValueError):
            _rbilinear(1, 2.1, bins_x, bins_y)

    def test_center_point(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.5, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_top(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.5, 3.0, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_bottom(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.5, 4.0, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_right(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(3.0, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_off_center_point_left(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.0, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_grid_corner(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.0, 2.0, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_topleft(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(0.1, 0.1, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_topright(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(4.99, 0.01, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_botright(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        weights = _rbilinear(4.99, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_corner_botleft(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(0.01, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_bottom_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(1.5, 6.99, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_right_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(4.99, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_top_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.5, 0.01, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_left_side(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(0.01, 3.5, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_small_bin_steps(self):
        bins_x = np.linspace(0, 2.5, 6)
        bins_y = np.linspace(0, 3.5, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0, 0.0],
                [0.0, 0.25, 0.25, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(1.0, 1.0, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_midbin1(self):
        bins_x = np.linspace(0, 5.0, 6)
        bins_y = np.linspace(0, 7.0, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0625, 0.1875, 0.0, 0.0],
                [0.0, 0.1875, 0.5625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.25, 3.25, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_midbin2(self):
        bins_x = np.linspace(0, 5.0, 6)
        bins_y = np.linspace(0, 7.0, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5625, 0.1875, 0.0],
                [0.0, 0.0, 0.1875, 0.0625, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(2.75, 3.75, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)

    def test_midbin3(self):
        bins_x = np.linspace(0, 5.0, 6)
        bins_y = np.linspace(0, 7.0, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.2601, 0.2499, 0.0],
                [0.0, 0.0, 0.2499, 0.2401, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        weights = _rbilinear(3 - 0.01, 4 - 0.01, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)


if __name__ == "__main__":
    unittest.main()
