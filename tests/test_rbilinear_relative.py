from typing import Dict, Tuple
import unittest

import numpy as np

from bloodmoon.images import _rbilinear, _rbilinear_relative


class TestAntibilinear(unittest.TestCase):
    def assertWeightsEqual(self, actual: Dict[Tuple[int, int], float], expected: np.ndarray):
        result = np.zeros(expected.shape)
        for (i, j), weight in actual.items():
            result[i, j] = weight
        try:
            np.testing.assert_array_almost_equal(result, expected, decimal=2)
        except AssertionError:
            print(f"Weights do not match expected matrix:\n\n{result}")
            raise

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
        cx, cy = 2.5, 3.5
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], 1.0)
        for key in [*components.keys()][1:]:
            self.assertAlmostEqual(components[key], 0.0)
        self.assertEqual(pivot, (3, 2))

    def test_offaxis_point1(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
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
        cx, cy = 2.25, 3.25
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], .5625)
        self.assertAlmostEqual(components[(-1, 0)], .1875)
        self.assertAlmostEqual(components[(0, -1)], .1875)
        self.assertAlmostEqual(components[(-1, -1)], .0625)
        self.assertEqual(pivot, (3, 2))

    def test_offaxis_point2(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1875, 0.0625, 0.0],
                [0.0, 0.0, 0.5625, 0.1875, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        cx, cy = 2.75, 3.25
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], .5625)
        self.assertAlmostEqual(components[(0, +1)], .1875)
        self.assertAlmostEqual(components[(-1, 0)], .1875)
        self.assertAlmostEqual(components[(-1, +1)], .0625)
        self.assertEqual(pivot, (3, 2))

    def test_offaxis_point3(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
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
        cx, cy = 2.75, 3.75
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], .5625)
        self.assertAlmostEqual(components[(0, +1)], .1875)
        self.assertAlmostEqual(components[(+1, 0)], .1875)
        self.assertAlmostEqual(components[(+1, +1)], .0625)
        self.assertEqual(pivot, (3, 2))


    def test_offaxis_point4(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1875, 0.5625, 0.0, 0.0],
                [0.0, 0.0625, 0.1875, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        cx, cy = 2.25, 3.75
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], .5625)
        self.assertAlmostEqual(components[(0, -1)], .1875)
        self.assertAlmostEqual(components[(+1, 0)], .1875)
        self.assertAlmostEqual(components[(+1, -1)], .0625)
        self.assertEqual(pivot, (3, 2))


    def test_offaxis_point5(self):
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
        cx, cy = 2.5, 3.
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertAlmostEqual(components[(0, 0)], 0.5)
        self.assertAlmostEqual(components[(-1, 0)], 0.5)
        self.assertEqual(pivot, (3, 2))


    def test_offaxis_point6(self):
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
        cx, cy = 2.5, 4.
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        self.assertEqual(pivot, (4, 2))
        self.assertAlmostEqual(components[(0, 0)], 0.5)
        self.assertAlmostEqual(components[(-1, 0)], 0.5)

    def test_offaxis_point6(self):
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
        cx, cy = 3.0, 3.5
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        # the pivot has always the largest numbers
        self.assertEqual(pivot, (3, 3))
        self.assertAlmostEqual(components[(0, 0)], 0.5)
        self.assertAlmostEqual(components[(0, -1)], 0.5)

    def test_offaxis_point6(self):
        bins_x = np.linspace(0, 5, 6)
        bins_y = np.linspace(0, 7, 8)
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.25, 0.0],
                [0.0, 0.0, 0.25, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        cx, cy = 3.0, 4.0
        components, pivot = _rbilinear_relative(cx, cy, bins_x, bins_y)
        weights = _rbilinear(cx, cy, bins_x, bins_y)
        self.assertWeightsEqual(weights, expected)
        # the pivot has always the largest numbers
        self.assertEqual(pivot, (4, 3))
        self.assertAlmostEqual(components[(0, 0)], 0.25)
        self.assertAlmostEqual(components[(0, -1)], 0.25)
        self.assertAlmostEqual(components[(-1, 0)], 0.25)
        self.assertAlmostEqual(components[(-1, -1)], 0.25)

if __name__ == "__main__":
    unittest.main()
