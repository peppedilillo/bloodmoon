import unittest

import numpy as np

from bloodmoon.images import _erosion


class TestErosion(unittest.TestCase):
    def setUp(self):
        self.assertArrayAlmostEqual = lambda x, y: np.testing.assert_array_almost_equal(x, y, decimal=2)

    def test_basic_erosion_1(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 0, 1],
            ]
        )
        step = 0.5
        cut = 0.25
        expected = np.array(
            [
                [0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.5],
                [0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.5],
                [0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.5],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_wide_pattern(self):
        arr = np.array(
            [
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            ]
        )
        step = 1.0
        cut = 4.5
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_small_step_small_cut(self):
        arr = np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )
        step = 0.5
        cut = 0.45
        expected = np.array(
            [
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_double_ones(self):
        arr = np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0],
            ]
        )
        step = 1.0
        cut = 0.5
        expected = np.array(
            [
                [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
                [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
                [0.0, 0.0, 0.75, 0.75, 0.0, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_triple_ones_with_large_cut(self):
        arr = np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ]
        )
        step = 0.5
        cut = 1.0
        expected = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_complex_pattern(self):
        arr = np.array(
            [
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0],
            ]
        )
        step = 0.5
        cut = 0.49
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
                [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
                [0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.0, 0.0, 0.51, 1.0, 0.51, 0.0, 0.02, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)

    def test_large_pattern_with_large_cut(self):
        arr = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            ]
        )
        step = 0.5
        cut = 1.2
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 1.0, 0.8, 0.0, 0.0],
            ]
        )
        result = _erosion(arr, step, cut)
        self.assertArrayAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
