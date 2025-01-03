import unittest

import numpy as np

from bloodmoon.images import _unframe


class TestUnframe(unittest.TestCase):
    def test_basic(self):
        arr = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        expected = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 3.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_basic2(self):
        arr = np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 0.0],
                [0.0, 4.0, 5.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_basic3(self):
        arr = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_no_empty_frame(self):
        arr = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), arr)

    def test_multiple_empty_frames(self):
        arr = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_rectangular(self):
        arr = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_different_value(self):
        arr = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        expected = np.array(
            [
                [5.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 5.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr, value=5.0), expected)

    def test_float_precision(self):
        arr = np.array(
            [
                [1e-10, 1.0, 1e-10],
                [1.0, 0.0, 1.0],
                [1e-10, 1.0, 1e-10],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        result = _unframe(arr)
        np.testing.assert_allclose(result, expected)

    def test_float_precision2(self):
        arr = np.array(
            [
                [1e-10, 1.0, 1e-10],
                [1.0, 1.0, 1.0],
                [1e-10, 1.0, 1e-10],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        result = _unframe(arr)
        np.testing.assert_allclose(result, expected)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            _unframe(np.array([1.0, 2.0, 3.0]))

        with self.assertRaises(ValueError):
            _unframe(np.zeros((2, 2, 2)))

    def test_minimal(self):
        arr = np.array(
            [
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(_unframe(arr), expected)

    def test_all_zeros(self):
        arr = np.zeros((3, 3))
        np.testing.assert_array_equal(_unframe(arr), arr)


if __name__ == "__main__":
    unittest.main()
