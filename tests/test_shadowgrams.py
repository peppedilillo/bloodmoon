import unittest

import numpy as np

from iros.assets import _path_test_mask
from iros.images import _shift
from iros.mask import CodedMaskCamera, shadowgram
from iros.mask import encode
from iros.mask import fetch_camera


class TestArrayShift(unittest.TestCase):
    def setUp(self):
        # Common test arrays
        self.arr_2x2 = np.array([[1, 2], [3, 4]])
        self.arr_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.arr_2x3 = np.array([[1, 2, 3], [4, 5, 6]])

    def test_basic_shifts(self):
        """Test basic positive shifts"""
        expected = np.array([[0, 0], [1, 2]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, (1, 0)), expected)

        expected = np.array([[0, 1], [0, 3]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, (0, 1)), expected)

    def test_negative_shifts(self):
        """Test negative shifts"""
        expected = np.array([[3, 4], [0, 0]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, (-1, 0)), expected)

        expected = np.array([[2, 0], [4, 0]])
        np.testing.assert_array_equal(_shift(self.arr_2x2, (0, -1)), expected)

    def test_both_dimensions(self):
        """Test shifting in both dimensions simultaneously"""
        expected = np.array([[0, 0, 0], [0, 1, 2], [0, 4, 5]])
        np.testing.assert_array_equal(_shift(self.arr_3x3, (1, 1)), expected)

        expected = np.array([[5, 6, 0], [8, 9, 0], [0, 0, 0]])
        np.testing.assert_array_equal(_shift(self.arr_3x3, (-1, -1)), expected)

    def test_large_shifts(self):
        """Test shifts larger than array dimensions"""
        expected = np.zeros((2, 2))
        np.testing.assert_array_equal(_shift(self.arr_2x2, (20, 0)), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, (0, 20)), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, (-20, 0)), expected)
        np.testing.assert_array_equal(_shift(self.arr_2x2, (0, -20)), expected)

    def test_zero_shift(self):
        """Test zero shift returns original array"""
        np.testing.assert_array_equal(_shift(self.arr_2x2, (0, 0)), self.arr_2x2)

    def test_different_shapes(self):
        """Test with non-square arrays"""
        expected = np.array([[0, 1, 2], [0, 4, 5]])
        np.testing.assert_array_equal(_shift(self.arr_2x3, (0, 1)), expected)

    def test_edge_cases(self):
        """Test edge cases like empty and single-element arrays"""
        # Empty array
        empty_arr = np.array([[]])
        np.testing.assert_array_equal(_shift(empty_arr, (1, 1)), empty_arr)

        # Single element array
        single_arr = np.array([[1]])
        expected = np.array([[0]])
        np.testing.assert_array_equal(_shift(single_arr, (1, 0)), expected)

    def test_different_dtypes(self):
        """Test with different data types"""
        # Float array
        float_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
        expected = np.array([[0.0, 0.0], [1.0, 2.0]])
        np.testing.assert_array_equal(_shift(float_arr, (1, 0)), expected)

        # Boolean array
        bool_arr = np.array([[True, False], [False, True]], dtype=bool)
        expected = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(_shift(bool_arr, (-1, -1)), expected)


def benchmark_shadowgram(camera: CodedMaskCamera, source_position: tuple[int, int]):
    i, j = source_position
    sky_model = np.zeros(camera.sky_shape)
    sky_model[i, j] = 1
    return np.round(encode(camera, sky_model)).astype(int)


class TestShadowgram(unittest.TestCase):
    def setUp(self):
        """Set up test data including precomputed benchmark shadowgrams."""
        self.camera = fetch_camera(_path_test_mask, (2, 1))
        n, m = self.camera.sky_shape
        self.test_positions = [
            (0, 0),  # Top-left corner
            (0, m - 1),  # Top-right corner
            (n - 1, 0),  # Bottom-left corner
            (n - 1, m - 1),  # Bottom-right corner
            (n // 2, m // 2),  # Center
            (n // 4, m // 4),  # Top-left quarter
            (n // 4, 3 * m // 4),  # Top-right quarter
            (3 * n // 4, m // 4),  # Bottom-left quarter
            (3 * n // 4, 3 * m // 4),  # Bottom-right quarter
        ]

        # Precompute benchmark shadowgrams
        self.benchmark_shadows = {pos: benchmark_shadowgram(self.camera, pos) for pos in self.test_positions}

    def test_shapes_match(self):
        """Test if both implementations produce the same shape outputs."""
        for pos in self.test_positions:
            with self.subTest(position=pos):
                shadow = shadowgram(self.camera, pos)
                benchmark = self.benchmark_shadows[pos]
                self.assertEqual(
                    shadow.shape,
                    benchmark.shape,
                    f"Shape mismatch at position {pos}: {shadow.shape} vs {benchmark.shape}",
                )

    def test_arrays_equal(self):
        """Test if both implementations produce identical outputs."""
        for pos in self.test_positions:
            with self.subTest(position=pos):
                shadow = shadowgram(self.camera, pos)
                benchmark = self.benchmark_shadows[pos]
                np.testing.assert_array_equal(shadow, benchmark, err_msg=f"Arrays not equal at position {pos}")

    def test_array_sums_match(self):
        """Test if the total counts in the shadowgrams match."""
        for pos in self.test_positions:
            with self.subTest(position=pos):
                shadow = shadowgram(self.camera, pos)
                benchmark = self.benchmark_shadows[pos]
                np.testing.assert_almost_equal(np.sum(shadow), np.sum(benchmark), decimal=8, err_msg=f"Total counts mismatch at position {pos}")
