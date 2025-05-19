import unittest

import numpy as np

from bloodmoon.mask import _bisect_interval


class TestBisectInterval(unittest.TestCase):
    def setUp(self):
        # Create a simple monotonic array for testing
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_normal_case(self):
        """Test with an interval fully within the array bounds"""
        result = _bisect_interval(self.arr, 2.5, 3.5)
        self.assertEqual(result, (1, 3))

    def test_exact_bounds(self):
        """Test when interval bounds exactly match array elements"""
        result = _bisect_interval(self.arr, 2.0, 4.0)
        self.assertEqual(result, (1, 3))

    def test_single_point_interval(self):
        """Test when start and stop are the same"""
        result = _bisect_interval(self.arr, 3.0, 3.0)
        self.assertEqual(result, (2, 2))

    def test_boundary_case(self):
        """Test with interval at array boundaries"""
        result = _bisect_interval(self.arr, 1.0, 5.0)
        self.assertEqual(result, (0, 4))

    def test_invalid_interval_below(self):
        """Test with interval starting below array bounds"""
        with self.assertRaises(ValueError):
            _bisect_interval(self.arr, 0.5, 3.0)

    def test_invalid_interval_above(self):
        """Test with interval ending above array bounds"""
        with self.assertRaises(ValueError):
            _bisect_interval(self.arr, 2.0, 5.5)
