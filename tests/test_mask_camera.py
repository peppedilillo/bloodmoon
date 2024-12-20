import unittest

import numpy as np

from iros.assets import _path_test_mask
from iros.mask import _bisect_interval
from iros.mask import decode
from iros.mask import encode
from iros.mask import fetch_camera
from iros.mask import psf


class TestWFM(unittest.TestCase):
    def setUp(self):
        self.wfm = fetch_camera(_path_test_mask, (2, 1))

    def test_shape_bulk(self):
        self.assertEqual(self.wfm.bulk.shape, self.wfm.detector_shape)

    def test_shape_detector(self):
        self.assertFalse(self.wfm.detector_shape == self.wfm.mask_shape)

    def test_sky_bins(self):
        xbins, ybins = self.wfm._bins_sky(self.wfm.upscale_f)
        assert len(np.unique(xbins)) == len(xbins)
        assert len(np.unique(ybins)) == len(ybins)
        assert len(np.unique(np.round(np.diff(xbins), 7))) == 1
        assert len(np.unique(np.round(np.diff(ybins), 7))) == 1

    def test_encode_shape(self):
        sky = np.zeros(self.wfm.sky_shape)
        self.assertEqual(encode(self.wfm, sky).shape, self.wfm.detector_shape)

    def test_encode_decode(self):
        n, m = self.wfm.sky_shape
        sky = np.zeros((n, m))
        sky[n // 2, m // 2] = 10000
        detector = encode(self.wfm, sky)
        decoded_sky, _ = decode(self.wfm, detector)
        self.assertTrue(np.any(decoded_sky))

    def test_decode_shape(self):
        detector = np.zeros(self.wfm.detector_shape)
        cc, var = decode(self.wfm, detector)
        cc_b, var_b = decode(self.wfm, detector)
        self.assertEqual(cc.shape, self.wfm.sky_shape)
        self.assertEqual(cc_b.shape, self.wfm.sky_shape)
        self.assertEqual(var.shape, self.wfm.sky_shape)
        self.assertEqual(var_b.shape, self.wfm.sky_shape)

    def test_psf_shape(self):
        self.assertEqual(psf(self.wfm).shape, self.wfm.mask_shape)

    # this may take some time
    @unittest.skip
    def test_all_sources_projects(self):
        n, m = self.wfm.sky_shape()
        for i in range(n):
            for j in range(m):
                sky = np.zeros(self.wfm.sky_shape())
                sky[i, j] = 1
                self.assertTrue(np.any(self.wfm.encode(sky)))


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

    def test_non_monotonic_array(self):
        """Test with non-monotonic array"""
        non_monotonic = np.array([1.0, 3.0, 2.0, 4.0, 5.0])
        with self.assertRaises(ValueError):
            _bisect_interval(non_monotonic, 2.0, 3.0)
