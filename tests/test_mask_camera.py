import unittest

import numpy as np

from iros.mask import CodedMaskCamera, encode, decode, psf
from iros.io import MaskDataLoader
from iros.assets import path_wfm_mask


class TestWFM(unittest.TestCase):
    def setUp(self):
        self.wfm = CodedMaskCamera(MaskDataLoader(path_wfm_mask), (2, 1))

    def test_shape_bulk(self):
        self.assertEqual(self.wfm.bulk.shape, self.wfm.detector_shape)

    def test_shape_detector(self):
        self.assertFalse(self.wfm.detector_shape == self.wfm.mask_shape)

    def test_sky_bins(self):
        bins_x, bins_y = self.wfm.bins_sky
        assert len(np.unique(bins_x)) == len(bins_x)
        assert len(np.unique(bins_y)) == len(bins_y)
        assert len(np.unique(np.round(np.diff(bins_x), 7))) == 1
        assert len(np.unique(np.round(np.diff(bins_y), 7))) == 1

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
