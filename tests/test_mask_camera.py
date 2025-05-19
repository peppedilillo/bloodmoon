import unittest

import numpy as np

from mbloodmoon.mask import codedmask
from mbloodmoon.mask import decode
from mbloodmoon.mask import encode
from mbloodmoon.mask import psf
from mbloodmoon.mask import variance

_path_test_mask = "/home/edoardo/Datadisk/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/wfm_mask.fits"

class TestWFM(unittest.TestCase):
    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=2, upscale_y=1)

    def test_shape_bulk(self):
        self.assertEqual(self.wfm.bulk.shape, self.wfm.shape_detector)

    def test_shape_detector(self):
        self.assertFalse(self.wfm.shape_detector == self.wfm.shape_mask)

    def test_sky_bins(self):
        xbins, ybins = self.wfm._bins_sky(self.wfm.upscale_f)
        assert len(np.unique(xbins)) == len(xbins)
        assert len(np.unique(ybins)) == len(ybins)
        assert len(np.unique(np.round(np.diff(xbins), 7))) == 1
        assert len(np.unique(np.round(np.diff(ybins), 7))) == 1

    def test_encode_shape(self):
        sky = np.zeros(self.wfm.shape_sky)
        self.assertEqual(encode(self.wfm, sky).shape, self.wfm.shape_detector)

    def test_encode_decode(self):
        n, m = self.wfm.shape_sky
        sky = np.zeros((n, m))
        sky[n // 2, m // 2] = 10000
        detector = encode(self.wfm, sky)
        decoded_sky = decode(self.wfm, detector)
        self.assertTrue(np.any(decoded_sky))

    def test_decode_shape(self):
        detector = np.zeros(self.wfm.shape_detector)
        cc = decode(self.wfm, detector)
        var = variance(self.wfm, detector)
        self.assertEqual(cc.shape, self.wfm.shape_sky)
        self.assertEqual(var.shape, self.wfm.shape_sky)

    def test_psf_shape(self):
        self.assertEqual(psf(self.wfm).shape, self.wfm.shape_mask)

    # this may take some time
    @unittest.skip
    def test_all_sources_projects(self):
        n, m = self.wfm.shape_sky
        for i in range(n):
            for j in range(m):
                sky = np.zeros(self.wfm.shape_sky)
                sky[i, j] = 1
                self.assertTrue(np.any(self.wfm.encode(sky)))
