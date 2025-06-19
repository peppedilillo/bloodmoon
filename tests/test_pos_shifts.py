import unittest

import numpy as np

from bloodmoon.assets import _path_test_mask
from bloodmoon.coords import pos2shift
from bloodmoon.coords import shift2pos
from bloodmoon.mask import codedmask


class TestShift2Pos(unittest.TestCase):
    """Test for the `shift2pos()` function in `mask.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=2, upscale_y=2)

    def test_near_axis(self):
        n, m = self.wfm.shape_sky
        xbin = self.wfm.bins_sky.x[1] - self.wfm.bins_sky.x[0]
        ybin = self.wfm.bins_sky.x[1] - self.wfm.bins_sky.x[0]
        self.assertEqual(shift2pos(self.wfm, 0.0, 0.0), (n // 2, m // 2))
        self.assertEqual(shift2pos(self.wfm, xbin / 4, 0.0), (n // 2, m // 2))
        self.assertEqual(shift2pos(self.wfm, xbin, 0.0), (n // 2, m // 2 + 1))
        self.assertEqual(shift2pos(self.wfm, -xbin / 4, 0.0), (n // 2, m // 2))
        self.assertEqual(shift2pos(self.wfm, -xbin, 0.0), (n // 2, m // 2 - 1))
        self.assertEqual(shift2pos(self.wfm, 0.0, ybin / 4), (n // 2, m // 2))
        self.assertEqual(shift2pos(self.wfm, 0.0, ybin), (n // 2 + 1, m // 2))
        self.assertEqual(shift2pos(self.wfm, 0.0, -ybin / 4), (n // 2, m // 2))
        self.assertEqual(shift2pos(self.wfm, 0.0, -ybin), (n // 2 - 1, m // 2))


class TestPos2Shift(unittest.TestCase):
    """Test for the `pos2shift()` function in `coords.py`."""
    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

    def test_p2s_s2p_idempotent(self):
        """
        Tests if computed shifts through `pos2shift()` refer to the
        same pixel indexes obtained with `shift2pos()`.
        """
        n, m = self.wfm.shape_sky
        for _ in range(1000):
            i, j = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((i, j), shift2pos(self.wfm, *pos2shift(self.wfm, i, j)))


if __name__ == "__main__":
    unittest.main()
