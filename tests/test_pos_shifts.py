import unittest

import numpy as np

from bloodmoon.assets import _path_test_mask
from bloodmoon.coords import pos2shift
from bloodmoon.coords import shift2pos
from bloodmoon.mask import codedmask


class TestShift2Pos(unittest.TestCase):
    """Test for the `shift2pos()` function in `mask.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

    def test_binning_boundaries(self):
        """Test for allowed and not allowed shifts wrt the binning."""
        # shifts with "_in" suffix refer to shifts inside binning
        # shifts with "_out" suffix refer to shifts outside binning
        f = 1e-5
        shiftx_sx_in = self.wfm.bins_sky.x[0] + f
        shiftx_sx_out = self.wfm.bins_sky.x[0] - f
        shiftx_dx_in = self.wfm.bins_sky.x[-1] - f
        shiftx_dx_out = self.wfm.bins_sky.x[-1] + f

        shifty_up_in = self.wfm.bins_sky.y[0] + f
        shifty_up_out = self.wfm.bins_sky.y[0] - f
        shifty_bm_in = self.wfm.bins_sky.y[-1] - f
        shifty_bm_out = self.wfm.bins_sky.y[-1] + f

        # test for the allowed shifts at the edges of the binning
        comb_yes = [
            (shiftx_sx_in, shifty_up_in),
            (shiftx_sx_in, shifty_bm_in),
            (shiftx_dx_in, shifty_up_in),
            (shiftx_dx_in, shifty_bm_in),
        ]
        testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_yes)

        with self.assertRaises(ValueError):
            # test for the shifts outside the binning
            comb_no = [
                (shiftx_sx_in, shifty_up_out),
                (shiftx_sx_in, shifty_bm_out),
                (shiftx_dx_in, shifty_up_out),
                (shiftx_dx_in, shifty_bm_out),
                (shiftx_sx_out, shifty_up_in),
                (shiftx_dx_out, shifty_up_in),
                (shiftx_sx_out, shifty_bm_in),
                (shiftx_dx_out, shifty_bm_in),
                (shiftx_sx_out, shifty_up_out),
                (shiftx_sx_out, shifty_bm_out),
                (shiftx_dx_out, shifty_up_out),
                (shiftx_dx_out, shifty_bm_out),
            ]
            testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_no)


class TestPos2Shift(unittest.TestCase):
    """Test for the `pos2shift()` function in `coords.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

    def test_p2s_and_s2p_are_inverse(self):
        """
        Tests if computed shifts through `pos2shift()` refer to the
        same pixel indexes obtained with `shift2pos()`.
        """
        n, m = self.wfm.shape_sky
        for _ in range(10000):
            y, x = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((y, x), shift2pos(self.wfm, *pos2shift(self.wfm, x, y)))

    def test_positive_and_negative_idxs(self):
        """Tests if positive and negative idxs refer to the same shifts."""
        n, m = self.wfm.shape_sky
        in_pos = [
            ((m, n), (-1, -1)),
            ((3 * m // 4, n), (-m // 4 - 1, -1)),
            ((m, 3 * n // 4), (-1, -n // 4 - 1)),
            ((0, 0), (-m - 1, -n - 1)),
        ]
        # `in_pos` contains array positions expressed with positive
        #  idxs and respective negative idxs
        for pos in in_pos:
            self.assertEqual(pos2shift(self.wfm, *pos[0]), pos2shift(self.wfm, *pos[1]))

    def test_idxs_boundaries(self):
        """Test for out-of-bound elements."""
        n, m = self.wfm.shape_sky
        with self.assertRaises(IndexError):
            out_pos = [
                (m + 1, n + 1),
                (-m - 1, n + 1),
                (m + 1, -n - 1),
                (-m - 2, -n - 2),
            ]
            for pos in out_pos:
                pos2shift(self.wfm, *pos)


if __name__ == "__main__":
    unittest.main()
