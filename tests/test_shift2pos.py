import unittest

from bloodmoon.assets import _path_test_mask
from bloodmoon.mask import shift2pos
import bloodmoon as bm


class TestShift2Pos(unittest.TestCase):
    """Test for the `shift2pos()` function in `mask.py`."""
    def setUp(self):
        self.wfm = bm.codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

    def test_shift2pos(self):
        f = 1e-5
        shiftx_sx_in = self.wfm.bins_sky.x[0] + f
        shiftx_sx_out = self.wfm.bins_sky.x[0] - f
        shiftx_dx_in = self.wfm.bins_sky.x[-1] - f
        shiftx_dx_out = self.wfm.bins_sky.x[-1] + f

        shifty_up_in = self.wfm.bins_sky.y[0] + f
        shifty_up_out = self.wfm.bins_sky.y[0] - f
        shifty_bm_in = self.wfm.bins_sky.y[-1] - f
        shifty_bm_out = self.wfm.bins_sky.y[-1] + f

        comb_yes = [
            (shiftx_sx_in, shifty_up_in), (shiftx_sx_in, shifty_bm_in),
            (shiftx_dx_in, shifty_up_in), (shiftx_dx_in, shifty_bm_in),
        ]
        testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_yes)

        with self.assertRaises(ValueError):
            comb_no = [
                (shiftx_sx_in, shifty_up_out), (shiftx_sx_in, shifty_bm_out), (shiftx_dx_in, shifty_up_out),
                (shiftx_dx_in, shifty_bm_out), (shiftx_sx_out, shifty_up_in), (shiftx_dx_out, shifty_up_in),
                (shiftx_sx_out, shifty_bm_in), (shiftx_dx_out, shifty_bm_in), (shiftx_sx_out, shifty_up_out),
                (shiftx_sx_out, shifty_bm_out), (shiftx_dx_out, shifty_up_out), (shiftx_dx_out, shifty_bm_out),
            ]
            testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_no)


if __name__ == "__main__":
    unittest.main()


# end