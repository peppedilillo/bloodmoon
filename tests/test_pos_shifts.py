import unittest

from random import choice
import numpy as np

from bloodmoon.assets import _path_test_mask
from bloodmoon.coords import pos2shift
from bloodmoon.coords import shift2pos
from bloodmoon.mask import codedmask


class TestShift2Pos(unittest.TestCase):
    """Test for the `shift2pos()` method in `coords.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=2, upscale_y=2)

    def test_binning_boundaries(self):
        """Test for allowed and not allowed shifts wrt the binning."""
        # Sky bins edges for upscaling x=2, y=2:
        #   - x-axis: (-209, 209)
        #   - x-axis: (-206.53333333, 206.53333333)

        # test for the allowed shifts at the edges of the binning
        comb_yes = [
            (-208.99999, -206.53332333),
            (-208.99999, 206.53332333),
            (208.99999, -206.53332333),
            (208.99999, 206.53332333),
        ]
        testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_yes)

        with self.assertRaises(ValueError):
            # test for the shifts outside the binning
            comb_no = [
                (-208.99999, -206.533343333),
                (-208.99999, 206.53334333),
                (208.99999, -206.533343333),
                (208.99999, 206.53334333),
                (-209.00001, -206.53332333),
                (209.00001, -206.53332333),
                (-209.00001, 206.53332333),
                (209.00001, 206.53332333),
                (-209.00001, -206.533343333),
                (-209.00001, 206.53334333),
                (209.00001, -206.533343333),
                (209.00001, 206.53334333),
            ]
            testing = tuple(shift2pos(self.wfm, *shifts) for shifts in comb_no)



class TestPos2Shift(unittest.TestCase):
    """Test for the `pos2shift()` method in `coords.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

    def test_positive_and_negative_idxs(self):
        """Tests if positive and negative idxs refer to the same shifts."""
        in_pos = [
            ((5015, 3097), (-1, -1)),
            ((3761, 3097), (-1255, -1)),
            ((5015, 2322), (-1, -776)),
            ((0, 0), (-5016, -3098)),
        ]
        # `in_pos` contains array positions expressed with positive
        #  idxs and respective negative idxs
        for pos in in_pos:
            self.assertEqual(pos2shift(self.wfm, *pos[0]), pos2shift(self.wfm, *pos[1]))

    def test_idxs_boundaries(self):
        """Test for out-of-bound elements."""
        # shape sky: (5015, 3097)
        out_pos = [
                (5016, 3098),
                (-5016, 3098),
                (5016, -3098),
                (-5017, -3099),
            ]
        
        with self.assertRaises(IndexError):
            for pos in out_pos:
                pos2shift(self.wfm, *pos)



class TestPosShiftTransform(unittest.TestCase):
    """Test for the `pos2shift()` and `shift2pos()` methods in `coords.py`."""

    def setUp(self):
        self.wfm = codedmask(_path_test_mask, upscale_x=2, upscale_y=2)

    def test_p2s_and_s2p_are_inverse(self):
        """
        Tests if computed shifts through `pos2shift()` refer to the
        same pixel indexes obtained with `shift2pos()`.
        """
        n, m = self.wfm.shape_sky
        for _ in range(10_000):
            y, x = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((y, x), shift2pos(self.wfm, *pos2shift(self.wfm, x, y)))

    def test_s2p_and_p2s_are_inverse(self):
        """
        Tests if computed pixel indexes through `shift2pos()` refer
        to the same binning shifts obtained with `pos2shift()`.
        """
        binsx, binsy = self.wfm.bins_sky
        for _ in range(10_000):
            shiftx = choice(binsx)
            shifty = choice(binsy)
            y, x = shift2pos(self.wfm, shiftx, shifty)
            self.assertEqual(
                (shiftx, shifty),
                pos2shift(self.wfm, x, y),
            )

    def test_specific_transforms(self):
        """Tests specific shifts to px indexes transformations."""
        binsx, binsy = self.wfm.bins_sky
        stepx, stepy = abs(binsx[0] - binsx[1]), abs(binsy[0] - binsy[1])

        shiftx, shifty = 0.0625, 0.1
        x, y = 1672, 1033

        # test if shifts inside half bin compares with (x, y) pixel indexes
        np.testing.assert_almost_equal(
            np.array(shift2pos(self.wfm, shiftx, shifty)),
            np.array((y, x)),
        )
        np.testing.assert_almost_equal(
            np.array(shift2pos(self.wfm, shiftx + 0.25 * stepx, shifty + 0.25 * stepy)),
            np.array((y, x)),
        )
        np.testing.assert_almost_equal(
            np.array(shift2pos(self.wfm, shiftx - 0.25 * stepx, shifty - 0.25 * stepy)),
            np.array((y, x)),
        )

        # test if shifts over half bin compares with next pixel indexes
        np.testing.assert_almost_equal(
            np.array(shift2pos(self.wfm, shiftx + 0.75 * stepx, shifty + 0.75 * stepy)),
            np.array((y + 1, x + 1)),
        )
        np.testing.assert_almost_equal(
            np.array(shift2pos(self.wfm, shiftx - 0.75 * stepx, shifty - 0.75 * stepy)),
            np.array((y - 1, x - 1)),
        )




if __name__ == "__main__":
    unittest.main()
