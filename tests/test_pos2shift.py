import unittest
import numpy as np

from bloodmoon.assets import _path_test_mask
from bloodmoon.coords import pos2shift
from bloodmoon.mask import shift2pos
import bloodmoon as bm


wfm = bm.codedmask(_path_test_mask, upscale_x=3, upscale_y=3)

class TestPos2Shift(unittest.TestCase):
    """Test for the `pos2shift()` function in `coords.py`."""
    def test_pos2shift(self):  
        n, m = wfm.sky_shape
        for _ in range(10000):
            y, x = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((y, x), shift2pos(wfm, *pos2shift(wfm, x, y)))
        
        in_pos = [
            ((m - 1, n - 1), (-1, -1)),
            ((3 * m // 4, n - 1), (-m // 4, -1)),
            ((m - 1, 3 * n // 4), (-1, -n // 4)),
            ((0, 0), (-m, -n)),
        ]
        for pos in in_pos: self.assertEqual(pos2shift(wfm, *pos[0]), pos2shift(wfm, *pos[1]))

        with self.assertRaises(IndexError):
            out_pos = [
                (m, n), (-m, n), (m, -n), (-m - 1, -n - 1),
            ]
            for pos in out_pos: pos2shift(wfm, *pos)


if __name__ == "__main__":
    unittest.main()


# end