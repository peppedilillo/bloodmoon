import unittest
import numpy as np

from bloodmoon.assets import _path_test_mask
from bloodmoon.mask import shift2pos
import bloodmoon as bm


wfm = bm.codedmask(_path_test_mask, upscale_x=3, upscale_y=3)


def pos2shift(
    x: int,
    y: int,
) -> tuple[float, float]:
    """Convert px indexes to sky-coord shifts."""

    def valid_idxs(pos: tuple[int, int]) -> bool:
        """Check sky indexes validity."""
        n, m = wfm.sky_shape
        x, y = pos
        if (y >= n) or (x >= m) or (y < -n) or (x < -m):
            raise IndexError(f"Indexes ({y}, {x}) are out of bound for sky shape {wfm.sky_shape}.")

    # bins resemble sky shape
    valid_idxs((x, y))
    binsx = wfm.bins_sky.x[:-1]; binsy = wfm.bins_sky.y[:-1]
    dbinx = binsx[1] - binsx[0]; dbiny = binsy[1] - binsy[0]
    return binsx[x] + dbinx/2, binsy[y] + dbiny/2


class TestPos2Equatorial(unittest.TestCase):
    """Test for the `pos2equatorial()` function in `coords.py`."""
    def test_pos2equatorial(self):
        # - here we will test the `pos2shift()` nested method,
        #   since `pos2equatorial()` is computed with `shift2equatorial()`    
        n, m = wfm.sky_shape
        for _ in range(10000):
            y, x = (np.random.randint(0, n), np.random.randint(0, m))
            self.assertEqual((y, x), shift2pos(wfm, *pos2shift(x, y)))
        
        in_pos = [
            ((m - 1, n - 1), (-1, -1)),
            ((3 * m // 4, n - 1), (-m // 4, -1)),
            ((m - 1, 3 * n // 4), (-1, -n // 4)),
            ((0, 0), (-m, -n)),
        ]
        for pos in in_pos: self.assertEqual(pos2shift(*pos[0]), pos2shift(*pos[1]))

        with self.assertRaises(IndexError):
            out_pos = [
                (m, n), (-m, n), (m, -n), (-m - 1, -n - 1),
            ]
            for pos in out_pos: pos2shift(*pos)


if __name__ == "__main__":
    unittest.main()


# end