from dataclasses import dataclass
import unittest

import numpy as np

import bloodmoon as bm
from bloodmoon.assets import _path_test_mask
from bloodmoon.coords import equatorial2pos
from bloodmoon.coords import pos2equatorial
from bloodmoon.types import CoordEquatorial


@dataclass(frozen=True)
class DummySDL:
    """Test SimulationDataLoader instance with camera pointing."""

    CAMZRA: float
    CAMZDEC: float
    CAMXRA: float
    CAMXDEC: float

    @property
    def pointings(self) -> dict[str, CoordEquatorial]:
        """
        Camera axis pointing information in equatorial frame.
        Angles are expressed in degrees.
        """
        return {
            "z": CoordEquatorial(ra=self.CAMZRA, dec=self.CAMZDEC),
            "x": CoordEquatorial(ra=self.CAMXRA, dec=self.CAMXDEC),
        }


class TestPosEquatorialTranform(unittest.TestCase):
    """Tests for the `equatorial2pos()` and `pos2equatorial()` function in `coords.py`."""

    def setUp(self):
        self.wfm = bm.codedmask(_path_test_mask, upscale_x=3, upscale_y=3)
        self.sdl = DummySDL(
            CAMZRA=266.4,
            CAMZDEC=-28.94,
            CAMXRA=266.4,
            CAMXDEC=61.06,
        )

    def test_eq2pos_and_pos2eq_are_inverse(self):
        """
        Tests if computed pixel indexes through `equatorial2pos()` refer to the
        same coordinates obtained with `pos2equatorial()`.
        """
        for _ in range(10_000):
            n, m = self.wfm.shape_sky
            input_row = int(np.random.uniform(0, n))
            input_col = int(np.random.uniform(0, m))

            # testing RA/Dec coordinates (output pixels = input pixels, i.e. pos2equatorial -> equatorial2pos)
            ra, dec = pos2equatorial(
                sdl=self.sdl,
                camera=self.wfm,
                y=input_row,
                x=input_col,
            )
            output_row, output_col = equatorial2pos(
                sdl=self.sdl,
                camera=self.wfm,
                ra=ra,
                dec=dec,
            )
            np.testing.assert_almost_equal(
                np.array([input_row, input_col]),
                np.array([output_row, output_col]),
                decimal=7,
            )

            # testing equatorial coordinates (output ra/dec = ra/dec, i.e. equatorial2pos -> pos2equatorial)
            output_ra, output_dec = pos2equatorial(
                sdl=self.sdl,
                camera=self.wfm,
                y=output_row,
                x=output_col,
            )
            np.testing.assert_almost_equal(
                np.array([ra, dec]),
                np.array([output_ra, output_dec]),
                decimal=7,
            )


if __name__ == "__main__":
    unittest.main()