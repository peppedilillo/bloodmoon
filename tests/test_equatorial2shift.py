import unittest
from dataclasses import dataclass

import numpy as np

from mbloodmoon.types import CoordEquatorial
from mbloodmoon.coords import shift2equatorial, equatorial2shift
from mbloodmoon.coords import _rotation_matrices
import mbloodmoon as bm

_path_test_mask = "/home/edoardo/Datadisk/Edos_Magnificent_Manor/PhD_AASS/Coding/IROS_Data/Simulations/wfm_mask.fits"


@dataclass(frozen=True)
class MiniSDL:
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


class TestEquatorial2Shift(unittest.TestCase):
    """Test for the `equatorial2shift()` function in `coords.py`."""

    def setUp(self):
        self.wfm = bm.codedmask(_path_test_mask, upscale_x=3, upscale_y=3)
        self.sdl = MiniSDL(
            CAMZRA=266.4,
            CAMZDEC=-28.94,
            CAMXRA=266.4,
            CAMXDEC=61.06,
        )

    def test_eq2s_and_s2eq_are_inverse(self):
        """
        Tests if computed shifts through `equatorial2shift()` refer to the
        same sky-shifts coordinates obtained with `shift2equatorial()`.
        """
        for _ in range(100_000):
            sky_bins = self.wfm.bins_sky
            input_shiftx = np.random.uniform(sky_bins.x[0], sky_bins.x[-1])
            input_shifty = np.random.uniform(sky_bins.y[0], sky_bins.y[-1])

            # testing shifts coordinates (output shifts = input shifts, i.e. shift2equatorial -> equatorial2shift)
            ra, dec = shift2equatorial(
                sdl=self.sdl,
                camera=self.wfm,
                shift_x=input_shiftx,
                shift_y=input_shifty,
            )
            output_shiftx, output_shifty = equatorial2shift(
                sdl=self.sdl,
                camera=self.wfm,
                ra=ra,
                dec=dec,
            )
            np.testing.assert_almost_equal(
                np.array([input_shiftx, input_shifty]),
                np.array([output_shiftx, output_shifty]),
                decimal=7,
            )

            # testing equatorial coordinates (output ra/dec = ra/dec, i.e. equatorial2shift -> shift2equatorial)
            output_ra, output_dec = shift2equatorial(
                sdl=self.sdl,
                camera=self.wfm,
                shift_x=output_shiftx,
                shift_y=output_shifty,
            )
            np.testing.assert_almost_equal(
                np.array([ra, dec]),
                np.array([output_ra, output_dec]),
                decimal=7,
            )

    def test_rotation_matrices(self):
        """Tests if rotation matrices are orthogonal."""
        sky2cam, cam2sky = _rotation_matrices(
            pointing_radec_z=self.sdl.pointings["z"],
            pointing_radec_x=self.sdl.pointings["x"],
        )
        np.testing.assert_almost_equal(
            np.matmul(sky2cam, cam2sky),
            np.eye(*sky2cam.shape),
        )


if __name__ == "__main__":
    unittest.main()