"""
Custom data types and containers for the WFM analysis pipeline.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class CoordEquatorial(NamedTuple):
    """Equatorial coordinates.

    Args:
        ra: right ascension
        dec: declination
    """

    ra: float
    dec: float


class CoordHorizontal(NamedTuple):
    """
    Horizontal coordinates.

    Args:
        az: azimuth
        al: altitude
    """

    az: float
    al: float


class BinsRectangular(NamedTuple):
    """Two-dimensional binning structure for rectangular coordinates.

    Args:
        x: Array of x-coordinate bin edges
        y: Array of y-coordinate bin edges
    """

    x: npt.NDArray
    y: npt.NDArray


class BinsEquatorial(NamedTuple):
    """Two-dimensional binning structure for equatorial coordinates.

    Args:
        ra: Array of right ascension coordinate bin edges
        dec: Array of dec coordinate bin edges
    """

    ra: npt.NDArray
    dec: npt.NDArray


class UpscaleFactor(NamedTuple):
    """Upscaling factors for x and y dimensions.

    Args:
        x: Upscaling factor for x dimension
        y: Upscaling factor for y dimension
    """

    x: int
    y: int
