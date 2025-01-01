from typing import NamedTuple

import numpy as np


class BinsRectangular(NamedTuple):
    """Two-dimensional binning structure for rectangular coordinates.

    Args:
        x: Array of x-coordinate bin edges
        y: Array of y-coordinate bin edges
    """

    x: np.array
    y: np.array


class BinsEquatorial(NamedTuple):
    """Two-dimensional binning structure for equatorial coordinates.

    Args:
        ra: Array of right ascension coordinate bin edges
        dec: Array of dec coordinate bin edges
    """

    ra: np.array
    dec: np.array


class UpscaleFactor(NamedTuple):
    """Upscaling factors for x and y dimensions.

    Args:
        x: Upscaling factor for x dimension
        y: Upscaling factor for y dimension
    """

    x: int
    y: int
