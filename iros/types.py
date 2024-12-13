from typing import NamedTuple

import numpy as np


class Bins2D(NamedTuple):
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
