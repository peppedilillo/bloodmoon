"""
Custom data types and containers for the WFM analysis pipeline.
"""

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

    def __repr__(self):
        def r(n):
            return round(n.item(), 3)

        reprx = (
            f"np.array([{r(self.x[0])}, {r(self.x[1])}, .., {r(self.x[-1])}])"
            if len(self.x) > 3 else f"{self.x}"
        )
        repry = (
            f"np.array([{r(self.y[0])}, {r(self.y[1])}, .., {r(self.y[-1])}])"
            if len(self.y) > 3 else f"{self.y}"
        )
        return f"BinsRectangular(x={reprx}, y={repry})"


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
