"""
General utility functions for the bloodmoon package.
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from bloodmoon.mask import CodedMaskCamera


@contextmanager
def clock(label: str, write: Callable = print):
    """A context manager for measuring processing times."""
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    write(f"{label} took {t2 - t1:.7}s")


def solid_angle(
    x: float | NDArray,
    y: float | NDArray,
    width: float,
    height: float,
    distance: float,
) -> NDArray:
    """
    Computes the solid angle covered by a rectangular plate of physical
    dimension `width` X `height` at a given `distance` from an observer
    located at `(x, y)` wrt the plate bottom-left corner.

                        |-------- width --------|
                                |
                         _______|_______________       _ _
                        |       |               |       |
                        |   4   |       3       |       |
                        |       |               |       |
                     ___|_______|_______________|____   |height
                        |       |               |       |
                        |   1  y|       2       |       |    
                        |_______|_______________|      _|_
                            x   |
                                |                 
    
    To compute the solid angle, the plate is divided in four sub-portions,
    with the observer located on one corner of each one. The plate solid
    angle can be computed by averaging the sub-portions solid angles seen
    by an on-axis observer.

    The sub-rectangules have areas:
        * `A_r1 = x * y`
        * `A_r2 = (width - x) * y`
        * `A_r3 = (width - x) * (height - y)`
        * `A_r4 = x * (height - y)`

    Args:
        x (float | NDArray):
            Observer coord along the x-axis wrt plate bottom-left corner.
            A non-zero solid angle requires that `0 <= x <= width / 2`.
        y (float | NDArray):
            Observer coord along the y-axis wrt plate bottom-left corner.
            A non-zero solid angle requires that `0 <= y <= height / 2`.
        width (float):
            Width of the rectangular plate.
        height (float):
            Height of the rectangular plate.
        distance (float):
            Distance between the plate and the observer.

    Returns:
        output (float | NDArray):
            Solid angle on the plate seen by the observer.
    
    Raises:
        ValueError: If input coords (x, y) not in the range [0, width / 2] x [0, height / 2].

    ## Notes
        - CFR with:
            * https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/imaging_utils.py#L58
            * https://vixra.org/pdf/2001.0603v2.pdf [Eq. 27, 34]
    """
    def on_axis_solid_angle(
        a: float | NDArray,
        b: float | NDArray,
    ) -> float | NDArray:
        """
        Computes the solid angle covered by a plate of dimension
        `a` X `b` wrt an observer at a given ox-axis `distance`.
        """
        alpha = a / (2 * distance)
        beta = b / (2 * distance)
        return 4 * np.arctan((alpha * beta) / np.sqrt(1 + alpha**2 + beta**2))
    
    if (
        np.any((x < 0) | (x > width / 2) | (y < 0) | (y > height / 2))
    ):
        raise ValueError(
            f"Invalid coords (x, y). Coords must be in the range [0, {width / 2}] x [0, {height / 2}]."
        )
    
    sub_portions = (
        (x, y),
        ((width - x), y),
        ((width - x), (height - y)),
        (x, (height - y)),
    )
    Omega = tuple(
        on_axis_solid_angle(2 * a, 2 * b) for a, b in sub_portions
    )
    return sum(Omega) / len(Omega)


def detector_solid_angle(camera: CodedMaskCamera) -> NDArray:
    """
    Computes the sky solid angle profile seen by each active element
    of the coded-mask camera detector.

    The solid angle is computed by considering only the instrument
    geometry, taking into account the mask physical dimension, which
    represents the base of a pyramid whose vertex sits on the center
    of each active element of the detector plane, and all the active
    elements coordinates (along the plane).
    The final array is masked with the detector bulk profile.

    Args:
        camera (CodedMaskCamera):
            Camera instance containing the system geometry info.
    
    Returns:
        output (NDArray):
            2D array representing the solid angle profile.
    
    ## Notes:
        - CFR with:
            * https://github.com/yuri-evangelista/CodedMasks/blob/26a5bb2fa08e37c645f85d55a3a1ef038fe7497d/mask_utils/imaging_utils.py#L104
    """
    UPX, UPY = camera.upscale_f

    # define mask physical dim (width, height)
    d = camera.specs['mask_detector_distance']
    maxx, minx = camera.specs['mask_maxx'], camera.specs['mask_minx']
    maxy, miny = camera.specs['mask_maxy'], camera.specs['mask_miny']
    maskplate_physdim = (maxx - minx, maxy - miny)

    # compute x- and y-coords for the detector elements solid angle
    #   - first define detector plane active elements positions
    #   - elements coords are clipped to remove binning artefact
    #   - the solid angle is masked with the bulk active elements
    _binsx, _binsy = camera.bins_detector
    centers_x, centers_y = (
        _binsx[:-1] + camera.specs['mask_deltax'] / (2 * UPX),
        _binsy[:-1] + camera.specs['mask_deltay'] / (2 * UPY),
    )
    x = np.clip(maxx - np.abs(centers_x[np.newaxis, :]), a_min=0, a_max=None)
    y = np.clip(maxy - np.abs(centers_y[:, np.newaxis]), a_min=0, a_max=None)
    return solid_angle(x, y, *maskplate_physdim, d) * (camera.bulk > 0)
