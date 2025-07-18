"""
Image processing and manipulation utilities for coded mask data analysis.

This module provides functions for:
- Image composition and decomposition
- Upscaling and interpolation
- Pattern shifting and erosion
- Bilinear interpolation
- Image statistics and peak finding

The functions handle both detector shadowgrams and reconstructed sky images.
"""

from bisect import bisect
from collections import OrderedDict
from typing import Callable, Optional
import warnings

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from .types import BinsRectangular
from .types import UpscaleFactor


def _upscale(
    m: npt.NDArray,
    upscale_x: int,
    upscale_y: int,
) -> npt.NDArray:
    """
    Oversamples a 2D array by repeating elements along the axes.

    Args:
        m: Input 2D array.
        upscale_x: Upscaling factor along the x-axis.
        upscale_y: Upscaling factor along the y-axis.

    Returns:
        output: Oversampled array.

    Notes:
        - the total sum is NOT conserved. Hence the function name is somewhat
          off, since there is no "scaling". A better name would be `enlarge` or
          similar. However, we used it for naming variables and parameters in
          many places so we are keeping it, for now.
    """
    for ax, factor in enumerate((upscale_y, upscale_x)):
        m = np.repeat(m, factor, axis=ax)
    return m


def compose(
    a: npt.NDArray,
    b: npt.NDArray,
    strict=True,
) -> tuple[np.ndarray, Callable]:
    """
    Composes two matrices `a` and `b` into one square embedding.
    The `b` matrix is rotated by 90 degree *clockwise*,
    i.e. np.rot90(b, k=-1) is applied before embedding.

         │
      ───┼──────────────j-index────────────────▶
         │     Δ                       Δ
         │   ◀────▶                  ◀────▶
         │   ┌────┬──────────────────┬────┐  ▲
         │   │    │ N                │    │  │
         │   ├────┼──────────────────┼────┤  │
         │   │    │                  │  E │  │
         │   │    │                  │    │  │
         │   │    │                  │    │  │
     i-index │    │                  │    │maxd
         │   │    │                  │    │  │
         │   │  W │                C │    │  │
         │   ├────┼──────────────────┼────┤  │
         │   │    │                S │    │  │
         │   └────┴──────────────────┴────┘  ▼
         │        ◀───────mind───────▶
         ▼
                        W+C+E == `a`
                   N+C+S ==  rotated(`b`)

    Args:
        a: First input matrix of shape (n,m) where n < m
        b: Second input matrix of same shape as `a`
        strict: if True raises an error if matrices have odd rows and even columns,
                or viceversa.

    Returns:
        Tuple containing:
            - ndarray: The composed square matrix of size maxd x maxd where
                      maxd = max(n,m)
            - Callable: A function f(i,j) that maps positions in the composed matrix
                       to positions in the original matrices a and b. For each position
                       it returns a tuple (pos_a, pos_b) where:
                       - pos_a: Optional tuple (i,j) in matrix a or None if position
                               doesn't map to a
                       - pos_b: Optional tuple (i,j) in matrix b or None if position
                               doesn't map to b
                       Full typing signature would be:
                       Callable[
                           [int, int], # input, composed matrix index
                           tuple[
                               Optional[tuple[int, int]], `a` matrix index
                               Optional[tuple[int, int]]  `b` matrix index
                           ]
                       ]

    Raises:
        AssertionError: If matrices a and b have different shapes
        ValueError: If `strict` and matrices have odd rows and even columns (and viceversa)
                    or if `a` and `b` have different shapes.

    Example:
        >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4 matrix
        >>> b = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2x4 matrix

        >>> composed, f = compose(a, b)
        >>> composed.shape
        (4, 4)
        >>> f(1, 1)  # center position
        ((0, 1), (1, 1))  # maps to both a and rotated b
    """
    if a.shape != b.shape:
        raise ValueError("Input matrices must have same shape")

    maxd, mind = max(a.shape), min(a.shape)
    # if matrices have odd rows and even columns, or viceversa, composition is ambiguous because
    # we can't put one piece over the other at the dead center. we have to chose if putting one
    # up or down a row. we solve this by silently cutting a column, or a row. We could pad
    # but I feel it would end even worse.
    if maxd % 2 != mind % 2:
        if strict:
            raise ValueError("Input matrices must have rows and columns with same parity if `strict` is True")
        if maxd == a.shape[1]:
            a = a[:, :-1]
            b = b[:, :-1]
        else:
            a = a[:-1, :]
            b = b[:-1, :]
        maxd -= 1

    delta = (maxd - mind) // 2
    if maxd == a.shape[1]:
        a_embedding = np.pad(a, pad_width=((delta, delta), (0, 0)))
        b_embedding = np.pad(np.rot90(b, k=-1), pad_width=((0, 0), (delta, delta)))
    else:
        a_embedding = np.pad(a, pad_width=((0, 0), (delta, delta)))
        b_embedding = np.pad(np.rot90(b, k=-1), pad_width=((delta, delta), (0, 0)))
    composed = a_embedding + b_embedding

    def _rotback2b(i, j):
        """Given c_i, c_j indices !of the compose's output 'c'! returns b_i, b_j."""
        return mind - 1 - j, i

    def f(i: int, j: int) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """
        Given a couple of indeces of the recombined image, returns two couples of
        indeces, one for the `a` matrix, and one for the `b` matrix.

        Args:
            i (int): row index in the composed matrix
            j (int): column index in the composed matrix

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]: A tuple containing
                - First element: Indices (i,j) in matrix a, or None if position doesn't map to a
                - Second element: Indices (i,j) in matrix b, or None if position doesn't map to b

        Raises:
            ValueError: If the position (i,j) is out of bounds of the composed matrix
        """
        if not ((0 <= i < maxd) and (0 <= j < maxd)):
            raise ValueError("position is out of bounds")
        if j < delta:
            # W quadrant
            if not (delta <= i < delta + mind):
                return None, None
            else:
                return (i - delta, j), None
        elif j < mind + delta:
            if i < delta:
                # N quadrant
                return None, _rotback2b(i, j - delta)
            elif i < maxd - delta:
                # C quadrant
                return (i - delta, j), _rotback2b(i, j - delta)
            else:
                # S quadrant
                return None, _rotback2b(i, j - delta)
        else:
            # E quadrant
            if not (delta <= i < delta + mind):
                return None, None
            else:
                return (i - delta, j), None

    return composed, f


def argmax(composed: npt.NDArray) -> tuple[int, int]:
    """Find indices of maximum value in array.

    Args:
        composed: Input array to search

    Returns:
        Tuple of (row, col) indices of maximum value
    """
    row, col = np.unravel_index(np.argmax(composed), composed.shape)
    return int(row), int(col)


def _rbilinear(
    cx: float,
    cy: float,
    bins_x: npt.NDArray,
    bins_y: npt.NDArray,
) -> OrderedDict[tuple, float]:
    """
    Reverse bilinear interpolation weights for a point in a 2D grid.
    Y coordinates are supposed to grow top to bottom.
    X coordinates grow left to right.

    The basic idea is to identify four poles and to assign them weights.
    The more the center is close to a pole, the more weight it gets.

           │            │            │
     ──────┼────────────┼────────────┼──────
           │A           │           B│
           │  ┌─────────┼──┐ ▲       │
           │  │     .─. │  │ │(1 - dy)
           │  │    ( c )│  │ │       │
           │  │     `─' │  │ ▼       │
     ──────┼──┼─────────┼──┼─▲───────┼──────
           │  │         │  │ │ dy    │
           │  └─────────┼──┘ ▼       │
           │   ◀───────▶│◀─▶     │
           │   (1 - dx) │ dx         │
           │C           │           D│
     ──────┼────────────┼────────────┼──────
           │            │            │

    To A (pivot) we assign a weight (1 - dx) * (1 - dy).
    To B we assign a weight dx * (1 - dy).
    To C we assign a weight (1 - dx) * dy.
    To D we assign a weight dx * dy.

    Args:
        cx: x-coordinate of the point
        cy: y-coordinate of the point
        bins_x: Sorted array of x-axis grid boundaries
        bins_y: Sorted array of y-axis grid boundaries

    Returns:
        Ordered dictionary mapping grid point indices to their interpolation weights
        The first dictionary elements map to the bin whose midpoint is closest to the input.

    Notes:
        * Assumes uniform grid spacing.
        * If the pivot falls on the grid (hence there is no unambiguous choice),
          the cell with the largest indeces is selected as the pivot.
          For example, in the next case, the pivot has index (4, 3):
          ```
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.25, 0.25, 0.0],
                [0.0, 0.0, 0.25, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
            ```
            See tests for more details.

    Raises:
        ValueError: If grid is invalid or point lies outside
    """
    if len(bins_x) < 2 or len(bins_y) < 2:
        raise ValueError("Grid boundaries must have at least 2 points")
    if not (np.all(np.diff(bins_x) > 0) and np.all(np.diff(bins_y) > 0)):
        raise ValueError("Grid bins must be strictly increasing")
    if not (bins_x[0] <= cx <= bins_x[-1] and bins_y[0] <= cy <= bins_y[-1]):
        raise ValueError("Center lies outside grid.")

    i, j = (bisect(bins_y, cy) - 1), bisect(bins_x, cx) - 1
    # why the `-2`?
    # ```
    # bins_x = [0, 1, 2]
    # cx = 1.99  # a value very close to the right border
    # assert(bisect(bins_x, cx) == 2)
    # i = bisect(bins_y, cy) - 1
    # ```
    # hence `i` should be 1 to fall on the border
    if i == 0 or j == 0 or i == len(bins_y) - 2 or j == len(bins_x) - 2:
        i = max(0, min(i, len(bins_y) - 2))
        j = max(0, min(j, len(bins_x) - 2))
        return OrderedDict([((i, j), 1.0)])

    mx, my = (bins_x[j] + bins_x[j + 1]) / 2, (bins_y[i] + bins_y[i + 1]) / 2
    deltax, deltay = cx - mx, cy - my
    a = (i, j)
    b = (i, j + 1) if deltax > 0 else (i, j - 1)
    c = (i + 1, j) if deltay > 0 else (i - 1, j)

    if deltax > 0 > deltay:
        d = (i - 1, j + 1)
    elif deltax > 0 and deltay > 0:
        d = (i + 1, j + 1)
    elif deltax < 0 < deltay:
        d = (i + 1, j - 1)
    else:
        d = (i - 1, j - 1)

    xstep, ystep = bins_x[1] - bins_x[0], bins_y[1] - bins_y[0]
    deltax, deltay = map(abs, (deltax, deltay))
    weights = OrderedDict(
        [
            (a, (ystep - deltay) * (xstep - deltax)),
            (b, (ystep - deltay) * deltax),
            (c, (xstep - deltax) * deltay),
            (d, deltay * deltax),
        ]
    )
    total = sum(weights.values())
    return OrderedDict([(k, v / total) for k, v in weights.items()])


def _rbilinear_relative(
    cx: float,
    cy: float,
    bins_x: npt.NDArray,
    bins_y: npt.NDArray,
) -> tuple[OrderedDict, tuple[int, int]]:
    """To avoid computing shifts many time, we create a slightly shadowgram and index over it.
    This operation requires the results for rbilinear to be expressed relatively to the pivot.
    """
    results_rbilinear = _rbilinear(cx, cy, bins_x, bins_y)
    ((pivot_i, pivot_j), _), *__ = results_rbilinear.items()
    # noinspection PyTypeChecker
    return OrderedDict([((k_i - pivot_i, k_j - pivot_j), w) for (k_i, k_j), w in results_rbilinear.items()]), (
        pivot_i,
        pivot_j,
    )


def _interp(
    tile: npt.NDArray,
    bins: BinsRectangular,
    interp_f: UpscaleFactor,
):
    """
    Upscales a regular grid of data and interpolates with cubic splines.

    Args:
        tile: the data value to interpolate
        bins: a Bins2D object. If data has shape (n, m), `bins` should have shape (n + 1,m + 1).
        interp_f: a `UpscaleFactor` object representing the upscaling to be applied on the data.

    Returns:
        a tuple of the interpolated data and their __midpoints__ (not bins!).

    """

    def find_method(xs: npt.NDArray, ys: npt.NDArray):
        mindim = min(min(xs.shape), min(ys.shape))
        if mindim > 3:
            return "cubic"
        elif mindim > 1:
            method = "linear"
            warnings.warn(
                f"Interpolator bins too small for method 'cubic', resorting to '{method}'. "
                f"Consider upscaling your mask if you haven't yet."
            )
        elif mindim > 0:
            method = "nearest"
            warnings.warn(
                f"Interpolator bins too small for method 'cubic', resorting to '{method}'. "
                f"Consider upscaling your mask if you haven't yet."
            )
        else:
            raise ValueError("Can not interpolate, interpolator grid is empty.")
        return method

    midpoints_x = (bins.x[1:] + bins.x[:-1]) / 2
    midpoints_y = (bins.y[1:] + bins.y[:-1]) / 2
    midpoints_x_fine = np.linspace(midpoints_x[0], midpoints_x[-1], interp_f.x * (len(midpoints_x) - 1) + 1)
    midpoints_y_fine = np.linspace(midpoints_y[0], midpoints_y[-1], interp_f.y * (len(midpoints_y) - 1) + 1)
    interp = RegularGridInterpolator(
        (midpoints_x, midpoints_y),
        tile.T,
        method=find_method(midpoints_x, midpoints_y),
    )
    grid_x_fine, grid_y_fine = np.meshgrid(midpoints_x_fine, midpoints_y_fine)
    tile_interp = interp((grid_x_fine, grid_y_fine))
    return tile_interp, BinsRectangular(x=midpoints_x_fine, y=midpoints_y_fine)


def _shift(
    a: npt.NDArray,
    shift_ext: tuple[int, int],
) -> npt.NDArray:
    """Shifts a 2D numpy array by the specified amount in each dimension.
    This exists because the scipy.ndimage one is slow.

    Args:
        a: Input 2D numpy array to be shifted.
        shift_ext: Tuple of (row_shift, column_shift) where positive values shift down/right
            and negative values shift up/left. Values larger than array dimensions
            result in an array of zeros.

    Returns:
        np.array: A new array of the same shape as the input, with elements shifted
            and empty spaces filled with zeros.

    Examples:
        >>> arr = np.array([[1, 2], [3, 4]])
        >>> _shift(arr, (1, 0))  # Shift down by 1
        array([[0, 0],
               [1, 2]])
        >>> _shift(arr, (0, -1))  # Shift left by 1
        array([[2, 0],
               [4, 0]])
    """
    n, m = a.shape
    shift_i, shift_j = shift_ext
    if abs(shift_i) >= n or abs(shift_j) >= m:
        # won't load into memory 66666666 x 66666666 bullshit matrix
        return np.zeros_like(a)
    vpadded = np.pad(a, ((0 if shift_i < 0 else shift_i, 0 if shift_i >= 0 else -shift_i), (0, 0)))
    vpadded = vpadded[:n, :] if shift_i > 0 else vpadded[-n:, :]
    hpadded = np.pad(
        vpadded,
        ((0, 0), (0 if shift_j < 0 else shift_j, 0 if shift_j >= 0 else -shift_j)),
    )
    hpadded = hpadded[:, :m] if shift_j > 0 else hpadded[:, -m:]
    return hpadded


def _erosion(
    arr: npt.NDArray,
    step: float,
    cut: float,
) -> npt.NDArray:
    """
    2D matrix erosion for simulating finite thickness effect in shadow projections.
    It takes a mask array and "thins" the mask elements across the columns' direction.

    Comes with NO safeguards: setting cuts larger than step may remove slits or make them negative.

    ⢯⣽⣿⣿⣿⠛⠉⠀⠀⠉⠉⢛⢟⡻⣟⡿⣿⢿⣿⣿⢿⣻⣟⡿⣟⡿⣿⣻⣟⣿⣟⣿⣻⣟⡿⣽⣻⠿⣽⣻⢟⡿⣽⢫⢯⡝
    ⢯⣞⣷⣻⠤⢀⠀⠀⠀⠀⠀⠀⠀⠑⠌⢳⡙⣮⢳⣭⣛⢧⢯⡽⣏⣿⣳⢟⣾⣳⣟⣾⣳⢯⣽⣳⢯⣟⣷⣫⢿⣝⢾⣫⠗⡜
    ⡿⣞⡷⣯⢏⡴⢀⠀⠀⣀⣤⠤⠀⠀⠀⠀⠑⠈⠇⠲⡍⠞⡣⢝⡎⣷⠹⣞⢧⡟⣮⢷⣫⢟⡾⣭⢷⡻⢶⣏⣿⢺⣏⢮⡝⢌
    ⢷⣹⢽⣚⢮⡒⠆⠀⢰⣿⠁⠀⠀⠀⢱⡆⠀⠀⠈⠀⠀⠄⠁⠊⠜⠬⡓⢬⠳⡝⢮⠣⢏⡚⢵⢫⢞⡽⣏⡾⢧⡿⣜⡣⠞⡠
    ⢏⣞⣣⢟⡮⡝⣆⢒⠠⠹⢆⡀⠀⢀⠼⠃⣀⠄⡀⢠⠠⢤⡤⣤⢀⠀⠁⠈⠃⠉⠂⠁⠀⠉⠀⠃⠈⠒⠩⠘⠋⠖⠭⣘⠱⡀
    ⡚⡴⣩⢞⣱⢹⠰⡩⢌⡅⠂⡄⠩⠐⢦⡹⢜⠀⡔⢡⠚⣵⣻⢼⡫⠔⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡄
    ⡑⠦⡑⢎⡒⢣⢣⡑⢎⡰⢁⡒⢰⢠⢣⠞⢁⠢⡜⢢⢝⣺⡽⢮⠑⡈⠀⠀⠀⢀⡀⠀⣾⡟⠁⠀⠀⠠⡀⠀⠀⠀⠀⠀⠀⠐
    ⢘⠰⡉⢆⠩⢆⠡⠜⢢⢡⠣⡜⢡⢎⠧⡐⢎⡱⢎⡱⢊⣾⡙⢆⠁⡀⠄⡐⡈⢦⢑⠂⠹⣇⠀⠀⠀⢀⣿⡀⠀⠀⠀⢀⠀⠄
    ⠈⢆⠱⢈⠒⡈⠜⡈⢆⠢⢱⡘⣎⠞⡰⣉⠎⡴⢋⢰⣻⡞⣍⠂⢈⠔⡁⠆⡑⢎⡌⠎⢡⠈⠑⠂⠐⠋⠁⠀⠀⡀⢆⠠⣉⠂
    ⡉⠔⡨⠄⢂⡐⠤⡐⣄⢣⢧⡹⡜⢬⡑⡌⢎⡵⢋⣾⡳⡝⠤⢀⠊⡔⡈⢆⡁⠮⡜⠬⢠⢈⡐⡉⠜⡠⢃⠜⣠⠓⣌⠒⠤⡁
    ⢌⠢⢡⠘⡄⢎⡱⡑⢎⡳⢎⠵⡙⢆⠒⡍⡞⣬⢛⡶⡹⠌⡅⢂⠡⠐⠐⠂⠄⡓⠜⡈⢅⠢⠔⡡⢊⠔⡡⢚⠤⣋⠤⡉⠒⠠
    ⢢⢑⢢⠱⡘⢦⠱⣉⠞⡴⢫⣜⡱⠂⡬⠜⣵⢊⠷⡸⠥⠑⡌⢂⠠⠃⢀⠉⠠⢜⠨⠐⡈⠆⡱⢀⠣⡘⠤⣉⠒⠄⠒⠠⢁⠡
    ⢌⡚⡌⢆⠳⣈⠦⣛⠴⣓⠮⣝⠃⠐⡁⠖⣭⢚⡴⢃⠆⢢⠑⡌⠀⠀⠌⠐⠠⢜⠢⡀⠡⠐⠡⠘⠠⢁⠂⡉⠐⡀⠂⠄⡈⠄
    ⠦⡱⡘⣌⠳⣌⠳⣌⠳⣍⠞⣥⢣⠀⠈⠑⠢⢍⠲⢉⠠⢁⠊⠀⠁⠀⠄⠡⠈⢂⠧⡱⣀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠐⠀⡀⠂
    ⠂⠥⠑⡠⢃⠌⡓⢌⠳⢌⡹⢄⠣⢆⠀⠀⠀⠈⠀⠀⠀⠀⠀⠈⠀⠀⡌⢢⡕⡊⠔⢡⠂⡅⠂⠀⠀⠀⠀⠀⠐⠈⠀⢀⠀⠀
    ⠈⠄⠡⠐⠠⠈⠔⣈⠐⢂⠐⡨⠑⡈⠐⡀⠀⠀⠀⠀⠀⠀⠀⡀⢤⡘⠼⣑⢎⡱⢊⠀⠐⡀⠁⠀⠀⠀⠐⠀⠀⢀⠀⠀⠀⠀
    ⠀⠈⠄⡈⠄⣁⠒⡠⠌⣀⠒⠠⠁⠄⠡⢀⠁⠀⢂⠠⢀⠡⢂⠱⠢⢍⠳⣉⠖⡄⢃⠀⠀⠄⠂⠀⢀⠈⠀⢀⠈⠀⠀⠀⠀⠀
    ⠀⡁⠆⠱⢨⡐⠦⡑⢬⡐⢌⢢⡉⢄⠃⡄⠂⠁⠠⠀⠄⠂⠄⠡⢁⠊⡑⠌⡒⢌⠢⢈⠀⠄⠂⠁⡀⠀⠂⡀⠄⠂⠀⠀⠀⠀
    ⠤⠴⣒⠦⣄⠘⠐⠩⢂⠝⡌⢲⡉⢆⢣⠘⠤⣁⢂⠡⠌⡐⠈⠄⢂⠐⡀⠂⢀⠂⠐⠠⢈⠀⡐⠠⠀⠂⢁⠀⠀⠀⠀⠀⠀⠀
    ⠌⠓⡀⠣⠐⢩⠒⠦⠄⣀⠈⠂⠜⡈⠦⠙⡒⢤⠃⡞⣠⠑⡌⠢⠄⢂⠐⠀⠀⠀⠀⠀⠀⠂⠀⠐⡀⠁⠠⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠁⡀⢈⠈⡑⠢⡙⠤⢒⠆⠤⢁⣀⠂⠁⠐⠁⠊⠔⠡⠊⠄⠂⢀⠀⠀⠀⠀⠀⠂⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠁⠀⠀⠀⡀⠀⠀⠀⠈⠁⠊⠅⠣⠄⡍⢄⠒⠤⠤⢀⣀⣀⣀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠁⠀⠀⠁⠀⠂⠀⠄⠀⠀⠀⠈⠀⠉⠀⠁⠂⠀⠀⠉⠉⠩⢉⠢⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    ⠀⠀⠀⠀⠂⠀⠀⠀⠀⠐⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠄⠁⠄⠀⠀⠀

    Args:
        arr: 2D input array of integers representing the projected shadow.
        step: The projection bin step.
        cut: Maximum cut width.

    Returns:
        Modified array with shadow effects applied

    Notes:
        * See tests for usage examples.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")

    # number of bins to cut
    ncuts = int(cut / step)
    cutted = arr * (arr & _shift(arr, (0, ncuts))) if ncuts else arr

    # array indexes to be fractionally reduced:
    #   - the bin with the decimal values is the one
    #     to the left or right wrt the cutted bins
    erosion_value = abs(cut / step - ncuts)
    border = (cutted - _shift(cutted, (0, int(np.sign(cut))))) > 0
    return cutted - border * erosion_value


def _unframe(a: npt.NDArray, value: float = 0.0) -> npt.NDArray:
    """Removes outer frames of a 2D array until a non-zero frame is found.

    A frame is considered empty if all values in its border are zeros. The function
    works from outside in, replacing each empty frame with the specified value until
    it finds a frame that contains non-zero values.

    Args:
        a (np.array): Input 2D array to process.
        value (float, optional): Value to replace the empty frames with. Defaults to `0.`.

    Returns:
        np.array: A copy of the input array with empty frames replaced.

    Raises:
        ValueError: If the input is not a two dimensional array.

    Examples:
        >>> arr = np.array([
        ...     [0, 1, 0, 0],
        ...     [0, 1, 2, 0],
        ...     [0, 3, 4, 0],
        ...     [0, 0, 0, 1]
        ... ])
        >>> unframe(arr)
        array([[0, 0, 0, 0],
               [0, 1, 2, 0],
               [0, 3, 4, 0],
               [0, 0, 0, 0]])
    """
    if a.ndim != 2:
        raise ValueError("Input is not a two dimensional array.")
    n, m = a.shape
    maxd, mind = sorted((n, m))
    out = a.copy()
    for i in range(mind // 2):
        upper_row = slice(i, i + 1), slice(i, m - i)
        right_column = slice(i, n - i), slice(m - i - 1, m - i)
        bottom_row = slice(n - i - 1, n - i), slice(i, m - i)
        left_column = slice(i, n - i), slice(i, i + 1)
        if not (
            np.any(np.isclose(a[*upper_row], 0.0))
            or np.any(np.isclose(a[*right_column], 0.0))
            or np.any(np.isclose(a[*bottom_row], 0.0))
            or np.any(np.isclose(a[*left_column], 0.0))
        ):
            break
        out[*upper_row] = value
        out[*right_column] = value
        out[*bottom_row] = value
        out[*left_column] = value
    return out
