from typing import Callable, Optional
from functools import partial, cache
from bisect import bisect

import numpy as np
from scipy.signal import convolve
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator

from iros.mask import CodedMaskCamera, _bisect_interval, Bins2D, UpscaleFactor


def compose(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, Callable[[int, int], tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]]]:
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
                        WCE == `a`
                   NCS ==  rotated(`b`)

    Args:
        a (ndarray): First input matrix of shape (n,m) where n < m
        b (ndarray): Second input matrix of same shape as `a`

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

    Raises:
        AssertionError: If matrices a and b have different shapes

    Example:
        >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2x4 matrix
        >>> b = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])  # 2x4 matrix

        >>> composed, f = compose(a, b)
        >>> composed.shape
        (4, 4)
        >>> f(1, 1)  # center position
        ((0, 1), (1, 1))  # maps to both a and rotated b
    """
    assert a.shape == b.shape
    maxd, mind = max(a.shape), min(a.shape)
    delta = (maxd - mind) // 2
    if maxd == a.shape[1]:
        a_embedding = np.pad(a, pad_width=((delta, delta), (0, 0)))
        b_embedding = np.pad(np.rot90(b, k=-1), pad_width=((0, 0), (delta, delta)))
    else:
        a_embedding = np.pad(a, pad_width=((0, 0), (delta, delta)))
        b_embedding = np.pad(np.rot90(b, k=-1), pad_width=((delta, delta), (0, 0)))
    composed = a_embedding + b_embedding

    def _rotb2b(i, j):
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
                return None, _rotb2b(i, j - delta)
            elif i < maxd - delta:
                # C quadrant
                return (i - delta, j), _rotb2b(i, j - delta)
            else:
                # S quadrant
                return None, _rotb2b(i, j - delta)
        else:
            # E quadrant
            if not (delta <= i < delta + mind):
                return None, None
            else:
                return (i - delta, j), None

    return composed, f


def argmax(composed: np.ndarray) -> tuple[int, int]:
    """Find indices of maximum value in array.

    Args:
        composed: Input array to search

    Returns:
        Tuple of (row, col) indices of maximum value
    """
    return tuple(map(int, np.unravel_index(np.argmax(composed), composed.shape)))


def rbilinear(cx: float, cy: float, bins_x: np.array, bins_y: np.array) -> dict[tuple, float]:
    """
    Reverse bilinear interpolation weights for a point in a 2D grid.
    Y coordinates are supposed to grow top to bottom.
    X coordinates grow left to right.

    Args:
        cx: x-coordinate of the point
        cy: y-coordinate of the point
        bins_x: Sorted array of x-axis grid boundaries
        bins_y: Sorted array of y-axis grid boundaries

    Returns:
        Dictionary mapping grid point indices to their interpolation weights

    Raises:
        ValueError: If grid is invalid or point lies outside
    """
    if len(bins_x) < 2 or len(bins_y) < 2:
        raise ValueError("Grid boundaries must have at least 2 points")
    if not (np.all(np.diff(bins_x) > 0) and np.all(np.diff(bins_y) > 0)):
        raise ValueError("Grid bins must be strictly increasing")
    if not (bins_x[0] < cx < bins_x[-1] and bins_y[0] < cy < bins_y[-1]):
        raise ValueError("Center lies outside grid.")

    i, j = (bisect(bins_y, cy) - 1), bisect(bins_x, cx) - 1
    if i == 0 or j == 0 or i == len(bins_y) - 2 or j == len(bins_x) - 2:
        return {(i, j): 1.0}

    mx, my = (bins_x[j] + bins_x[j + 1]) / 2, (bins_y[i] + bins_y[i + 1]) / 2
    deltax, deltay = cx - mx, cy - my
    a = (i, j)
    b = (i, j + 1) if deltax > 0 else (i, j - 1)
    c = (i + 1, j) if deltay > 0 else (i - 1, j)

    if deltax > 0 and deltay < 0:
        d = (i - 1, j + 1)
    elif deltax > 0 and deltay > 0:
        d = (i + 1, j + 1)
    elif deltax < 0 and deltay > 0:
        d = (i + 1, j - 1)
    else:
        d = (i - 1, j - 1)

    deltax, deltay = map(abs, (deltax, deltay))
    weights = {
        a: (1 - deltay) * (1 - deltax),
        b: (1 - deltay) * deltax,
        c: (1 - deltax) * deltay,
        d: deltay * deltax,
    }
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


@cache
def _packing_factor(camera: CodedMaskCamera) -> tuple[float, float]:
    """
    Returns the density of slits along the x and y axis.

    Args:
        camera: a CodedMaskCamera object.

    Returns:
        A tuple of the x and y packing factors.
    """
    rows_notnull = camera.mask[np.any(camera.mask != 0, axis=1), :]
    cols_notnull = camera.mask[:, np.any(camera.mask != 0, axis=0)]
    pack_x, pack_y = np.mean(np.mean(rows_notnull, axis=1)), np.mean(np.mean(cols_notnull, axis=0))
    return tuple(map(float, (pack_x, pack_y)))


def chop(camera: CodedMaskCamera, pos: tuple[int, int], sky: np.array) -> tuple[np.array, Bins2D]:
    """
    Returns a slice of `sky` centered around `pos` and sized slightly larger than slit size.

    Args:
        camera: a CodedMaskCameraObject.
        pos: the (row, col) indeces of the slice center.
        sky: a sky image.

    Returns:
        A tuple of the slice value and its bins.
    """
    bins = camera.bins_sky
    i, j = pos
    packing_x, packing_y = map(lambda x: x * 2, _packing_factor(camera))
    min_i, max_i = _bisect_interval(bins.y, bins.y[i] - camera.mdl["slit_deltay"] / packing_y, bins.y[i] + camera.mdl["slit_deltay"] / packing_y)
    min_j, max_j = _bisect_interval(bins.x, bins.x[j] - camera.mdl["slit_deltax"] / packing_x, bins.x[j] + camera.mdl["slit_deltax"] / packing_x)
    return sky[min_i:max_i, min_j:max_j], Bins2D(x=bins.x[min_j: max_j + 1], y=bins.y[min_i: max_i + 1], )


def _interp(tile: np.array, bins: Bins2D, interp_f):
    """
    Upscales a regular grid of data and interpolates with cubic splines.

    Args:
        tile: the data value to interpolate
        bins: a Bins2D object. If data has shape (n, m), `bins` should have shape (n + 1,m + 1).
        interp_f: a `UpscaleFactor` object representing the upscaling to be applied on the data.

    Returns:
        a tuple of the interpolated data and their __midpoints__ (not bins!).

    """
    midpoints_x = (bins.x[1:] + bins.x[:-1]) / 2
    midpoints_y = (bins.y[1:] + bins.y[:-1]) / 2
    midpoints_x_fine = np.linspace(midpoints_x[0], midpoints_x[-1], len(midpoints_x) * interp_f.x + 1 )
    midpoints_y_fine = np.linspace(midpoints_y[0], midpoints_y[-1], len(midpoints_y) * interp_f.y + 1 )
    interp = RegularGridInterpolator((midpoints_x, midpoints_y), tile.T, method="cubic")
    grid_x_fine, grid_y_fine = np.meshgrid(midpoints_x_fine, midpoints_y_fine)
    tile_interp = interp((grid_x_fine, grid_y_fine))
    return tile_interp, Bins2D(x=midpoints_x_fine, y=midpoints_y_fine)


def interpmax(camera: CodedMaskCamera, pos, sky, interp_f: UpscaleFactor=UpscaleFactor(10, 10)):
    """
    Interpolates and maximizes data around pos.

    Args:
        camera: a CodedMaskCamera object.
        pos: the (row, col) indeces of the slice center.
        sky: the sky image.
        interp_f: a `UpscaleFactor` object representing the upscaling to be applied on the data.

    Returns:
        Sky-shift position of the interpolated maximum.
    """
    tile_interp, bins_fine = _interp(*chop(camera, pos, sky), interp_f)
    max_tile_i, max_tile_j = argmax(tile_interp)
    return tuple(map(float, (bins_fine.x[max_tile_j], bins_fine.y[max_tile_i])))


params_psfx = {
    "center": 0,
    "alpha": 0.0016,
    "beta": 0.6938,
}

params_psfy = {
    "center": 0,
    "alpha": 0.3214,
    "beta": 0.6246,
}


def modsech(x, norm, center, alpha, beta):
    return norm / np.cosh(np.abs((x - center) / alpha) * beta)


psfx = partial(modsech, norm=1., **params_psfx)
psfy = partial(modsech, norm=1., **params_psfy)


def norm_constant(center, alpha, beta):
    xs = np.linspace(-50*alpha, +50*alpha, 10000)
    return 1 / trapezoid(
        y=modsech(xs, norm=1, center=center, alpha=alpha, beta=beta),
        x=xs,
    )


def norm_modsech(center, alpha, beta):
    norm = norm_constant(center, alpha, beta)
    return partial(modsech, norm=norm, center=center, alpha=alpha, beta=beta)


norm_psfx = norm_modsech(**params_psfx)
norm_psfy = norm_modsech(**params_psfy)


def filter_detector(camera: CodedMaskCamera, detector: np.array):
    bins = camera.bins_detector
    min_bin, max_bin = _bisect_interval(np.round(bins.y, 2), -5, +5)
    bin_edges = bins.y[min_bin: max_bin + 1]
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    kernel = norm_psfy(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np.sum(kernel)
    return convolve(detector, kernel, mode="same")