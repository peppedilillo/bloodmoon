from bisect import bisect
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from iros.mask import _bisect_interval, CodedMaskCamera, _detector_footprint
from iros.types import BinsRectangular
from iros.mask import CodedMaskCamera
from iros.mask import UpscaleFactor


def compose(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, Callable]:
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
    row, col = np.unravel_index(np.argmax(composed), composed.shape)
    return int(row), int(col)


def _rbilinear(cx: float, cy: float, bins_x: np.array, bins_y: np.array) -> dict[tuple, float]:
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
    weights = {
        a: (ystep - deltay) * (xstep - deltax),
        b: (ystep - deltay) * deltax,
        c: (xstep - deltax) * deltay,
        d: deltay * deltax,
    }
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


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
    return float(pack_x), float(pack_y)


def _chop(camera: CodedMaskCamera, pos: tuple[int, int]) -> tuple[tuple, BinsRectangular]:
    """
    Returns a slice of sky centered around `pos` and sized slightly larger than slit size.

    Args:
        camera: a CodedMaskCameraObject.
        pos: the (row, col) indeces of the slice center.

    Returns:
        A tuple of the slice value (length n) and its bins (length n + 1).
    """
    bins = camera.bins_sky
    i, j = pos
    packing_x, packing_y = map(lambda x: x * 2, _packing_factor(camera))
    min_i, max_i = _bisect_interval(bins.y, bins.y[i] - camera.mdl["slit_deltay"] / packing_y, bins.y[i] + camera.mdl["slit_deltay"] / packing_y)
    min_j, max_j = _bisect_interval(bins.x, bins.x[j] - camera.mdl["slit_deltax"] / packing_x, bins.x[j] + camera.mdl["slit_deltax"] / packing_x)
    return (min_i, max_i, min_j, max_j), BinsRectangular(
        x=bins.x[min_j:max_j + 1],
        y=bins.y[min_i:max_i + 1],
    )


def _interp(tile: np.array, bins: BinsRectangular, interp_f):
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
    midpoints_x_fine = np.linspace(midpoints_x[0], midpoints_x[-1], len(midpoints_x) * interp_f.x + 1)
    midpoints_y_fine = np.linspace(midpoints_y[0], midpoints_y[-1], len(midpoints_y) * interp_f.y + 1)
    interp = RegularGridInterpolator((midpoints_x, midpoints_y), tile.T, method="cubic")
    grid_x_fine, grid_y_fine = np.meshgrid(midpoints_x_fine, midpoints_y_fine)
    tile_interp = interp((grid_x_fine, grid_y_fine))
    return tile_interp, BinsRectangular(x=midpoints_x_fine, y=midpoints_y_fine)


def _interpmax(camera: CodedMaskCamera, pos, sky, interp_f: UpscaleFactor = UpscaleFactor(10, 10)):
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
    (min_i, max_i, min_j, max_j), bins = _chop(camera, pos)
    tile_interp, bins_fine = _interp(sky[min_i:max_i, min_j:max_j], bins, interp_f)
    max_tile_i, max_tile_j = argmax(tile_interp)
    return tuple(map(float, (bins_fine.x[max_tile_j], bins_fine.y[max_tile_i])))


# TODO: These should go to a separate configuration file
_PSFX_WFM_PARAMS = {
    "center": 0,
    "alpha": 0.0016,
    "beta": 0.6938,
}

_PSFY_WFM_PARAMS = {
    "center": 0,
    "alpha": 0.2592,
    "beta": 0.5972,
}


def _modsech(x: np.array, norm: float, center: float, alpha: float, beta: float) -> np.array:
    """
    PSF fitting function template.

    Args:
        x: a numpy array or value, in millimeters
        norm: normalization parameter
        center: center parameter
        alpha: alpha shape parameter
        beta: beta shape parameter

    Returns:
        numpy array or value, depending on the input
    """
    return norm / np.cosh(np.abs((x - center) / alpha) * beta)


def psfy_wfm(x: np.array) -> np.array:
    """
    PSF function in y direction as fitted from WFM simulations.

    Args:
        x: a numpy array or value, in millimeters

    Returns:
        numpy array or value
    """
    return _modsech(x, **_PSFY_WFM_PARAMS)


def _convolution_kernel_psfy(camera) -> np.array:
    """
    Returns PSF convolution kernel.
    At present, it ignores the `x` direction, since PSF characteristic lenght is much shorter
    than typical bin size, even at moderately large upscales.

    Args:
        camera: a CodedMaskCamera object.

    Returns:
        A column array convolution kernel.
    """
    bins = camera.bins_detector
    min_bin, max_bin = _bisect_interval(bins.y, -camera.mdl["slit_deltay"], camera.mdl["slit_deltay"])
    bin_edges = bins.y[min_bin: max_bin + 1]
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    kernel = psfy_wfm(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np. sum(kernel)
    return kernel


def _shift(a: np.array, shift_ext: tuple[int, int]) -> np.array:
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
        # we won't load into memory your 66666666 x 6666666666 bullshit matrix
        return np.zeros_like(a)
    vpadded = np.pad(a, ((0 if shift_i < 0 else shift_i, 0 if shift_i >= 0 else -shift_i), (0, 0)))
    vpadded = vpadded[:n, :] if shift_i > 0 else vpadded[-n:, :]
    hpadded = np.pad(vpadded, ((0, 0), (0 if shift_j < 0 else shift_j, 0 if shift_j >= 0 else -shift_j)))
    hpadded = hpadded[:, :m] if shift_j > 0 else hpadded[:, -m:]
    return hpadded


def _erosion(
        arr: np.array,
        step: float,
        cut: float
) -> np.array:
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
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")

    # how many bins, summing on both sides, should we cut?
    ncuts = cut / step
    # remove as many bins as we can by shifting
    nshifts = int(ncuts // 2)
    if nshifts:
        rshift = _shift(arr, (0, +nshifts))
        lshift = _shift(arr, (0, -nshifts))
        arr_ = arr * ((rshift > 0) & (lshift > 0))
    else:
        arr_ = arr

    # fix borders
    decimal = (ncuts - 2 * nshifts)

    # this is why we only accept integer array inputs.
    _lborder_mask = (arr_ - _shift(arr_, (0, +1)) > 0)
    _rborder_mask = (arr_ - _shift(arr_, (0, -1)) > 0)
    lborder_mask = _lborder_mask & (~_rborder_mask)
    rborder_mask = _rborder_mask & (~_lborder_mask)
    cborder_mask = _lborder_mask & _rborder_mask

    return (
            arr_ +
            (1 - decimal / 2) * lborder_mask - arr_ * lborder_mask +
            (1 - decimal / 2) * rborder_mask - arr_ * rborder_mask +
            (1 - decimal) * cborder_mask - arr_ * cborder_mask
    )


def shadowgram(camera: CodedMaskCamera, source_position: tuple[int, int]) -> np.array:
    """Fast computation of detector shadowgram for a point source.

    Shifts and crops the mask pattern to generate the detector response.
    The output is unnormalized and binary (ones and zeros).

    Does NOT apply bulk correction, nor nomalize.

    Args:
        camera: CodedMaskCamera instance containing mask pattern and geometry.
        source_position: Tuple of (i,j) integers specifying source position in sky coordinates,
            where (sky_shape[0]//2, sky_shape[1]//2) is the center.

    Returns:
        np.array: Binary detector shadowgram of shape (detector_height, detector_width).
    """
    i, j = source_position
    n, m = camera.sky_shape
    shift_i, shift_j = (n // 2 - i), (m // 2 - j)
    shifted_mask = _shift(camera.mask, (shift_i, shift_j))
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    return shifted_mask[i_min:i_max, j_min:j_max]
