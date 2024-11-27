from typing import NamedTuple, Callable, Optional
from bisect import bisect_left, bisect_right
from functools import cached_property

import numpy as np
from astropy.io.fits.fitsrec import FITS_rec
from scipy.stats import binned_statistic_2d
from scipy.signal import correlate


from .io import MaskDataLoader


class Bins2D(NamedTuple):
    """Two-dimensional binning structure for mask and detector coordinates.

    Args:
        x: Array of x-coordinate bin edges
        y: Array of y-coordinate bin edges
    """
    x: np.array
    y: np.array


def _bin(
    start: float,
    stop: float,
    step: float,
) -> tuple[np.array, np.array]:
    """Returns equally spaced points between start and stop, included.

    Args:
        start: Minimum x-coordinate
        stop: Minimum y-coordinate
        step: Maximum x-coordinate

    Returns:
        Bin edges array.
    """
    return np.linspace(start, stop, int((stop - start) / step) + 1)


class UpscaleFactor(NamedTuple):
    """Upscaling factors for x and y dimensions.

    Args:
        x: Upscaling factor for x dimension
        y: Upscaling factor for y dimension
    """
    x: int
    y: int


def _upscale(
    m: np.ndarray,
    upscale_f: UpscaleFactor,
) -> np.ndarray:
    """Upscale a 2D array by repeating elements along each axis.

    Args:
        m: Input 2D array
        upscale_f: UpscaleFactor containing scaling factors for each dimension

    Returns:
        Upscaled array with dimensions multiplied by respective scaling factors
    """
    fx, fy = upscale_f.x, upscale_f.y
    # VERY careful here, the next is not a typo.
    # if i'm upscaling by (2, 1). it means i'm doubling the elements
    # over the x direction, while keeping the same element over the y direction.
    # this means doubling the number of columns in the mask array, while
    # keeping the number of rows the same.
    m = np.repeat(m, fy, axis=0)
    m = np.repeat(m, fx, axis=1)
    return m


def _fold(ml: FITS_rec, mask_bins: Bins2D,) -> np.array:
    """Convert mask data from FITS record to 2D binned array.

    Args:
        ml: FITS record containing mask data
        mask_bins: Binning structure for the mask

    Returns:
        2D array containing binned mask data
    """
    return binned_statistic_2d(ml["X"], ml["Y"], ml["VAL"], statistic="max", bins=[mask_bins.x, mask_bins.y])[0].T


def _resize(a: np.array, b: np.array,) -> np.array:
    """Resizes the `a` matrix to the size of the smallest submatrix of `b` with non-zero border"

    Args:
        a: Array to be resized
        b: Reference array defining the non-zero border

    Returns:
        Resized submatrix of input array
    """
    non_zero_rows = np.where(b.any(axis=1))[0]
    non_zero_cols = np.where(b.any(axis=0))[0]
    row_start, row_end = non_zero_rows[0], non_zero_rows[-1] + 1
    col_start, col_end = non_zero_cols[0], non_zero_cols[-1] + 1
    submatrix = a[row_start:row_end, col_start:col_end]
    return submatrix


class CodedMaskCamera:
    """Class representing a coded mask camera system.

    Handles mask pattern, detector geometry, and related calculations for coded mask imaging.

    Args:
        mask_data: Loader object containing mask and detector specifications
        upscale_f: Tuple of upscaling factors for x and y dimensions

    Raises:
        ValueError: If detector plane is larger than mask or if upscale factors are not positive
    """
    def __init__(
        self,
        mask_data: MaskDataLoader,
        upscale_f: tuple[int, int] = (1, 1),
    ):
        # guarantee that the bisect operation just below are performed on well suited arrays.
        if not (
            # fmt: off
            mask_data["detector_minx"] >= mask_data["mask_minx"] and
            mask_data["detector_maxx"] <= mask_data["mask_maxx"] and
            mask_data["detector_miny"] >= mask_data["mask_miny"] and
            mask_data["detector_maxy"] <= mask_data["mask_maxy"]
            # fmt: on
        ):
            raise ValueError("Detector plane is larger than mask.")
        self.mdl = mask_data

        if not (upscale_f[0] > 0 and upscale_f[1] > 0):
            raise ValueError("Upscale factors must be positive integers.")
        self.upscale_f = UpscaleFactor(*upscale_f)

    def _bins_mask(self, upscale_f: UpscaleFactor, ) -> Bins2D:
        """Generate binning structure for mask with given upscale factors."""
        return Bins2D(
            _bin(self.mdl["mask_minx"], self.mdl["mask_maxx"], self.mdl["mask_deltax"] / upscale_f.x),
            _bin(self.mdl["mask_miny"], self.mdl["mask_maxy"], self.mdl["mask_deltay"] / upscale_f.y),
        )

    @property
    def bins_mask(self) -> dict[str, np.array]:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)._asdict()

    def _bins_detector(self, upscale_f: UpscaleFactor) -> Bins2D:
        """Generate binning structure for detector with given upscale factors."""
        bins = self._bins_mask(self.upscale_f)
        return Bins2D(
            _bin(
                bins.x[bisect_right(bins.x, self.mdl["detector_minx"]) - 1],
                bins.x[bisect_left(bins.x, self.mdl["detector_maxx"])],
                self.mdl["mask_deltax"] / upscale_f.x,
            ),
            _bin(
                bins.y[bisect_right(bins.y, self.mdl["detector_miny"]) - 1],
                bins.y[bisect_left(bins.y, self.mdl["detector_maxy"])],
                self.mdl["mask_deltay"] / upscale_f.y,
            ),
        )

    @property
    def bins_detector(self) -> dict[str, np.array]:
        """Binning structure for the detector."""
        return self._bins_detector(self.upscale_f)._asdict()

    def _bins_sky(self, upscale_f: UpscaleFactor) -> Bins2D:
        """Binning structure for the reconstructed sky image."""
        o, p = self.mask_shape
        bins = self._bins_detector(upscale_f)
        binstep, nbins = bins.x[1] - bins.x[0], o // 2 - 1
        xbins = np.concatenate(
            (
                np.linspace(bins.x[0] - (nbins + 1) * binstep, bins.x[-0] - binstep, nbins + 1),
                bins.x,
                np.linspace(bins.x[-1] + binstep, bins.x[-1] + nbins * binstep, nbins),
            )
        )
        binstep, nbins = bins.y[1] - bins.y[0], p // 2 - 1
        ybins = np.concatenate(
            (
                np.linspace(bins.y[0] - (nbins + 1) * binstep, bins.y[-0] - binstep, nbins + 1),
                bins.y,
                np.linspace(bins.y[-1] + binstep, bins.y[-1] + nbins * binstep, nbins),
            )
        )
        return Bins2D(x=xbins, y=ybins)

    @property
    def bins_sky(self) -> dict[str, np.array]:
        return self._bins_sky(self.upscale_f)._asdict()

    @cached_property
    def mask(self) -> np.array:
        """2D array representing the coded mask pattern."""
        return _upscale(_fold(self.mdl.get_mask_data(), self._bins_mask(UpscaleFactor(1, 1))).astype(int), self.upscale_f)

    @cached_property
    def decoder(self) -> np.array:
        """2D array representing the mask pattern used for decoding."""
        return _upscale(_fold(self.mdl.get_decoder_data(), self._bins_mask(UpscaleFactor(1, 1))).astype(int), self.upscale_f)

    @cached_property
    def bulk(self) -> np.array:
        """2D array representing the bulk (sensitivity) array of the mask."""
        framed_bulk = _fold(self.mdl.get_bulk_data(), self._bins_mask(UpscaleFactor(1, 1)))
        # this is a temporary fix for the fact that the bulk may contain non-integer value at border
        framed_bulk[~np.isclose(framed_bulk, np.zeros_like(framed_bulk))] = 1
        return _upscale(_resize(framed_bulk, framed_bulk), self.upscale_f)

    @cached_property
    def balancing(self) -> np.array:
        """2D array representing the correlation between decoder and bulk patterns."""
        return correlate(self.decoder, self.bulk, mode="full")

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        bins = self._bins_mask(self.upscale_f)
        xlen = bins.x[bisect_left(bins.x, self.mdl["detector_maxx"])] - bins.x[bisect_right(bins.x, self.mdl["detector_minx"]) - 1]
        ylen = bins.y[bisect_left(bins.y, self.mdl["detector_maxy"])] - bins.y[bisect_right(bins.y, self.mdl["detector_miny"]) - 1]
        return (
            int(ylen / (self.mdl["mask_deltay"] / self.upscale_f.y)),
            int(xlen / (self.mdl["mask_deltax"] / self.upscale_f.x)),
        )

    @property
    def mask_shape(self) -> tuple[int, int]:
        """Shape of the mask array (rows, columns)."""
        return (
            int((self.mdl["mask_maxy"] - self.mdl["mask_miny"]) / (self.mdl["mask_deltay"] / self.upscale_f.y)),
            int((self.mdl["mask_maxx"] - self.mdl["mask_minx"]) / (self.mdl["mask_deltax"] / self.upscale_f.x)),
        )

    @property
    def sky_shape(self) -> tuple[int, int]:
        """Shape of the reconstructed sky image (rows, columns)."""
        n, m = self.detector_shape
        o, p = self.mask_shape
        return n + o - 1, m + p - 1


def encode(camera: CodedMaskCamera, sky: np.ndarray) -> np.array:
    """Generate detector shadowgram from sky image through coded mask.

    Args:
        camera: CodedMaskCamera object containing mask pattern
        sky: 2D array representing sky image

    Returns:
        2D array representing detector shadowgram
    """
    unnormalized_shadowgram = correlate(camera.mask, sky, mode="valid")
    return unnormalized_shadowgram


def decode(camera: CodedMaskCamera, detector: np.array) -> tuple[np.array, np.array]:
    """Reconstruct sky image from detector counts using cross-correlation.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns
        detector: 2D array of detector counts

    Returns:
        Tuple containing:
            - Balanced cross-correlation sky image
            - Variance map of the reconstructed sky image
    """
    cc = correlate(camera.decoder, detector, mode="full")
    var = correlate(np.square(camera.decoder), detector, mode="full")
    sum_det, sum_bulk = map(np.sum, (detector, camera.bulk))
    cc_bal = cc - camera.balancing * sum_det / sum_bulk
    var_bal = var + np.square(camera.balancing) * sum_det / np.square(sum_bulk) ** 2 - 2 * cc * camera.balancing / sum_bulk
    return cc_bal, var_bal


def psf(camera: CodedMaskCamera) -> np.array:
    """Calculate Point Spread Function (PSF) of the coded mask system.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns

    Returns:
        2D array representing the system's PSF
    """
    return correlate(camera.mask, camera.decoder, mode="same")


def compose(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, Callable[[int, int], tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]]]:
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
    a_embedding = np.pad(a, pad_width=((delta, delta), (0, 0)))
    b_embedding = np.pad(np.rot90(b, k=-1), pad_width=((0, 0), (delta, delta)))
    composed = a_embedding + b_embedding

    def _rotb2b(i, j):
        return  mind - 1 - j, i

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


def maximize(composed: np.ndarray, f: Callable) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """
    Maximize the composed image and returns corresponding locations on the component images.

    Args:
        composed:
        f:

    Returns:

    """
    composed_max = np.unravel_index(np.argmax(composed), composed.shape)
    return f(*composed_max)
