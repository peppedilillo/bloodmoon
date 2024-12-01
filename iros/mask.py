from bisect import bisect_left
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

from astropy.io.fits.fitsrec import FITS_rec
import numpy as np
from scipy.signal import correlate
from scipy.stats import binned_statistic_2d

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


def _fold(
    ml: FITS_rec,
    mask_bins: Bins2D,
) -> np.array:
    """Convert mask data from FITS record to 2D binned array.

    Args:
        ml: FITS record containing mask data
        mask_bins: Binning structure for the mask

    Returns:
        2D array containing binned mask data
    """
    return binned_statistic_2d(ml["X"], ml["Y"], ml["VAL"], statistic="max", bins=[mask_bins.x, mask_bins.y])[0].T


def _bisect_interval(a: np.array, start: float, stop: float):
    """
    Given a monotonically increasing array of floats and a float interval (start, stop)
    in it, returns the indices of the smallest sub array containing the interval.

    Args:
        a (np.array): A monotonically increasing array of floats.
        start (float): The lower bound of the interval. Must be greater than or equal to
            the first element of the array.
        stop (float): The upper bound of the interval. Must be less than or equal to
            the last element of the array.

    Returns:
        tuple: A pair of integers (left_idx, right_idx) where:
            - left_idx is the index of the largest value in 'a' that is less than or equal to 'start'
            - right_idx is the index of the smallest value in 'a' that is greater than or equal to 'stop'

    Raises:
        ValueError: If the interval [start, stop] is not contained within the array bounds
            or if the input array is not monotonically increasing.
    """
    if not (start >= a[0] and stop <= a[-1]):
        raise ValueError("The interval isn't contained in the input array")
    if not np.all(np.diff(a) > 0):
        raise ValueError("The array isn't monotonically increasing")
    return bisect_right(a, start) - 1, bisect_left(a, stop)


"""
last one 
i swear

　　　 　　／＞　　 フ
　　　 　　| 　_　 _ l
　 　　 　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿_ヽ_)__) """
@dataclass(frozen=True)
class CodedMaskCamera:
    """Dataclass containing a coded mask camera system.

    Handles mask pattern, detector geometry, and related calculations for coded mask imaging.

    Args:
        mdl: Mask data loader object containing mask and detector specifications
        upscale_f: Tuple of upscaling factors for x and y dimensions

    Raises:
        ValueError: If detector plane is larger than mask or if upscale factors are not positive
    """

    mdl: MaskDataLoader
    upscale_f: UpscaleFactor

    def _bins_mask(
        self,
        upscale_f: UpscaleFactor,
    ) -> Bins2D:
        """Generate binning structure for mask with given upscale factors."""
        return Bins2D(
            _bin(self.mdl["mask_minx"], self.mdl["mask_maxx"], self.mdl["mask_deltax"] / upscale_f.x),
            _bin(self.mdl["mask_miny"], self.mdl["mask_maxy"], self.mdl["mask_deltay"] / upscale_f.y),
        )

    @property
    def bins_mask(self) -> Bins2D:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)

    def _bins_detector(self, upscale_f: UpscaleFactor) -> Bins2D:
        """Generate binning structure for detector with given upscale factors."""
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        return Bins2D(
            _bin(bins.x[xmin], bins.x[xmax], self.mdl["mask_deltax"] / upscale_f.x),
            _bin(bins.y[ymin], bins.y[ymax], self.mdl["mask_deltay"] / upscale_f.y),
        )

    @property
    def bins_detector(self) -> Bins2D:
        """Binning structure for the detector."""
        return self._bins_detector(self.upscale_f)

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
    def bins_sky(self) -> Bins2D:
        return self._bins_sky(self.upscale_f)

    @cached_property
    def mask(self) -> np.array:
        """2D array representing the coded mask pattern."""
        return _upscale(_fold(self.mdl.mask, self._bins_mask(UpscaleFactor(1, 1))).astype(int), self.upscale_f)

    @cached_property
    def decoder(self) -> np.array:
        """2D array representing the mask pattern used for decoding."""
        return _upscale(_fold(self.mdl.decoder, self._bins_mask(UpscaleFactor(1, 1))), self.upscale_f)

    @cached_property
    def bulk(self) -> np.array:
        """2D array representing the bulk (sensitivity) array of the mask."""
        framed_bulk = _fold(self.mdl.bulk, self._bins_mask(UpscaleFactor(1, 1)))
        framed_bulk[~np.isclose(framed_bulk, np.zeros_like(framed_bulk))] = 1
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        return _upscale(framed_bulk, self.upscale_f)[ymin:ymax, xmin:xmax]

    @cached_property
    def balancing(self) -> np.array:
        """2D array representing the correlation between decoder and bulk patterns."""
        return correlate(self.decoder, self.bulk, mode="full")

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        xmin = np.floor(self.mdl["detector_minx"] / (self.mdl["mask_deltax"] / self.upscale_f.x))
        xmax = np.ceil(self.mdl["detector_maxx"] / (self.mdl["mask_deltax"] / self.upscale_f.x))
        ymin = np.floor(self.mdl["detector_miny"] / (self.mdl["mask_deltay"] / self.upscale_f.y))
        ymax = np.ceil(self.mdl["detector_maxy"] / (self.mdl["mask_deltay"] / self.upscale_f.y))
        return int(ymax - ymin), int(xmax - xmin)

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


def fetch_camera(
    mask_filepath: str | Path,
    upscale_f: tuple[int, int] = (1, 1),
) -> CodedMaskCamera:
    """
    An interface to CodedMaskCamera.

    Args:
        mask_filepath: a str or a path object pointing to the mask filepath
        upscale_f: the upscaling factor in x and y coordinates

    Returns: a CodedMaskCamera object.

    """
    # guarantee that the bisect operation just below are performed on well suited arrays.
    mdl = MaskDataLoader(mask_filepath)

    if not (
        # fmt: off
        mdl["detector_minx"] >= mdl["mask_minx"] and
        mdl["detector_maxx"] <= mdl["mask_maxx"] and
        mdl["detector_miny"] >= mdl["mask_miny"] and
        mdl["detector_maxy"] <= mdl["mask_maxy"]
        # fmt: on
    ):
        raise ValueError("Detector plane is larger than mask.")

    if not (upscale_f[0] > 0 and upscale_f[1] > 0):
        raise ValueError("Upscale factors must be positive integers.")

    return CodedMaskCamera(mdl, UpscaleFactor(*upscale_f))


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
    """Reconstruct balanced sky image from detector counts using cross-correlation.

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
    var_bal = (
        var + np.square(camera.balancing) * sum_det / np.square(sum_bulk) ** 2 - 2 * cc * camera.balancing / sum_bulk
    )
    return cc_bal, var_bal


def psf(camera: CodedMaskCamera) -> np.array:
    """Calculate Point Spread Function (PSF) of the coded mask system.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns

    Returns:
        2D array representing the system's PSF
    """
    return correlate(camera.mask, camera.decoder, mode="same")


def count(camera, data):
    """Create 2D histogram of detector counts from event data.

    Args:
        camera: CodedMaskCamera object containing detector binning
        data: Array of event data with `X` and `Y` coordinates

    Returns:
        2D array of binned detector counts
    """
    bins = camera.bins_detector
    counts, *_ = np.histogram2d(data["Y"], data["X"], bins=[bins.y, bins.x])
    return counts
