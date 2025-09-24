"""
Core functionality for coded mask imaging analysis.

This module implements the primary algorithms for:
- Shadowgram generation and encoding
- Image reconstruction and decoding
- Point spread function calculation
- Source detection and counting
"""

from bisect import bisect_left
from bisect import bisect_right
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.signal import correlate
from toolz.curried import get_in

from .coords import pos2shift
from .images import _interp
from .images import _unframe
from .images import _upscale
from .images import argmax
from .io import load_from_fits
from .io import validate_fits
from .types import BinsRectangular
from .types import UpscaleFactor


def _bisect_interval(
    a: npt.NDArray,
    start: float,
    stop: float,
) -> tuple[int, int]:
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

    Notes:
        - To improve performance the function will not check for array monotonicity.
    """
    if not (start >= a[0] and stop <= a[-1]):
        raise ValueError(f"Interval ({start:+.2f}, {stop:+.2f}) out bounds input array ({a[0]:+.2f}, {a[-1]:+.2f})")
    return bisect_right(a, start) - 1, bisect_left(a, stop)


@dataclass(frozen=True)
class CodedMaskSpecs:
    detector_minx: float
    detector_maxx: float
    detector_miny: float
    detector_maxy: float
    mask_deltax: float
    mask_deltay: float
    mask_thickness: float
    mask_minx: float
    mask_miny: float
    mask_maxx: float
    mask_maxy: float
    slit_deltax: float
    slit_deltay: float
    # The mask-detector distance can be defined in several ways:
    # - Distance between detector top and mask bottom
    # - Distance between detector top and mask top
    # - Distance between detector top and mask midpoint
    # The key requirement is consistency: whichever definition is used here
    # must match the correction applied in vignetting (see comment in
    # `apply_vignetting`).
    # We define the distance as the separation between detector top and mask top.
    # This choice is empirically motivated: testing showed this definition yields
    # the best results, though we don't fully understand why. Note that this
    # differs from the data convention, where mask-detector distance refers to
    # the separation between detector top and mask bottom.
    mask_detector_distance: float


"""
last one 
i swear

　　　 　　／＞　　 フ
　　　 　　| 　_　 _`
　 　　 　／` ミ＿xノ
　　 　 /　　　 　 |
　　　 /　 ヽ　　 ﾉ
　 　 │　　|　|　|
　／￣|　　 |　|　|
　| (￣ヽ＿_ヽ_)__) """


@dataclass(frozen=True)
class CodedMaskCamera:
    """
    Dataclass containing a coded mask camera system.

    Handles mask pattern, detector geometry, and related calculations for coded mask imaging.
    Uses callable thunks for lazy loading of mask data to maintain hashability and performance.

    Args:
        get_mask: Callable that returns mask pattern as 2D array
        get_decoder: Callable that returns decoder pattern as 2D array
        get_bulk: Callable that returns bulk pattern as 2D array
        specs: CodedMaskSpecs containing geometric parameters and dimensions
        upscale_f: Tuple of upscaling factors for x and y dimensions

    Raises:
        ValueError: If detector plane is larger than mask or if upscale factors are not positive
    """

    # note that we are taking thunks here, rather than the actual array.
    # two reasons for this:
    #   1. python functions are hashable and using thunks keeps camera objects hashable themselves.
    #   2. reading data may take time. thunk delays this expensive operation.
    get_mask: Callable[[], npt.NDArray]
    get_decoder: Callable[[], npt.NDArray]
    get_bulk: Callable[[], npt.NDArray]
    specs: CodedMaskSpecs
    upscale_f: UpscaleFactor

    @cached_property
    def shape_detector(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        return len(self.bins_detector.y) - 1, len(self.bins_detector.x) - 1

    @cached_property
    def shape_mask(self) -> tuple[int, int]:
        """Shape of the mask array (rows, columns)."""
        # there is no need for this since we can just `mask.shape` but since we have the other already..
        return len(self.bins_mask.y) - 1, len(self.bins_mask.x) - 1


    @cached_property
    def shape_sky(self) -> tuple[int, int]:
        """Shape of the reconstructed sky image (rows, columns)."""
        n, m = self.shape_detector
        o, p = self.shape_mask
        return n + o - 1, m + p - 1

    def _bins_mask(
        self,
        upscale_f: UpscaleFactor,
    ) -> BinsRectangular:
        """Returns bins for mask with given upscale factors."""
        l, r = self.specs.mask_minx, self.specs.mask_maxx
        b, t = self.specs.mask_miny, self.specs.mask_maxy
        xsteps = int((r - l) / (self.specs.mask_deltax / upscale_f.x)) + 1
        ysteps = int((t - b) / (self.specs.mask_deltay / upscale_f.y)) + 1
        return BinsRectangular(np.linspace(l, r, xsteps), np.linspace(b, t, ysteps))

    @cached_property
    def bins_mask(self) -> BinsRectangular:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)

    def _bins_detector(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """
        Returns bins for detector with given upscale factors.
        The detector bins are aligned to the mask bins.
        To guarantee this, we may need to extend the detector bin a bit over the mask.

         ◀────────────mask────────────▶
         │    │    │    │    │    │    │
         └────┴────┴────┴────┴────┴────┘
        -3   -2   -1    0    +1   +2   +3
              ┌─┬──┬────┬────┬──┬─┐
              │    │    │    │    │
                │               │
                ◀───detector────▶
                │               │
           detector_min   detector_max
        """
        bins = self._bins_mask(upscale_f)
        jmin, jmax = _bisect_interval(bins.x, self.specs.detector_minx, self.specs.detector_maxx)
        imin, imax = _bisect_interval(bins.y, self.specs.detector_miny, self.specs.detector_maxy)
        return BinsRectangular(self.bins_mask.x[jmin : jmax + 1], self.bins_mask.y[imin : imax + 1])

    @cached_property
    def bins_detector(self) -> BinsRectangular:
        """Binning structure for the detector."""
        return self._bins_detector(self.upscale_f)

    def _bins_sky(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """
        Returns bins for the reconstructed sky image.cd
        While the mask and detector bins are aligned, the sky-bins are not.

            │    │    │    │    │    │    │
            ◀────┴────┴──mask───┴────┴───▶┘
            0    1    2    3    4    5    6

                      │    │    │
                      ◀───det───▶
                      0    1    2

         │    │    │    │     │    │    │    │
         ◀────┴────┴────┴─sky─┴────┴────┴────▶
         0    1    2    3     4    5    6    7
        """
        binsd, binsm = self._bins_detector(upscale_f), self._bins_mask(upscale_f)
        xstep, ystep = binsm.x[1] - binsm.x[0], binsm.y[1] - binsm.y[0]
        return BinsRectangular(
            np.linspace(
                binsd.x[0] + binsm.x[0] + xstep / 2,
                binsd.x[-1] + binsm.x[-1] - xstep / 2,
                self.shape_sky[1] + 1,
            ),
            np.linspace(
                binsd.y[0] + binsm.y[0] + ystep / 2,
                binsd.y[-1] + binsm.y[-1] - ystep / 2,
                self.shape_sky[0] + 1,
            ),
        )

    @cached_property
    def bins_sky(self) -> BinsRectangular:
        """Returns bins for the sky-shift domain"""
        return self._bins_sky(self.upscale_f)

    @cached_property
    def mask(self) -> npt.NDArray:
        """2D array representing the coded mask pattern."""
        return _upscale(self.get_mask(), *self.upscale_f)

    @cached_property
    def decoder(self) -> npt.NDArray:
        """2D array representing the mask pattern used for decoding."""
        return _upscale(self.get_decoder(), *self.upscale_f)

    @cached_property
    def bulk(self) -> npt.NDArray:
        """2D array representing the bulk (sensitivity) array of the mask."""
        bulk = self.get_bulk()
        bulk[~np.isclose(bulk, np.zeros_like(bulk))] = 1
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.specs.detector_minx, self.specs.detector_maxx)
        ymin, ymax = _bisect_interval(bins.y, self.specs.detector_miny, self.specs.detector_maxy)
        # why `xmin: xmax` rather than `xmin: xmax + 1`?
        # `bins.x[xmin:xmax + 1]` is the smallest subarray of `bins.x` spanning `det_minx` and `det_maxx`
        # the bin edges number of the subarray is `xmax - xmin + 1`.
        # the number of matrix elements in the subarray is `xmax - xmin + 1 - 1 == xmax - xmin`
        return _upscale(bulk, *self.upscale_f)[ymin:ymax, xmin:xmax]

    @cached_property
    def balancing(self) -> npt.NDArray:
        """2D array representing the correlation between decoder and bulk patterns."""
        return correlate(self.decoder, self.bulk, mode="full")


def codedmask(
    mask_filepath: str | Path,
    upscale_x: int = 1,
    upscale_y: int = 1,
) -> CodedMaskCamera:
    """
    Create a CodedMaskCamera from FITS file data.

    Loads mask patterns and specifications from a FITS file and creates a
    CodedMaskCamera instance with lazy-loaded array data.

    Args:
        mask_filepath: Path to the mask FITS file
        upscale_x: Upscaling factor for x direction (default: 1)
        upscale_y: Upscaling factor for y direction (default: 1)

    Returns:
        CodedMaskCamera object containing mask patterns and specifications

    Raises:
        ValueError: If detector plane is larger than mask or upscale factors are invalid
        NotImplementedError: If mask_filepath is not a valid FITS file
    """
    if validate_fits(mask_filepath):
        get_mask, get_decoder, get_bulk, specs_dict = load_from_fits(mask_filepath)
        specs = CodedMaskSpecs(**specs_dict)
        if not (
            # fmt: off
                specs.detector_minx >= specs.mask_minx and
                specs.detector_maxx <= specs.mask_maxx and
                specs.detector_miny >= specs.mask_miny and
                specs.detector_maxy <= specs.mask_maxy
            # fmt: on
        ):
            raise ValueError("Detector plane is larger than mask.")

        if not ((isinstance(upscale_x, int) and upscale_x > 0) and (isinstance(upscale_y, int) and upscale_y > 0)):
            raise ValueError("Upscale factors must be positive integers.")

        return CodedMaskCamera(
            get_mask=get_mask,
            get_decoder=get_decoder,
            get_bulk=get_bulk,
            specs=specs,
            upscale_f=UpscaleFactor(x=upscale_x, y=upscale_y),
        )
    raise NotImplementedError("Only reading masks from fits file is supported.")


def encode(
    camera: CodedMaskCamera,
    sky: np.ndarray,
) -> npt.NDArray:
    """
    Generate detector shadowgram from sky image through coded mask.

    Args:
        camera: CodedMaskCamera object containing mask pattern
        sky: 2D array representing sky image

    Returns:
        2D array representing detector shadowgram
    """
    unnormalized_shadowgram = correlate(camera.mask, sky, mode="valid")
    return unnormalized_shadowgram


def decode(
    camera: CodedMaskCamera,
    detector: npt.NDArray,
) -> npt.NDArray:
    """
    Reconstruct balanced sky image from detector counts using cross-correlation.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns
        detector: 2D array of detector counts

    Returns:
        Balanced cross-correlation sky image
            - Variance map of the reconstructed sky image
    """
    cc = correlate(camera.decoder, detector, mode="full")
    sum_det, sum_bulk = map(np.sum, (detector, camera.bulk))
    cc_bal = cc - camera.balancing * sum_det / sum_bulk
    return cc_bal


def variance(
    camera: CodedMaskCamera,
    detector: npt.NDArray,
) -> npt.NDArray:
    """
    Reconstruct balanced sky variance from detector counts.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns
        detector: 2D array of detector counts

    Returns:
        Variance map of the reconstructed sky image
    """
    cc = correlate(camera.decoder, detector, mode="full")
    var = correlate(np.square(camera.decoder), detector, mode="full")
    sum_det, sum_bulk = map(np.sum, (detector, camera.bulk))
    var_bal = var + np.square(camera.balancing) * sum_det / np.square(sum_bulk) - 2 * cc * camera.balancing / sum_bulk
    return var_bal


def snratio(
    sky: npt.NDArray,
    var: npt.NDArray,
) -> npt.NDArray:
    """
    Calculate signal-to-noise ratio from sky signal and variance arrays.

    Args:
        sky: Array containing sky signal values.
        var: Array containing variance values. Negative values are clipped to 0.

    Returns:
        NDArray: Signal-to-noise ratio calculated as sky/sqrt(variance).

    Notes:
        - Variance's boundary frames with elements close to zero are replaced with infinity.
        - Variance's minimum is clipped at 0 if any negative value are present in the array.
    """
    variance_clipped = np.clip(var, a_min=0.0, a_max=None) if np.any(var < 0) else var
    variance_unframed = _unframe(variance_clipped, value=np.inf)
    return sky / np.sqrt(variance_unframed)


def psf(camera: CodedMaskCamera) -> npt.NDArray:
    """
    Calculate Point Spread Function (PSF) of the coded mask system.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns

    Returns:
        2D array representing the system's PSF
    """
    return correlate(camera.mask, camera.decoder, mode="same")


def count(
    camera: CodedMaskCamera,
    data: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Create 2D histogram of detector counts from event data.

    Args:
        camera: CodedMaskCamera object containing detector binning
        data: Array of event data with `X` and `Y` coordinates

    Returns:
        2D array of binned detector counts
    """
    bins = camera.bins_detector
    counts, *_ = np.histogram2d(data["Y"], data["X"], bins=[bins.y, bins.x])
    # - we mask the binned detector counts for the camera bulk to remove
    #   those photons that fall in the dead zone
    # PN: we do NOT multiply for the bulk, since we do not want to modify
    #     the observed data (the bulk represents the sensitivity profile
    #     of the detector and generally has values in the [0, 1] interval)
    bulk_mask = camera.bulk > 0
    return counts * bulk_mask, bins


def _detector_footprint(camera: CodedMaskCamera) -> tuple[int, int, int, int]:
    """Shadowgram helper function."""
    bins_detector = camera.bins_detector
    bins_mask = camera.bins_mask
    i_min, i_max = _bisect_interval(bins_mask.y, bins_detector.y[0], bins_detector.y[-1])
    j_min, j_max = _bisect_interval(bins_mask.x, bins_detector.x[0], bins_detector.x[-1])
    return i_min, i_max, j_min, j_max


def cutout(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
    fx: float = 1,
    fy: float = 1,
) -> tuple[tuple, BinsRectangular]:
    """
    Returns a cutout of sky centered around `pos` with length multiple of the mask slit size.

    Args:
        camera: a CodedMaskCameraObject.
        pos: the (row, col) indexes of the slice center.

    Returns:
        A tuple of the slice value (length n) and its bins (length n + 1).
    """
    bins = camera.bins_sky
    sx, sy = pos2shift(camera, *pos)
    min_i, max_i = _bisect_interval(
        bins.y,
        max(sy - camera.specs.slit_deltay * (fy / 2), bins.y[0]),
        min(sy + camera.specs.slit_deltay * (fy / 2), bins.y[-1]),
    )
    min_j, max_j = _bisect_interval(
        bins.x,
        max(sx - camera.specs.slit_deltax * (fx / 2), bins.x[0]),
        min(sx + camera.specs.slit_deltax * (fx / 2), bins.x[-1]),
    )
    return (min_i, max_i, min_j, max_j), BinsRectangular(
        x=bins.x[min_j : max_j + 1],
        y=bins.y[min_i : max_i + 1],
    )


def interpmax(
    camera: CodedMaskCamera,
    pos,
    sky,
    interp_f: UpscaleFactor = UpscaleFactor(9, 9),
) -> tuple[float, float]:
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
    (min_i, max_i, min_j, max_j), bins = cutout(camera, pos, 4, 1)
    tile_interp, bins_fine = _interp(sky[min_i:max_i, min_j:max_j], bins, interp_f)
    max_tile_i, max_tile_j = argmax(tile_interp)
    return float(bins_fine.x[max_tile_j]), float(bins_fine.y[max_tile_i])
