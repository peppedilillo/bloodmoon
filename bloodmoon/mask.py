"""
Core functionality for coded mask imaging analysis.

This module implements the primary algorithms for:
- Shadowgram generation and encoding
- Image reconstruction and decoding
- Point spread function calculation
- Source detection and counting
- Vignetting and detector effects modeling

These components form the foundation of the WFM data analysis pipeline.
"""

from bisect import bisect
from bisect import bisect_left
from bisect import bisect_right
from dataclasses import dataclass
from functools import cache
from functools import cached_property
from pathlib import Path

from astropy.io.fits.fitsrec import FITS_rec
import numpy as np
import numpy.typing as npt
from scipy.signal import convolve
from scipy.signal import correlate
from scipy.stats import binned_statistic_2d

from .images import _erosion
from .images import _interp
from .images import _rbilinear_relative
from .images import _shift
from .images import _unframe
from .images import argmax
from .images import upscale
from .io import MaskDataLoader
from .types import BinsRectangular
from .types import UpscaleFactor


def _bin(
    start: float,
    stop: float,
    step: float,
) -> npt.NDArray:
    """Returns equally spaced points between start and stop, included.

    Args:
        start: Minimum x-coordinate
        stop: Minimum y-coordinate
        step: Maximum x-coordinate

    Returns:
        Bin edges array.
    """
    return np.linspace(start, stop, int((stop - start) / step) + 1)


def _fold(
    ml: FITS_rec,
    mask_bins: BinsRectangular,
) -> npt.NDArray:
    """Convert mask data from FITS record to 2D binned array.

    Args:
        ml: FITS record containing mask data
        mask_bins: Binning structure for the mask

    Returns:
        2D array containing binned mask data
    """
    return binned_statistic_2d(ml["X"], ml["Y"], ml["VAL"], statistic="max", bins=[mask_bins.x, mask_bins.y])[0].T


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

    @property
    def specs(self) -> dict:
        """Returns a dictionary of mask parameters useful for image reconstruction."""
        return self.mdl.specs

    def _bins_mask(
        self,
        upscale_f: UpscaleFactor,
    ) -> BinsRectangular:
        """Generate binning structure for mask with given upscale factors."""
        return BinsRectangular(
            _bin(self.mdl["mask_minx"], self.mdl["mask_maxx"], self.mdl["mask_deltax"] / upscale_f.x),
            _bin(self.mdl["mask_miny"], self.mdl["mask_maxy"], self.mdl["mask_deltay"] / upscale_f.y),
        )

    @cached_property
    def bins_mask(self) -> BinsRectangular:
        """Binning structure for the mask pattern."""
        return self._bins_mask(self.upscale_f)

    def _bins_detector(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """Generate binning structure for detector with given upscale factors."""
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        return BinsRectangular(
            _bin(bins.x[xmin], bins.x[xmax], self.mdl["mask_deltax"] / upscale_f.x),
            _bin(bins.y[ymin], bins.y[ymax], self.mdl["mask_deltay"] / upscale_f.y),
        )

    @cached_property
    def bins_detector(self) -> BinsRectangular:
        """Binning structure for the detector."""
        return self._bins_detector(self.upscale_f)

    def _bins_sky(self, upscale_f: UpscaleFactor) -> BinsRectangular:
        """Binning structure for the reconstructed sky image."""
        binsd, binsm = self._bins_detector(upscale_f), self._bins_mask(upscale_f)
        xstep, ystep = (
            binsm.x[1] - binsm.x[0],
            binsm.y[1] - binsm.y[0],
        )
        return BinsRectangular(
            np.linspace(binsd.x[0] + binsm.x[0] + xstep, binsd.x[-1] + binsm.x[-1], self.sky_shape[1] + 1),
            np.linspace(binsd.y[0] + binsm.y[0] + ystep, binsd.y[-1] + binsm.y[-1], self.sky_shape[0] + 1),
        )

    @cached_property
    def bins_sky(self) -> BinsRectangular:
        """Returns bins for the sky-shift domain"""
        return self._bins_sky(self.upscale_f)

    @cached_property
    def mask(self) -> npt.NDArray:
        """2D array representing the coded mask pattern."""
        return upscale(_fold(self.mdl.mask, self._bins_mask(UpscaleFactor(1, 1))).astype(int), *self.upscale_f)

    @cached_property
    def decoder(self) -> npt.NDArray:
        """2D array representing the mask pattern used for decoding."""
        return upscale(_fold(self.mdl.decoder, self._bins_mask(UpscaleFactor(1, 1))), *self.upscale_f)

    @cached_property
    def bulk(self) -> npt.NDArray:
        """2D array representing the bulk (sensitivity) array of the mask."""
        framed_bulk = _fold(self.mdl.bulk, self._bins_mask(UpscaleFactor(1, 1)))
        framed_bulk[~np.isclose(framed_bulk, np.zeros_like(framed_bulk))] = 1
        bins = self._bins_mask(self.upscale_f)
        xmin, xmax = _bisect_interval(bins.x, self.mdl["detector_minx"], self.mdl["detector_maxx"])
        ymin, ymax = _bisect_interval(bins.y, self.mdl["detector_miny"], self.mdl["detector_maxy"])
        return upscale(framed_bulk, *self.upscale_f)[ymin:ymax, xmin:xmax]

    @cached_property
    def balancing(self) -> npt.NDArray:
        """2D array representing the correlation between decoder and bulk patterns."""
        return correlate(self.decoder, self.bulk, mode="full")

    @cached_property
    def detector_shape(self) -> tuple[int, int]:
        """Shape of the detector array (rows, columns)."""
        xmin = np.floor(self.mdl["detector_minx"] / (self.mdl["mask_deltax"] / self.upscale_f.x))
        xmax = np.ceil(self.mdl["detector_maxx"] / (self.mdl["mask_deltax"] / self.upscale_f.x))
        ymin = np.floor(self.mdl["detector_miny"] / (self.mdl["mask_deltay"] / self.upscale_f.y))
        ymax = np.ceil(self.mdl["detector_maxy"] / (self.mdl["mask_deltay"] / self.upscale_f.y))
        return int(ymax - ymin), int(xmax - xmin)

    @cached_property
    def mask_shape(self) -> tuple[int, int]:
        """Shape of the mask array (rows, columns)."""
        return (
            int((self.mdl["mask_maxy"] - self.mdl["mask_miny"]) / (self.mdl["mask_deltay"] / self.upscale_f.y)),
            int((self.mdl["mask_maxx"] - self.mdl["mask_minx"]) / (self.mdl["mask_deltax"] / self.upscale_f.x)),
        )

    @cached_property
    def sky_shape(self) -> tuple[int, int]:
        """Shape of the reconstructed sky image (rows, columns)."""
        n, m = self.detector_shape
        o, p = self.mask_shape
        return n + o - 1, m + p - 1


def codedmask(
    mask_filepath: str | Path,
    upscale_x: int = 1,
    upscale_y: int = 1,
) -> CodedMaskCamera:
    """
    An interface to CodedMaskCamera.

    Args:
        mask_filepath: a str or a path object pointing to the mask filepath
        upscale_x: upscaling factor over the x direction
        upscale_y: upscaling factor over the y direction

    Returns:
        a CodedMaskCamera object.

    Raises:
        ValueError: for invalid upscale factors.
    """
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

    if not ((isinstance(upscale_x, int) and upscale_x > 0) and (isinstance(upscale_y, int) and upscale_y > 0)):
        raise ValueError("Upscale factors must be positive integers.")

    return CodedMaskCamera(mdl, UpscaleFactor(x=upscale_x, y=upscale_y))


def encode(
    camera: CodedMaskCamera,
    sky: np.ndarray,
) -> npt.NDArray:
    """Generate detector shadowgram from sky image through coded mask.

    Args:
        camera: CodedMaskCamera object containing mask pattern
        sky: 2D array representing sky image

    Returns:
        2D array representing detector shadowgram
    """
    unnormalized_shadowgram = correlate(camera.mask, sky, mode="valid")
    return unnormalized_shadowgram


def variance(
    camera: CodedMaskCamera,
    detector: npt.NDArray,
) -> npt.NDArray:
    """Reconstruct balanced sky variance from detector counts.

    Args:
        camera: CodedMaskCamera object containing mask and decoder patterns
        detector: 2D array of detector counts

    Returns:
        Variance map of the reconstructed sky image
    """
    cc = correlate(camera.decoder, detector, mode="full")
    var = correlate(np.square(camera.decoder), detector, mode="full")
    sum_det, sum_bulk = map(np.sum, (detector, camera.bulk))
    var_bal = (
        var + np.square(camera.balancing) * sum_det / np.square(sum_bulk) ** 2 - 2 * cc * camera.balancing / sum_bulk
    )
    return var_bal


def snratio(
    sky: npt.NDArray,
    var: npt.NDArray,
) -> npt.NDArray:
    """Calculate signal-to-noise ratio from sky signal and variance arrays.

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


def decode(
    camera: CodedMaskCamera,
    detector: npt.NDArray,
) -> npt.NDArray:
    """Reconstruct balanced sky image from detector counts using cross-correlation.

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


def psf(camera: CodedMaskCamera) -> npt.NDArray:
    """Calculate Point Spread Function (PSF) of the coded mask system.

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
    """Create 2D histogram of detector counts from event data.

    Args:
        camera: CodedMaskCamera object containing detector binning
        data: Array of event data with `X` and `Y` coordinates

    Returns:
        2D array of binned detector counts
    """
    bins = camera.bins_detector
    counts, *_ = np.histogram2d(data["Y"], data["X"], bins=[bins.y, bins.x])
    return counts, bins


def _detector_footprint(camera: CodedMaskCamera) -> tuple[int, int, int, int]:
    """Shadowgram helper function."""
    bins_detector = camera.bins_detector
    bins_mask = camera.bins_mask
    i_min, i_max = _bisect_interval(bins_mask.y, bins_detector.y[0], bins_detector.y[-1])
    j_min, j_max = _bisect_interval(bins_mask.x, bins_detector.x[0], bins_detector.x[-1])
    return i_min, i_max, j_min, j_max


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
    return float(pack_x), float(pack_y)


def strip(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
) -> tuple[tuple, BinsRectangular]:
    """
    Returns a thin slice of sky centered around `pos`.
    The strip has height 1 in the y direction and length equal to slit length in x direction.

    Args:
        camera: a CodedMaskCameraObject.
        pos: the (row, col) indeces of the slice center.

    Returns:
        A tuple of the slice value (length n) and its bins (length n + 1).
    """
    bins = camera.bins_sky
    i, j = pos
    min_i, max_i = _bisect_interval(
        bins.y,
        max(bins.y[i] - camera.mdl["slit_deltay"] / 2, bins.y[0]),
        min(bins.y[i] + camera.mdl["slit_deltay"] / 2, bins.y[-1]),
    )
    min_j, max_j = _bisect_interval(
        bins.x,
        max(bins.x[j] - camera.mdl["slit_deltax"] / 2, bins.x[0]),
        min(bins.x[j] + camera.mdl["slit_deltax"] / 2, bins.x[-1]),
    )
    return (min_i, max_i, min_j, max_j), BinsRectangular(
        x=bins.x[min_j : max_j + 1],
        y=bins.y[min_i : max_i + 1],
    )


def chop(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
) -> tuple[tuple, BinsRectangular]:
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
    packing_x, packing_y = _packing_factor(camera)
    min_i, max_i = _bisect_interval(
        bins.y,
        max(bins.y[i] - camera.mdl["slit_deltay"] / (2 * packing_y), bins.y[0]),
        min(bins.y[i] + camera.mdl["slit_deltay"] / (2 * packing_y), bins.y[-1]),
    )
    min_j, max_j = _bisect_interval(
        bins.x,
        max(bins.x[j] - camera.mdl["slit_deltax"] / (2 * packing_x), bins.x[0]),
        min(bins.x[j] + camera.mdl["slit_deltax"] / (2 * packing_x), bins.x[-1]),
    )
    return (min_i, max_i, min_j, max_j), BinsRectangular(
        x=bins.x[min_j : max_j + 1],
        y=bins.y[min_i : max_i + 1],
    )


def _interpmax(
    camera: CodedMaskCamera,
    pos,
    sky,
    interp_f: UpscaleFactor,
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
    (min_i, max_i, min_j, max_j), bins = strip(camera, pos)
    # we want to use the cubic interpolator so we take a larger window using chop
    # if strip get us a slice too small. this may happen for masks with no upscaling.
    if not (max_i - min_i > 1 and min_j - max_j > 1):
        (min_i, max_i, min_j, max_j), bins = chop(camera, pos)
    tile_interp, bins_fine = _interp(sky[min_i:max_i, min_j:max_j], bins, interp_f)
    max_tile_i, max_tile_j = argmax(tile_interp)
    return float(bins_fine.x[max_tile_j]), float(bins_fine.y[max_tile_i])


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


def _modsech(
    x: npt.NDArray,
    norm: float,
    center: float,
    alpha: float,
    beta: float,
) -> npt.NDArray:
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
    return norm / np.cosh(np.abs((x - center) / alpha) ** beta)


def psfy_wfm(x: npt.NDArray) -> npt.NDArray:
    """
    PSF function in y direction as fitted from WFM simulations.

    Args:
        x: a numpy array or value, in millimeters

    Returns:
        numpy array or value
    """
    return _modsech(x, norm=1, **_PSFY_WFM_PARAMS)


def _convolution_kernel_psfy(camera: CodedMaskCamera) -> npt.NDArray:
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
    bin_edges = bins.y[min_bin : max_bin + 1]
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    kernel = psfy_wfm(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_vignetting(
    camera: CodedMaskCamera,
    shadowgram: npt.NDArray,
    shift_x: float,
    shift_y: float,
) -> npt.NDArray:
    """
    Apply vignetting effects to a shadowgram based on source position.
    Vignetting occurs when mask thickness causes partial shadowing at off-axis angles.
    This function models this effect by applying erosion operations in both x and y
    directions based on the source's angular displacement from the optical axis.

    Args:
        camera: CodedMaskCamera instance containing mask and detector geometry
        shadowgram: 2D array representing the detector shadowgram before vignetting
        shift_x: Source displacement from optical axis in x direction (mm)
        shift_y: Source displacement from optical axis in y direction (mm)

    Returns:
        2D array representing the detector shadowgram with vignetting effects applied.
        Values are float between 0 and 1, where lower values indicate stronger vignetting.

    Notes:
        - The vignetting effect increases with larger off-axis angles
        - The effect is calculated separately for x and y directions then combined
        - The mask thickness parameter from the camera model determines the strength
          of the effect
    """
    bins = camera.bins_detector

    angle_x_rad = abs(np.arctan(shift_x / camera.mdl["mask_detector_distance"]))
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_x_rad)
    sg1 = _erosion(shadowgram, bins.x[1] - bins.x[0], red_factor)

    angle_y_rad = abs(np.arctan(shift_y / camera.mdl["mask_detector_distance"]))
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_y_rad)
    sg2 = _erosion(shadowgram.T, bins.y[1] - bins.y[0], red_factor)
    return sg1 * sg2.T


def model_shadowgram(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    fluence: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> npt.NDArray:
    """
    Generates a shadowgram for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis
    - Flux scaling

    Args:
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        fluence: Source intensity/fluence value
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled detector image from the source

    Notes:
        - Results are normalized to fluence, e.g. the sum of the result equals `fluence`.
    """
    # relative component map
    RCMAP = {
        0: slice(1, -1),
        +1: slice(2, None),
        -1: slice(None, -2),
    }

    n, m = camera.sky_shape
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    _mask = apply_vignetting(camera, camera.mask, shift_x, shift_y) if vignetting else camera.mask
    _mask = convolve(_mask, _convolution_kernel_psfy(camera), mode="same") if psfy else _mask
    components, (pivot_i, pivot_j) = _rbilinear_relative(shift_x, shift_y, camera.bins_sky.x, camera.bins_sky.y)
    r, c = (n // 2 - pivot_i), (m // 2 - pivot_j)
    mask_shifted_processed = _shift(_mask, (r, c))

    framed_shadowgram = mask_shifted_processed[i_min - 1 : i_max + 1, j_min - 1 : j_max + 1]
    model = (
        sum(framed_shadowgram[RCMAP[pos_i], RCMAP[pos_j]] * weight for (pos_i, pos_j), weight in components.items())
        * camera.bulk
    )
    model /= np.sum(model)
    return model * fluence


def model_sky(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    fluence: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> npt.NDArray:
    """
    Generate a model of the reconstructed sky image for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis
    - Flux scaling

    Args:
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        fluence: Source intensity/fluence value
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled sky reconstruction after all effects
        and processing steps have been applied

    Notes:
        - For optimization, consider using the dedicated, cached function of `optim.py`
    """
    return decode(camera, model_shadowgram(camera, shift_x, shift_y, fluence, vignetting, psfy))


def shift2pos(camera: CodedMaskCamera, shift_x: float, shift_y: float) -> tuple[int, int]:
    """
    Convert continuous sky-shift coordinates to nearest discrete pixel indices.

    Args:
        camera: CodedMaskCamera instance containing binning information
        shift_x: x-coordinate in sky-shift space (mm)
        shift_y: y-coordinate in sky-shift space (mm)

    Returns:
        Tuple of (row, column) indices in the discrete sky image grid

    Notes:
        TODO: Needs boundary checks for shifts outside valid range
    """
    return bisect(camera.bins_sky.y, shift_y) - 1, bisect(camera.bins_sky.x, shift_x) - 1
