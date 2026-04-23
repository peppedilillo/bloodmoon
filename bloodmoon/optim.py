"""
Optimization routines for source parameter estimation.

This module provides algorithms for:
- Source position estimation
- Flux estimation
- Two-stage combined direction/flux estimation
- Model fitting with instrumental effects
"""

from functools import lru_cache
from typing import Callable, Iterable
import warnings

from numpy import typing as npt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import convolve

from .coords import pos2shift
from .images import _erosion
from .images import fshift
from .io import SimulationDataLoader
from .mask import _detector_footprint
from .mask import CodedMaskCamera
from .mask import count
from .mask import cutout
from .mask import decode
from .mask import snratio
from .mask import variance


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


def _wfm_psfy(x: npt.NDArray) -> npt.NDArray:
    """
    PSF function in y direction as fitted from WFM simulations.

    Args:
        x: a numpy array or value, in millimeters

    Returns:
        numpy array or value
    """
    PSFY_WFM_PARAMS = {
        "norm": 1.0,
        "center": 0.0,
        "alpha": 0.5459735904725987,
        "beta": 0.7363355668833482,
    }
    return _modsech(x, **PSFY_WFM_PARAMS)


def _wfm_psfy_kernel(camera: CodedMaskCamera) -> npt.NDArray:
    """
    Returns PSF normalised convolution kernel.
    At present, it ignores the `x` direction, since PSF characteristic lenght
    is much shorter than typical bin size, even at moderately large upscales.

    Args:
        camera: a CodedMaskCamera object.

    Returns:
        A column array convolution kernel.
    """
    PSFY_GAUSS_PARAMS = {
        "sigma": 0.17749677955602094,
        "mode": 'constant',
        "cval": 0.0,
    }

    # we take a whole slit to have a good kernel spatial extension
    px_ydim = camera.specs.mask_deltay / camera.upscale_f.y
    slit_dim = camera.specs.slit_deltay
    # the kernel must have the same binning as the mask elements
    bins = np.linspace(-slit_dim, slit_dim, int(2 * slit_dim / px_ydim) + 1)
    kernel = _wfm_psfy(bins).reshape(len(bins), -1)
    # from tests, the modsech should be modulated with a Gaussian
    psfy = gaussian_filter(kernel, **PSFY_GAUSS_PARAMS)
    return psfy / np.sum(psfy)


@lru_cache(maxsize=1)
def _wfm_psfy_kernel_cached(camera: CodedMaskCamera):
    """Caching helper."""
    return _wfm_psfy_kernel(camera)


def apply_detector_resolution(
    camera: CodedMaskCamera,
    shadowgram: npt.NDArray,
) -> npt.NDArray:
    """
    Applies finite detector spatial resolution effects to a shadowgram.

    Args:
        camera: CodedMaskCamera instance containing mask and detector geometry
        shadowgram: 2D array representing the detector shadowgram
    
    Returns:
        2D array representing the detector shadowgram
        with spatial resolution effects applied.
    """
    return convolve(
        shadowgram, _wfm_psfy_kernel_cached(camera), mode="same",
    )


def apply_vignetting(
    camera: CodedMaskCamera,
    shadowgram: npt.NDArray,
    shift_x: float,
    shift_y: float,
) -> npt.NDArray:
    r"""
    Apply vignetting effects to a shadowgram based on source position.
    Vignetting occurs when mask thickness causes partial shadowing at off-axis angles.
    This function models this effect by applying erosion operations in both x and y
    directions based on the source's angular displacement from the optical axis.


                <--------> MASK APERTURE

              \       \  \
    ___________\       \  \____________
               |\       \ |x            MASK ELEMENT
    ___________| \       \|_x___________
                  \       \  x
                   \       \  x
                    \       \  x
     ________________\_______\__x_________  DETECTOR
     <--------------->        <->
           SHIFT             EROSION

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
    def project_mask_thickness(shift: float, bin_dim: float) -> float:
        """
        Projects the mask thickness on the mask binning elements, and
        corrects the projection value to allows for correct erosion
        of the mask physical elements starting from the pixels' edges.
        """
        # since the mask detector distance is defined as the distance between the
        # detector top and the mask top, erosion shall cut on the left-side of the
        # shadowgram when sources have negative `angle`.
        # if the mask detector distance was defined as the distance between the
        # detector top and the mask bottom, erosion should have been applied to the
        # right side, i.e. `proj` should be multiplied by -1.
        angle = np.arctan(shift / camera.specs.mask_detector_distance)
        proj = camera.specs.mask_thickness * np.tan(angle)
        shift_px = shift / bin_dim
        # the mask thickness projection has to be corrected by considering the
        # erosion pixel start point, due to the discretisation of the projection
        # https://github.com/yuri-evangelista/CodedMasks/blob/main/mask_050_1040x17/new_erosion_20251024.ipynb
        bin_erosion_start = (1.0 - abs(shift_px - int(shift_px))) * bin_dim
        return proj + np.sign(proj) * bin_erosion_start
    
    bins = camera.bins_detector
    bin_dim_x, bin_dim_y = (
        bins.x[1] - bins.x[0],
        bins.y[1] - bins.y[0],
    )

    red_factor_x = project_mask_thickness(shift_x, bin_dim_x)
    sg_x = _erosion(shadowgram, bin_dim_x, red_factor_x)

    # - we apply the y-axis erosion to `sg_x`, otherwise the decimal
    #   values of the input shifted shadowgram would be squared
    # - the erosion on the two axes is still independent, as it must be
    red_factor_y = project_mask_thickness(shift_y, bin_dim_y)
    sg_y = _erosion(sg_x.T, bin_dim_y, red_factor_y)

    return sg_y.T


@lru_cache(maxsize=1)
def _detector_footprint_cached(camera: CodedMaskCamera):
    """Caching helper"""
    return _detector_footprint(camera)


def _mask_pattern_projection(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> npt.NDArray:
    """
    Generates the mask pattern projection for a point source by applying the
    fractional shift of the camera mask array (mask pattern projection), and
    the instrumental effects (i.e., vignetting and detector spatial resolution).

    To process the mask pattern, a specific operation order is followed:
        * first, the mask array is shifted
        * second, the vignetting is applied to the shifted array
        * lastly, the detector sp. res. is applied

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        Processed mask pattern array for the source.
    """
    # shift mask pattern
    pxdimy, pxdimx = (
        camera.specs.mask_deltay / camera.upscale_f.y,
        camera.specs.mask_deltax / camera.upscale_f.x,
    )
    fr, fc = (
        (-1.0) * shift_y / pxdimy,
        (-1.0) * shift_x / pxdimx,
    )
    mask_shifted = fshift(camera.mask.astype(float), fr, fc)
    # apply vignetting effect
    mask_vignetted = (
        apply_vignetting(camera, mask_shifted, shift_x, shift_y)
        if vignetting else mask_shifted
    )
    # apply detector spatial resolution effect
    mask_projected = (
        apply_detector_resolution(camera, mask_vignetted)
        if psfy else mask_vignetted
    )
    return mask_projected


def model_shadowgram(
    camera: CodedMaskCamera,
    shift_x: float,
    shift_y: float,
    vignetting: bool = True,
    psfy: bool = True,
) -> npt.NDArray:
    """
    Generates a normalized shadowgram for a point source.

    The model may feature:
    - Mask pattern projection
    - Vignetting effects
    - PSF convolution over y axis

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled detector image from the source

    Notes:
        * Results are normalized, i.e. sums up to one.
    """
    # project mask pattern from point source
    sg = _mask_pattern_projection(
        camera, shift_x, shift_y, vignetting, psfy,
    )
    # extract normalised detector image
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    detector = sg[i_min:i_max, j_min:j_max]
    detector *= camera.bulk
    detector /= np.sum(detector)
    return detector


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
        camera: CodedMaskCamera instance containing all geometric parameters
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        fluence: Source intensity/fluence value
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled sky reconstruction after all effects
        and processing steps have been applied
    """
    detector = model_shadowgram(
        camera, shift_x, shift_y, vignetting=vignetting, psfy=psfy,
    )
    return decode(camera, detector * fluence)


def process_skyimg(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    pos: tuple[int, int],
) -> npt.NDArray:
    """
    Processes the sky image for optimisation.
    Specifically, it crops the input sky 2D array around the extracted
    source position with an extension proportional to the system source
    PSF, and then it flattens the cropped array.

    Args:
        camera: CodedMaskCamera intance with system geometry info.
        sky: 2D array representing the sky image.
        pos: Pixel indexes for the cropping.
    
    Returns:
        output: 1D source-cropped and flattened sky image.
    """
    # here we crop the source PSF slit plus an offset to account for
    # shifts and to accomodate the `curve_fit` optimisation:
    #   - along y (coarse dir) we insert an offset of `5 * upscaling`, which
    #     is ~ 1/6 of the upsampled PSF slit dimension;
    #   - along x (fine dir) we insert an offset of `2 + upscaling`, to
    #     avoid the bkg contributes from the surrounding pixels;
    # NOTE: if the offset is smaller than at least the shifts bounds in
    # `optimize()`, the optimisation procedure may fail for some sources
    psf_slit_px_y, psf_slit_px_x = (
        int(camera.specs.slit_deltay * camera.upscale_f.y / camera.specs.mask_deltay),
        int(camera.specs.slit_deltax * camera.upscale_f.x / camera.specs.mask_deltax),
    )
    offset_y, offset_x = (
        5 * camera.upscale_f.y, 2 + camera.upscale_f.x,
    )

    i, j = pos
    crop_y, crop_x = (
        psf_slit_px_y + offset_y, psf_slit_px_x + offset_x,
    )
    slice_y, slice_x = (
        slice(i - crop_y, i + crop_y + 1), slice(j - crop_x, j + crop_x + 1),
    )
    cropped = sky[slice_y, slice_x]
    return cropped.flatten()


# this is essentially a wrapper to `mask.model_sky`, i am creating it
# because its interface # follows the rules required by `optimize`.
def _ModelShiftFluence(
    camera: CodedMaskCamera,
    pos: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
) -> Callable[[npt.NDArray, float, float, float], npt.NDArray]:
    """
    A slow, vanilla implementation of the model for both direction and fluence optimization.
    Intended for debugging and benchmarking.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        pos: tuple of row, col indexes indicating the source peak position. The source
        sky image is cropped around `pos`.
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        A Callable, which is the routine for computing the model.
    """

    def f(x: npt.NDArray, shift_x: float, shift_y: float, fluence: float) -> npt.NDArray:
        """
        A simple, slow version of the model for both direction and fluence optimization.
        The input `x` represents an independent variable, and it has only been inserted
        to match the inputs of the scipy `curve_fit` procedure.

        Args:
            x: Placeholder for independent variable as in `curve_fit` doc
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            Flattened and cropped 2D source-modeled sky image
        """
        modeled = model_sky(camera, shift_x, shift_y, fluence, vignetting, psfy)
        return process_skyimg(camera, modeled, pos)
    
    return f


def optimize(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    camera_coding_power: float = 0.85,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """
    Performs the optimization to fit a point source model to sky image data.

    This function performs the optimization by simultaneously fit the candidate
    position and fluence. The starting position is inferred from the candidate
    pixel position, while the starting fluence is represented by the counts at
    the candidate extracted pixel indexes. If the `psfy` effect is active, the
    fluence start value is corrected for the camera coding power.

    Args:
        camera: CodedMaskCamera instance containing detector and mask parameters
        sky: 2D array of the reconstructed sky image to fit
        arg_sky: Initial guess for source position as (row, col) indices
        vignetting: If true, the model used for optimization will simulate vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.
        verbose: If True, the optimisation results are printed out.

    Returns:
        Tuple containing the best-fit parameters `(x, y, fluence)` where:
                - x, y are the optimized sky-shift coordinates
                - fluence is the optimized source intensity

    Notes:
        - Bounds are set based on initial guess and physical constraints
    """
    px_dim_x, px_dim_y = (
        camera.specs.mask_deltax / camera.upscale_f.x,
        camera.specs.mask_deltay / camera.upscale_f.y,
    )
    model_shift_flux = _ModelShiftFluence(camera, arg_sky, vignetting, psfy)
    sx_start, sy_start = pos2shift(camera, *arg_sky)
    sky_peak = sky[*arg_sky]
    fluence_start = (
        sky_peak / camera_coding_power if psfy else sky_peak
    )
    sky_ydata = process_skyimg(camera, sky, arg_sky)
    
    # - the shifts are allowed to fluctuate in a 3 x 3 pixel box since
    #   the extracted position is close enough to the true source pos
    #   Also, since multiple sources may be superimposed or close, the
    #   optimisation procedure may introduce biases in the source fit
    # - the fluence cannot be smaller than the one observed at the peak,
    #   and we insert a lower value just for precaution (if simulating
    #   for example an infinite detector spatial resolution)
    results, _ = curve_fit(
        model_shift_flux,
        xdata=np.arange(len(sky_ydata)),
        ydata=sky_ydata,
        p0=[sx_start, sy_start, fluence_start],
        bounds=[
            (
                max(sx_start - 1.5 * px_dim_x, camera.bins_sky.x[0]),
                max(sy_start - 1.5 * px_dim_y, camera.bins_sky.y[0]),
                0.95 * sky_peak,
            ),
            (
                min(sx_start + 1.5 * px_dim_x, camera.bins_sky.x[-1]),
                min(sy_start + 1.5 * px_dim_y, camera.bins_sky.y[-1]),
                1.25 * sky_peak,
            ),
        ],
    )
    # store the final optimized positions and fluence
    sx, sy, fluence = map(float, results)

    if verbose:
        print(
            f'\n'
            f'## Optimisation Results:\n'
            f'  - fluence START: {fluence_start}\n'
            f'  - shifts START (x, y): {sx_start}, {sy_start}\n'

            f'  - fluence OPTIM.: {fluence}\n'
            f'  - shifts OPTIM. (x, y): {sx}, {sy}\n'

            f'  - fluence GAIN %: {(fluence - fluence_start) * 100 / fluence_start:.3f}\n'
            f'  - shift_x GAIN %: {np.sign(sx_start) * (sx - sx_start) * 100 / sx_start:.3f}\n'
            f'  - shift_y GAIN %: {np.sign(sy_start) * (sy - sy_start) * 100 / sy_start:.3f}\n'
        )

    return sx, sy, fluence


"""
jesus pleasee look upon it

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠴⠋⡽⢃⣀⣇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠔⠉⣠⠞⢠⡞⠁⣏⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠤⣀⡞⠁⢀⠔⠁⣰⠏⢀⣤⠁⡇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⡞⠀⣰⠃⢀⠞⠁⣰⠋⣸⣄⠇
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⡼⠁⣰⠃⢀⠏⠀⢰⠃⢠⠇⢸⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⠏⠜⠁⡰⠃⠀⡜⠀⢠⠇⠀⡜⡀⠈⡇
⠀⠀⠀⠀⠀⠀⠀⢀⡏⠀⠀⠀⠀⠀⠀⠀⠠⠋⠀⡸⢡⠃⠀⡇
⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⢣⠃⢀⡞⠁
⠀⠀⠀⠀⠀⠀⠀⡾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡟⠳⠄⡜⠀⠀
⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀⠀⢀⠇⠀⠀
⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⡘⠀⠀⠀
⣀⣠⣤⣶⣦⣴⠃⠀⠀⠀⠀⠀⠀⠀⠀⢠⠏⠀⠀⡰⠁⠀⠀⠀
⠈⢿⣿⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⣴⣿⡄⠀⠀⠀
⠀⠀⢻⣿⣿⣿⣿⣿⡄⠀⠀⣠⡴⠋⠀⠀⠀⠰⣿⣿⣿⡄⠀⠀
⠀⠀⠈⣿⣿⣿⣿⣿⣿⣀⠞⣿⣷⡀⠀⠀⠀⠀⣿⣿⣿⡇⠀⠀
⠀⠀⠀⢸⣿⣿⣿⣿⣿⣏⠀⢹⣿⣿⣶⣤⣤⣴⣿⣿⣿⠇⠀⠀
⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀
⠀⠀⠀⠀⢸⣿⣿⠿⠟⠉⠀⠀⠀⠙⠻⠿⠿⠿⠟⠋⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""


def iros(
    camera: CodedMaskCamera,
    sdl_cam1a: SimulationDataLoader,
    sdl_cam1b: SimulationDataLoader,
    max_iterations: int,
    snr_threshold: float = 0.0,
    vignetting: bool = True,
    psfy: bool = True,
) -> Iterable:
    """Performs Iterative Removal of Sources (IROS) for dual-camera WFM observations.

    This function implements an iterative source detection and removal algorithm for
    the WFM coded mask instrument. For each iteration, it:
    1. Ranks source candidates by SNR and integrated intensity
    2. Matches compatible source positions between orthogonal cameras
    3. Fits source parameters
    4. Removes fitted sources from the sky image
    5. Repeats until no significant sources remain or max iterations reached

    Args:
        camera: CodedMaskCamera instance containing mask/detector geometry and parameters
        sdl_cam1a: SimulationDataLoader for the first WFM  camera
        sdl_cam1b: SimulationDataLoader for the second WFM camera
        max_iterations: Maximum number of source removal iterations to perform
        snr_threshold: Optional float. If provided, iteration stops when maximum
            residual SNR falls below this value. Defaults to 0. (no threshold).
        vignetting: Optional bool. If `True`, the model used for optimization will simulate vignetting.
        psfy: Optional bool. If `True`, the model used for optimization will simulate detector
        position reconstruction effects.

    Yields:
        For each iteration, yields:
            - A tuple of two (x, y, fluence, significance) tuples, one for each camera's
              detected source, where x,y are sky-shift coordinates in mm, fluence is source intensity,
               significance in standard deviations.
            - A tuple of two residual sky images after source removal, one for each camera
            Note: Results are ordered to match sdl_cam1a, sdl_cam1b order

    Raises:
        ValueError: If cameras are not oriented orthogonally (90° rotation in azimuth)
        RuntimeError: If source parameter optimization fails (with detailed error message)

    Notes:
        Performance Considerations:
        - Computation scales with mask resolution. Keep upscaling factors low
          (upscale_x * upscale_y ~< 10) for reasonable performance

        Algorithm Details:
        - Requires orthogonal camera views (90° rotation) for source localization
        - Ranks candidates by SNR and integrated intensity within aperture
        - Optimizes source parameters in local windows around candidates
        - When using reconstructed data, accounts for vignetting and PSF effects

    Example:
    >>> for sources, residuals in iros(camera, sdl_cam1a, sdl_cam1b, max_iterations=2):
    >>>     source_1a, source_1b = sources
    >>>     residual_1a, residual_1b = residuals
    >>>     ...
    """
    from astropy.coordinates import angular_separation

    # verify cameras are oriented orthogonally (90° rotation in azimuth).
    # this is required for the source position matching algorithm.
    # then sort the data loaders into a tuple so that the second's data loader
    # x axis is at +90° from the first one.
    # fmt: off
    if not np.isclose(
        angular_separation(
            *map(np.deg2rad, (*sdl_cam1a.rotations["z"], *sdl_cam1b.rotations["z"]))
        ),
        0.
    ) or not np.isclose(
        np.abs(
            delta_rot_x := angular_separation(
                *map(np.deg2rad, (*sdl_cam1a.rotations["x"], *sdl_cam1b.rotations["x"])))
        ),
        np.pi / 2
    ):
        raise ValueError("Cameras must be rotated by 90° degrees over azimuth.")
    else:
        if delta_rot_x > 0:
            sdls = (sdl_cam1a, sdl_cam1b)
        else:
            sdls = (sdl_cam1b, sdl_cam1a)
    # fmt: on

    def direction_match(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> bool:
        """Determines if source positions from both cameras correspond to the same sky location.
        Compares source positions accounting for the 90° camera rotation. Positions are
        considered matching if they are within one slit width from each other after rotation.
        TODO: not urgent, but in a future we should make this work for arbitrary camera rotations.
        """
        ax, ay = camera.bins_sky.x[a[1]], camera.bins_sky.y[a[0]]
        # we apply -90deg rotation to camera b source
        bx, by = -camera.bins_sky.y[b[0]], camera.bins_sky.x[b[1]]
        min_slit = min(camera.specs.slit_deltax, camera.specs.slit_deltay)
        return abs(ax - bx) < min_slit and abs(ay - by) < min_slit

    def match(pending: tuple) -> tuple:
        """Cross-check the last entry in pending to match against all other pending directions"""
        pa, pb = pending
        if not pa or not pb:
            return tuple()

        # we are going to call this each time we get a new couple of candidate indices.
        # we avoid evaluating matches for all pairs at all calls, which would result in
        # repeated evaluations of the same pairs (would result in O(n^3) worst case for
        # `find_candidates()`
        *_, latest_a = pa
        for b in pb:
            if direction_match(latest_a, b):
                return latest_a, b

        *_, latest_b = pb
        for a in pa:
            if direction_match(a, latest_b):
                return a, latest_b
        return tuple()

    def init_get_arg(skies: tuple, snrs: tuple, batchsize: int = 1000) -> Callable:
        """This hides a reservoirs-batch mechanism for quickly selecting candidates,
        and initializes the data structures it relies on."""
        # we sort source directions by significance.
        # this is kind of costly because the sky arrays may be very large.
        # sorted directions are moved to a reservoir.
        reservoirs = [np.argsort(sky, axis=None) for sky in skies]

        # integrating source intensities over aperture for all matrix elements is
        # computationally unfeasable. To avoid this, we execute this computation over small batches.
        batches = [np.array([]), np.array([])]

        def slit_intensity():
            """Integrates source intensity over mask's aperture."""
            intensities = ([], [])
            for int_, sky, batch in zip(
                intensities,
                skies,
                batches,
            ):
                for arg in batch:
                    (min_i, max_i, min_j, max_j), _ = cutout(camera, arg)
                    slit = sky[min_i:max_i, min_j:max_j]
                    int_.append(np.sum(slit))
            return intensities

        def fill():
            """Fill the batches with sorted candidates"""
            for i, _ in enumerate(sdls):
                tail, head = reservoirs[i][:-batchsize], reservoirs[i][-batchsize:]
                batches[i] = np.array([np.unravel_index(id, skies[i].shape) for id in head])
                reservoirs[i] = tail

            # integrates over mask element aperture and sum between cameras
            argsort_intensities = np.argsort(np.sum(slit_intensity(), axis=0))

            # sort candidates in present batch by their integrated-combined intensity
            for i, _ in enumerate(sdls):
                batches[i] = batches[i][argsort_intensities]

        def empty():
            """Checks if batches are empty"""
            return all(not len(b) for b in batches)

        def get() -> tuple | None:
            """Think of this as a faucet getting you one decent direction combo at a time."""
            if empty():
                fill()
                if empty():
                    return None

            out = tuple(batch[-1] for batch in batches)
            for i, _ in enumerate(sdls):
                batches[i] = batches[i][:-1]
            return out

        return get if max(tuple(snr[*cand] for cand, snr in zip(get(), snrs))) > snr_threshold else lambda: None

    def find_candidates(skies: tuple, snrs: tuple, max_pending=6666) -> tuple:
        """Returns candidate, compatible sources for the two cameras.
        Worst case complexity is O(n^2) but amortized costs are much smaller."""
        get_arg = init_get_arg(skies, snrs)
        pending = ([], [])

        while not (matches := match(pending)):
            args = get_arg()
            if args is None:
                break
            for stack, arg in zip(pending, args):
                stack.append(arg)
                if len(stack) > max_pending:
                    stack.pop(0)
        return matches if matches else tuple()

    def subtract(
        arg: tuple[int, int],
        sky: npt.NDArray,
        snr_map: npt.NDArray,
    ) -> tuple[tuple[float, float, float, float], npt.NDArray]:
        """Runs optimizer and subtract source."""
        try:
            shiftx, shifty, fluence = optimize(
                camera=camera,
                sky=sky,
                arg_sky=arg,
                vignetting=vignetting,
                psfy=psfy,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e

        significance = float(snr_map[*arg])  # candidate significance at extraction pos
        model = model_sky(
            camera=camera,
            shift_x=shiftx,
            shift_y=shifty,
            fluence=fluence,
            vignetting=vignetting,
            psfy=psfy,
        )
        residual = sky - model
        return (shiftx, shifty, fluence, significance), residual

    def compute_snratios(
        skymaps: tuple[npt.NDArray, npt.NDArray],
        varmaps: tuple[npt.NDArray, npt.NDArray],
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Computes skies SNR."""
        # variance is clipped to improve numerical stability for off-axis sources,
        # which may result in very few counts.
        # TODO: improve on this only sorting matrix elements over a threshold.
        snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skymaps, varmaps))
        return snrs

    detectors = tuple(count(camera, sdl.data)[0] for sdl in sdls)
    variances = tuple(variance(camera, d) for d in detectors)
    skies = tuple(decode(camera, d) for d in detectors)
    for i in range(max_iterations):
        snrs = compute_snratios(skies, variances)
        candidates = find_candidates(skies, snrs)
        if not candidates:
            break
        try:
            sources, skies = zip(*(subtract(index, sky, snr) for index, sky, snr in zip(candidates, skies, snrs)))
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield ((sources, skies) if sdls == (sdl_cam1a, sdl_cam1b) else (sources[::-1], skies))
