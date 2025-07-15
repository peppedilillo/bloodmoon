"""
Optimization routines for source parameter estimation.

This module provides algorithms for:
- Source position estimation
- Flux estimation
- Two-stage combined direction/flux estimation
- Model fitting with instrumental effects
"""

from functools import lru_cache
from typing import Callable, Iterable, Literal
import warnings

from numpy import typing as npt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

from .coords import shift2pos
from .images import _erosion
from .images import _rbilinear
from .images import _rbilinear_relative
from .images import _shift
from .io import SimulationDataLoader
from .mask import _bisect_interval
from .mask import _detector_footprint
from .mask import CodedMaskCamera
from .mask import count
from .mask import cutout
from .mask import decode
from .mask import interpmax
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
        "center": 0,
        "alpha": 0.3214,
        "beta": 0.6246,
    }
    return _modsech(x, norm=1, **PSFY_WFM_PARAMS)


def _wfm_psfy_kernel(camera: CodedMaskCamera) -> npt.NDArray:
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
    kernel = _wfm_psfy(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np.sum(kernel)
    return kernel


@lru_cache(maxsize=1)
def _wfm_psfy_kernel_cached(camera: CodedMaskCamera):
    """Caching helper."""
    return _wfm_psfy_kernel(camera)


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

    angle_x_rad = np.arctan(shift_x / camera.mdl["mask_detector_distance"])
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_x_rad)
    # since the mask detector distance is assumed to be the distance between the
    # detector top and the mask bottom, erosion shall cut on the right-side of the
    # shadowgram when sources have negative `angle_x_rad`.
    # given the implementation of `erosion` we have presently to multiply `red_factor`
    # by -1 to achieve a cut on the right direction.
    # TODO: change `erosion` and its tests so that multiplying by -1 isn't needed
    sg1 = _erosion(shadowgram, bins.x[1] - bins.x[0], red_factor)

    angle_y_rad = np.arctan(shift_y / camera.mdl["mask_detector_distance"])
    red_factor = camera.mdl["mask_thickness"] * np.tan(angle_y_rad)
    sg2 = _erosion(shadowgram.T, bins.y[1] - bins.y[0], red_factor)
    return sg1 * sg2.T


@lru_cache(maxsize=1)
def _detector_footprint_cached(camera: CodedMaskCamera):
    """Caching helper"""
    return _detector_footprint(camera)


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
        shift_x: Source position x-coordinate in sky-shift space (mm)
        shift_y: Source position y-coordinate in sky-shift space (mm)
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: simulates vignetting effects
        psfy: simulates detector reconstruction effects

    Returns:
        2D array representing the modeled detector image from the source

    Notes:
        * Results are normalized, i.e. sums up to one.
    """

    def process_mask(shift_x, shift_y):
        mask_maybe_vignetted = (
            apply_vignetting(
                camera,
                camera.mask,
                shift_x,
                shift_y,
            )
            if vignetting
            else camera.mask
        )
        mask_maybe_vignetted_maybe_psfy = (
            convolve(
                mask_maybe_vignetted,
                _wfm_psfy_kernel_cached(camera),
                mode="same",
            )
            if psfy
            else mask_maybe_vignetted
        )
        return mask_maybe_vignetted_maybe_psfy

    # relative component map
    components = _rbilinear(shift_x, shift_y, camera.bins_sky.x, camera.bins_sky.y)
    n, m = camera.shape_sky
    detector = np.zeros(camera.shape_detector)
    i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
    for (c_i, c_j), weight in components.items():
        r, c = (n // 2 - c_i), (m // 2 - c_j)
        mask_p = process_mask(camera.bins_sky.x[c_j], camera.bins_sky.y[c_i])  # mask processed
        sg = _shift(mask_p, (r, c))  # mask shifted processed
        detector += sg[i_min:i_max, j_min:j_max] * weight
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
    return decode(camera, model_shadowgram(camera, shift_x, shift_y, vignetting=vignetting, psfy=psfy)) * fluence


def _ModelFluence(  # noqa
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    A fast caching version of the model for optimization, leveraging correlation linearity.
    Intended for fluence optimization, not for optimizing source direction.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Two callables. The first is the routine for computing the model, the second
        is a routine for freeing the cache.
    """
    cache = {}

    def cache_clear():
        cache.clear()
        return

    def f(shift_x: float, shift_y: float, fluence: float) -> npt.NDArray:
        """
        This is a faster version of compute_model that caches the decoded shadowgram
        pattern for repeated evaluations with the same source position but different
        fluence values. This makes it suitable for fluence optimization.

        Args:
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            2D array representing the modeled sky reconstruction

        Notes:
            - Uses last-value caching for the spatial pattern
            - Only recomputes pattern when position changes
            - Scales cached pattern by fluence value
        """
        if (shift_x, shift_y) in cache:
            # note we cache the normalized sky model from the normalized shadowgram.
            # hence the sky model should be adjusted by the shift.
            # print("cache hit")
            return cache[(shift_x, shift_y)] * fluence
        # print("cache miss")
        sg = model_shadowgram(camera, shift_x, shift_y, vignetting=vignetting, psfy=psfy)
        _d = decode(camera, sg)
        cache[(shift_x, shift_y)] = _d
        return _d * fluence

    return f, cache_clear


# this is essentially a wrapper to `mask.model_sky`,i am creating it because it's interface
# follows the rules required by `optimize`.
def _ModelShiftFluenceUncached(  # noqa
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    A slow, vanilla implementation of the model for both direction and fluence optimization.
    Intended for debugging and benchmarking.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Two callables. The first is the routine for computing the model, the second
        is a routine for freeing the cache.

    Notes:
        * Although we label this as `Uncached` because it is not using the leveraging
          correlation linearity as `_ModelShiftFluence` and `_ModelShift` do, this model
          is still using some caching to speed up dumb computes such as detector footprint
          and psfy kernel evaluation.
    """

    def f(shift_x: float, shift_y: float, fluence: float) -> npt.NDArray:
        """
        A simple, slow version of the model for both direction and fluence optimization.

        Args:
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            2D array representing the modeled sky reconstruction
        """
        return model_sky(camera, shift_x, shift_y, fluence, vignetting=vignetting, psfy=psfy)

    # there is no cache here, hence no need to clean anything.
    # we return a lambda anyway for compatibility with the other models
    return f, lambda: None


def _ModelShiftFluence(
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    A cached implementation of the model for both direction and fluence optimization.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Two callables. The first is the routine for computing the model, the second
        is a routine for freeing the cache.

    Notes:
        * Applies the same erosion to all the `rbilinear` components. This makes the output
          different from that of `ModelShiftFluenceUncached`, but the difference is small.
    """
    # this dictionary maps an offset (see _rbilinear_relative) to a slice.
    # these slices are used to select the correct piece of mask projection.
    RCMAP = {
        0: slice(1, -1),
        +1: slice(2, None),
        -1: slice(None, -2),
    }
    cache = {}

    def cache_clear():
        cache.clear()

    def process_mask(shift_x, shift_y):
        mask_maybe_vignetted = (
            apply_vignetting(
                camera,
                camera.mask,
                shift_x,
                shift_y,
            )
            if vignetting
            else camera.mask
        )
        mask_maybe_vignetted_maybe_psfy = (
            convolve(
                mask_maybe_vignetted,
                _wfm_psfy_kernel_cached(camera),
                mode="same",
            )
            if psfy
            else mask_maybe_vignetted
        )
        return mask_maybe_vignetted_maybe_psfy

    def normalized_component(framed_shadowgram, relative_position):
        pos_i, pos_j = relative_position
        return (s := framed_shadowgram[RCMAP[pos_i], RCMAP[pos_j]] * camera.bulk) / np.sum(s)

    def f(shift_x: float, shift_y: float, fluence: float) -> npt.NDArray:
        """
        This version decomposes the model into constituent components and caches them
        separately. This allows for precise interpolation between grid points while
        maintaining computational efficiency through caching.

        Args:
            shift_x: Source position x-coordinate in sky-shift space (mm)
            shift_y: Source position y-coordinate in sky-shift space (mm)
            fluence: Source intensity/fluence value

        Returns:
            2D array representing the modeled sky reconstruction

        Notes:
            - Caches individual spatial components
            - Suitable for source position optimization
        """
        components, pivot = _rbilinear_relative(shift_x, shift_y, camera.bins_sky.x, camera.bins_sky.y)
        relative_positions = tuple(components.keys())
        if (pivot, *relative_positions) in cache:
            decoded_components = cache[(pivot, *relative_positions)]
        else:
            n, m = camera.shape_sky
            pivot_i, pivot_j = pivot
            i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
            r, c = (n // 2 - pivot_i), (m // 2 - pivot_j)

            # we call with pivot because calling with shifts to ensure consistent cached/vignetting combos
            mask_p = process_mask(camera.bins_sky.x[pivot_j], camera.bins_sky.y[pivot_i])  # mask processed
            mask_sp = _shift(mask_p, (r, c))  # mask shifted processed
            sg_f = mask_sp[i_min - 1 : i_max + 1, j_min - 1 : j_max + 1]  # shadowgram framed

            # this makes me suffer, there should be a way to not compute decode four times..
            # TODO: is it possible to obtain the same behaviour without four decodings?
            decoded_components = tuple(
                map(
                    lambda x: decode(camera, x),
                    (normalized_component(sg_f, rpos) for rpos in relative_positions),
                )
            )
            cache[(pivot, *relative_positions)] = decoded_components
        sky_model = sum(dc * w for dc, w in zip(decoded_components, components.values()))
        return sky_model * fluence

    return f, cache_clear


def _Loss(model_f: Callable) -> Callable:  # noqa
    """
    Returns a loss function for source parameter optimization, given a routine for computing models.

    Args:
        model_f: Callable that generates model predictions. Expected to have signature:
            model_f(shift_x: float, shift_y: float, fluence: float, camera: CodedMaskCamera) -> np.array

    Returns:
        Callable that computes the loss with signature:
            f(args: np.array, truth: np.array, camera: CodedMaskCamera) -> float
        where:
            - args is [shift_x, shift_y, fluence]
            - truth is the observed sky image
    """

    def f(args: npt.NDArray, truth: npt.NDArray, camera: CodedMaskCamera) -> float:
        """
        Compute MSE loss between model prediction and truth.

        Args:
            args: Array of [shift_x, shift_y, fluence] parameters to evaluate
            truth: Full observed sky image to compare against
            camera: CodedMaskCamera instance containing geometry information
                    No need for this, but we take the parameter for compatibility with
                    optimization model interfaces.

        Returns:
            float: Mean Squared Error between model and truth in local window
        """
        model = model_f(*args)
        mae = np.mean(np.square(model - truth))
        return float(mae)

    return f


def optimize(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    model: Literal["fast", "accurate"] = "fast",
) -> tuple[float, float, float]:
    """
    Perform two-stage optimization to fit a point source model to sky image data.

    This function performs a two-stage optimization:
    1. Coarse optimization of fluence only, keeping position fixed
    2. Fine, simultaneous optimization of position and fluence.
       This step is warm-started with the flux value inferred from the coarse step.

    The process uses different model at each stage to balance speed and accuracy.

    Args:
        camera: CodedMaskCamera instance containing detector and mask parameters
        sky: 2D array of the reconstructed sky image to fit
        arg_sky: Initial guess for source position as (row, col) indices
        vignetting: If true, the model used for optimization will simulate vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Tuple containing the best-fit parameters `(x, y, fluence)` where:
                - x, y are the optimized sky-shift coordinates
                - fluence is the optimized source intensity

    Notes:
        - Initial position is refined using interpolation
        - Bounds are set based on initial guess and physical constraints
    """
    from bloodmoon.images import argmax

    sx_start, sy_start = interpmax(camera, arg_sky, sky)  # pos2shift(camera, *argmax(sky))
    fluence_start = sky.max()

    # initialize the function to compute coarse, fluence-dependent shadowgram model.
    # to reduce the number of cross-correlation the function is cached. it is our
    # responsibility to clear cache, freeing memory, after we will be done with the
    # the coarse fluence step.
    model_fluence, model_fluence_clear = _ModelFluence(camera, vignetting, psfy)
    loss = _Loss(model_fluence)
    results = minimize(
        lambda args: loss((sx_start, sy_start, args[0]), sky, camera),
        x0=np.array((fluence_start,)),
        method="L-BFGS-B",
        bounds=[
            (0.75 * fluence_start, 1.5 * fluence_start),
        ],
        options={
            "maxiter": 10,
            "ftol": 1e-4,
        },
    )
    # we use the best fluence value as the initial value for the next step.
    fluence = results.x[0]
    # releases model cache memory.
    model_fluence_clear()

    # initialize the function to fine coarse, fluence and position dependent shadowgram model.
    # this is slower to compute and requires more memory. again it leverages caches to reduce
    # the number of cross-correlation computations, and it is our responsibility to free
    # memory after we will be done.
    if model == "fast":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluence(camera, vignetting, psfy)
    elif model == "accurate":
        model_shift_flux, model_shift_flux_clear = _ModelShiftFluenceUncached(camera, vignetting, psfy)
    else:
        raise ValueError("Model value not supported. The `model` arguments should be `fast` or `accurate`.")

    loss = _Loss(model_shift_flux)
    results = minimize(
        lambda args: loss((args[0], args[1], args[2]), sky, camera),
        x0=np.array((sx_start, sy_start, fluence)),
        method="Nelder-Mead",
        bounds=[
            (
                max(sx_start - camera.mdl["slit_deltax"], camera.bins_sky.x[0]),
                min(sx_start + camera.mdl["slit_deltax"], camera.bins_sky.x[-1]),
            ),
            (
                max(sy_start - camera.mdl["slit_deltay"], camera.bins_sky.y[0]),
                min(sy_start + camera.mdl["slit_deltay"], camera.bins_sky.y[-1]),
            ),
            (0.9 * fluence, 1.1 * fluence),
        ],
        options={
            "xatol": 1e-6,
        },
    )
    # store the final optimized positions and fluence.
    sx, sy, fluence = map(float, results.x[:3])
    # releases model cache memory.
    model_shift_flux_clear()
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
        min_slit = min(camera.mdl["slit_deltax"], camera.mdl["slit_deltay"])
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

    def init_get_arg(snrs: tuple, batchsize: int = 1000) -> Callable:
        """This hides a reservoirs-batch mechanism for quickly selecting candidates,
        and initializes the data structures it relies on."""
        # we sort source directions by significance.
        # this is kind of costly because the sky arrays may be very large.
        # sorted directions are moved to a reservoir.
        reservoirs = [np.argsort(snr, axis=None) for snr in snrs]

        # integrating source intensities over aperture for all matrix elements is
        # computationally unfeasable. To avoid this, we execute this computation over small batches.
        batches = [np.array([]), np.array([])]

        def slit_intensity():
            """Integrates source intensity over mask's aperture."""
            intensities = ([], [])
            for int_, snr, batch in zip(
                intensities,
                snrs,
                batches,
            ):
                for arg in batch:
                    (min_i, max_i, min_j, max_j), _ = cutout(camera, arg)
                    slit = snr[min_i:max_i, min_j:max_j]
                    int_.append(np.sum(slit))
            return intensities

        def fill():
            """Fill the batches with sorted candidates"""
            for i, _ in enumerate(sdls):
                tail, head = reservoirs[i][:-batchsize], reservoirs[i][-batchsize:]
                batches[i] = np.array([np.unravel_index(id, snrs[i].shape) for id in head])
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

        return get if max(map(np.max, snrs)) > snr_threshold else lambda: None

    def find_candidates(snrs: tuple, max_pending=6666) -> tuple:
        """Returns candidate, compatible sources for the two cameras.
        Worst case complexity is O(n^2) but amortized costs are much smaller."""
        get_arg = init_get_arg(snrs)
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

        significance = float(snr_map[*shift2pos(camera, shiftx, shifty)])
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
        candidates = find_candidates(snrs)
        if not candidates:
            break
        try:
            sources, skies = zip(*(subtract(index, sky, snr) for index, sky, snr in zip(candidates, skies, snrs)))
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield ((sources, skies) if sdls == (sdl_cam1a, sdl_cam1b) else (sources[::-1], skies))
