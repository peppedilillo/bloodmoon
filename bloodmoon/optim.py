from bisect import bisect
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

from .images import _rbilinear_relative
from .images import _shift
from .mask import _chop
from .mask import _convolution_kernel_psfy
from .mask import _detector_footprint
from .mask import _interpmax
from .mask import apply_vignetting
from .mask import CodedMaskCamera
from .mask import decode
from .mask import model_shadowgram
from .types import UpscaleFactor


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


@lru_cache(maxsize=1)
def _convolution_kernel_psfy_cached(camera: CodedMaskCamera):
    """Cached helper."""
    return _convolution_kernel_psfy(camera)


@lru_cache(maxsize=1)
def _detector_footprint_cached(camera: CodedMaskCamera):
    """Cached helper"""
    return _detector_footprint(camera)


def _init_model_coarse(
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    This is a faster version of compute_model that caches the decoded shadowgram
    pattern for repeated evaluations with the same source position but different
    fluence values. This makes it suitable for fluence optimization.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Two callables. The first is the routine for computing the model, the second
        is a routine for freeing the cache.
    """
    cache = [
        (None, None),
    ]

    def cache_hash():
        return cache[0][0]

    def cached(shift):
        return cache_hash() == hash(shift)

    def cache_set(shift, value):
        cache[0] = hash(shift), value

    def cache_get():
        return cache[0][1]

    def cache_clear():
        cache.clear()
        cache.append(
            (None, None),
        )

    def f(shift_x: float, shift_y: float, fluence: float) -> np.array:
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
        if cached((shift_x, shift_y)):
            # note we cache the normalized sky model from the normalized shadowgram.
            # hence the sky model should be adjusted by the shift.
            # print("cache hit")
            return cache_get() * fluence
        # print("cache miss")
        sg = model_shadowgram(camera, shift_x, shift_y, 1, vignetting=vignetting, psfy=psfy)
        cache_set((shift_x, shift_y), decode(camera, sg))
        return cache_get() * fluence

    return f, cache_clear


def _init_model_fine(
    camera: CodedMaskCamera,
    vignetting: bool = True,
    psfy: bool = True,
) -> tuple[Callable, Callable]:
    """
    This version decomposes the model into constituent components and caches them
    separately. This allows for precise interpolation between grid points while
    maintaining computational efficiency through caching.

    Args:
        camera: CodedMaskCamera instance containing all geometric parameters
        vignetting: If true, shadowgram model simulates vignetting.
        psfy: If true, the model used for optimization will simulate detector position
        reconstruction effects.

    Returns:
        Two callables. The first is the routine for computing the model, the second
        is a routine for freeing the cache.
    """
    RCMAP = {
        0: slice(1, -1),
        +1: slice(2, None),
        -1: slice(None, -2),
    }

    cache = {}

    def cached(key):
        return key in cache

    def cache_set(key, value):
        cache[key] = value

    def cache_get(key):
        return cache[key]

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
                _convolution_kernel_psfy_cached(camera),
                mode="same",
            )
            if psfy
            else mask_maybe_vignetted
        )
        return mask_maybe_vignetted_maybe_psfy

    def normalized_component(framed_shadowgram, relative_position):
        pos_i, pos_j = relative_position
        return (s := framed_shadowgram[RCMAP[pos_i], RCMAP[pos_j]] * camera.bulk) / np.sum(s)

    def f(shift_x: float, shift_y: float, fluence: float) -> np.array:
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
        if cached((pivot, *relative_positions)):
            # print("cache hit")
            decoded_components = cache_get((pivot, *relative_positions))
        else:
            # print("no cache hit")
            n, m = camera.sky_shape
            pivot_i, pivot_j = pivot
            i_min, i_max, j_min, j_max = _detector_footprint_cached(camera)
            r, c = (n // 2 - pivot_i), (m // 2 - pivot_j)

            # we call with pivot because calling with shifts to ensure consistent cached/vignetting combos
            mask_processed = process_mask(camera.bins_sky.x[pivot_j], camera.bins_sky.y[pivot_i])
            mask_shifted_processed = _shift(mask_processed, (r, c))
            framed_shadowgram = mask_shifted_processed[i_min - 1 : i_max + 1, j_min - 1 : j_max + 1]

            # this makes me suffer, there should be a way to not compute decode four times..
            # TODO: is it possible to obtain the same behaviour without four decodings?
            decoded_components = tuple(
                map(
                    lambda x: decode(camera, x),
                    (normalized_component(framed_shadowgram, rpos) for rpos in relative_positions),
                )
            )
            cache_set((pivot, *relative_positions), decoded_components)
        sky_model = sum(dc * w for dc, w in zip(decoded_components, components.values()))
        return sky_model * fluence

    return f, cache_clear


def _loss(model_f: Callable) -> Callable:
    """
    Returns a loss function for source parameter optimization with a given strategy
    for computing models.

    Args:
        model_f: Callable that generates model predictions. Should have signature:
            model_f(shift_x: float, shift_y: float, fluence: float, camera: CodedMaskCamera) -> np.array

    Returns:
        Callable that computes the loss with signature:
            f(args: np.array, truth: np.array, camera: CodedMaskCamera) -> float
        where:
            - args is [shift_x, shift_y, fluence]
            - truth is the observed sky image
            - camera is the CodedMaskCamera instance
    """

    def f(args: np.array, truth: np.array, camera: CodedMaskCamera) -> float:
        """
        Compute MSE loss between model prediction and truth within a local window, roughly
        sized as a slit (see `chop`).

        Args:
            args: Array of [shift_x, shift_y, fluence] parameters to evaluate
            truth: Full observed sky image to compare against
            camera: CodedMaskCamera instance containing geometry information

        Returns:
            float: Mean Squared Error between model and truth in local window

        Notes:
            - Window size is determined by camera.mdl["slit_delta{x,y}"]
            - Model is generated using the provided model_f function
            - Only computes error within the local window to improve robustness
        """
        shift_x, shift_y, fluence = args
        model = model_f(*args)
        (min_i, max_i, min_j, max_j), _ = _chop(camera, shift2pos(camera, shift_x, shift_y))
        truth_chopped = truth[min_i:max_i, min_j:max_j]
        model_chopped = model[min_i:max_i, min_j:max_j]
        residual = truth_chopped - model_chopped
        mse = np.mean(np.square(residual))
        return mse

    return f


def optimize(
    camera: CodedMaskCamera,
    sky: np.array,
    arg_sky: tuple[int, int],
    vignetting: bool = True,
    psfy: bool = True,
    verbose: bool = False,
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
        verbose: If true, prints the output from the optimizer.

    Returns:
        Tuple containing the best-fit parameters `(x, y, fluence)` where:
                - x, y are the optimized sky-shift coordinates
                - fluence is the optimized source intensity

    Notes:
        - Initial position is refined using interpolation
        - Bounds are set based on initial guess and physical constraints
    """
    # TODO: the upscaling factor should probably go into a configuration thing.
    shift_start_x, shift_start_y = _interpmax(camera, arg_sky, sky, UpscaleFactor(10, 10))
    fluence_start = sky.max()

    # initialize the function to compute coarse, fluence-dependent shadowgram model.
    # to reduce the number of cross-correlation the function is cached. it is our
    # responsibility to clear cache, freeing memory, after we will be done with the
    # the coarse fluence step.
    _compute_model_coarse, _compute_model_coarse_cache_clear = _init_model_coarse(camera, vignetting, psfy)
    loss_coarse = _loss(_compute_model_coarse)
    results = minimize(
        lambda args: loss_coarse((shift_start_x, shift_start_y, args[0]), sky, camera),
        x0=np.array((fluence_start,)),
        method="L-BFGS-B",
        bounds=[
            (0.75 * fluence_start, 1.5 * fluence_start),
        ],
        options={
            "maxiter": 10,
            "iprint": 1 if verbose else -1,
            "ftol": 10e-5,
        },
    )
    # we use the best fluence value as the initial value for the next step.
    coarse_fluence = results.x[0]
    # releases model cache memory.
    _compute_model_coarse_cache_clear()

    # initialize the function to fine coarse, fluence and position dependent shadowgram model.
    # this is slower to compute and requires more memory. again it leverages caches to reduce
    # the number of cross-correlation computations, and it is our responsibility to free
    # memory after we will be done.
    _compute_model_fine, _compute_model_fine_cache_clear = _init_model_fine(camera, vignetting, psfy)
    loss_fine = _loss(_compute_model_fine)
    results = minimize(
        lambda args: loss_fine((args[0], args[1], args[2]), sky, camera),
        x0=np.array((shift_start_x, shift_start_y, coarse_fluence)),
        method="L-BFGS-B",
        bounds=[
            (
                max(shift_start_x - camera.mdl["slit_deltax"] / 2, camera.bins_sky.x[0]),
                min(shift_start_x + camera.mdl["slit_deltax"] / 2, camera.bins_sky.x[-1]),
            ),
            (
                max(shift_start_y - camera.mdl["slit_deltay"] / 2, camera.bins_sky.y[0]),
                min(shift_start_y + camera.mdl["slit_deltay"] / 2, camera.bins_sky.y[-1]),
            ),
            (0.95 * coarse_fluence, 1.05 * coarse_fluence),
        ],
        options={
            "maxiter": 10,
            "iprint": 1 if verbose else -1,
            "ftol": 10e-5,
        },
    )
    # store the final optimized positions and fluence.
    x, y, fluence = results.x[:3]

    # releases model cache memory.
    _compute_model_fine_cache_clear()
    return x, y, fluence
