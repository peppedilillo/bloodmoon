"""
Optimization routines for source parameter estimation.

This module provides algorithms for:
- Source position refinement
- Flux estimation
- Two-stage optimization process
- Model fitting with instrumental effects
- Caching strategies for performance

The optimization handles both spatial and intensity parameters simultaneously.
"""

from functools import lru_cache
from typing import Callable, Iterable, Literal
import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.signal import convolve

from .images import _rbilinear_relative
from .images import _shift
from .io import SimulationDataLoader
from .mask import _convolution_kernel_psfy
from .mask import _detector_footprint
from .mask import _interpmax
from .mask import apply_vignetting
from .mask import chop
from .mask import CodedMaskCamera
from .mask import count
from .mask import decode
from .mask import model_shadowgram
from .mask import model_sky
from .mask import shift2pos
from .mask import snratio
from .mask import strip
from .mask import variance
from .types import UpscaleFactor


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

    def f(args: npt.NDArray, truth: npt.NDArray, camera: CodedMaskCamera) -> float:
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
        (min_i, max_i, min_j, max_j), _ = chop(camera, shift2pos(camera, shift_x, shift_y))
        truth_chopped = truth[min_i:max_i, min_j:max_j]
        model_chopped = model[min_i:max_i, min_j:max_j]
        residual = truth_chopped - model_chopped
        mse = np.mean(np.square(residual))
        return float(mse)

    return f


def optimize(
    camera: CodedMaskCamera,
    sky: npt.NDArray,
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
    x, y, fluence = map(float, results.x[:3])

    # releases model cache memory.
    _compute_model_fine_cache_clear()
    return x, y, fluence


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
    dataset: Literal["detected", "reconstructed"] = "reconstructed",
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
        dataset: Which dataset to analyze. Either "detected" (simulated data prior to reconstruction)
            or "reconstructed" (position-reconstructed data). Defaults to "reconstructed"

    Yields:
        For each iteration, yields:
            - A tuple of two (x, y, fluence) tuples, one for each camera's detected source,
              where x,y are sky-shift coordinates in mm and fluence is source intensity
            - A tuple of two residual sky images after source removal, one for each camera
            Note: Results are ordered to match sdl_cam1a, sdl_cam1b order

    Raises:
        ValueError: If cameras are not oriented orthogonally (90° rotation in azimuth)
        ValueError: If dataset argument is not "detected" or "reconstructed"
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

    if dataset not in ["detected", "reconstructed"]:
        raise ValueError("Argument `dataset` must be either `detected` or `reconstructed`.")

    def direction_match(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> bool:
        """Determines if source positions from both cameras correspond to the same sky location.
        Compares source positions accounting for the 90° camera rotation. Positions are
        considered matching if they are within one slit width of each other after rotation.
        TODO: not urgent, but in a future we should make this work for arbitray camera rotations."""
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
        # `find_candidates`)
        *_, latest_a = pa
        for b in pb:
            if direction_match(latest_a, b):
                return latest_a, b

        *_, latest_b = pb
        for a in pa:
            if direction_match(a, latest_b):
                return a, latest_b
        return tuple()

    def init_get_arg(skys: tuple, batchsize: int = 1000) -> Callable:
        """This hides a reservoirs-batch mechanism for quickly selecting candidates,
        and initializes the data structures it relies on."""
        # variance is clipped to improve numerical stability for off-axis sources,
        # which may result in very few counts.
        snrs = tuple(snratio(sky, np.clip(var_, a_min=1, a_max=None)) for sky, var_ in zip(skys, variances))

        # we sort source directions by significance.
        # this is kind of costly because the sky arrays may be very large.
        # TODO: improve on this only sorting matrix elements over a threshold.
        # sorted directions are moved to a reservoir.
        reservoirs = [np.argsort(snr, axis=None) for snr in snrs]

        # integrating source intensities over aperture for all matrix elements is
        # computationally unfeasable. to avoid this, we execute this computation over small batches.
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
                    (min_i, max_i, min_j, max_j), _ = strip(camera, arg)
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

    def find_candidates(skys: tuple, max_pending=6666) -> tuple:
        """Returns candidate, compatible sources for the two cameras.
        Worst case complexity is O(n^2) but amortized costs are much smaller."""
        get_arg = init_get_arg(skys)
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

    def subtract(arg: tuple[int, int], sky: np.ndarray):
        """Runs optimizer and subtract source."""
        try:
            source = optimize(
                camera,
                sky,
                arg,
                psfy=True if dataset == "reconstructed" else False,
                vignetting=True if dataset == "reconstructed" else False,
            )
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}") from e
        model = model_sky(camera, *source)
        residual = sky - model
        return source, residual

    detectors = tuple(count(camera, sdl.data)[0] for sdl in sdls)
    variances = tuple(variance(camera, d) for d in detectors)
    skys = tuple(decode(camera, d) for d in detectors)
    for i in range(max_iterations):
        candidates = find_candidates(skys)
        if not candidates:
            break
        try:
            sources, skys = zip(*(subtract(index, sky) for index, sky in zip(candidates, skys)))
        except RuntimeError as e:
            warnings.warn(f"Optimizer failed at iteration {i}:\n\n{e}")
            continue
        yield (sources, skys) if sdls == (sdl_cam1a, sdl_cam1b) else (sources[::-1], skys)
