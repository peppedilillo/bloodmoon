from typing import Callable

from astropy.coordinates import angular_separation
import matplotlib.pyplot as plt
import numpy as np

from bloodmoon import codedmask
from bloodmoon import count
from bloodmoon import decode
from bloodmoon import model_sky
from bloodmoon import optimize
from bloodmoon import simulation
from bloodmoon import snratio
from bloodmoon import variance
from bloodmoon.assets import _path_test_mask
from bloodmoon.images import compose
from bloodmoon.images import upscale
from bloodmoon.io import SimulationDataLoader
from bloodmoon.mask import CodedMaskCamera
from bloodmoon.mask import strip

PLOT_COUNTER = [0]


def _plotsky(sky, points=tuple(), title="", vmax=None, dpi=300):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=dpi)
    ax.imshow(sky, vmax=np.quantile(sky, 0.9999) if vmax is None else vmax, vmin=0)
    for p in points:
        ax.scatter(p[1], p[0], s=80, facecolors="none", edgecolors="white")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"../notebooks/pics_reconstructed/it{PLOT_COUNTER[0]}.png")
    plt.close()
    PLOT_COUNTER[0] += 1
    return fig, ax


def plotskys(skys):
    sky_1a = skys["cam1a"]
    sky_1b = skys["cam1b"]
    sky_upscaled_1a = upscale(sky_1a, upscale_x=1, upscale_y=8)
    sky_upscaled_1b = upscale(sky_1b, upscale_x=1, upscale_y=8)
    composed, _ = compose(
        sky_upscaled_1a,
        sky_upscaled_1b,
        strict=False,
    )
    _plotsky(composed, dpi=150)


def rootmeansquare(sky: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(sky - sky.mean())))


def prms() -> tuple[Callable, Callable]:
    rms = [None]

    def latest_rms() -> float | None:
        return rms[0]

    def f(residual: np.array) -> float | None:
        rms_new, rms_old = rootmeansquare(residual), latest_rms()
        rms[0] = rms_new
        if rms_old is None:
            return None
        val = 100 * (rms_new - rms_old) / rms_old
        print(f"percentual rms variation {val:+.2f}")
        return val

    return f, latest_rms


def prms_smaller(than: float) -> Callable:
    prms1, rms1 = prms()
    prms2, rms2 = prms()

    def f(residual1: np.array, residual2: np.array) -> bool:
        prms1_ = prms1(residual1)
        prms2_ = prms2(residual2)
        if prms1_ is None:
            assert prms2_ is None
            return False
        return abs(prms1_) < than and abs(prms2_) < than

    return f


def main(
    camera: CodedMaskCamera,
    sdl: SimulationDataLoader,
    max_iterations: int,
    stop_when: Callable = lambda: False,
):
    """
    Bloodmoon's IROS implementation.

    Args:
        camera:
        sdl:
        max_iterations:
        stop_when:

    Returns:

    """
    # we check for the two camera to be rotated in azimuth by 90°s degrees because
    # the `close` function relies on this fact to identify compatibles candidate pairs.
    lon_x_a1 = sdl.rotations["cam1a"]["x"].az
    lon_x_b1 = sdl.rotations["cam1b"]["x"].az
    lat_x_a1 = sdl.rotations["cam1a"]["x"].al
    lat_x_b1 = sdl.rotations["cam1b"]["x"].al
    lon_z_a1 = sdl.rotations["cam1a"]["z"].az
    lon_z_b1 = sdl.rotations["cam1b"]["z"].az
    lat_z_a1 = sdl.rotations["cam1a"]["z"].al
    lat_z_b1 = sdl.rotations["cam1b"]["z"].al

    if not (
        np.isclose(angular_separation(*map(np.deg2rad, (lon_x_a1, lat_x_a1, lon_x_b1, lat_x_b1))), np.pi)
        and np.isclose(angular_separation(*map(np.deg2rad, (lon_z_a1, lat_z_a1, lon_z_b1, lat_z_b1))), 0.0)
    ):
        raise ValueError("Cameras must be rotated by 90° degrees over azimuth.")

    def close(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> bool:
        """Checks that `a` and `b` indeces corresponds to similiar source direction.
        Tolerances are determined by the size of the smallest aperture over the fine direction.
        At present this function relies on the fact that `cam1b` is rotated by 90° from `cam1a`
        over the azimuth plane of the instrument frame.
        TODO: not urgent, but in a future we should make this work for arbitray camera rotations."""
        ax, ay = camera.bins_sky.x[a[1]], camera.bins_sky.y[a[0]]
        # we apply -90deg rotation to camera b source
        bx, by = -camera.bins_sky.y[b[0]], camera.bins_sky.x[b[1]]
        min_slit = min(camera.mdl["slit_deltax"], camera.mdl["slit_deltay"])
        return abs(ax - bx) < min_slit and abs(ay - by) < min_slit

    def match(pending: dict) -> tuple:
        """Cross-check the last entry in pending to match against all other pending directions"""
        c1, c2 = sdl.camkeys
        if not pending[c1] or not pending[c2]:
            return tuple()

        # we are going to call this each time we get a new couple of candidate indices.
        # we avoid evaluating matches for all pairs at all calls, which would result in
        # repeated evaluations of the same pairs (would result in O(n^3) worst case for
        # `find_candidates`)
        latest_a = pending[c1][-1]
        for b in pending[c2]:
            if close(latest_a, b):
                return latest_a, b

        latest_b = pending[c2][-1]
        for a in pending[c1]:
            if close(a, latest_b):
                return a, latest_b
        return tuple()

    def init_get_arg(skys: dict, batchsize: int = 1000) -> Callable:
        """This hides a reservoir-batch mechanism for quickly selecting candidates,
        and initializes the data structures it relies on."""
        # variance is clipped to improve numerical stability for off-axis sources,
        # which may result in very few counts.
        snrs = {c: snratio(skys[c], np.clip(vars[c], a_min=1, a_max=None)) for c in sdl.camkeys}
        # we sort source directions by significance.
        # this is kind of costly because the sky arrays may be very large.
        # TODO: improve on this only sorting matrix elements over a threshold.
        # sorted directions are moved to a reservoir.
        reservoir = {c: np.argsort(snrs[c], axis=None) for c in sdl.camkeys}
        # integrating source intensities over aperture for all matrix elements is
        # computationally unfeasable. to avoid this, we execute this computation over small batches.
        batches = {c: np.array([]) for c in sdl.camkeys}

        def slit_intensity():
            """Integrates source intensity over mask's aperture."""
            intensities = {c: [] for c in sdl.camkeys}
            for c in sdl.camkeys:
                for arg in batches[c]:
                    (min_i, max_i, min_j, max_j), _ = strip(camera, arg)
                    slit = snrs[c][min_i:max_i, min_j:max_j]
                    intensities[c].append(np.sum(slit))
            return intensities

        def fill():
            """Fill the batches with sorted candidates"""
            for c in sdl.camkeys:
                tail, head = reservoir[c][:-batchsize], reservoir[c][-batchsize:]
                batches[c] = np.array([np.unravel_index(id, snrs[c].shape) for id in head])
                reservoir[c] = tail

            # integrates over mask element aperture and sum between cameras
            intensities = slit_intensity()
            argsort_intensities = np.argsort(np.sum([intensities[c] for c in sdl.camkeys], axis=0))

            # sort candidates in present batch by their integrated-combined intensity
            for c in sdl.camkeys:
                batches[c] = batches[c][argsort_intensities]

        def empty():
            """Checks if batches are empty"""
            return all(not len(batches[c]) for c in sdl.camkeys)

        def get() -> dict | None:
            """Think of this as a faucet getting you one decent direction combo at a time."""
            if empty():
                fill()
                if empty():
                    return None

            out = {c: batches[c][-1] for c in sdl.camkeys}
            for c in sdl.camkeys:
                batches[c] = batches[c][:-1]
            return out

        return get

    def find_candidates(skys: dict) -> dict:
        """Returns candidate, compatible sources for the two cameras.
        Worst case complexity is O(n^2) but amortized costs are much smaller."""
        get_arg = init_get_arg(skys)
        pending = {c: [] for c in sdl.camkeys}

        while not (matches := match(pending)):
            args = get_arg()
            if args is None:
                break
            for c in sdl.camkeys:
                pending[c].append(args[c])
        return {c: m for c, m in zip(sdl.camkeys, matches)} if matches else {}

    def subtract(arg: tuple[int, int], sky: np.array):
        """Runs optimizer and subtract source."""
        source = optimize(camera, skys[c], arg)
        model = model_sky(camera, *source)
        residual = sky - model
        return source, residual

    def _stop_when(skys: dict) -> bool:
        """A helper."""
        return stop_when(*tuple(skys[c] for c in sdl.camkeys))

    detectors_ = {c: count(camera, sdl.reconstructed[c])[0] for c in sdl.camkeys}
    skys = {c: decode(camera, detectors_[c]) for c in sdl.camkeys}
    plotskys(skys)

    vars = {c: variance(camera, detectors_[c]) for c in sdl.camkeys}
    sources = {c: [] for c in sdl.camkeys}
    for i in range(max_iterations):
        if _stop_when(skys):
            break
        candidates = find_candidates(skys)
        if not candidates:
            break
        for c, candidate_index in candidates.items():
            source, residual = subtract(candidate_index, skys[c])
            sources[c].append(source)
            print(f"{i}: source values {c}: {source}")
            skys[c] = residual
        plotskys(skys)
    return sources


if __name__ == "__main__":
    sdl = simulation("../../simulations/galcenter")
    wfm = codedmask(_path_test_mask, upscale_x=5, upscale_y=1)

    main(
        wfm,
        sdl,
        stop_when=prms_smaller(than=0.1),  # lambda x,y: c1(x, y) or c2(x, y),
        max_iterations=20,
    )
