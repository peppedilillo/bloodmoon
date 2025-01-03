from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from bloodmoon import simulation, camera, count, decode, optimize, model_sky, chop
from bloodmoon.images import upscale, argmax, compose
from bloodmoon.assets import _path_test_mask
from bloodmoon.io import SimulationDataLoader
from bloodmoon.mask import CodedMaskCamera


PLOT_COUNTER = [0]


def _plotsky(sky, points=tuple(), title="", vmax=None, dpi=300):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=dpi)
    ax.imshow(sky, vmax=np.quantile(sky, 0.9999) if vmax is None else vmax, vmin=0)
    for p in points:
        ax.scatter(p[1], p[0], s=80, facecolors='none', edgecolors='white')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"../notebooks/pics_detected/it{PLOT_COUNTER[0]}.png")
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


def _source_smaller_residual_maximum(camera: CodedMaskCamera) -> Callable:
    thr_residual = [0]
    sky = [None]

    def f(residual: np.array):
        if sky[0] is None:
            sky[0] = residual
            return False
        else:
            model = sky[0] - residual
            (min_i, max_i, min_j, max_j), _ = chop(camera, argmax(model))
            max_residual = np.max(np.abs(residual[min_i:max_i, min_j:max_j]))
            max_model = np.max(model)
            thr_residual[0] = max(thr_residual[0], max_residual)
            print(f"model maximum= {max_model:.1f}, residual_crit= {thr_residual[0]:.1f}")
            sky[0] = residual
            if max_model < thr_residual[0]:
                return True
            return False

    return f


def source_smaller_residual_maximum(camera):
    f1 = _source_smaller_residual_maximum(camera)
    f2 = _source_smaller_residual_maximum(camera)

    def f(residual1: np.array, residual2: np.array) -> bool:
        return f1(residual1) or f2(residual1)
    return f


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

    def close(a: tuple[int, int], b: tuple[int, int],) -> bool:
        ax, ay = camera.bins_sky.x[a[1]], camera.bins_sky.y[a[0]]
        bx, by = -camera.bins_sky.y[b[0]], camera.bins_sky.x[b[1]]  # applying rotation
        return (abs(ax - bx) < camera.mdl["slit_deltax"] and
                abs(ay - by) < camera.mdl["slit_deltay"])

    def match(pending: dict) -> tuple:
        c1, c2 = sdl.camkeys
        if not pending[c1] or not pending[c2]:
            return tuple()

        latest_a = pending[c1][-1]
        for b in pending[c2]:
            if close(latest_a, b):
                return latest_a, b

        latest_b = pending[c2][-1]
        for a in pending[c1]:
            if close(a, latest_b):
                return a, latest_b
        return tuple()

    def find_candidates(skys: dict) -> dict:
        argsorted = {c: np.argsort(skys[c], axis=None) for c in sdl.camkeys}
        pending = {c: [] for c in sdl.camkeys}

        while not (matches := match(pending)) and all(len(argsorted[c]) for c in sdl.camkeys):
            for c in sdl.camkeys:
                arg = np.unravel_index(argsorted[c][-1], skys[c].shape)
                pending[c].append(arg)
                argsorted[c] = argsorted[c][:-1]
        return {c: m for c, m in zip(sdl.camkeys, matches)} if matches else {}

    def subtract(candidate_index: tuple[int, int], sky: np.array):
        source = optimize(camera, skys[c], candidate_index, psfy=False, vignetting=False)
        model = model_sky(camera, *source)
        residual = sky - model
        return source, residual

    def _stop_when(skys: dict) -> bool:
        return stop_when(*tuple(skys[c] for c in sdl.camkeys))

    detectors_ = {c: count(camera, sdl.detected[c])[0] for c in sdl.camkeys}
    skys = {c: decode(camera, detectors_[c]) for c in sdl.camkeys}
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
            print(f"source values {c}: {source}")
            skys[c] = residual
        plotskys(skys)

    return sources


if __name__ == "__main__":
    sdl = simulation("../../simulations/galcenter")
    wfm = camera(_path_test_mask, upscale_x=5, upscale_y=1)
    # c1 = prms_smaller(than=.1)
    # c2 = source_smaller_residual_maximum(wfm)

    main(
        wfm,
        sdl,
        stop_when=prms_smaller(than=.1),  # lambda x,y: c1(x, y) or c2(x, y),
        max_iterations=20,
    )
