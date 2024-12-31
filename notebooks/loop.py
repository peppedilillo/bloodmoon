from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from iros.assets import _path_test_mask
from iros.images import compose
from iros.images import _upscale
from iros.io import fetch_simulation
from iros.mask import _chop
from iros.mask import CodedMaskCamera
from iros.mask import count
from iros.mask import decode
from iros.mask import fetch_camera
from iros.mask import model_sky
from iros.optim import optimize
from iros.types import UpscaleFactor


class Rectangle(NamedTuple):
    p1: tuple[float, float]
    p2: tuple[float, float]


def intersect(r_a: Rectangle, r_b: Rectangle) -> bool:
    ax1, ay1 = r_a.p1
    ax2, ay2 = r_a.p2
    bx1, by1 = r_b.p1
    bx2, by2 = r_b.p2

    # one  is completely to the left of the other
    if ax2 < bx1 or bx2 < ax1:
        return False
    # one rectangle is completely above the other
    if ay2 < by1 or by2 < ay1:
        return False
    return True


def centered(r_a: Rectangle, r_b: Rectangle, tolerance: float = 0.25) -> bool:
    ax1, ay1 = r_a.p1
    ax2, ay2 = r_a.p2
    bx1, by1 = r_b.p1
    bx2, by2 = r_b.p2

    center_a_x = (ax1 + ax2) / 2
    center_a_y = (ay1 + ay2) / 2
    center_b_x = (bx1 + bx2) / 2
    center_b_y = (by1 + by2) / 2

    width_a = abs(ax2 - ax1)
    height_a = abs(ay2 - ay1)
    width_b = abs(bx2 - bx1)
    height_b = abs(by2 - by1)

    return (
        abs(center_a_x - center_b_x) < min(width_a, width_b) * tolerance and
        abs(center_a_y - center_b_y) < min(height_a, height_b) * tolerance
    )


def init_finder(camera: CodedMaskCamera):
    def close(a: tuple[int, int], b: tuple[int, int]):
        *_, bins_a = _chop(camera, a)
        *_, bins_b = _chop(camera, b)
        r_a = Rectangle(
            (bins_a.x[0], bins_a.y[0]),
            (bins_a.x[-1], bins_a.y[-1]),
        )
        r_b = Rectangle(
            (bins_b.x[0], bins_b.y[0]),
            (bins_b.x[-1], bins_b.y[-1]),
        )
        r_b = Rectangle(
            (-r_b.p2[1], r_b.p2[0]),
            (-r_b.p1[1], r_b.p1[0]),
        )
        return centered(r_a, r_b)

    def match(as_, bs) -> tuple:
        for b in bs:
            if close(as_[-1], b):
                return as_[-1], b
        for a in as_:
            if close(a, bs[-1]):
                return a, bs[-1]
        return tuple()

    def find_candidates(skys) -> dict:
        pending = {c: [] for c in sdl.camkeys}
        argsorted = {c: np.argsort(skys[c], axis=None) for c in sdl.camkeys}
        while not (matches := match(*(pending[c] for c in sdl.camkeys))):
            for c in sdl.camkeys:
                pending[c].append(np.unravel_index(argsorted[c][-1], skys[c].shape))
                argsorted[c] = argsorted[c][:-1]
        return {c: m for c, m in zip(sdl.camkeys, matches)} if matches else {}
    return find_candidates


def rootmeansquare(sky:np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(sky - sky.mean())))


def plot_intersection(r_a: Rectangle, r_b: Rectangle, camera):
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.add_patch(patches.Rectangle(r_a.p1, r_a.p2[0] - r_a.p1[0], r_a.p2[1] - r_a.p1[1]))
    ax.add_patch(patches.Rectangle(r_b.p1, r_b.p2[0] - r_b.p1[0], r_b.p2[1] - r_b.p1[1]))
    ax.set_xlim(camera.bins_sky.x[0], camera.bins_sky.x[-1])
    ax.set_ylim(camera.bins_sky.y[0], camera.bins_sky.y[-1])
    ax.set_aspect('equal')
    plt.show()
    return


COUNTER = [0]


def _plotsky(sky, points=tuple(), title="", vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=300)
    c0 = ax.imshow(sky, vmax=np.quantile(sky, 0.9999) if vmax is None else vmax, vmin=0)
    for p in points:
        ax.scatter(p[1], p[0], s=80, facecolors='none', edgecolors='white')
    #fig.colorbar(c0, ax=ax, label="Correlation", location="bottom", shrink=.25)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"pic_detected/it{COUNTER[0]}.png")
    COUNTER[0] += 1
    return fig, ax


def plotskys(skys):
    print("Plotting and saving sky images")
    sky_1a = skys["cam1a"]
    sky_1b = skys["cam1b"]
    sky_upscaled_1a = _upscale(sky_1a, UpscaleFactor(1, 8))
    sky_upscaled_1b = _upscale(sky_1b, UpscaleFactor(1, 8))
    composed, _ = compose(
        sky_upscaled_1a,
        sky_upscaled_1b,
        strict=False,
    )
    _plotsky(composed)

if __name__ == "__main__":
    from time import time
    tic = time()

    print("Fetching simulation and computing initial images")
    sdl = fetch_simulation("../../simulations/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb")
    wfm = fetch_camera(_path_test_mask, (5, 1))
    detectors_ = {c: count(wfm, sdl.reconstructed[c]) for c in sdl.camkeys}
    skys_ = {c: decode(wfm, detectors_[c]) for c in sdl.camkeys}
    # plotskys(skys_)

    frms = None
    find_candidates = init_finder(wfm)
    skys = {k: v.copy() for k, v in skys_.items()}
    rms = {c: rootmeansquare(skys_[c]) for c in sdl.camkeys}
    sources = {c: [] for c in sdl.camkeys}
    nit = 0
    while nit < 20:
        print(f"\n\n>> Iteration number {nit}")
        if not (candidates := find_candidates(skys)):
            break
        print(f">> Found candidates: "
              f"\n>>  * cam1a: {tuple(map(int, candidates["cam1a"]))}"
              f"\n>>  * cam1b: {tuple(map(int, candidates["cam1b"]))}")
        for c, m in candidates.items():
            (shift, flux) = optimize(wfm, skys[c], m, verbose=False)
            print(f">> Ended optimization ({c}) at shift (x= {shift[0]:+.2f}, y= {shift[1]:+.2f}), flux= {flux:.2f}.")
            model = model_sky(wfm, *shift, flux)
            skys[c] -= model
            frms = 100 * ((new_rms := rootmeansquare(skys[c])) - rms[c]) / rms[c]
            print(f">> RMS p.c. variation ({c}) sky image: {frms:+.3f}%")
            rms[c] = new_rms
            sources[c].append((shift, flux))
        nit += 1

        toc = time()
        print(f"\nElapsed time since start: {toc - tic:.2f} s")
        # plotskys(skys)
