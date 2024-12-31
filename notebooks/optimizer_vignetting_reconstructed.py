from bisect import bisect
from functools import cache
from functools import lru_cache
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

from iros.assets import _path_test_mask
from iros.images import _rbilinear
from iros.images import _shift
from iros.images import argmax
from iros.images import _erosion
from iros.io import fetch_simulation
from iros.mask import _bisect_interval, _chop, _interpmax
from iros.mask import _detector_footprint
from iros.mask import CodedMaskCamera
from iros.mask import count
from iros.mask import decode
from iros.mask import fetch_camera
from iros.types import UpscaleFactor
from iros.utils import clock


# def erode(
#         arr: np.array,
#         step: float,
#         cut: float,
# ) -> np.array:
#     """Simulates finite thickness effect in shadow projection.
#
#     Args:
#         arr: 2D input array of integers representing the projected shadow.
#         step: The bin step
#         cut: Maximum cut width
#
#     Returns:
#         Modified array with shadow effects applied
#     """
#
#     # how many bins, summing on both sides, should we cut?
#     ncuts =  cut / step
#     # remove as many bins as we can by shifting
#     nshifts = int(ncuts // 2)
#     if nshifts:
#         rshift = _shift(arr, (0, +nshifts))
#         lshift = _shift(arr, (0, -nshifts))
#         arr_ = arr * ((rshift > 0) & (lshift > 0))
#     else:
#         arr_ = arr
#
#     # fix borders
#     decimal = (ncuts - 2 * nshifts)
#
#     _lborder_mask = (arr_ - _shift(arr_, (0, +1)) > 0)
#     _rborder_mask = (arr_ - _shift(arr_, (0, -1)) > 0)
#     lborder_mask = _lborder_mask & (~_rborder_mask)
#     rborder_mask = _rborder_mask & (~_lborder_mask)
#     cborder_mask = _lborder_mask & _rborder_mask
#
#     return (
#             arr_ +
#             (1 - decimal / 2) * lborder_mask - arr_ * lborder_mask +
#             (1 - decimal / 2) * rborder_mask - arr_ * rborder_mask +
#             (1 - decimal) * cborder_mask - arr_ * cborder_mask
#     )


def apply_vignetting(camera: CodedMaskCamera, shadowgram: np.array, shift_x, shift_y):
    bins = camera.bins_detector

    angle_x_rad = abs(np.arctan(shift_x / camera.mdl["mask_detector_distance"]))
    red_factor = 1 * camera.mdl["mask_thickness"] * np.tan(angle_x_rad)
    sg1 = _erosion(shadowgram, bins.x[1] - bins.x[0], red_factor)

    angle_y_rad = abs(np.arctan(shift_y / camera.mdl["mask_detector_distance"]))
    red_factor = 1 * camera.mdl["mask_thickness"] * np.tan(angle_y_rad)
    sg2 = _erosion(shadowgram.T, bins.y[1] - bins.y[0], red_factor)

    # fig, ax = plt.subplots(1,1)
    # c0 = ax.matshow(camera.mask - sg1 * sg2.T)
    # fig.colorbar(c0, ax=ax, location="bottom", pad=0.05, shrink=.45)
    # plt.show()
    return sg1 * sg2.T


params_psfy = {
    "center": 0,
    "alpha": 0.2592,
    "beta": 0.5972,
}


def modsech(x, norm, center, alpha, beta):
    return norm / np.cosh(np.abs((x - center) / alpha) ** beta)


def norm_modsech(center, alpha, beta):
    return partial(modsech, norm=1, center=center, alpha=alpha, beta=beta)


psfy = norm_modsech(**params_psfy)


@lru_cache(maxsize=2)
def get_kernel(camera):
    bins = camera.bins_detector
    min_bin, max_bin = _bisect_interval(bins.y, -camera.mdl["slit_deltay"], camera.mdl["slit_deltay"])
    bin_edges = bins.y[min_bin: max_bin + 1]
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    kernel = psfy(midpoints).reshape(len(midpoints), -1)
    kernel = kernel / np. sum(kernel)
    return kernel


@cache
def shift2pos(camera, shift_x, shift_y):
    # TODO: needs guardrail for boundaries
    return bisect(camera.bins_sky.y, shift_y) - 1, bisect(camera.bins_sky.x, shift_x) - 1


def compute_model(camera, shift_x, shift_y, flux):
    processed_mask = convolve(apply_vignetting(camera, camera.mask, shift_x, shift_y), get_kernel(camera), mode="same")
    n, m = camera.sky_shape
    i_min, i_max, j_min, j_max = _detector_footprint(camera)
    model = np.zeros(camera.detector_shape)
    for (pos, weight) in _rbilinear(shift_x, shift_y, *camera.bins_sky).items():
        i, j = pos
        r, c = (n // 2 - i), (m // 2 - j)
        shifted_mask = _shift(processed_mask, (r, c))
        shadow = shifted_mask[i_min: i_max, j_min: j_max]
        model += shadow * weight
    model *= camera.bulk
    model /= np.sum(model)
    return decode(camera, model * flux)


class Optimizer:
    def __init__(self, camera: CodedMaskCamera):
        self.camera = camera

    def __call__(self, sky):
        shift_start_x, shift_start_y = _interpmax(self.camera, argmax(sky), sky, UpscaleFactor(10, 10))
        print(f"Interpolated maximum shift x = {shift_start_x:.3f}mm, y = {shift_start_y:.3f}mm")
        flux_start = sky.max()

        print("Starting coarse flux optimization")
        results = minimize(
            lambda args: self.loss((shift_start_x, shift_start_y, args[0]), sky),
            x0=np.array((flux_start,)),
            method="L-BFGS-B",
            bounds=[
                (0.75 * flux_start, 1.5 * flux_start),
            ],
            options={
                "maxiter": 10,
                "iprint": 1,
                "ftol": 10e-4,
            }
        )
        flux = results.x[0]
        print(f"Flux end values is {flux:.2f}")

        print("Starting fine optimization")
        results = minimize(
            lambda args: self.loss((args[0], args[1], args[2]), sky),
            x0=np.array((shift_start_x, shift_start_y, flux,)),
            method="L-BFGS-B",
            bounds=[
                (shift_start_x - self.camera.mdl["slit_deltax"] / 2, shift_start_x + self.camera.mdl["slit_deltax"] / 2),
                (shift_start_y - self.camera.mdl["slit_deltay"] / 2, shift_start_y + self.camera.mdl["slit_deltay"] / 2),
                (0.95 * flux, 1.05 * flux),
            ],
            options={
                "maxiter": 20,
                "iprint": 1,
                "ftol": 10e-5,
            }
        )
        x, y, flux = results.x[:3]
        print(f"Final shift x = {x:.3f}mm, x = {y:.3f}mm, flux = {flux:.1f}")
        return ((x, y), flux), results

    def loss(self, args, sky):
        shift_x, shift_y, flux = args
        model = compute_model(self.camera, *args)
        (min_i, max_i, min_j, max_j), _ = _chop(self.camera, shift2pos(self.camera, shift_x, shift_y))
        truth_chopped = sky[min_i:max_i, min_j:max_j]
        model_chopped = model[min_i:max_i, min_j:max_j]
        residual = truth_chopped - model_chopped
        loss = np.mean(np.square(residual))
        return loss


def plotsky(sky, points=tuple(), title="", vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    c0 = ax.imshow(sky, vmax=-sky.min() if vmax is None else vmax, vmin=0)
    for p in points:
        ax.scatter(p[1], p[0], s=80, facecolors='none', edgecolors='white')
    fig.colorbar(c0, ax=ax, label="Correlation", location="bottom", pad=0.05, shrink=.45)
    ax.set_title(title)
    plt.show()
    return fig, ax


def plotres(sky, model, row, col, camera):
    fig, axs = plt.subplots(2, 1, figsize=(7, 7))
    axs[0].stairs((sky - model)[row], edges=camera.bins_sky.x, label="subtracted")
    axs[0].stairs(model[row], edges=camera.bins_sky.x, label="model")
    axs[0].stairs(sky[row], edges=camera.bins_sky.x,  label="truth")
    axs[0].set_title("Slices")
    axs[1].stairs((sky - model)[:, col], edges=camera.bins_sky.y,  label="subtracted")
    axs[1].stairs(model[:, col], edges=camera.bins_sky.y, label="model")
    axs[1].stairs(sky[:, col], edges=camera.bins_sky.y, label="truth")
    plt.legend()
    plt.show()
    return fig, axs


def main():
    print(f"Computing first sky image.")
    # sdl = fetch_simulation("/home/deppep/Documents/wfm_sims/id10") # faint crowded
    # sdl = fetch_simulation("/home/deppep/Documents/wfm_sims/id12") # single 4.15
    sdl = fetch_simulation("../../simulations/id00")  # double strong
    # sdl = fetch_simulation("/home/deppep/Documents/wfm_sims/20241011_galctr_rxte_sax_2-30keV_1ks_2cams_sources_cxb")
    wfm = fetch_camera(_path_test_mask, (5, 3))

    detector = count(wfm, sdl.reconstructed["cam1a"])
    sky = decode(wfm, detector)

    print(f"Starting optimization.")
    with clock("optimization"):
        optimize = Optimizer(wfm)
        (shift, flux), receipt = optimize(sky)
    print(f"Ended optimization.")
    model = compute_model(wfm, *shift, flux)
    return (
        lambda : plotres(sky, model, *argmax(sky), wfm),
        lambda res=True: (
            plotsky(sky - model, points=(argmax(sky),), vmax=(sky - model).max() / 2) if res else
            plotsky(sky, points=(argmax(sky),), vmax=sky.max() / 2)),
    )


if __name__ == "__main__":
    _plotres, _plotsky = main()
    _plotres()
    _plotsky()

    # print(f"Computing square sky image.")
    # wfm_ = fetch_camera(path_wfm_mask, (5, 8))
    # detector_ = count(wfm_, sdl.detected["cam1a"])
    # sky_ = decode(wfm_, detector_)
    # model_ = compute_model(wfm_, *shift, flux)