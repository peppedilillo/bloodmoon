from bisect import bisect
from functools import cache

import numpy as np
from scipy.optimize import minimize

from iros.assets import path_wfm_mask
from iros.images import _chop
from iros.images import _interpmax
from iros.images import _rbilinear
from iros.images import argmax
from iros.images import shadowgram
from iros.io import fetch_simulation
from iros.mask import CodedMaskCamera
from iros.mask import count
from iros.mask import decode
from iros.mask import fetch_camera
from iros.mask import UpscaleFactor


@cache
def shift2pos(camera, shift_x, shift_y):
    # TODO: needs guardrail for boundaries
    return bisect(camera.bins_sky.y, shift_y) - 1, bisect(camera.bins_sky.x, shift_x) - 1


@cache
def model_template(camera, posweights):
    model = np.zeros(camera.detector_shape)
    for (pos, weight) in posweights:
        model += shadowgram(camera, pos).astype(float) * weight
    model *= camera.bulk
    model /= np.sum(model)
    return model


def compute_model(camera, shift_x, shift_y, flux):
    components = _rbilinear(shift_x, shift_y, *camera.bins_sky)
    posweights = tuple(((p, w) for p, w in components.items()))
    model_normalized = model_template(camera, posweights)
    return decode(camera, model_normalized * flux)


class Optimizer:
    def __init__(self, camera: CodedMaskCamera):
        self.camera = camera

    def __call__(self, sky):
        shift_start_x, shift_start_y = _interpmax(self.camera, argmax(sky), sky, UpscaleFactor(10, 10))
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
                "maxiter": 20,
                "iprint": 1,
            }
        )
        flux = results.x[0]
        print(f"Flux end values is {flux:.2f}")

        print("Starting fine optimization")
        results = minimize(
            lambda args: self.loss((args[0], shift_start_y, args[1]), sky),
            x0=np.array((shift_start_x, flux,)),
            method="L-BFGS-B",
            bounds=[
                (shift_start_x - self.camera.mdl["slit_deltax"], shift_start_x + self.camera.mdl["slit_deltax"]),
                (0.95 * flux, 1.05 * flux),
            ],
            options={
                "maxiter": 20,
                "iprint": 1,
            }
        )
        x, flux = results.x[:2]

        return ((x, shift_start_y), flux), results

    def loss(self, args, sky):
        shift_x, shift_y, flux = args
        model = compute_model(self.camera, *args)
        (min_i, max_i, min_j, max_j), _ = _chop(self.camera, shift2pos(self.camera, shift_x, shift_y))
        truth_chopped = sky[min_i:max_i, min_j:max_j]
        model_chopped = model[min_i:max_i, min_j:max_j]
        residual = truth_chopped - model_chopped
        loss = np.mean(np.square(residual))
        return loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def plotsky(sky, points=tuple(), title="", vmax=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        c0 = ax.imshow(sky, vmax=-sky.min() if vmax is None else vmax, vmin=0)
        for p in points:
            ax.scatter(p[1], p[0], s=80, facecolors='none', edgecolors='k')
        fig.colorbar(c0, ax=ax, label="Correlation", location="bottom", pad=0.05, shrink=.45)
        ax.set_title(title)
        plt.show()
        return fig, ax


    def plotres(sky, model, row, col, camera):
        fig, axs = plt.subplots(2, 1, figsize=(9, 8))
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


    print(f"Computing first sky image.")
    sdl = fetch_simulation("../../simulations/id13")
    wfm = fetch_camera(path_wfm_mask, (8, 1))

    detector = count(wfm, sdl.detected["cam1a"])
    sky = decode(wfm, detector)

    print(f"Starting optimization.")
    optimize = Optimizer(wfm)
    (shift, flux), receipt = optimize(sky)
    print(f"Ended optimization.")
    model = compute_model(wfm, *shift, flux)

    # print(f"Computing square sky image.")
    # wfm_ = fetch_camera(path_wfm_mask, (5, 8))
    # detector_ = count(wfm_, sdl.detected["cam1a"])
    # sky_ = decode(wfm_, detector_)
    # model_ = compute_model(wfm_, *shift, flux)
