import matplotlib.pyplot as plt
from iros.optim import optimize
from iros.mask import fetch_camera, count, decode, model_sky

from iros.images import argmax
from iros.io import fetch_simulation
from iros.assets import path_wfm_mask

from iros.utils import catchtime


def plotsky(sky, points=tuple(), title="", vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    c0 = ax.imshow(sky, vmax=-sky.min() if vmax is None else vmax, vmin=0)
    for p in points:
        ax.scatter(p[1], p[0], s=80, facecolors='none', edgecolors='white')
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


def main():
    print(f"Computing first sky image.")
    # sdl = fetch_simulation("/home/deppep/Documents/wfm_sims/id10") # faint crowded
    # sdl = fetch_simulation("/home/deppep/Documents/wfm_sims/id12") # single 4.15
    sdl = fetch_simulation("../../simulations/id00")  # double strong
    wfm = fetch_camera(path_wfm_mask, (8, 1))

    detector = count(wfm, sdl.reconstructed["cam1a"])
    sky = decode(wfm, detector)

    print(f"Starting optimization.")
    with catchtime("optimization"):
        (shift, flux), model, receipt = optimize(wfm, sky, argmax(sky),)
    print(f"Ended optimization.")

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
