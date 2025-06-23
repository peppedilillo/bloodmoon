# compile these:
MASK_PATH = "wfm_mask.fits"
CATALOG_PATH = "/path/to/catalog.csv"  # expected to contain RA and DEC for the sources
SIMDIR_PATH = "/path/to/simulation_directory/"  # expected to contain a `detected` simulation
IROS_MAX_ITERATIONS = 5

import matplotlib.pyplot as plt
# script starts
import numpy as np
import pandas as pd

from bloodmoon import codedmask
from bloodmoon import iros
from bloodmoon import simulation
from bloodmoon import simulation_files
from bloodmoon.coords import equatorial2shift
from bloodmoon.coords import shift2angle
from bloodmoon.io import SimulationDataLoader
from bloodmoon.utils import clock


def table(shifts: list[tuple]) -> pd.DataFrame:
    sxs, sys = zip(*shifts)
    return pd.DataFrame(
        {
            "SHIFTX": sxs,
            "SHIFTY": sys,
            "THETAX": [*map(lambda x: shift2angle(wfm, x), sxs)],
            "THETAY": [*map(lambda y: shift2angle(wfm, y), sys)],
        }
    )


def enrich(sdl: SimulationDataLoader, catalog: pd.DataFrame) -> pd.DataFrame:
    shifts = [equatorial2shift(sdl, wfm, ra, dec) for i, (ra, dec) in catalog[["RA", "DEC"]].iterrows()]
    sxs, sys = zip(*shifts)
    return pd.concat(
        (
            catalog,
            pd.DataFrame(
                {
                    "SHIFTX": sxs,
                    "SHIFTY": sys,
                    "THETAX": [*map(lambda x: shift2angle(wfm, x), sxs)],
                    "THETAY": [*map(lambda y: shift2angle(wfm, y), sys)],
                }
            ),
        ),
        axis=1,
    )


def run_iros(iterations: int):
    def callback(i: int, x):
        print(f"on iteration {i}..")
        sources, _ = x
        source1a, source1b = sources
        sx1a, sy1a, *_ = source1a
        sx1b, sy1b, *_ = source1b
        return (sx1a, sy1a), (sx1b, sy1b)

    loop = iros(wfm, sdl1a, sdl1b, psfy=False, vignetting=True, max_iterations=iterations)
    return [callback(i, s) for i, s in enumerate(loop, start=1)]


def association_table(measured: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    out = {}
    for i, row in measured.iterrows():
        deltax = catalog["SHIFTX"] - row["SHIFTX"]
        deltay = catalog["SHIFTY"] - row["SHIFTY"]
        out[i] = int(np.argmin(deltax**2 + deltay**2))
    return out


wfm = codedmask(MASK_PATH, upscale_x=5, upscale_y=1)
catalog_df = pd.read_csv(CATALOG_PATH)
filepaths = simulation_files(SIMDIR_PATH)

sdl1a = simulation(filepaths["cam1a"]["detected"])
sdl1b = simulation(filepaths["cam1b"]["detected"])

# adds shift columns to the catalog
catalog1a_df = enrich(sdl1a, catalog_df)
catalog1a_df.to_csv("catalog1a.txt", index=False)
catalog1b_df = enrich(sdl1b, catalog_df)
catalog1b_df.to_csv("catalog1b.txt", index=False)

# run iros
with clock("iros"):
    shifts = run_iros(IROS_MAX_ITERATIONS)
measured1a_df, measured1b_df = [*map(table, [*zip(*shifts)])]

# builds associations between the measured sources and the sources in the catalog
meas2cat1a = association_table(measured1a_df, catalog1a_df)
cat2meas1a = {v: k for k, v in meas2cat1a.items()}
meas2cat1b = association_table(measured1b_df, catalog1b_df)
cat2meas1b = {v: k for k, v in meas2cat1b.items()}

# plots the residual as off-axis angle for cam1a relative to axis x
xs = np.array([catalog1a_df["THETAX"][k] for k in cat2meas1a.keys()])
ys = np.array([measured1a_df["THETAX"][v] for v in cat2meas1a.values()])
ys = (ys - xs) * 60  # arcmin

plt.figure(figsize=(12, 8))
plt.scatter(xs, ys)
plt.ylabel("measured - true [arcmin]")
plt.title("CAM1A X-axis")
plt.savefig("cam1a_xaxis_theta.png")
plt.close()

# plots the residual as off-axis angle for cam1a relative to axis y
xs = np.array([catalog1a_df["THETAX"][k] for k in cat2meas1a.keys()])
ys = np.array([measured1a_df["THETAY"][v] for v in cat2meas1a.values()])
ys = ys * 60  # arcmin

plt.figure(figsize=(12, 8))
plt.scatter(xs, ys)
plt.ylabel("measured - true [arcmin]")
plt.title("CAM1A Y-axis")
plt.savefig("cam1a_yaxis_theta.png")
plt.close()

# plots the residual shifts over the x axis.
xs = np.array([catalog1a_df["SHIFTX"][k] for k in cat2meas1a.keys()])
ys = np.array([measured1a_df["SHIFTX"][v] for v in cat2meas1a.values()])
ys = ys - xs

plt.figure(figsize=(12, 8))
plt.scatter(xs, ys)
plt.ylabel("measured - true [mm]")
plt.title("CAM1A X-axis")
plt.savefig("cam1a_xaxis_shift.png")
plt.close()
