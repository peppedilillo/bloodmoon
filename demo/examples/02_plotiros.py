from pathlib import Path
from bloodmoon import codedmask, simulation, simulation_files, count, decode
from bloodmoon.images import compose

import matplotlib.pyplot as plt

# select data
path_mask = "wfm_mask.fits"
path_sim = Path("data/")
wfm = codedmask("wfm_mask.fits", upscale_x=5)
filepaths = simulation_files(path_sim)

# reads the data and puts them in thin container
sdl1a = simulation(filepaths["cam1a"]["reconstructed"])
sdl1b = simulation(filepaths["cam1b"]["reconstructed"])

# make detector image
det1a, _ = count(wfm, sdl1a)
det1b, _ = count(wfm, sdl1b)

# decode the images into sky pictures:
sky1a = decode(wfm, det1a)
sky1b = decode(wfm, det1b)

# compose the sky pictures and plots:
composed, _ = compose(sky1a, sky1b)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(composed, vmin=0, vmax=-composed.min())
plt.show()

