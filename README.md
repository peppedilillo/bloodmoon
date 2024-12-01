To install, clone the repo and `pip install '.[dev]'` it.

# Examples

## Image reconstruction

Load simulation and mask data with upscaling. Compute detector counts. Compute balanced reconstructed image by cross-correlation. Compose balanced reconstructions from two camera into a single image. Find the maximum in the reconstructed composed image and in the components images.

```python 
from iros.io import fetch_simulation
from iros.mask import fetch_camera, encode, decode, count
from iros.images import compose, argmax
from iros.assets import path_wfm_mask

sdl = fetch_simulation("/home/deppep/Dropbox/Progetti/masks/simulations/id00/")
wfm = fetch_camera(path_wfm_mask, (5, 8))

detector_1a = count(wfm, sdl.detected["cam1a"])
balanced_1a, var_1a = decode(wfm, detector_1a)

detector_1b = count(wfm, sdl.detected["cam1b"])
balanced_1b, var_1b = decode(wfm, detector_1b)

composed, composed_f = compose(balanced_1a, balanced_1b)
max_composed = argmax(composed)
max_balanced_1a, max_balanced_1b = composed_f(*max_composed)
```
