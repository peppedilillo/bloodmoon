"""
bloodmoon: A Python Library for WFM Coded Mask Analysis

bloodmoon provides tools for analyzing data from the Wide Field Monitor (WFM)
coded mask instrument. It supports both simulated and real data analysis with
features for image reconstruction, source detection, and parameter estimation.

Main Components:
---------------
camera : Function
    Creates a CodedMaskCamera instance from mask FITS file

simulation : Function
    Loads and manages WFM simulation data

count : Function
    Creates detector images from photon event data

decode : Function
    Reconstructs sky images using balanced cross-correlation

model_shadowgram : Function
    Generates realistic detector shadowgrams

model_sky : Function
    Creates simulated sky images with instrumental effects

optimize : Function
    Estimates source parameters through two-stage optimization

Example:
--------
>>> import bloodmoon
>>> wfm = bloodmoon.camera("wfm_mask.fits")  # Load camera
>>> sdl = bloodmoon.simulation("simdata/")    # Load simulation
>>> detector, bins = bloodmoon.count(wfm, sdl.reconstructed["cam1a"])
>>> sky = bloodmoon.decode(wfm, detector)     # Reconstruct sky image

For detailed documentation on specific functions, use help() on the individual
components or refer to the module docstrings.
"""

from .mask import camera
from .mask import count
from .mask import decode
from .mask import model_sky
from .mask import model_shadowgram
from .optim import optimize
from .io import simulation
