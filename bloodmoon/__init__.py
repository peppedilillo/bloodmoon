"""
bloodmoon: A Python Library for WFM Coded Mask Analysis

bloodmoon provides tools for analyzing data from the Wide Field Monitor (WFM)
coded mask instrument. It supports both simulated and real data analysis with
features for image reconstruction, source detection, and parameter estimation.

Main Components:
---------------
codedmask : Function
    Creates a CodedMaskCamera instance from mask FITS file

simulation : Function
    Loads and manages WFM simulation data

count : Function
    Creates detector images from photon event data

decode : Function
    Reconstructs sky images using balanced cross-correlation

sky_variance : Function
    Computes balanced sky image variance

sky_significance : Function
    Computes balanced sky image signal-to-noise ratio

model_shadowgram : Function
    Generates realistic detector shadowgrams

model_sky : Function
    Creates simulated sky images with instrumental effects

optimize : Function
    Estimates source parameters through two-stage optimization

iros : Function
    Source subtraction by Iterative Removal of Sources method.


For detailed documentation on specific functions, use help() on the individual
components or refer to the module docstrings.
"""

from .io import simulation
from .io import simulation_files
from .mask import codedmask
from .mask import CodedMaskCamera
from .mask import CodedMaskSpecs
from .mask import count
from .mask import cutout
from .mask import decode
from .mask import sky_variance
from .mask import sky_significance
from .optim import iros
from .optim import model_shadowgram
from .optim import model_sky
from .optim import optimize
