"""
Data loading and handling for WFM mask and simulation data.

This module provides dataclasses and utilities for:
- Loading mask parameters and patterns from FITS files
- Managing simulation data including photon events and pointing information
- Accessing detector, reconstruction, and source information
- Parsing configuration data from FITS headers
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header
from numpy import typing as npt
import numpy as np
from scipy.stats import binned_statistic_2d

from .types import BinsRectangular
from .types import CoordEquatorial
from .types import CoordHorizontal


def validate_fits(filepath: Path | str) -> bool:
    """
    Validate presence and format of FITS file.

    Checks if the specified file exists and has a valid FITS format signature.
    Supports both string paths and Path objects.

    Args:
        filepath: Path to the FITS file to validate

    Returns:
        True if file exists and has valid FITS format

    Raises:
        FileNotFoundError: If FITS file does not exist
        ValueError: If file is not in valid FITS format
    """

    def validate_signature(filepath: Path) -> bool:
        """
        Following astropy's approach, reads the first FITS card (80 bytes)
        and checks for the SIMPLE keyword signature.

        Args:
            filepath: Path object pointing to the file to validate

        Returns:
            bool: True if file has a valid FITS signature, False otherwise
        """
        try:
            with open(filepath, "rb") as file:
                # FITS signature is supposed to be in the first 30 bytes, but to
                # allow reading various invalid files we will check in the first
                # card (80 bytes).
                simple = file.read(80)
        except OSError:
            return False

        fits_signature = b"SIMPLE  =                    T"

        match_sig = simple[:29] == fits_signature[:-1] and simple[29:30] in (b"T", b"F")
        return match_sig

    if not Path(filepath).is_file():
        raise FileNotFoundError(f"FITS file '{filepath}' does not exist.")
    elif not validate_signature(Path(filepath)):
        raise ValueError("File not in valid FITS format.")
    return True


def simulation_files(dirpath: str | Path) -> dict[str, dict[str, Path]]:
    """
    Locate and validate all required FITS files in the root directory.

    Args:
        dirpath: Path to the FITS file.

    Returns:
        Nested dictionary mapping camera IDs to their respective file paths
        for detected, reconstructed, and source data.

    Raises:
        ValueError: If expected files are missing or if multiple matches are found
    """

    def check_and_pick(parent: Path, pattern: str) -> Path:
        matches = tuple(parent.glob(pattern))
        if not matches:
            raise ValueError(f"A file matching the pattern {str(parent / pattern)} is expected but missing.")
        f, *extra_matches = matches
        if extra_matches:
            raise ValueError(
                f"Found unexpected extra matches for glob pattern {str(parent / pattern)}."
                f"File with pattern {pattern} should be unique"
            )
        return f

    dirpath = Path(dirpath)
    return {
        "cam1a": {
            "detected": check_and_pick(dirpath, "cam1a/*detected*.fits"),
            "reconstructed": check_and_pick(dirpath, "cam1a/*reconstructed.fits"),
            "sources": check_and_pick(dirpath, "cam1a/*sources.fits"),
        },
        "cam1b": {
            "detected": check_and_pick(dirpath, "cam1b/*detected*.fits"),
            "reconstructed": check_and_pick(dirpath, "cam1b/*reconstructed.fits"),
            "sources": check_and_pick(dirpath, "cam1b/*sources.fits"),
        },
    }


@dataclass(frozen=True)
class SimulationDataLoader:
    """
    Container for WFM coded mask simulation data.

    The class provides access to photon events and instrument configuration from a
    FITS file containing WFM simulation data for a single camera.

    Attributes:
        filepath (Path): Path to the FITS file

    Properties:
        data: Photon event data from FITS extension 1
        header: Primary FITS header
        pointings (dict[str, CoordEquatorial]): Camera axis directions in equatorial frame
            - 'z': Optical axis pointing (RA/Dec)
            - 'x': Camera x-axis pointing (RA/Dec)
        rotations (dict[str, CoordHorizontal]): Camera axis directions in the instrument's frame
            - 'z': Optical axis pointing (azimuth/altitude)
            - 'x': Camera x-axis pointing (azimuth/altitude)
    """

    filepath: Path

    @cached_property
    def data(self) -> FITS_rec:
        return fits.getdata(self.filepath, ext=1, header=False)

    @cached_property
    def header(self) -> Header:
        return fits.getheader(self.filepath, ext=0)

    @cached_property
    def pointings(self) -> dict[str, CoordEquatorial]:
        """
        Extract camera axis pointing information in equatorial frame from file header.
        Angles are expressed in degrees.

        Returns:
            Nested dictionary containing RA/Dec coordinates for both cameras'
            z and x axes.
        """
        return {
            "z": CoordEquatorial(ra=self.header["CAMZRA"], dec=self.header["CAMZDEC"]),
            "x": CoordEquatorial(ra=self.header["CAMXRA"], dec=self.header["CAMXDEC"]),
        }

    @cached_property
    def rotations(self) -> dict[str, CoordHorizontal]:
        """
        Extract camera axis directions in the instrument frame from reconstructed file header.
        Angles expressed in degrees.

        Returns:
            Nested dictionary containing RA/Dec coordinates for both cameras'
            z and x axes.
        """
        return {
            "z": CoordHorizontal(az=self.header["CAMZPH"], al=90 - self.header["CAMZTH"]),
            "x": CoordHorizontal(az=self.header["CAMXPH"], al=90 - self.header["CAMXTH"]),
        }


def simulation(filepath: str | Path) -> SimulationDataLoader:
    """
    Checks validity of filepath and intializes SimulationDataLoader.

    Args:
        filepath: path to FITS file.

    Returns:
        a SimulationDataLoader dataclass.
    """
    if validate_fits(Path(filepath)):
        sdl = SimulationDataLoader(filepath)
    return sdl


def _fold(
    ml: FITS_rec,
    mask_bins: BinsRectangular,
) -> npt.NDArray:
    """
    Convert mask data from FITS record to 2D binned array.

    Args:
        ml: FITS record containing mask data
        mask_bins: Binning structure for the mask

    Returns:
        2D array containing binned mask data
    """
    return binned_statistic_2d(ml["X"], ml["Y"], ml["VAL"], statistic="max", bins=[mask_bins.x, mask_bins.y])[0].T


def load_from_fits(filepath: str | Path) -> tuple:
    """
    Load mask data and specifications from FITS file.

    Extracts mask patterns, decoder patterns, bulk patterns, and geometric
    specifications from a coded mask FITS file. Returns callable thunks for
    lazy loading of array data.

    Args:
        filepath: Path to the mask FITS file

    Returns:
        Tuple containing:
            - get_mask: Callable that returns mask pattern as 2D array
            - get_decoder: Callable that returns decoder pattern as 2D array
            - get_bulk: Callable that returns bulk pattern as 2D array
            - specs: Dictionary of mask specifications and geometric parameters
    """
    h0 = dict(fits.getheader(filepath, ext=0))
    specs = {
        "detector_minx": h0["PLNXMIN"],
        "detector_maxx": h0["PLNXMAX"],
        "detector_miny": h0["PLNYMIN"],
        "detector_maxy": h0["PLNYMAX"],
        "mask_thickness": h0["MASKTHK"],
        "mask_detector_distance": h0["MDDIST"] + h0["MASKTHK"],
    }
    h2 = dict(fits.getheader(filepath, ext=2))
    specs |= {
        "mask_minx": h2["MINX"],
        "mask_miny": h2["MINY"],
        "mask_maxx": h2["MAXX"],
        "mask_maxy": h2["MAXY"],
        "slit_deltax": h2["DXSLIT"],
        "slit_deltay": h2["DYSLIT"],
    }
    h3 = dict(fits.getheader(filepath, ext=3))
    specs |= {
        "mask_deltax": h3["ELXDIM"],
        "mask_deltay": h3["ELYDIM"],
    }

    l, r = specs["mask_minx"], specs["mask_maxx"]
    b, t = specs["mask_miny"], specs["mask_maxy"]
    mask_bins = BinsRectangular(
        np.linspace(l, r, int((r - l) / specs["mask_deltax"]) + 1),
        np.linspace(b, t, int((t - b) / specs["mask_deltay"]) + 1),
    )

    get_mask = lambda: _fold(fits.getdata(filepath, ext=2), mask_bins).astype(int)
    get_decoder = lambda: _fold(fits.getdata(filepath, ext=3), mask_bins).astype(int)
    get_bulk = lambda: _fold(fits.getdata(filepath, ext=4), mask_bins).astype(int)
    return get_mask, get_decoder, get_bulk, specs


"""
too much dataclasses

⠀⠀⠀⠀⠀⠀⣀⣠⣤⣤⣤⣤⣀⡀
⠀⠀⠀⣠⡶⡿⢿⣿⣛⣟⣿⡿⢿⢿⣷⣦⡀
⠀⢰⣯⣷⣿⣿⣿⢟⠃⢿⣟⣿⣿⣾⣷⣽⣺⢆⠀
⠀⢸⣿⢿⣾⢧⣏⡴⠀⠈⢿⣘⣿⢿⣿⣿⣿⣿⡆
⠀⢹⣿⢠⡶⠒⢶⠀⠀⣠⠒⠒⠢⡀⢿⣿⣿⣿⡇
⠀⣿⣿⠸⣄⣠⡾⠀⠀⠻⣀⣀⡼⠁⢸⣿⣿⣿⣿
⠀⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿
⢰⣿⣿⠀⠀⠀⡔⠢⠤⠔⠒⢄⠀⠀⢸⣿⣿⣿⣿⡇
⢸⣿⣿⣄⠀⠸⡀⠀⠀⠀⠀⢀⡇⠠⣸⣿⣿⣿⣿⡇
⢸⣿⣿⣿⣷⣦⣮⣉⢉⠉⠩⠄⢴⣾⣿⣿⣿⣿⡇
⢸⣿⣿⢻⣿⣟⢟⡁⠀⠀⠀⠀⢇⠻⣿⣿⣿⣿⣿
⢸⠿⣿⡈⠋⠀⠀⡇⠀⠀⠀⢰⠃⢠⣿⡟
"""
