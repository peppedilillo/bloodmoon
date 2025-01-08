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

from bloodmoon.types import CoordEquatorial
from bloodmoon.types import CoordHorizontal


def _validate_fits(filepath: Path) -> bool:
    """Following astropy's approach, reads the first FITS card (80 bytes) and checks for
    the SIMPLE keyword signature.

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


def simulation_files(dirpath: str | Path) -> dict[str, dict[str, Path]]:
    """
    Locate and validate all required FITS files in the root directory.

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
            "detected": check_and_pick(dirpath, "cam1a/*detected_plane.fits"),
            "reconstructed": check_and_pick(dirpath, "cam1a/*reconstructed.fits"),
            "sources": check_and_pick(dirpath, "cam1a/*sources.fits"),
        },
        "cam1b": {
            "detected": check_and_pick(dirpath, "cam1b/*detected_plane.fits"),
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
        mask_detector_distance (float): Distance between mask and detector in mm
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
        a MaskDataLoader dataclass.
    """
    dr = Path(filepath)
    if not dr.is_file():
        raise FileNotFoundError("The simulation file does not exists.")
    if not _validate_fits(filepath):
        raise ValueError("File not in valid FITS format.")
    return SimulationDataLoader(filepath)


@dataclass(frozen=True)
class MaskDataLoader:
    """
    Container for WFM coded mask parameters and patterns.

    The class provides access to mask geometry, decoder patterns, and associated
    parameters from a single FITS file containing WFM mask data.

    Attributes:
        filepath: Path to the FITS file

    Properties:
        specs: Dictionary of mask and detector dimensions
        mask: Mask pattern data from extension 2
        decoder: Decoder pattern data from extension 3
        bulk: Bulk pattern data from extension 4
    """

    filepath: Path

    def __getitem__(self, key: str) -> float:
        """Access mask parameters via dictionary-style lookup."""
        return self.specs[key]

    @cached_property
    def specs(self) -> dict[str, float]:
        """
        Extract and convert mask parameters from FITS headers (extensions 0 and 2).

        Returns:
            Dictionary of mask parameters (dimensions, bounds, distances) as float values
        """
        h = dict(fits.getheader(self.filepath, ext=0)) | dict(fits.getheader(self.filepath, ext=2))
        return {
            k: float(v)
            for k, v in {
                "mask_minx": h["MINX"],
                "mask_miny": h["MINY"],
                "mask_maxx": h["MAXX"],
                "mask_maxy": h["MAXY"],
                "mask_deltax": h["ELXDIM"],
                "mask_deltay": h["ELYDIM"],
                "mask_thickness": h["MASKTHK"],
                "slit_deltax": h["DXSLIT"],
                "slit_deltay": h["DYSLIT"],
                "detector_minx": h["PLNXMIN"],
                "detector_maxx": h["PLNXMAX"],
                "detector_miny": h["PLNYMIN"],
                "detector_maxy": h["PLNYMAX"],
                "mask_detector_distance": h["MDDIST"],
            }.items()
        }

    @property
    def mask(self) -> fits.FITS_rec:
        """
        Load mask data from mask FITS file.

        Returns:
            FITS record array containing mask data
        """
        return fits.getdata(self.filepath, ext=2)

    @property
    def decoder(self) -> fits.FITS_rec:
        """
        Load decoder data from mask FITS file.

        Returns:
            FITS record array containing decoder data
        """
        return fits.getdata(self.filepath, ext=3)

    @property
    def bulk(self) -> fits.FITS_rec:
        """
        Load bulk data from mask FITS file.

        Returns:
            FITS record array containing bulk data
        """
        return fits.getdata(self.filepath, ext=4)


def fetch_mask(filepath: str | Path) -> MaskDataLoader:
    """
    Checks data and intializes MaskDataLoader.

    Args:
        filepath: path to mask FITS file.

    Returns:
        a MaskDataLoader dataclass.
    """
    fp = Path(filepath)
    if not fp.is_file():
        raise FileNotFoundError("Mask file does not exists")
    if not _validate_fits(filepath):
        raise ValueError("File not in valid FITS format.")
    return MaskDataLoader(Path(filepath))


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
