from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from astropy.io import fits
from astropy.io.fits.fitsrec import FITS_rec
from astropy.io.fits.header import Header


def _get_simulation_fits_data(filedict: dict, kind: str, headers: bool = False) -> dict:
    """
    Load data or headers from FITS simulation files for both WFM cameras.

    Args:
        filedict: Dictionary containing file paths for both cameras (i.e., Dataloader.simulation_files)
        kind: Type of data to load ('detected', 'reconstructed', or 'sources')
        headers: If True, return headers instead of data

    Returns:
        Dictionary mapping camera IDs to their respective data or headers
    """

    def fits2data(f: Path, headers: bool):
        data, header = fits.getdata(f, ext=1, header=True)
        if headers:
            return header
        return data

    return {k: fits2data(filedict[k][kind], headers) for k in ["cam1a", "cam1b"]}


@dataclass(frozen=True)
class SimulationDataLoader:
    """
    An immutable dataclass for loading WFM simulation data from FITS files.

    The class handles data from two cameras (cam1a and cam1b), each containing detected,
    reconstructed, and source data. It provides access to both the data and headers of
    these files, as well as pointing information for the cameras in those simulations.

    Attributes:
        root: Path to root directory containing the simulation data files
    """

    root: Path

    @cached_property
    def simulation_files(self):
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
                raise ValueError(f"Found unexpected extra matches for glob pattern {str(parent / pattern)}." f"File with pattern {pattern} should be unique")
            return f

        return {
            "cam1a": {
                "detected": check_and_pick(self.root, "cam1a/*detected_plane.fits"),
                "reconstructed": check_and_pick(self.root, "cam1a/*reconstructed.fits"),
                "sources": check_and_pick(self.root, "cam1a/*sources.fits"),
            },
            "cam1b": {
                "detected": check_and_pick(self.root, "cam1b/*detected_plane.fits"),
                "reconstructed": check_and_pick(self.root, "cam1b/*reconstructed.fits"),
                "sources": check_and_pick(self.root, "cam1b/*sources.fits"),
            },
        }

    @cached_property
    def pointings(self) -> dict[str, dict[str, tuple]]:
        """
        Extract camera pointing information from reconstruction file headers.

        Returns:
            Nested dictionary containing RA/Dec coordinates for both cameras'
            z and x axes.
        """
        header_1a = fits.getheader(self.simulation_files["cam1a"]["reconstructed"], ext=0)
        header_1b = fits.getheader(self.simulation_files["cam1b"]["reconstructed"], ext=0)
        return {
            "cam1a": {
                "radec_z": (header_1a["CAMZRA"], header_1a["CAMZDEC"]),
                "radec_x": (header_1a["CAMXRA"], header_1a["CAMXDEC"]),
            },
            "cam1b": {
                "radec_z": (header_1b["CAMZRA"], header_1b["CAMZDEC"]),
                "radec_x": (header_1b["CAMXRA"], header_1b["CAMXDEC"]),
            },
        }

    @property
    def detected(self) -> dict[str, FITS_rec]:
        """
        Load photons data without plane position reconstruction effects from both cameras.

        Returns:
            Dictionary mapping camera IDs to their detected plane data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "detected", headers=False)

    @property
    def reconstructed(self) -> dict[str, FITS_rec]:
        """
        Load reconstructed photons data from both cameras.

        Returns:
            Dictionary mapping camera IDs to their reconstructed data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "reconstructed", headers=False)

    @property
    def source(self) -> dict[str, FITS_rec]:
        """
        Load source data from both cameras.

        Returns:
            Dictionary mapping camera IDs to their source data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "sources", headers=False)

    @property
    def header_detected(self) -> dict[str, Header]:
        """
        Load FITS headers from detected plane files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their detected plane headers
        """
        return _get_simulation_fits_data(self.simulation_files, "detected", headers=True)

    @property
    def header_reconstructed(self) -> dict[str, Header]:
        """
        Load FITS headers from reconstructed files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their reconstructed data headers
        """
        return _get_simulation_fits_data(self.simulation_files, "reconstructed", headers=True)

    @property
    def header_source(self) -> dict[str, Header]:
        """
        Load FITS headers from source files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their source data headers
        """
        return _get_simulation_fits_data(self.simulation_files, "sources", headers=True)


def fetch_simulation(data_root: str | Path):
    """
    Checks data and intializes MaskDataLoader.

    Args:
        data_root: path to mask FITS file.

    Returns:
        a MaskDataLoader dataclass.
    """
    dr = Path(data_root)
    if not dr.is_dir():
        raise NotADirectoryError("The simulation path is not a directory.")
    return SimulationDataLoader(Path(data_root))


@dataclass(frozen=True)
class MaskDataLoader:
    """
    Frozen dataclass for loading and parsing mask-related FITS data from a single file.
    Handles mask parameters and various data extensions (mask, decoder, and bulk data).

    Attributes:
        filepath: path to mask file
    """

    filepath: Path

    def __getitem__(self, key: str) -> float:
        """Access mask parameters via dictionary-style lookup."""
        return self.parameters[key]

    @cached_property
    def parameters(self) -> dict[str, float]:
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

    @property
    def header_mask(self) -> fits.Header:
        """
        Load mask header from mask FITS file.

        Returns:
            FITS header containing mask data
        """
        return fits.getheader(self.filepath, ext=2)

    @property
    def header_decoder(self) -> fits.Header:
        """
        Load decoder header from mask FITS file.

        Returns:
            FITS header containing decoder data
        """
        return fits.getheader(self.filepath, ext=3)

    @property
    def header_bulk(self) -> fits.Header:
        """
        Load bulk header from mask FITS file.

        Returns:
            FITS header containing bulk data
        """
        return fits.getheader(self.filepath, ext=4)


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
    return MaskDataLoader(Path(filepath))


"""
ain't all these 
dataclasses
too much?
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
⢸⠿⣿⡈⠋⠀⠀⡇⠀⠀⠀⢰⠃⢠⣿⡟⣿⣿⢻ 
"""
