from functools import cached_property
from pathlib import Path
from dataclasses import dataclass

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
class ObservationDataLoader:
    """
    An immutable dataclass for loading WFM simulation data from FITS files.

    The class handles data from two cameras (cam1a and cam1b), each containing detected,
    reconstructed, and source data. It provides access to both the data and headers of
    these files, as well as pointing information for the cameras in those simulations.

    Attributes:
        root: Root directory containing the simulation data files
    """

    root: str

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

        rp = Path(self.root)

        return {
            "cam1a": {
                "detected": check_and_pick(rp, "cam1a/*detected_plane.fits"),
                "reconstructed": check_and_pick(rp, "cam1a/*reconstructed.fits"),
                "sources": check_and_pick(rp, "cam1a/*sources.fits"),
            },
            "cam1b": {
                "detected": check_and_pick(rp, "cam1b/*detected_plane.fits"),
                "reconstructed": check_and_pick(rp, "cam1b/*reconstructed.fits"),
                "sources": check_and_pick(rp, "cam1b/*sources.fits"),
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

    def get_detected_data(self) -> dict[str, FITS_rec]:
        """
        Load photons data without plane position reconstruction effects from both cameras.

        Returns:
            Dictionary mapping camera IDs to their detected plane data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "detected", headers=False)

    def get_reconstructed_data(self) -> dict[str, FITS_rec]:
        """
        Load reconstructed photons data from both cameras.

        Returns:
            Dictionary mapping camera IDs to their reconstructed data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "reconstructed", headers=False)

    def get_source_data(self) -> dict[str, FITS_rec]:
        """
        Load source data from both cameras.

        Returns:
            Dictionary mapping camera IDs to their source data arrays
        """
        return _get_simulation_fits_data(self.simulation_files, "sources", headers=False)

    def get_detected_header(self) -> dict[str, Header]:
        """
        Load FITS headers from detected plane files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their detected plane headers
        """
        return _get_simulation_fits_data(self.simulation_files, "detected", headers=True)

    def get_reconstructed_header(self) -> dict[str, Header]:
        """
        Load FITS headers from reconstructed files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their reconstructed data headers
        """
        return _get_simulation_fits_data(self.simulation_files, "reconstructed", headers=True)

    def get_source_header(self) -> dict[str, Header]:
        """
        Load FITS headers from source files for both cameras.

        Returns:
            Dictionary mapping camera IDs to their source data headers
        """
        return _get_simulation_fits_data(self.simulation_files, "sources", headers=True)


@dataclass(frozen=True)
class MaskDataLoader:
    """
    Frozen dataclass for loading and parsing mask-related FITS data from a single file.
    Handles mask parameters and various data extensions (mask, decoder, and bulk data).

    Attributes:
        file: String path to mask file
    """

    file: str

    def __getitem__(self, key: str) -> float:
        """Access mask parameters via dictionary-style lookup."""
        return self.parameters[key]

    @cached_property
    def filepath(self) -> Path:
        """Convert file string to Path object."""
        return Path(self.file)

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
                "detector_minx": h["PLNXMIN"],
                "detector_maxx": h["PLNXMAX"],
                "detector_miny": h["PLNYMIN"],
                "detector_maxy": h["PLNYMAX"],
                "mask_detector_distance": h["MDDIST"],
            }.items()
        }

    def get_mask_data(self) -> fits.FITS_rec:
        """
        Load mask data from mask FITS file.

        Returns:
            FITS record array containing mask data
        """
        return fits.getdata(self.filepath, ext=2)

    def get_decoder_data(self) -> fits.FITS_rec:
        """
        Load decoder data from mask FITS file.

        Returns:
            FITS record array containing decoder data
        """
        return fits.getdata(self.filepath, ext=3)

    def get_bulk_data(self) -> fits.FITS_rec:
        """
        Load bulk data from mask FITS file.

        Returns:
            FITS record array containing bulk data
        """
        return fits.getdata(self.filepath, ext=4)

    def get_mask_header(self) -> fits.Header:
        """
        Load mask header from mask FITS file.

        Returns:
            FITS header containing mask data
        """
        return fits.getheader(self.filepath, ext=2)

    def get_decoder_header(self) -> fits.Header:
        """
        Load decoder header from mask FITS file.

        Returns:
            FITS header containing decoder data
        """
        return fits.getheader(self.filepath, ext=3)

    def get_bulk_header(self) -> fits.Header:
        """
        Load bulk header from mask FITS file.

        Returns:
            FITS header containing bulk data
        """
        return fits.getheader(self.filepath, ext=4)