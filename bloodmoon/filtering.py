"""
Data filters for photons energy range, sources flux and sources positions.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .types import CoordEquatorial


def filter_data(
    data: np.recarray,
    *,
    E_min: int | float | None,
    E_max: int | float | None,
    coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
) -> np.recarray:
    """
    Filters the input `data` record based on the photons energy
    and/or incoming direction.
    
    Args:
        data (np.recarray):
            Input simulated data container.
        E_min (int | float | None):
            Minimum photons energy in [keV] for the data filtering.
        E_max (int | float | None):
            Maximum photons energy in [keV] for the data filtering.
        coords (CoordEquatorial | Sequence[CoordEquatorial] | None):
            Input photons RA/Dec in [deg] to filter out.
    
    Returns:
        output: Output filtered data container.
    """
    def make_mask(
        emin: int | float | None,
        emax: int | float | None,
        coords: CoordEquatorial | Sequence[CoordEquatorial] | None,
    ) -> npt.NDArray[np.bool]:
        """Generates mask for the input data."""
        energy_mask = np.ones(len(data), dtype=bool)
        coords_mask = np.ones(len(data), dtype=bool)

        if any(emin, emax):
            if emin is None: emin = 0
            if emax is None: emax = np.inf
            energy_mask &= (data["ENERGY"] > emin & data["ENERGY"] < emax)

        if coords:
            for c in coords:
                coords_mask &= ~(
                    np.isclose(data["RA"], c.ra, atol=1e-7) &
                    np.isclose(data["DEC"], c.dec, atol=1e-7)
                )
        
        return energy_mask & coords_mask

    if not any((E_min, E_max, coords)):
        return data
    mask = make_mask(E_min, E_max, coords)
    return data[mask]











def data_filter(
    record: np.recarray,
    energy_range: int | float | tuple[int | float, int | float] | None,
    coords: tuple[float, float] | Sequence[tuple[float, float]] | None,
) -> np.recarray:
    """
    Filters the input `record` based on the photons energy and/or position.
    
    Args:
        record: Input simulated data container.
        energy_range: Energy range in keV for the data filtering. If a specific energy
                      is given, this will be considered as the maximum filter value.
                      If a tuple is given, it's interpreted as (`E_min`, `E_max`).
        coords: Input photons RA/Dec (or sequence of RA/Dec) to filter out.
    
    Returns:
        output: Output filtered data container.
    """
    def _energy_mask(
        mask: npt.NDArray,
        values: int | float | tuple[int | float, int | float],
    ) -> npt.NDArray:
        """Creates an energy mask for the input `record`."""
        if isinstance(values, (int, float)):
            mask &= (record["ENERGY"] < values)
        else:
            mask &= (record["ENERGY"] > values[0]) & (record["ENERGY"] < values[1])
        return mask

    def _coords_mask(
        mask: npt.NDArray,
        values: tuple[float, float],
    ) -> npt.NDArray:
        """Creates a RA/Dec mask for the input `record`."""
        # to address float64 to float32 conv, we remove
        # the photons coming from the specified RA/Dec
        mask &= ~(
            (np.abs(record["RA"] - values[0]) < 1e-7) &
            (np.abs(record["DEC"] - values[1]) < 1e-7)
        )
        #mask &= (record["RA"] != values[0]) | (record["DEC"] != values[1])
        return mask

    mask = np.ones(len(record), dtype=bool)

    if energy_range is not None:
        mask = _energy_mask(mask, energy_range)
    
    if coords is not None:
        if isinstance(coords[0], float):
            mask = _coords_mask(mask, coords)
        else:
            _cmask = np.ones(len(record), dtype=bool)
            for c in coords:
                _cmask = _coords_mask(_cmask, c)
            mask &= _cmask
    
    return record[mask]





















def flux_filter(
    record: np.recarray,
    flux_range: int | float | tuple[int | float, int | float],
) -> np.recarray:
    """
    Filters the input catalog `record` for a given flux range.

    Args:
        record: Input simulated data container.
        flux_range: Flux range in ph/cm2/s for the data filtering. If a specific flux
                    is given, this will be considered as the minimum filter value.
                    If a tuple is given, it's interpreted as (`F_min`, `F_max`).

    Returns:
        output: Output filtered data container.
    """
    mask = np.ones(len(record), dtype=bool)
    if isinstance(flux_range, (int, float)):
        mask &= (record["FLUX"] > flux_range)
    else:
        mask &= (record["FLUX"] > flux_range[0]) & (record["FLUX"] < flux_range[1])
    
    return record[mask]


def source_filter(
    record: np.recarray,
    n: int | tuple[int, int]
) -> np.recarray:
    """
    Select the `n` brightest sources from the input catalog `record`,
    or a given interval of sources.

    Args:
        record: Input simulated data container.
        n: Filtered interval of sources, up to the n-th brightest
           source or from `n[0]` to `n[1]` if `n` is a tuple.

    Returns:
        output: Output filtered data container.
    
    Notes:
        - `n` follows the std Python indexing rules.
    """
    sorted_rec = np.sort(record, order="NPHOTONS")[::-1]
    runs = len(sorted_rec) // len(np.unique(sorted_rec["NAME"]))
    return sorted_rec[:runs * n] if isinstance(n, int) else sorted_rec[runs * n[0] : runs * n[1]]


def catalog_filter(
    catalog: np.recarray,
    n: int | tuple[int, int] | None,
    flux_range: int | float | tuple[int | float, int | float] | None = None,
) -> np.recarray:
    """
    Filters the input `catalog` record based on the sources fluence OR flux.
    If `n` is given, it selects the `n` brightest sources from the input
    record, or a given interval of sources. If `flux_range` is given, it
    filters the input record for a given flux range.
    
    Args:
        catalog: Input simulated data container.
        n: Filtered interval of sources, up to the n-th brightest
           source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range: Flux range in ph/cm2/s for the data filtering. If a specific flux
                    is given, this will be considered as the minimum filter value.
                    If a tuple is given, it's interpreted as (`F_min`, `F_max`).
    
    Returns:
        output: Output filtered data container.
    
    Raises:
        ValueError: If `n` or `flux_range` are both specified for catalogs filtering.
    """
    if n and flux_range:
        raise ValueError("Specify either 'n' or 'flux_range' to filter the catalog.")
    
    if n is not None:
        return source_filter(catalog, n)
    elif flux_range is not None:
        return flux_filter(catalog, flux_range)
    
    return catalog

