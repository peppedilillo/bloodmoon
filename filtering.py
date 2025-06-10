"""
Data filters for photons energy range, sources flux and sources positions.
"""

from collections.abc import Sequence

import numpy as np
from bloodmoon.types import CoordEquatorial


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
    mask = np.ones(len(data), dtype=bool)

    if E_min is not None:
        mask &= (data["ENERGY"] > E_min)
    if E_max is not None:
        mask &= (data["ENERGY"] < E_max)
    
    if coords is not None:
        for c in ((coords,) if isinstance(coords, CoordEquatorial) else coords):
            mask &= ~(np.isclose(data["RA"], c.ra) & np.isclose(data["DEC"], c.dec))
    
    return data[mask]


def flux_filter(
    data: np.recarray,
    F_min: int | float | None,
    F_max: int | float | None,
) -> np.recarray:
    """
    Filters the input `data` record based on the sources flux.
    
    Args:
        data (np.recarray):
            Input simulated data container.
        F_min (int | float | None):
            Minimum flux range in [ph/cm2/s] for the data filtering.
        F_max (int | float | None):
            Maximum flux range in [ph/cm2/s] for the data filtering.
    
    Returns:
        output: Output filtered data container.
    """
    mask = np.ones(len(data), dtype=bool)
    if F_min is not None:
        mask &= (data["FLUX"] > F_min)
    if F_max is not None:
        mask &= (data["FLUX"] < F_max)
    return data[mask]


def source_filter(
    data: np.recarray,
    n: int | tuple[int, int],
) -> np.recarray:
    """
    Select the `n` brightest sources from the input catalog `data`,
    or a given interval of sources.

    Args:
        data (np.recarray):
            Input simulated data container.
        n (int | tuple[int, int]):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.

    Returns:
        output (np.recarray): Output filtered data container.
    
    Notes:
        - `n` follows the std Python indexing rules.
    """
    sorted_rec = np.sort(data, order="NPHOTONS")[::-1]
    runs = len(sorted_rec) // len(np.unique(sorted_rec["NAME"]))
    return sorted_rec[:runs * n] if isinstance(n, int) else sorted_rec[runs * n[0] : runs * n[1]]


def catalog_filter(
    catalog: np.recarray,
    *,
    n: int | tuple[int, int] | None,
    flux_range: tuple[int | float | None, int | float | None] | None = None,
) -> np.recarray:
    """
    Filters the input `catalog` record based on the sources fluence OR flux.
    If `n` is given, it selects the `n` brightest sources from the input
    record, or a given interval of sources. If `flux_range` is given, it
    filters the input record for a given flux range.
    
    Args:
        catalog (np.recarray):
            Input simulated data container.
        n (int | tuple[int, int] | None):
            Filtered interval of sources, up to the n-th brightest
            source or from `n[0]` to `n[1]` if `n` is a tuple.
        flux_range (tuple[int | float | None, int | float | None] | None, optional (default=None)):
            Flux range in ph/cm2/s for the data filtering. The
            input tuple is interpreted as (`F_min`, `F_max`).
    
    Returns:
        output (np.recarray): Output filtered data container.
    
    Raises:
        ValueError: If `n` or `flux_range` are both specified for catalogs filtering.
    """
    if n and flux_range:
        raise ValueError("Specify either 'n' or 'flux_range' to filter the catalog.")
    
    if n is not None:
        return source_filter(catalog, n)
    elif flux_range is not None:
        return flux_filter(catalog, *flux_range)
    
    return catalog

