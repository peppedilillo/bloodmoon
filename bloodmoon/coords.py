"""
Coordinate transformation utilities for the WFM coded mask instrument.

This module provides functions to convert between different coordinate systems:
- Sky-shift coordinates (mm on the detector plane)
- Angular coordinates (degrees from optical axis)
- Equatorial coordinates (RA/Dec)

The transformations account for the instrument geometry and pointing direction.
"""

from bisect import bisect
# python is a disgusting piece of shit. cf. PEP 749
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .mask import CodedMaskCamera

from .io import SimulationDataLoader
from .types import BinsEquatorial
from .types import BinsRectangular
from .types import CoordEquatorial


def shift2pos(
    camera: "CodedMaskCamera",
    shift_x: float,
    shift_y: float,
) -> tuple[int, int]:
    """
    Convert continuous sky-shift coordinates to nearest discrete pixel indices.

    Args:
        camera: CodedMaskCamera instance containing binning information
        shift_x: x-coordinate in sky-shift space (mm)
        shift_y: y-coordinate in sky-shift space (mm)

    Returns:
        Tuple of (row, column) indices in the discrete sky image grid
    """
    return (
        bisect(camera.bins_sky.y, shift_y) - 1,
        bisect(camera.bins_sky.x, shift_x) - 1,
    )


def pos2shift(
    camera: "CodedMaskCamera",
    i: int,
    j: int,
) -> tuple[float, float]:
    """
    Convert a sky-shift index to a continuous sky-shift coordinate.

    Args:
        camera: CodedMaskCamera instance containing binning information
        i: row index
        j: column index

    Returns:
        Tuple of (shift_x, shift_j) float coordinates
    """
    xmidpoints = (camera.bins_sky.x[1:] + camera.bins_sky.x[:-1]) / 2
    ymidpoints = (camera.bins_sky.y[1:] + camera.bins_sky.y[:-1]) / 2
    return float(xmidpoints[j]), float(ymidpoints[i])


def shift2angle(camera: "CodedMaskCamera", shift: float) -> float:
    """
    Convert sky-coordinate shift in respective angular coordinate in the
    coded mask camera reference frame.

    Args:
        camera: The camera object containing the WFM cameras parameters.
        shift: Sky-coordinate shift.

    Returns:
        angle: Angular sky-coordinate in [deg].

    Usage:
        If the shift is measured in the x direction, the returned angle is
        the declination of the sky-versor projection onto the xz plane.

    Notes:
        - `shift` must have same physical dimension of mask-detector distance, i.e. [mm].
    """
    return np.rad2deg(np.arctan(shift / camera.specs["mask_detector_distance"]))


def angle2shift(camera: "CodedMaskCamera", angle: float) -> float:
    """
    Convert angular sky-coordinate in the coded mask camera reference
    frame in respective sky-coordinate shift.

    Args:
        camera: The camera object containing the WFM cameras parameters.
        angle: Angular sky-coordinate in [deg].

    Returns:
        shift: Sky-coordinate shift in [mm].

    Usage:
        If the angle is the declination of the sky-versor projection on the
        the xz plane, returns the shift in the x direction.
    """
    return camera.specs["mask_detector_distance"] * np.tan(np.deg2rad(angle))


def pos2equatorial(
    sdl: SimulationDataLoader,
    camera: "CodedMaskCamera",
    i: int,
    j: int,
) -> CoordEquatorial:
    """
    Convert sky pixel position to corresponding sky-shift coordinates.

    Args:
        sdl: SimulationDataLoader containing camera pointings
        camera: A CodedMaskCamera object containing sky shape and binning information.
        i: row index
        j: column index

    Returns:
        CoordEquatorial containing:
            - ra: Right ascension in degrees [0, 360].
            - dec: Declination in degrees [-90, 90].

    Notes:
        - the sky-coord shifts are in [mm] wrt optical axis.
        - RA is normalized to [0, 360) degree range.
        - resulting RA/Dec refer to the center of the pixel.
        - negative indexes are allowed.
    """
    return shift2equatorial(sdl, camera, *pos2shift(camera, i, j))


def shift2equatorial(
    sdl: SimulationDataLoader,
    camera: "CodedMaskCamera",
    shift_x: float,
    shift_y: float,
) -> CoordEquatorial:
    """Convert sky-shift coordinates to equatorial coordinates (RA/Dec) for a specific camera.

    Args:
        sdl: SimulationDataLoader containing camera pointings
        camera: CodedMaskCamera object containing mask pattern and mask-detector distance
        shift_x: X coordinate in sky-shift space (mm)
        shift_y: Y coordinate in sky-shift space (mm)

    Returns:
        CoordEquatorial containing:
            - ra: Right ascension in degrees [0, 360]
            - dec: Declination in degrees [-90, 90]

    Notes:
        - Input coordinates and distance must use consistent units
        - RA is normalized to [0, 360) degree range
        - Zero point in sky-shift space is the optical axis
    """
    return _shift2equatorial(
        shift_x,
        shift_y,
        sdl.pointings["z"],
        sdl.pointings["x"],
        camera.specs["mask_detector_distance"],
    )


def _shift2equatorial(
    shift_x: float,
    shift_y: float,
    pointing_radec_z: CoordEquatorial,
    pointing_radec_x: CoordEquatorial,
    distance_detector_mask: float,
) -> CoordEquatorial:
    """Implementation to `shift2equatorial`.

    Args:
        shift_x: X coordinate on the sky-shift plane in spatial units (e.g., mm or cm).
            Dimension should match shift_y and distance_detector_mask.
        shift_y: X coordinate on the sky-shift plane in spatial units (e.g., mm or cm).
            Dimension should match shift_x and distance_detector_mask.
        pointing_radec_z: Pointing direction of the detector's z-axis in
            (RA, Dec) coordinates in degrees.
        pointing_radec_x: Pointing direction of the detector's x-axis in
            (RA, Dec) coordinates in degrees. Used to define the detector's roll angle.
        distance_detector_mask: Distance between the detector and mask planes in the same
            spatial units as midpoints_xs and midpoints_ys.

    Returns:
        CoordEquatorial containing:
            - ra: Right ascension in degrees [0, 360]
            - dec: Declination in degrees [-90, 90]
    """
    _, rotmat_cam2sky = _rotation_matrices(
        pointing_radec_z,
        pointing_radec_x,
    )
    r = np.sqrt(shift_x * shift_x + shift_y * shift_y + distance_detector_mask * distance_detector_mask)
    v = np.array([shift_x, shift_y, distance_detector_mask]) / r
    wx, wy, wz = np.matmul(rotmat_cam2sky, v)
    # the versors above are in the rectangular coordinates, we transform into angles
    dec = 0.5 * np.pi - np.arccos(wz)
    ra = np.arctan2(wy, wx)
    ra += 2 * np.pi if ra < 0 else 0.0
    dec = np.rad2deg(dec)
    ra = np.rad2deg(ra)
    return CoordEquatorial(*map(float, (ra, dec)))


def equatorial2shift(
    sdl: SimulationDataLoader,
    camera: "CodedMaskCamera",
    ra: float,
    dec: float,
) -> tuple[float, float]:
    """
    Convert equatorial coordinates (RA/Dec) to sky-shift coordinates for a specific camera.
    Args:
        sdl: SimulationDataLoader containing camera pointings
        camera: CodedMaskCamera object containing mask pattern and mask-detector distance
        ra: Right ascension in degrees [0, 360]
        dec: Declination in degrees [-90, 90]
    Returns:
        A tuple of float containing:
            - shift x: X coordinate in sky-shift space [mm]
            - shift y: Y coordinate in sky-shift space [mm]
    Notes:
        - Input coordinates and distance must use consistent units
        - Zero point in sky-shift space is the optical axis
    """
    return _equatorial2shift(
        ra,
        dec,
        sdl.pointings["z"],
        sdl.pointings["x"],
        camera.specs["mask_detector_distance"],
    )


def _equatorial2shift(
    ra: float,
    dec: float,
    pointing_radec_z: CoordEquatorial,
    pointing_radec_x: CoordEquatorial,
    distance_detector_mask: float,
) -> tuple[float, float]:
    """
    Implementation to `equatorial2shift()`.
    Args:
        ra: Right ascension in degrees [0, 360]
        dec: Declination in degrees [-90, 90]
        pointing_radec_z: Pointing direction of the detector's z-axis in
            (RA, Dec) coordinates in degrees.
        pointing_radec_x: Pointing direction of the detector's x-axis in
            (RA, Dec) coordinates in degrees. Used to define the detector's roll angle.
        distance_detector_mask: Distance between the detector and mask planes in the same
            spatial units as midpoints_xs and midpoints_ys.
    Returns:
        A tuple of float containing:
            - shift_x: X coordinate on the sky-shift plane in spatial units.
                Dimension should match shift_y and distance_detector_mask.
            - shift_y: X coordinate on the sky-shift plane in spatial units.
                Dimension should match shift_x and distance_detector_mask.
    """
    rotmat_sky2cam, _ = _rotation_matrices(
        pointing_radec_z,
        pointing_radec_x,
    )
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    w = np.array(
        [
            np.cos(ra) * np.cos(dec),
            np.sin(ra) * np.cos(dec),
            np.sin(dec),
        ]
    )
    vx, vy, vz = np.matmul(rotmat_sky2cam, w)
    # the sky-shifts are computed from the versor `v` using the mask-detector distance
    shift_x = vx * distance_detector_mask / vz
    shift_y = vy * distance_detector_mask / vz
    return float(shift_x), float(shift_y)


def _rotation_matrices(
    pointing_radec_z: tuple[float, float],
    pointing_radec_x: tuple[float, float],
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Computes two 3x3 rotation matrices that transform coordinates between the Earth equatorial
    reference frame (RA/Dec) and the camera's local reference frame.
    The transformation is defined by specifying the camera's z-axis and x-axis directions
    in equatorial coordinates.

    Args:
        pointing_radec_z: Camera's z-axis direction in equatorial coordinates.
            Either tuple[float, float] or np.array of (RA, Dec) in degrees.
            RA in [0, 360], Dec in [-90, 90].
        pointing_radec_x: Camera's x-axis direction in equatorial coordinates.
            Either tuple[float, float] or np.array of (RA, Dec) in degrees.
            RA in [0, 360], Dec in [-90, 90].

    Returns:
        A tuple containing:
            - rotmat_sky2cam (np.ndarray): 3x3 rotation matrix to transform vectors
              from equatorial to camera coordinates
            - rotmat_cam2sky (np.ndarray): 3x3 rotation matrix to transform vectors
              from camera to equatorial coordinates (transpose of rotmat_sky2cam)

    Notes:
        - The rotation matrices are orthogonal, so rotmat_cam2sky is the transpose
          of rotmat_sky2cam
        - The x and z axes provided must be approximately perpendicular for the
          resulting transformation to be valid
        - The matrices operate on vectors in Cartesian coordinates, not directly
          on RA/Dec angles
        - All internal angle calculations are performed in radians
    """
    ra_z, dec_z = pointing_radec_z
    ra_x, dec_x = pointing_radec_x

    theta_z = np.deg2rad(90 - dec_z)
    phi_z = np.deg2rad(ra_z)

    theta_x = np.deg2rad(90 - dec_x)
    phi_x = np.deg2rad(ra_x)

    sin_theta_x = np.sin(theta_x)
    x_axis = np.array([sin_theta_x * np.cos(phi_x), sin_theta_x * np.sin(phi_x), np.cos(theta_x)])

    sin_theta_z = np.sin(theta_z)
    z_axis = np.array([sin_theta_z * np.cos(phi_z), sin_theta_z * np.sin(phi_z), np.cos(theta_z)])

    y_axis = np.array(
        [
            z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
            z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
            z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
        ]
    )

    rotmat_sky2cam = np.vstack((x_axis, y_axis, z_axis))
    rotmat_cam2sky = rotmat_sky2cam.T

    return rotmat_sky2cam, rotmat_cam2sky


def shiftgrid2equatorial(
    sdl: SimulationDataLoader,
    camera: "CodedMaskCamera",
    shift_xs: npt.NDArray,
    shift_ys: npt.NDArray,
) -> BinsEquatorial:
    """
    Converts sky-shift coordinates to equatorial sky coordinates (RA/Dec).
    This function performs a coordinate transformation from a rectangular grid of points
    on a sky-shift plane to their corresponding positions in the sky using equatorial
    coordinates. To achieve the transformation it requires the pointings in equatorial
    coordinates of the x and z axis of the camera.
    For batch computations.

    Args:
        sdl: SimulationDataLoader containing camera pointings
        camera: CodedMaskCamera object containing mask pattern and mask-detector distance
        shift_xs: X coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_ys.
        shift_ys: Y coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_xs.

    Returns:
        BinsEquatorial record containing:
            - `dec` field: Grid of declination values in degrees, same shape as input arrays
            - `ra` field: Grid of right ascension values in degrees, same shape as input arrays.
              Values are in the range [0, 360] degrees.

    Notes:
        - Inputs (midpoints_xs, midpoints_ys,  distance_detector_mask) should be in a consistent unit system
        - The output RA values are normalized to [0, 360) degrees
        - The output Dec values are in the range [-90, 90] degrees

    Example:
        >>> from bloodmoon import codedmask, simulation
        >>> from bloodmoon.coords import shiftgrid2equatorial
        >>>
        >>> wfm = codedmask("mask.fits")
        >>> sdl = simulation("datapath.fits")
        >>>
        >>> ras, decs = shiftgrid2equatorial(sdl, wfm, *wfm.bins_sky)
    """
    return _shiftgrid2equatorial(
        *np.meshgrid(shift_xs, shift_ys),
        sdl.pointings["z"],
        sdl.pointings["x"],
        camera.specs["mask_detector_distance"],
    )


def _shiftgrid2equatorial(
    midpoints_sky_xs: npt.NDArray,
    midpoints_sky_ys: npt.NDArray,
    pointing_radec_z: tuple[float, float],
    pointing_radec_x: tuple[float, float],
    distance_detector_mask: float,
) -> BinsEquatorial:
    """Implementation to `shiftgrid2equatorial`.

    Args:
        midpoints_sky_xs: X coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape should match midpoints_ys.
            Dimension should match midpoint_ys and distance_detector_mask.
        midpoints_sky_ys: Y coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape should match midpoints_xs.
            Dimension should match midpoint_xs and distance_detector_mask.
        pointing_radec_z: Pointing direction of the detector's z-axis in
            (RA, Dec) coordinates in degrees.
        pointing_radec_x: Pointing direction of the detector's x-axis in
            (RA, Dec) coordinates in degrees. Used to define the detector's roll angle.
        distance_detector_mask: Distance between the detector and mask planes in the same
            spatial units as midpoints_xs and midpoints_ys.

    Returns:
        BinsEquatorial record containing:
            - `dec` field: Grid of declination values in degrees, same shape as input arrays
            - `ra` field: Grid of right ascension values in degrees, same shape as input arrays.
              Values are in the range [0, 360] degrees.
    """
    _, rotmat_cam2sky = _rotation_matrices(
        pointing_radec_z,
        pointing_radec_x,
    )
    # point distances from the mask center.
    r = np.sqrt(
        midpoints_sky_xs * midpoints_sky_xs
        + midpoints_sky_ys * midpoints_sky_ys
        + distance_detector_mask * distance_detector_mask
    )
    # these are the versors from the mask center to the detector elements
    versors_local_ys = midpoints_sky_ys / r
    versors_local_xs = midpoints_sky_xs / r
    versors_local_zs = distance_detector_mask / r
    # this multiplies all detector vectors with the rotation matrix
    _v = np.hstack(
        (
            versors_local_xs.ravel().reshape(-1, 1, 1),
            versors_local_ys.ravel().reshape(-1, 1, 1),
            versors_local_zs.ravel().reshape(-1, 1, 1),
        )
    )
    versors_eq = np.matmul(rotmat_cam2sky, _v)
    # the versors above are in the rectangular coordinates, we transform into angles
    decs = 0.5 * np.pi - np.arccos(versors_eq[:, 2].ravel())
    ras = np.arctan2(versors_eq[:, 1].ravel(), versors_eq[:, 0].ravel())
    ras[ras < 0] += 2 * np.pi
    decs = np.rad2deg(decs.reshape(midpoints_sky_xs.shape))
    ras = np.rad2deg(ras.reshape(midpoints_sky_ys.shape))
    return BinsEquatorial(ra=ras, dec=decs)
