import numpy as np

from .types import BinsEquatorial
from .types import BinsRectangular


def _rotation_matrices(
    pointing_radec_z: tuple[float, float],
    pointing_radec_x: tuple[float, float],
) -> tuple[np.array, np.array]:
    """
    This function computes two 3x3 rotation matrices that transform coordinates between
    the Earth equatorial reference frame (RA/Dec) and the camera's local reference frame.
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
        tuple[np.ndarray, np.ndarray]: A tuple containing:
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


def to_sky_coordinates(
    midpoints_sky_xs: np.array,
    midpoints_sky_ys: np.array,
    pointing_radec_z: tuple[float, float],
    pointing_radec_x: tuple[float, float],
    distance_detector_mask: float,
) -> BinsEquatorial:
    """
    Converts sky-shift coordinates to equatorial sky coordinates (RA/Dec).

    This function performs a coordinate transformation from a rectangular grid of points
    on a sky-shift plane to their corresponding positions in the sky using equatorial
    coordinates. To achieve the transformation it requires the pointings in equatorial
    coordinates of the x and z axis of the camera.

    The transformation process involves:
    1. Converting detector plane coordinates to direction vectors
    2. Applying the appropriate rotation matrices for the pointing direction
    3. Converting the rotated vectors to spherical coordinates (RA/Dec)

    Args:
        midpoints_sky_xs (np.array): X coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_ys.
        midpoints_sky_ys (np.array): Y coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_xs.
        pointing_radec_z (tuple[float, float]): Pointing direction of the detector's z-axis in
            (RA, Dec) coordinates in degrees.
        pointing_radec_x (tuple[float, float]): Pointing direction of the detector's x-axis in
            (RA, Dec) coordinates in degrees. Used to define the detector's roll angle.
        distance_detector_mask (float): Distance between the detector and mask planes in the same
            spatial units as midpoints_xs and midpoints_ys.

    Returns:
        BinsEquatorial record containing:
            - `dec` field: Grid of declination values in degrees, same shape as input arrays
            - `ra` field: Grid of right ascension values in degrees, same shape as input arrays.
              Values are in the range [0, 360] degrees.

    Notes:
        - Inputs (midpoints_xs, midpoints_ys,  distance_detector_mask) should be in a consistent unit system
        - The output RA values are normalized to [0, 360) degrees
        - The output Dec values are in the range [-90, 90] degrees
    """
    rotmat_sky2cam, rotmat_cam2sky = _rotation_matrices(
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
    versors_local_ys = midpoints_sky_ys / r  # TODO: for reasons, francesco ceraudo has a minus here. double check.
    versors_local_xs = midpoints_sky_xs / r  # TODO: for reasons, francesco ceraudo has a minus here. double check.
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


def to_angles(
    midpoints_xs: np.array,
    midpoints_ys: np.array,
    distance_detector_mask: float,
) -> BinsRectangular:
    """
    Expresses the sky-shift coordinates in terms of angle between source and the detector center.

    Args:
        midpoints_xs (np.array): X coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_ys.
        midpoints_ys (np.array): Y coordinates of the grid points on the sky-shift plane in spatial units
            (e.g., mm or cm). Shape and dimension should match midpoints_xs.
        distance_detector_mask (float): Distance between the detector and mask planes in the same
            spatial units as midpoints_xs and midpoints_ys.


    Returns:
        Bins2D record containing:
            - `x` field: Angular offsets in the X direction in degrees.
              Negative angles indicate positions left of center. Same shape as input arrays.
            - `y` field: Angular offsets in the Y direction in degrees.
              Negative angles indicate positions below center. Same shape as input arrays.
    """
    angles_xs = np.rad2deg(np.arctan(midpoints_xs / distance_detector_mask))
    angles_ys = np.rad2deg(np.arctan(midpoints_ys / distance_detector_mask))
    return BinsRectangular(x=angles_xs, y=angles_ys)
