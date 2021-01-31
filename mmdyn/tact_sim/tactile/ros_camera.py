"""ROS Camera models
Code taken from ROS-Perception package:
https://github.com/ros-perception/vision_opencv/blob/ros2/image_geometry/image_geometry/cameramodels.py

Some modifications have been made by me to remove
ROS dependencies and make it work with PyBullet.
"""

import cv2
import math
import numpy as np


def mkmat(rows, cols, L):
    mat = np.matrix(L, dtype='float64')
    mat.resize((rows, cols))
    return mat


class ROI:

    def __init__(self, width=0, height=0, x_offset=0, y_offset=0):
        """
        ROI all zeros is considered the same as full resolution
        """
        self._width = width
        self._height = height
        self._x_offset = x_offset
        self._y_offset = y_offset

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def x_offset(self):
        return self._x_offset

    @property
    def y_offset(self):
        return self._y_offset


class ROSPinholeCameraModel:
    """
    A pinhole camera is an idealized monocular camera.
    """
    def __init__(self):
        self._K = None
        self._D = None
        self._R = None
        self._P = None
        self._full_K = None
        self._full_P = None
        self._width = None
        self._height = None
        self._binning_x = None
        self._binning_y = None
        self._raw_roi = None
        self._resolution = None
        self._map_x = None
        self._map_y = None

    def from_camera_params(self, k, r, p, width, height, d=None, binning_x=1, binning_y=1, roi=None):
        """
        Set the camera parameters using the camera parameters similar to ROS. See below for more information:
        http://docs.ros.org/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        """
        self._K = mkmat(3, 3, k)
        if d:
            self._D = mkmat(len(d), 1, d)
        else:
            self._D = None
        self._R = mkmat(3, 3, r)
        self._P = mkmat(3, 4, p)
        self._full_K = mkmat(3, 3, k)
        self._full_P = mkmat(3, 4, p)
        self._width = width
        self._height = height
        self._binning_x = max(1, binning_x)
        self._binning_y = max(1, binning_y)
        self._resolution = (width, height)
        self._raw_roi = roi if roi is not None else ROI(width=0, height=0, x_offset=0, y_offset=0)
        self._map_x = None
        self._map_y = None

        # Adjust K and P for binning and ROI
        self._K[0, 0] /= self._binning_x
        self._K[1, 1] /= self._binning_y
        self._K[0, 2] = (self._K[0, 2] - self._raw_roi.x_offset) / self._binning_x
        self._K[1, 2] = (self._K[1, 2] - self._raw_roi.y_offset) / self._binning_y
        self._P[0, 0] /= self._binning_x
        self._P[1, 1] /= self._binning_y
        self._P[0, 2] = (self._P[0, 2] - self._raw_roi.x_offset) / self._binning_x
        self._P[1, 2] = (self._P[1, 2] - self._raw_roi.y_offset) / self._binning_y

    def rectify_image(self, raw):
        """
        Applies the rectification specified by camera parameters `K`
        and `D` to image `raw` and returns the resulting image `rectified`.
        Args:
            raw (:class:`CvMat` or :class:`IplImage`)   : Raw image

        Returns:
            :class:`CvMat` or :class:`IplImage`         : Rectified image
        """
        rectified = []
        self._map_x = np.ndarray(shape=(self._height, self._width, 1),
                                 dtype='float32')
        self._map_y = np.ndarray(shape=(self._height, self._width, 1),
                                 dtype='float32')
        cv2.initUndistortRectifyMap(self._K, self._D, self._R, self._P,
                                    (self._width, self._height), cv2.CV_32FC1, self._map_x, self._map_y)
        cv2.remap(raw, self._map_x, self._map_y, cv2.INTER_CUBIC, rectified)

    def rectify_point(self, uv_raw):
        """
        Applies the rectification specified by camera parameters `K` and `D` to
        point (u, v) and returns the pixel coordinates of the rectified point.
        Args:
            uv_raw           (u, v) : Pixel coordinates

        Returns:
            (u, v)                  : Rectified image
        """

        src = mkmat(1, 2, list(uv_raw))
        src.resize((1, 1, 2))
        dst = cv2.undistortPoints(src, self._K, self._D, R=self._R, P=self._P)
        return dst[0, 0]

    def project_3D_to_pixel(self, point):
        """
        Returns the rectified pixel coordinates (u, v)
        of the 3D point, using the camera `P` matrix.
        This is the inverse of `projectPixelTo3dRay`.
        Args:
            point (x, y, z)     : 3D point

        Returns:
            (u, v)              : Pixel point
        """
        src = mkmat(4, 1, [point[0], point[1], point[2], 1.0])
        dst = self._P * src
        x = dst[0, 0]
        y = dst[1, 0]
        w = dst[2, 0]
        if w != 0:
            return x / w, y / w
        else:
            return float('nan'), float('nan')

    def project_pixel_to_3DRay(self, uv):
        """
        Returns the unit vector which passes from the camera center to
        through rectified pixel (u, v), using the camera :math:`P` matrix.
        This is the inverse of `project3dToPixel`.
        Args:
            uv (u, v)           : rectified pixel coordinates

        Returns:
            (x, y, z)           : 3D point
        """
        x = (uv[0] - self.cx) / self.fx
        y = (uv[1] - self.cy) / self.fy
        norm = math.sqrt(x * x + y * y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return x, y, z

    def get_delta_u(self, deltaX, Z):
        """
        Compute delta u, given Z and delta X in Cartesian space.
        For given Z, this is the inverse of `getDeltaX`.
        Args:
            deltaX (float)      : delta X, in cartesian space
            Z (float)           : Z, in cartesian space

        Returns:
            float               : delta u

        """
        fx = self._P[0, 0]
        if Z == 0:
            return float('inf')
        else:
            return fx * deltaX / Z

    def get_delta_v(self, deltaY, Z):
        """
        Compute delta v, given Z and delta Y in Cartesian space.
        For given Z, this is the inverse of `getDeltaY`.
        Args:
            deltaY (float)      : delta Y, in cartesian space
            Z (float)           : Z, in cartesian space

        Returns:
            float               : delta v

        """
        fy = self._P[1, 1]
        if Z == 0:
            return float('inf')
        else:
            return fy * deltaY / Z

    def get_delta_x(self, deltaU, Z):
        """
        Compute delta X, given Z in cartesian space and delta u in
        pixels. For given Z, this is the inverse of `getDeltaU`.
        Args:
            deltaU (float)      : delta u in pixels
            Z (float)           : Z, in cartesian space

        Returns:
            float               : delta x
        """
        fx = self._P[0, 0]
        return Z * deltaU / fx

    def get_delta_y(self, deltaV, Z):
        """
        Compute delta Y, given Z in cartesian space and delta v in pixels.
        For given Z, this is the inverse of `getDeltaV`.
        Args:
            deltaV (float)      : delta v in pixels
            Z (float)           : Z, in cartesian space

        Returns:
            float               : delta v

        """
        fy = self._P[1, 1]
        return Z * deltaV / fy

    @property
    def full_resolution(self):
        return self._resolution

    @property
    def intrinsic_matrix(self):
        return self._K

    @property
    def distortion_coeffs(self):
        return self._D

    @property
    def rotation_matrix(self):
        return self._R

    @property
    def projection_matrix(self):
        return self._P

    @property
    def full_intrinsic_matrix(self):
        return self._full_K

    @property
    def full_projection_matrix(self):
        return self._full_P

    @property
    def cx(self):
        return self._P[0, 2]

    @property
    def cy(self):
        return self._P[1, 2]

    @property
    def fx(self):
        return self._P[0, 0]

    @property
    def fy(self):
        return self._P[1, 1]

    @property
    def Tx(self):
        return self._P[0, 3]

    @property
    def Ty(self):
        return self._P[1, 3]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class ROSStereoCameraModel:
    """
    An idealized stereo camera.
    """

    def __init__(self):
        self.left_cam = ROSPinholeCameraModel()
        self.right_cam = ROSPinholeCameraModel()

        self.Q = None

    def from_camera_params(self, left_cam_params, right_cam_params):
        """
        Set the camera parameters. See below for more information:
        Args:
            left_cam_params (dict)      : Left cam parameters
            right_cam_params (dict)     : Right cam parameters
        """
        self.left_cam.from_camera_params(**left_cam_params)
        self.right_cam.from_camera_params(**right_cam_params)

        # [ Fx, 0,  Cx,  Fx*-Tx ]
        # [ 0,  Fy, Cy,  0      ]
        # [ 0,  0,  1,   0      ]

        fx = self.right_cam.projection_matrix[0, 0]
        fy = self.right_cam.projection_matrix[1, 1]
        cx = self.right_cam.projection_matrix[0, 2]
        cy = self.right_cam.projection_matrix[1, 2]
        tx = -self.right_cam.projection_matrix[0, 3] / fx

        # Q is:
        #    [ 1, 0,  0, -Clx ]
        #    [ 0, 1,  0, -Cy ]
        #    [ 0, 0,  0,  Fx ]
        #    [ 0, 0, 1 / Tx, (Crx-Clx)/Tx ]

        self.Q = np.zeros((4, 4), dtype='float64')
        self.Q[0, 0] = 1.0
        self.Q[0, 3] = -cx
        self.Q[1, 1] = 1.0
        self.Q[1, 3] = -cy
        self.Q[2, 3] = fx
        self.Q[3, 2] = 1 / tx

    def project_3D_to_pixel(self, point):
        """
        Returns the rectified pixel coordinates (u, v) of the 3D point for each camera
        using the cameras' `P` matrices. This is the inverse of `projectPixelTo3d`.
        Args:
            point (x, y, z)                             : 3D point

        Returns:
            ((u_left, v_left), (u_right, v_right))      : Rectified point of each camera
        """
        left_pixel = self.left_cam.project_3D_to_pixel(point)
        right_pixel = self.right_cam.project_3D_to_pixel(point)
        return left_pixel, right_pixel

    def project_pixel_to_3D(self, left_uv, disparity):
        """
        Returns the 3D point (x, y, z) for the given pixel position, using
        the cameras' `P` matrices. This is the inverse of `project3dToPixel`.
        Note that a disparity of zero implies that the 3D point is at infinity.
        Args:
            left_uv (u, v)          : Rectified pixel coordinates
            disparity (float)       : Disparity, in pixels

        Returns:
            (x, y, z)               : 3D point
        """
        src = mkmat(4, 1, [left_uv[0], left_uv[1], disparity, 1.0])
        dst = self.Q * src
        x = dst[0, 0]
        y = dst[1, 0]
        z = dst[2, 0]
        w = dst[3, 0]
        if w != 0:
            return x / w, y / w, z / w
        else:
            return 0.0, 0.0, 0.0

    def get_z(self, disparity):
        """
        Returns the depth at which a point is observed with a given disparity.
        This is the inverse of `getDisparity`.
        Note that a disparity of zero implies Z is infinite.
        Args:
            disparity (float)       : Disparity, in pixels

        Returns:
            float                   : Depth
        """
        if disparity == 0:
            return float('inf')
        Tx = -self.right_cam.projection_matrix[0, 3]
        return Tx / disparity

    def get_disparity(self, Z):
        """
        Returns the disparity observed for a point at depth Z.
        This is the inverse of `getZ`.
        Args:
            Z (float)               : Depth

        Returns:
            float                   : Disparity

        """
        if Z == 0:
            return float('inf')
        Tx = -self.right_cam.projection_matrix[0, 3]
        return Tx / Z
