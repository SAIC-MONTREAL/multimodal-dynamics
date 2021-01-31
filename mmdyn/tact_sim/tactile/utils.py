import cv2
import time
from pathlib import Path
import numpy as np
import open3d


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


class Video:
    """Video
    Video recording utilities.
    """

    def __init__(self, width=640, height=480, RGB=True, file_name='video_output', logdir='.'):
        """
        Args:
            width (int)             : Frame width.
            height (int)            : Frame height.
            file_name (str)         : File name.
            logdir (str)            : Log directory.
        """
        self._RGB = RGB
        time_str = time.strftime("%Y%m%d-%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_name = Path(logdir).joinpath(file_name + time_str + '.avi')
        self._video = cv2.VideoWriter(str(video_name), fourcc, 20.0, (width, height))

    def write(self, frame):
        """
        Writes the frame into the video_writer instance.
        Args:
            frame (np.array)          : Image to be written in the video.
        """
        if self._RGB:
            self._video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            self._video.write(frame)

    def close(self):
        """
        Ends the video and closes all CV2 windows.
        """
        self._video.release()
        cv2.destroyAllWindows()


class PointCloud:
    """Point cloud
    Wrapper around Open3D PointCloud for easier use.
    """

    def __init__(self):
        self._pcd = open3d.geometry.PointCloud()

    def set_points(self, points, colors=None, estimate_normals=False, camera_location=(0, 0, 0), **kwargs):
        """
        Sets the points and colors of the point cloud.
        Args:
            points (np.array)       : 3D Position of the points in the format [3, N_points] where columns are [X; Y; Z].
            colors (np.array)       : Colors of the points in the format [3, N_ponts] where columns are [R; G; B].
            camera_location (list or np.array)      : Location of the camera of orientation of the surface normals.
            estimate_normals (bool) : If true, estimates point normals.
            **kwargs                : Arguments for estimate_normal method.
        """
        self._pcd.points = open3d.utility.Vector3dVector(points.transpose())
        if colors is not None:
            self._pcd.colors = open3d.utility.Vector3dVector(colors[:3, :].transpose() / 255.)
        if estimate_normals:
            self.estimate_normals(camera_location=camera_location, **kwargs)

    def estimate_normals(self, camera_location, **kwargs):
        """
        Computes the surface normal of the point cloud. It also
        orients the surface normals towards the camera location.
        Args:
            camera_location (list or np.array)      : Location of the camera of orientation of the surface normals.
            **kwargs                                : Arguments for estimate normal method.
        """
        if len(self.points) > 0:
            self._pcd.estimate_normals(**kwargs)
            self._pcd.orient_normals_towards_camera_location(camera_location=camera_location)
            self._pcd.normalize_normals()

    def show(self):
        """
        Shows the pointcloud. Not that this method blocks
        the process. So avoid using it in a loop. If we really
        need the non-blocking visualization, there is a trick to
        make it work:
        http://www.open3d.org/docs/release/tutorial/Advanced/non_blocking_visualization.html
        """
        if len(self.points) > 0:
            open3d.visualization.draw_geometries([self._pcd])

    @property
    def points(self):
        # transpose it to be consistent with Camera class
        return np.asarray(self._pcd.points).transpose()

    @property
    def colors(self):
        # transpose it to be consistent with Camera class
        return np.asarray(self._pcd.colors).transpose()

    @property
    def normals(self):
        # transpose it to be consistent with Camera class
        return np.asarray(self._pcd.normals).transpose()

    @property
    def pcd(self):
        return self._pcd


class ImageBuffer:
    """
    Buffer for caching the images for tactile sensor depth calculation.
    """

    def __init__(self, img_width, img_height, size, n_channel=3):
        self.rgb_buf = np.zeros((size, img_width * img_height * n_channel), dtype=np.uint8)
        self.depth_buf = np.zeros((size, img_width * img_height), dtype=np.float32)
        self.seg_buf = np.zeros((size, img_width * img_height), dtype=np.uint8)
        self.z_buf = np.zeros(size, dtype=np.float32)
        self.t_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.img_width, self.img_height, self.n_channel = img_width, img_height, n_channel

    def store(self, rgb_img, depth_img, seg_img, obj_z, t):
        """
        Store the data in the buffer.
        Args:
            rgb_img (np.array)          : RGB image.
            depth_img (np.array)        : Depth image.
            seg_img (np.array)          : Segmentation image.
            obj_z (float)               : Z position of the objects in the scene.
            t (float)                   : Time of the simulator.
        """
        self.rgb_buf[self.ptr] = rgb_img[:, :, :self.n_channel].reshape((1, -1))
        self.depth_buf[self.ptr] = depth_img.reshape((1, -1))
        self.seg_buf[self.ptr] = seg_img.reshape((1, -1))
        self.z_buf[self.ptr] = obj_z
        self.t_buf[self.ptr] = t
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def get(self, s=None, body_id=None, query='idx'):
        """
        Gets an image from the buffer.
        If s is not defined, it returns the latest image.
        If s is defined, it searches for it based on the
        query type (index, z, or time).
        If body ID is defined, masks the image using the
        segmentation data.
        If there is no such point, returns the last entry of the buffer.
        Args:
            s (float)                   : Search query either on Z position or time of the objects in the scene.
            body_id (int)               : Body id of the object in query. If
            query (str)                 : Type of the search query: 'idx', 'time', or 'z'

        Returns:
            dict                        : Dictionary of the required data or False.
        """
        # default is to return the last entry to in the buffer
        idx = self.ptr - 1

        if s is not None:
            # find the datapoint with the closest point to the query
            if query == 'z':
                idx = (np.abs(self.z_buf - s)).argmin()
            elif query == 'time':
                idx = (np.abs(self.t_buf - s)).argmin()
            else:
                idx = min(int(s), self.ptr - 1)

        return {
            'rgb_img': self.rgb_buf[idx].reshape((self.img_height, self.img_width, self.n_channel)),
            'depth_img': self.depth_buf[idx].reshape((self.img_height, self.img_width)),
            'seg_img': self.seg_buf[idx].reshape((self.img_height, self.img_width)),
            'z': self.z_buf[idx],
            't': self.t_buf[idx]
        }

    def reset(self):
        """
        Resets the buffer.
        """
        self.rgb_buf = np.zeros((self.size, self.img_width * self.img_height * self.n_channel), dtype=np.uint8)
        self.depth_buf = np.zeros((self.size, self.img_width * self.img_height), dtype=np.float32)
        self.seg_buf = np.zeros((self.size, self.img_width * self.img_height), dtype=np.uint8)
        self.z_buf = np.zeros(self.size, dtype=np.float32)
        self.t_buf = np.zeros(self.size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    @property
    def min_z(self):
        try:
            return np.min(self.z_buf[:self.ptr - 1])
        except ValueError:
            return 0.

    @property
    def max_z(self):
        try:
            return np.max(self.z_buf[:self.ptr - 1])
        except ValueError:
            return 0.

    @property
    def min_t(self):
        try:
            return np.min(self.t_buf[:self.ptr - 1])
        except ValueError:
            return 0.

    @property
    def max_t(self):
        try:
            return np.max(self.t_buf[:self.ptr - 1])
        except ValueError:
            return 0.

    @property
    def pointer(self):
        return self.ptr
