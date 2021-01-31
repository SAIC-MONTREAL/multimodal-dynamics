import os
import pybullet as p
import numpy as np
from pathlib import Path
import math
import random

from mmdyn.tact_sim.utils.pybullet import add_object
from mmdyn.tact_sim.tactile.camera import Camera
from mmdyn.tact_sim.tactile.utils import PointCloud, ImageBuffer, normalize
from mmdyn.tact_sim.tactile.shader import Shader
from mmdyn.tact_sim.tactile.contact import Contact
from mmdyn.tact_sim import config


class Sensor:
    """
    Sensor with an Integrated Camera.
    """

    def __init__(self, position, orientation, mesh_scale, sensor_vector, mass=10000,
                 camera_up_vector=(0, 1, 0), image_width=640, image_height=480,
                 camera_fovy=60, camera_aspect=1, camera_near=0.01, camera_far=1,
                 simple_model=False, constrained=False, virtual_links=False):
        """
        Args:
            position (list)         : Initial position of the COM of the sensor in world coordinates.
            orientation (list)      : Initial orientation of the COM of the sensor in world coordinates.
            mesh_scale (list)       : Scale of the sensor OBJ. Sensor size is equivalent to this if simple_model=True.
            sensor_vector (list)    : The vector pointing towards the facing direction of the camera in the local frame
            camera_up_vector (list) : Up vector of the camera
            image_width (int)       : Image width
            image_height (int)      : Image height
            camera_fovy (float)     : The vertical lens opening angle.
            camera_aspect (float)   : The aspect ratio (width/height) of the lens.
            camera_near (float)     : Distance to the front clipping plane.
            camera_far (float)      : Distance to the back clipping plane.
            simple_model (bool)     : If true, uses a simple cube as the sensor.
        """
        self._position = np.array(position)
        self._orientation = np.array(orientation)
        self._sensor_size = np.array(mesh_scale) if simple_model else np.array([1.6, 1.6, .5])
        self._init_sensor_vector = sensor_vector
        self._time = 0.
        self._virtual_links = virtual_links
        self._constrained = constrained
        self._max_force = 10000
        model = Path(os.path.dirname(os.path.realpath(__file__))).parents[1].joinpath(
            'graphics', 'descriptions', 'sensor', 'tactile_sensor.obj') if not simple_model else "cube.obj"
        self._sensor_id = add_object(graphic_file=model,
                                     collision_file=model,
                                     base_position=position,
                                     base_orientation=orientation,
                                     mesh_scale=mesh_scale,
                                     mass=mass,
                                     color=[x / 255 for x in [255, 157, 0, 256]],
                                     virtual_links=virtual_links)
        if constrained:
            self._sensor_constraint = p.createConstraint(parentBodyUniqueId=self._sensor_id,
                                                         parentLinkIndex=-1,
                                                         childBodyUniqueId=-1,
                                                         childLinkIndex=-1,
                                                         jointType=p.JOINT_FIXED,
                                                         jointAxis=[0, 0, 0],
                                                         parentFramePosition=[0, 0, 0],
                                                         childFramePosition=[0, 0, 0],
                                                         childFrameOrientation=[0, 0, 0])
        self._camera = Camera(width=image_width,
                              height=image_height,
                              camera_up_vector=camera_up_vector)
        self._camera.set_projection_matrix(fovy=camera_fovy,
                                           aspect=camera_aspect,
                                           near=camera_near,
                                           far=camera_far)

        # surface normal vector and spanning vectors
        surface_vectors = [0 if x == 1 else 1 for x in sensor_vector]
        self._init_surface_vec_1, self._init_surface_vec_2 = np.zeros(3), np.zeros(3)
        self._init_surface_vec_1[np.nonzero(surface_vectors)[0][0]] = 1
        self._init_surface_vec_2[np.nonzero(surface_vectors)[0][1]] = 1
        self._sensor_vector, self._surface_vec_1, self._surface_vec_2 = np.array([]), np.array([]), np.array([])

        # place holder for debug lines
        self.debug_line = []
        for i in range(5):
            self.debug_line.append(p.addUserDebugLine([0., 0., 0.], [1., 0., 0.], [1, 0, 0]))

    def _update_pose(self):
        """
        Gets the updated position and orientation of the sensor from Pybullet.
        """
        self._position, self._orientation = p.getBasePositionAndOrientation(self._sensor_id)
        self._time += config.TIME_STEP
        self._position = np.array(self._position)
        self._orientation = np.array(self._orientation)

    def set_pose(self, position, orientation, quaternion=True):
        """
        Sets the position and orientation of the sensor in Pybullet.
        Args:
            position (list)         : Desired position of the sensor.
            orientation (list)      : Desired orientation of the sensor.
            quaternion (bool)       : If true, the orientation is a Quaternion, otherwise it is Euler angles.
        """
        if not quaternion:
            orientation = p.getQuaternionFromEuler(orientation)
        p.resetBasePositionAndOrientation(self._sensor_id, position, orientation)

    def _update_sensor(self):
        """
        Updates the parameters of the sensor based on its updated position and orientation.
        Always call this after the call to `_update_pose()`.
        """
        rot_matrix = p.getMatrixFromQuaternion(self._orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        self._sensor_vector = normalize(rot_matrix.dot(self._init_sensor_vector))
        self._surface_vec_1 = normalize(rot_matrix.dot(self._init_surface_vec_1))
        self._surface_vec_2 = normalize(rot_matrix.dot(self._init_surface_vec_2))
        camera_up_vector = normalize(rot_matrix.dot(self._camera.init_camera_up_vector))

        p.addUserDebugLine(self._position - self._sensor_vector *
                           abs(np.dot(self._init_sensor_vector, self._sensor_size)) / 2,
                           self._position + self._sensor_vector, [1, 0, 0], replaceItemUniqueId=self.debug_line[0])

        self._camera.set_view_matrix(self._position - self._sensor_vector *
                                     abs(np.dot(self._init_sensor_vector, self._sensor_size)) / 2,
                                     self._position + self._sensor_vector, camera_up_vector)

    def get_command(self, controller):
        """
        Applies the commands of the controller to the sensor.
        Args:
            controller (list)                   : List of controller GUI IDs
        """
        cmd = []
        for c in controller:
            cmd.append(p.readUserDebugParameter(c))
        return cmd

    def plan_motion(self, speed=40):
        """
        Plans the motion and create the next sensor command.
        Returns:
            (list) :            List of sensor commands in local coordinates (pos_x, pos_y, pos_z, orn_x, orn_y, orn_z)
        """
        rand = random.random()
        if rand < 0.3:
            cmd = [0, 0, speed / 5, 0, 0, 0]
        elif rand < 0.3:
            cmd = self.prev_cmd
        else:
            cmd = [random.uniform(-speed, speed), random.uniform(-speed, speed), 0, 0, 0, 0]
        self.prev_cmd = cmd
        return cmd

    def apply_command(self, cmd, velocity=True, local_coord=True):
        """
        Applies the command to the sensor.
        Args:
            cmd (list)          : Position and orientation commands.
            velocity (bool)     : Set to true if the commands are velocity, they are assumed to be position otherwise.
            local_coord (bool)  : Set to false to use global coordinates.
        """
        if self._virtual_links:
            for j in range(p.getNumJoints(self._sensor_id)):
                if velocity:
                    p.setJointMotorControl2(self._sensor_id, j, p.VELOCITY_CONTROL,
                                            targetPosition=0,
                                            targetVelocity=cmd[j],
                                            velocityGain=1.,
                                            force=self._max_force)
                else:
                    p.setJointMotorControl2(self._sensor_id, j, p.POSITION_CONTROL,
                                            targetPosition=cmd[j],
                                            targetVelocity=0,
                                            positionGain=1,
                                            velocityGain=1,
                                            force=self._max_force)
        else:
            if velocity:
                delta_position = np.array(cmd[0:3]) * config.TIME_STEP
                delta_orientation = np.array(cmd[3:6]) * config.TIME_STEP

                base_position, base_orientation = p.getBasePositionAndOrientation(self._sensor_id)

                if local_coord:
                    rot_mat = p.getMatrixFromQuaternion(base_orientation)
                    rot_mat = np.array(rot_mat).reshape(3, 3)
                    new_position = rot_mat.dot(delta_position) + np.array(base_position)
                else:
                    new_position = delta_position + np.array(base_position)

                base_orientation = np.array(p.getEulerFromQuaternion(base_orientation))
                new_orientation = (base_orientation + delta_orientation).tolist()
                new_orientation = p.getQuaternionFromEuler(new_orientation)
            else:
                assert not local_coord, "Position controller only works with global coordinates."
                new_position = cmd[0:3]
                new_orientation = p.getQuaternionFromEuler(cmd[3:6])

            if self._constrained:
                p.changeConstraint(self._sensor_constraint, new_position, new_orientation, maxForce=self._max_force)
            else:
                p.resetBasePositionAndOrientation(self._sensor_id, new_position, new_orientation)

    def get_sensor_image(self):
        """
        Gets the sensor RGB and depth image. For now, this uses the
        Pybullet camera which is in turn using OpenGL camera rendering.
        Returns:
            np.array, np.array, np.array        : Sensor data in RGB image, depth image and segmentation image
        """
        # update sensor position and parameters
        self._update_pose()
        self._update_sensor()
        return self._camera.get_pybullet_image()

    def get_sensor_pointcloud(self, rgb_img=None, depth_img=None):
        """
        Gets the 3D point cloud of the from the sensor data. Also saves
        the point cloud in a class property. If the data is not provided
        in the arguments, it first calls the camera for getting the image.
        Args:
            rgb_img (np.array)              : RGB image of the sensor.
            depth_img (np.array)            : Depth image of the sensor

        Returns:
            PointCloud                      : Point cloud
        """
        if rgb_img is None or depth_img is None:
            rgb_img, depth_img, _ = self.get_sensor_image()

        points, colors = self._camera.unproject_canvas_to_pointcloud(rgb_img, depth_img)
        pcd = PointCloud()
        pcd.set_points(points, colors, estimate_normals=True, camera_location=self._position)
        return pcd

    @property
    def position(self):
        return self._position

    @property
    def orientation(self):
        return self._orientation

    @property
    def sensor_size(self):
        return self._sensor_size

    @property
    def sensor_id(self):
        return self._sensor_id

    @property
    def camera(self):
        return self._camera


class TactileSensor(Sensor):
    """
    Tactile Sensor with an integrated Camera.
    """

    def __init__(self, shader, layer_thickness=0.005, buffer_size=200, solver_epsilon=1,
                 k_spring=1, darkening_factor=10, use_force=False, *args, **kwargs):
        """
        Args:
            shader (Shader)                 : Shader object used for rendering the tactile image.
            layer_thickness (float)         : Thickness of the tactile layer.
            buffer_size (int)               : Size of the image buffer for caching image data.
            solver_epsilon (float)          : Error threshold for solving the surface displacement.
            k_spring (float)                : K of the spring at each pixel in the surface of the sensor.
            darkening_factor (int)          : A factor for darkening the tactile parts.
            use_force (bool)                : If true, uses buffer for estimating the penetration of objects. Otherwise, does not compute penetration.
            *args                           : Args for the super class.
            **kwargs                        : Kwargs for the super class.
        """
        super().__init__(*args, **kwargs)

        self._shader = shader
        self._layer_thickness = layer_thickness
        self._image_buf = ImageBuffer(self.camera.width, self.camera.height, buffer_size, n_channel=3)
        self._solver_epsilon = solver_epsilon
        self._k_spring = k_spring
        self._darkening_factor = darkening_factor
        self._use_force = use_force

        # background color of the tactile sensor, this should be matched with the real sensor
        self.background_color = np.array([178, 178, 204, 255])
        # self.background_color = np.array([50, 50, 50, 255])
        # max allowable depth in the buffer, any value more than this is considered outside the tactile sensing region
        self.max_buffer_depth = self.camera.real_depth_to_buffer(
            self._layer_thickness + abs(np.dot(self._init_sensor_vector, self._sensor_size))
        )
        # for saving contact information
        self._contacts = None

    def _set_lights(self, i_specular=2.0, i_diffuse=2.0):
        """
        Sets four light sources on each edge of the sensor.
        The light sources are red, green, blue and white, respectively.
        Args:
            i_specular (float)          : Specular intensity of each light source.
            i_diffuse (float)           : Diffuse intensity of each light source.
        """
        # z_offset, z_dir_offset = 0.1, 0.15
        z_offset, z_dir_offset = 0.0, 0.0
        z = self._sensor_vector * (self._sensor_size / 2 - z_offset)
        z_dir = self._sensor_vector * z_dir_offset

        positions = [
            self._position + self._surface_vec_1 * self._sensor_size + z,
            self._position - self._surface_vec_1 * self._sensor_size + z,
            self._position + self._surface_vec_2 * self._sensor_size + z,
            self._position - self._surface_vec_2 * self._sensor_size + z
        ]
        directions = [
            -self._surface_vec_1 + z_dir,
            self._surface_vec_1 + z_dir,
            -self._surface_vec_2 + z_dir,
            self._surface_vec_2 + z_dir,
        ]
        i_speculars = [
            [i_specular, 0, 0],
            [0, i_specular, 0],
            [0, 0, i_specular],
            [i_specular, i_specular, i_specular]
        ]
        i_diffuses = [
            [i_diffuse, 0, 0],
            [0, i_diffuse, 0],
            [0, 0, i_diffuse],
            [i_diffuse, i_diffuse, i_diffuse]
        ]
        self._shader.set_lights(
            positions=positions,
            directions=directions,
            i_speculars=i_speculars,
            i_diffuses=i_diffuses
        )

    def get_sensor_image(self):
        """
        Returns the raw and clipped sensor image. The clipped image
         is generated based on  the thickness of the tactile layer.
        Returns:
            (np.array, np.array, np.array, np.array)     : RGB image, clipped RGB image, clipped depth image, seg image.
        """
        # update sensor position and parameters
        self._update_pose()
        self._update_sensor()
        rgb_img, depth_img, seg_img = self._camera.get_pybullet_image()

        # update the contact information
        self._contacts = Contact(self._sensor_id)

        # clip the images to the depth of the tactile sensor
        mask = np.where(depth_img >= self.max_buffer_depth)
        depth_img[mask] = self.max_buffer_depth

        # basically all the RGB image should be the color of the tactile surface
        clipped_rgb_img = np.copy(rgb_img)
        clipped_rgb_img[:, :, :] = self.background_color

        # clip the segmentation image
        clipped_seg_img = np.copy(seg_img)
        clipped_seg_img[mask] = -1

        if self._use_force:
            # store the raw clipped images in the image buffer with the Z position of the object
            obj_id = p.getBodyUniqueId(p.getNumBodies() - 1)
            position, _ = p.getBasePositionAndOrientation(obj_id)
            self._image_buf.store(clipped_rgb_img, depth_img, clipped_seg_img, position[-1], self._time)

            # compute the penetration based on the equilibrium of contact information and the spring models
            img_equilibrium = self.compute_equilibrium()

            return rgb_img, img_equilibrium['rgb_img'], img_equilibrium['depth_img'], seg_img, img_equilibrium['seg_img']

        else:
            return rgb_img, clipped_rgb_img, depth_img, seg_img, clipped_seg_img

    def get_sensor_pointcloud(self, rgb_img=None, depth_img=None, mask=False):
        """
        Gets the 3D point cloud of the tactile sensor data by
        clipping the camera image to the thickness of the tactile sensor.
        It is important to also return the mask because the mask is used
        later in the shader to adjust the illumination of the points in
        the image. If the data is not provided in the arguments, it first
        calls the camera for getting the image.
        Args:
            rgb_img (np.array)              : Clipped RGB image of the sensor.
            depth_img (np.array)            : Clipped depth image of the sensor
            mask (bool)                     : (Experimental) Removes the clipped point from the pointcloud for faster computation.

        Returns:
            PointCloud                      : Point cloud and the mask that has been applied to it.
        """
        if rgb_img is None or depth_img is None:
            _, rgb_img, depth_img, _ = self.get_sensor_image()

        points, colors = self._camera.unproject_canvas_to_pointcloud(rgb_img, depth_img)

        if mask:
            # remove points outside the tactile sensing region from the point cloud in 3D word coordinates
            mask = np.where(points[-1, :] <
                            self.layer_thickness + self.camera.camera_eye_position[-1] + self.sensor_size[-1] / 2)
            points = points[:, mask].squeeze()
            colors = colors[:, mask].squeeze()

        pcd = PointCloud()
        pcd.set_points(points, colors, estimate_normals=True, camera_location=self._position)
        return pcd

    def get_tactile_image(self, rgb_img, depth_img, pointcloud):
        """
        Computes the tactile image using the Shader class.
        This implements Phong's reflection model.
        The pointcloud and RGB image should be from the same
        time instance. Also the input RGB image should be clipped.
        Args:
            shader (Shader)             : Shader instance.
            rgb_img (np.array)          : Raw and clipped RGB image as returned by `get_sensor_image`.
            pointcloud (np.array)       : Pointcloud as return by `get_sensor_pointcloud`.

        Returns:
            np.array                    : Shaded RGB image emulating the surface of the tactile sensor.
        """
        self._set_lights(i_specular=2.0, i_diffuse=2.0)

        illumination = self._shader.illumination(pointcloud.points, pointcloud.normals, self._camera.camera_eye_position)
        tactile_img = self._shader.shade_image(rgb_img, illumination)

        # darken the points on the object surface
        dark_map = self.max_buffer_depth - depth_img
        dark_map = np.repeat(dark_map[:, :, np.newaxis], 3, axis=2)
        tactile_img = tactile_img - self._darkening_factor * dark_map / self._layer_thickness

        alpha = 255 * np.ones((self.camera.height, self.camera.width, 1))
        tactile_img = np.concatenate((tactile_img, alpha), axis=2)

        # round and cast to uint8
        tactile_img = np.rint(tactile_img).astype(np.uint8)

        return tactile_img

    def compute_equilibrium(self):
        """
        Computes the penetration for the objects on the sensor by doing a binary search
        on the cached images and matching the spring force with the most recent contact information.
        This assumes the tactile surface has a
        spring at each pixel and solves for the equilibrium of contact forces and the
        displacement in the depth image.
        Returns:
            (dict)                  : Recomputed RGB, Depth and Segmentation images.
        """
        l, r = 0, self._image_buf.pointer
        img = self._image_buf.get(l)

        for body in self._contacts.unique_ids:
            contact_force = self._contacts.total_force(body)

            while l <= r:
                m = int(round((l + r) / 2))
                img = self._image_buf.get(m, query='idx')
                spring_force = np.sum(self._k_spring * (self.max_buffer_depth - img['depth_img']))

                if abs(spring_force - contact_force) < self._solver_epsilon:
                    return img
                elif spring_force > contact_force:
                    r = m - 1
                else:
                    l = m + 1
        return img

    def reset(self):
        """Resets the sensor."""
        self._image_buf.reset()
        self._update_pose()
        self._update_sensor()

    def is_blank(self, seg_img):
        return np.all(seg_img == -1)

    @property
    def layer_thickness(self):
        return self._layer_thickness

    @property
    def contacts(self):
        return self._contacts


def make_sensor(position=(0., 0., 0.5), orientation=(0, 0, 0, 1), size=(1., 1., 1.), mass=10000,
                sensor_vector=(0., 0., 1.), thickness=0.01, use_force=False, constrained=False, virtual_links=False):
    # create the shader
    shader = Shader(
        k_specular=0.5,
        k_diffuse=1.,
        k_ambient=0.8,
        alpha=5,
        ambient_lightning=1.0,
        directional_light=True
    )

    # camera intrinsic properties
    near = abs(np.dot(size, sensor_vector)) * 0.9
    far = 10
    fovy = 2 * math.atan(size[0] / 2 / abs(np.dot(size, sensor_vector))) / math.pi * 180

    # create the tactile sensor
    sensor = TactileSensor(
        shader,
        layer_thickness=thickness,
        buffer_size=200,
        solver_epsilon=1.,
        k_spring=1.,
        darkening_factor=1,
        position=position,
        orientation=orientation,
        mesh_scale=size,
        mass=mass,
        sensor_vector=sensor_vector,
        camera_up_vector=[0., 1., 0.],
        image_width=640,
        image_height=480,
        camera_fovy=fovy,
        camera_aspect=1,
        camera_near=near,
        camera_far=far,
        simple_model=True,
        use_force=use_force,
        constrained=constrained,
        virtual_links=virtual_links,
    )

    return sensor

