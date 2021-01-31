import numpy as np
import math


class Light:
    """Light
    Light source and its properties.
    """

    def __init__(self, position, direction, i_specular, i_diffuse):
        """
        Args:
            position (list)         : 3D Position of the light source in the format [X, Y, Z].
            direction (list)        : Direction vector of the light soruce [X, Y, Z].
            i_specular (list)       : Specular intensity of the light source for each channel in the format [R, G, B].
            i_diffuse (list)        : Diffuse intensity of the light source for each channel in the format [R, G, B].
        """
        self._position = position
        self._direction = direction
        self._i_specular = i_specular
        self._i_diffuse = i_diffuse

    @property
    def position(self):
        return np.reshape(self._position, (3, 1))

    @property
    def direction(self):
        return np.reshape(self._direction, (3, 1))

    @property
    def i_specular(self):
        return np.reshape(self._i_specular, (3, 1))

    @property
    def i_diffuse(self):
        return np.reshape(self._i_diffuse, (3, 1))


class Shader:
    """Shader
    Implements Phong's reflection model.
    """

    def __init__(self, k_specular=0.15, k_diffuse=0.5, k_ambient=1, alpha=5,
                 ambient_lightning=1, directional_light=True):
        """
        Args:
            k_specular (float)          : Specular reflection constant.
            k_diffuse (float)           : Diffuse reflection constant.
            k_ambient (float)           : Ambient reflection constant.
            alpha (float)               : Shininess constant of the material.
            ambient_lightning (float)   : Ambient lightning.
            directional_light (bool)      : If true, uses directional light. Otherwise uses point light. 
        """
        self._directional_light = directional_light
        self._k_specular = k_specular
        self._k_diffuse = k_diffuse
        self._k_ambient = k_ambient
        self._alpha = alpha
        self._ambient_lightning = ambient_lightning
        self._lights = []

    def set_lights(self, positions, directions, i_speculars, i_diffuses):
        """
        Sets the lights of the environment.
        Args:
            positions (list)            : List of the light positions.
            directions (list)           : List of the light directions.
            i_speculars (list)          : List of the light specular intensities.
            i_diffuses (list)           : List of the light diffuse intensities.
        """
        self._lights = []
        assert len(positions) == len(i_speculars) == len(i_diffuses), "All properties must have the same length."
        for position, direction, i_specular, i_diffuse in zip(positions, directions, i_speculars, i_diffuses):
            self._lights.append(Light(position, direction, i_specular, i_diffuse))

    def illumination(self, points, surface_normals, viewer):
        """
        Computes the surface illumination using Phong's reflection model.
        Args:
            points (np.array)               : Position of the 3D points.
            surface_normals (np.array)      : Surface normals of the 3D points.
            viewer (np.array)               : Position of the viewer.

        Returns:
            np.array                        : Illumination of each point.
        """
        # TODO: normalize these?
        # make sure everything has the right shape
        points = np.reshape(points, (3, -1))
        surface_normals = np.reshape(surface_normals, (3, -1))
        viewer = np.reshape(viewer, (3, -1))

        I_p = self._k_ambient * self._ambient_lightning
        for light in self._lights:
            V = viewer - points

            if self._directional_light:
                L = light.direction
            else:
                L = light.position - points

            # clip values less than zero to zero, because they are shining from behind the surface
            normal_dot_light = np.sum(L * surface_normals, axis=0)
            normal_dot_light = np.clip(normal_dot_light, 0, math.inf)

            R = 2 * normal_dot_light * surface_normals - L

            I_p += (self._k_diffuse * normal_dot_light * light.i_diffuse +
                    self._k_specular * ((np.sum(R * V, axis=0)) ** self._alpha) * light.i_specular)

        return I_p

    def shade_image(self, rgb_img, illumination):
        """
        Shades the input RGB image with the input illumination.
        Args:
            rgb_img (np.array)          : The input RGB image in the format (height, width, channel)
            illumination (np.array)     : The input illumination of each point in the format (3, n_points)

        Returns:
            np.array                    : Shaded RGB image in the format (height, width, channel)
        """
        height, width = rgb_img.shape[0], rgb_img.shape[1]
        illumination = illumination.transpose().reshape(height, width, -1)
        shaded_image = np.clip(rgb_img[:, :, :3] * illumination, 0, 255)

        return shaded_image

    @property
    def lights(self):
        return self._lights
