import random
import math
import numpy as np
import trimesh
import pybullet as p

from mmdyn.tact_sim.tactile.utils import normalize

import mmdyn.tact_sim.utils.transforms as transforms


def sample_point_on_mesh(mesh, base_position=(0, 0, 0), base_orientation=(0, 0, 0, 1), scale=1):
    """
    Samples points on a mesh surface and return the point, the surface normal
    and the quaternion representing the orientation of that mesh triangle on
    the surface. Also draws the lines for better visualization.
    Args:
        mesh (trimesh.Base.Trimesh)         : Trimesh object.
        base_position (list)                : Position of the local frame w.r.t the global frame.
        base_orientation (list)             : Orientation of the local frame w.r.t the global frame.
        scale (float)                       : Scale of the loaded mesh in PyBullet.

    Returns:
        (np.array, np.array, np.array) : (point, normal, rotation matrix)
    """
    if isinstance(scale, list):
        scale = scale[0]
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    transformation_mat = transforms.get_transformation_matrix(translation=base_position, rotation=base_orientation)
    rotation_mat = transforms.get_rotation_matrix(rotation=base_orientation)

    point, face_idx = trimesh.sample.sample_surface(mesh, count=1)
    tri_points = mesh.vertices[mesh.faces[face_idx]]
    normal = mesh.face_normals[face_idx]

    point = scale * point
    tri_points = (scale * tri_points).squeeze()

    point = transforms.apply_transformation(point, transformation_mat).squeeze()
    tri_points = transforms.apply_transformation(tri_points, transformation_mat).squeeze()
    normal = transforms.apply_rotation(normal, rotation_mat).squeeze()

    v_1 = normalize(normal)
    v_2 = normalize(tri_points[1, :] - tri_points[0, :])
    v_3 = normalize(np.cross(v_1, v_2))

    rot_mat = np.stack((v_1, v_2, v_3), axis=-1)
    rot_mat = np.pad(rot_mat, ((0, 1), (0, 1)), mode='constant', constant_values=0)
    rot_mat[3, 3] = 1

    # visualization lines
    for x in [v_1, v_2, v_3]:
        p.addUserDebugLine(point.tolist(), (point + x).tolist(), [0, 1, 0])

    return point, normal, rot_mat


def sample_pose(mean_position, random_chance=0.5, gaussian_mean=0., gaussian_std=0.1, random_orn=False, random_yaw=False):
    """
    Samples a single pose for loading a single object.
    Args:
        mean_position (list or np.array)    : Base position of the object.
        gaussian_mean (float)               : Mean of the Gaussian noise.
        gaussian_std (float)                : STD of the Gaussian noise.
        random_yaw (bool)                   : If true,

    Returns:
        (np.array, np.array)                : Position and orientation of a single object.
    """
    position = np.array(mean_position) + np.random.normal(gaussian_mean, gaussian_std, size=3)
    position[-1] = mean_position[-1]

    if random_yaw:
        # generate an a random yaw angle
        orientation = p.getQuaternionFromEuler([0., 0., random.random() * 2 * math.pi])
    elif random_orn:
        if random.random() < random_chance:
            # generate a uniform Quaternion
            x = np.random.random(size=3)
            orientation = [
                math.sqrt(1 - x[0]) * math.sin(2 * math.pi * x[1]),
                math.sqrt(1 - x[0]) * math.cos(2 * math.pi * x[1]),
                math.sqrt(x[0]) * math.sin(2 * math.pi * x[2]),
                math.sqrt(x[0]) * math.cos(2 * math.pi * x[2])
            ]
        else:
            orientation = [0, 0, 0, 1]
    else:
        orientation = p.getQuaternionFromEuler([0, 0, 0])

    return np.array(position), np.array(orientation)


def sample_positions(mean_position, n_objects, orientation=(0, 0, 0, 1), gaussian_mean=0., gaussian_std=0.1):
    """
    Samples random positions for loading the objects.
    Args:
        mean_position (list)            : Base position of the object.
        n_objects (int)                 : Number of the objects.
        orientation (list)              : Base orientation of the object.
        gaussian_mean (float)           : Mean of the Gaussian noise.
        gaussian_std (float)            : STD of the Gaussian noise.

    Returns:
        (list, list)                    : Positions and orientations of the objects.
    """
    positions, orientations = [], []
    for i in range(n_objects):
        sampled_position = np.array(mean_position) + np.random.normal(gaussian_mean, gaussian_std, size=3)
        sampled_position[-1] = mean_position[-1]
        positions.append(sampled_position)
        orientations.append(orientation)
    return positions, orientations
