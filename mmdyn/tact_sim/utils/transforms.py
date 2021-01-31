
import numpy as np
import pybullet as p


def get_transformation_matrix(translation, rotation):
    """
    Computes the transformation matrix with the specified translation and rotation.
    Args:
        translation (list)          : Translation
        rotation (list)             : Rotation (in Quaternion)

    Returns:
        np.array : (4x4) transformation matrix
    """
    transformation_matrix = np.zeros((4, 4))

    rot_matrix = p.getMatrixFromQuaternion(rotation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    translation = np.array(translation).reshape(3, )

    transformation_matrix[0:3, 0:3] = rot_matrix
    transformation_matrix[0:3, 3] = translation
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def get_rotation_matrix(rotation):
    """
    Computes the rotation matrix with the specified rotation.
    Args:
        rotation (list)         : Rotation (in Quaternion)

    Returns:
        np.array : (3x3) rotation matrix
    """
    rot_matrix = p.getMatrixFromQuaternion(rotation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    return rot_matrix


def apply_transformation(points, transformation_mat):
    """
    Applies the transformation matrix to the points.
    Args:
        points (np.array)                   : 3D Points in the shape (N, 3)
        transformation_mat (np.array)       : Transformation matrix in the shape (4, 4)

    Returns:
        (np.array) : Transformed points in the shape (N, 3)
    """
    points = points.transpose()
    points = np.pad(points, ((0, 1), (0, 0)), mode='constant', constant_values=1)
    points = np.matmul(transformation_mat, points)
    points = points[:3, :].transpose()

    return points


def apply_rotation(points, rotation_mat):
    """
    Applies rotation matrix to the points.
    Args:
        points (np.array)                   : 3D Points in the shape (N, 3)
        rotation_mat (np.array)             : Rotation matrix in the shape (3, 3)

    Returns:
        (np.array) : Rotated points in the shape (N, 3)
    """
    points = points.transpose()
    points = np.matmul(rotation_mat, points).transpose()

    return points
