import numpy as np
import mmdyn.tact_sim.utils.transformations as transformations
import math

class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.

class Orientation:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.

class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

class Header:
    def __init__(self):
        self.frame_id = "world"

class PoseStamped():
    def __init__(self):
        position = Position()
        orientation = Orientation()
        pose = Pose(position, orientation)
        header = Header()
        self.pose = pose
        self.header = header

    def value(self):
        print ('frame_id:    ', self.header.frame_id)
        print ('position:    ', vars(self.pose.position))
        print ('orientation: ', vars(self.pose.orientation))


def get_2d_pose(pose3d):
    #1. extract rotation about z-axis
    T = matrix_from_pose(pose3d)
    euler_angles_list = transformations.euler_from_matrix(T, 'rxyz')
    pose2d = np.array([pose3d.pose.position.x,
                       pose3d.pose.position.y,
                       euler_angles_list[2],
                       ])
    return pose2d

def C3_2d(theta):
    C = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]]
                 )

    return C

def C3(theta):
    C = np.array([[np.cos(theta), np.sin(theta), 0],
                  [-np.sin(theta), np.cos(theta), 0],
                  [0,0,1]]
                 )
    return C

def unwrap(angles, min_val=-np.pi, max_val=np.pi):
    if type(angles) is not 'ndarray':
        angles = np.array(angles)

    angles_unwrapped = []
    for counter in range(angles.shape[0]):
        angle = angles[counter]
        if angle < min_val:
            angle +=  2 * np.pi
        if angle > max_val:
            angle -=  2 * np.pi
        angles_unwrapped.append(angle)
    return np.array(angles_unwrapped)

def pose_from_matrix(matrix, frame_id="world"):
    trans = transformations.translation_from_matrix(matrix)
    quat = transformations.quaternion_from_matrix(matrix)
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose, frame_id=frame_id)
    return pose

def list2pose_stamped(pose, frame_id="world"):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg

def unit_pose():
    return list2pose_stamped([0,0,0,0,0,0,1])

def convert_reference_frame(pose_source, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target, frame_id=frame_id)
    return pose_target

def convert_reference_frame_list(pose_source_list, pose_frame_target, pose_frame_source, frame_id = "yumi_body"):
    pose_target_list = []
    for pose_source in pose_source_list:
        pose_target_list.append(convert_reference_frame(pose_source,
                                                        pose_frame_target,
                                                        pose_frame_source,
                                                        frame_id))
    return pose_target_list

def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]

def get_transform(pose_frame_target, pose_frame_source):
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target)
    T_source_world = matrix_from_pose(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(T_relative_world, frame_id=pose_frame_source.header.frame_id)
    return pose_relative_world

def matrix_from_pose(pose):
    pose_list = pose_stamped2list(pose)
    trans = pose_list[0:3]
    quat = pose_list[3:7]
    T = transformations.quaternion_matrix(quat)
    T[0:3,3] = trans
    return T

def rotate_quat_y(pose):
    '''set orientation of right gripper as a mirror reflection of left gripper about y-axis'''
    quat = pose.pose.orientation
    R = transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    z_vec = R[0:3, 2]  # np.cross(x_vec, y_vec)
    y_vec = -R[0:3, 1]  # normal
    x_vec = np.cross(y_vec, z_vec)  # np.array([0,0,-1])
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)
    # Normalized object frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = mat2quat(hand_orient_norm)
    return quat

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    R = rotation_matrix(0.123, (1, 2, 3))
    q = quaternion_from_matrix(R)
    np.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def mat2quat(orient_mat_3x3):
    orient_mat_4x4 = [[orient_mat_3x3[0][0],orient_mat_3x3[0][1],orient_mat_3x3[0][2],0],
                       [orient_mat_3x3[1][0],orient_mat_3x3[1][1],orient_mat_3x3[1][2],0],
                       [orient_mat_3x3[2][0],orient_mat_3x3[2][1],orient_mat_3x3[2][2],0],
                       [0,0,0,1]]

    orient_mat_4x4 = np.array(orient_mat_4x4)
    quat = quaternion_from_matrix(orient_mat_4x4)
    return quat

def interpolate_pose(pose_initial, pose_final, N, frac=1):
    frame_id = pose_initial.header.frame_id
    pose_initial_list = pose_stamped2list(pose_initial)
    pose_final_list = pose_stamped2list(pose_final)
    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
     # onvert to pyquaterion convertion (w,x,y,z)
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                          np.linspace(trans_initial[1], trans_final[1], num=N),
                          np.linspace(trans_initial[2], trans_final[2], num=N)]
    pose_interp = []
    for counter in range(int(frac * N)):
        quat_interp = transformations.quaternion_slerp(quat_initial,
                                                          quat_final,
                                                          float(counter) / (N-1))
        pose_tmp = [trans_interp_total[0][counter],
                            trans_interp_total[1][counter],
                            trans_interp_total[2][counter],
                            quat_interp[0], #return in ROS ordering w,x,y,z
                            quat_interp[1],
                            quat_interp[2],
                            quat_interp[3],
                            ]
        pose_interp.append(list2pose_stamped(pose_tmp, frame_id=frame_id))
    return pose_interp

def offset_local_pose(pose_world, offset):
    #1. convert to gripper reference frame
    pose_gripper = convert_reference_frame(pose_world,
                                             pose_world,
                                             unit_pose(),
                                             frame_id = "local")

    #3. add offset to grasp poses in gripper frames
    pose_gripper.pose.position.x += offset[0]
    pose_gripper.pose.position.y += offset[1]
    pose_gripper.pose.position.z += offset[2]
    #4. convert back to world frame
    pose_new_world = convert_reference_frame(pose_gripper,
                                             unit_pose(),
                                             pose_world,
                                             frame_id = "world")
    return pose_new_world

def transform_pose(pose_source, pose_transform):
    T_pose_source = matrix_from_pose(pose_source)
    T_transform_source = matrix_from_pose(pose_transform)
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = pose_from_matrix(T_pose_final_source, frame_id=pose_source.header.frame_id)
    return pose_final_source

def transform_body(pose_source_world, pose_transform_target_body):
    #convert source to target frame
    pose_source_body = convert_reference_frame(pose_source_world,
                                                 pose_source_world,
                                                 unit_pose(),
                                                 frame_id="body_frame")
    #perform transformation in body frame
    pose_source_rotated_body = transform_pose(pose_source_body,
                                              pose_transform_target_body)
    # rotate back
    pose_source_rotated_world = convert_reference_frame(pose_source_rotated_body,
                                                         unit_pose(),
                                                         pose_source_world,
                                                         frame_id="yumi_body")
    return pose_source_rotated_world

def rotate_local_pose(pose_world, offset):
    angle_x = offset[0]
    angle_y = offset[1]
    angle_z = offset[2]
    pose_transform_tmp = pose_from_matrix(transformations.euler_matrix(angle_x, angle_y, angle_z, 'sxyz'),
                                                frame_id="tmp")

    pose_rotated_world = transform_body(pose_world, pose_transform_tmp)
    return pose_rotated_world

def rotate_local_pose_list(pose_world_list, offset_list):
    pose_rotated_world_list = []
    for i, pose_world in enumerate(pose_world_list):
            pose_rotated_world_list.append(rotate_local_pose(pose_world, offset_list[i]))
    return pose_rotated_world_list

def offset_local_pose(pose_world, offset):
    #1. convert to gripper reference frame
    pose_gripper = convert_reference_frame(pose_world,
                                                         pose_world,
                                                         unit_pose(),
                                                         frame_id = "gripper_frame")

    #3. add offset to grasp poses in gripper frames
    pose_gripper.pose.position.x += offset[0]
    pose_gripper.pose.position.y += offset[1]
    pose_gripper.pose.position.z += offset[2]
    #4. convert back to world frame
    pose_new_world = convert_reference_frame(pose_gripper,
                                                         unit_pose(),
                                                         pose_world,
                                                         frame_id = "yumi_body")
    return pose_new_world
