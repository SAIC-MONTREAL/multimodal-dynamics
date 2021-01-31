"""Utilities for general functionalities in PyBullet"""

import math
import pybullet as p
import pybullet_data


def setup_pybullet(time_step=1. / 240,
                   load_plane=True,
                   gravity=True,
                   plane_urdf="plane100.urdf",
                   renders=True):
    """
    Setup Pybullet simulator and connect python to gui interface
    Args:
        time_step           (float) : Simulation time step.
        plane_urdf            (str) : Plane name.
        renders:             (bool) : If true, renders the environment.

    Returns:
        None
    """
    if renders:
        cid = p.connect(p.GUI)
        if cid < 0:
            p.connect(p.gui)
        p.resetDebugVisualizerCamera(1, 0, -20, [0., 0., 1.])
    else:
        p.connect(p.DIRECT)

    pybullet_data.getDataPath()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(time_step)
    if load_plane:
        p.loadURDF(plane_urdf)
    if gravity:
        p.setGravity(0, 0, -10)


def add_object(graphic_file="duck.obj",
               collision_file="duck_vhacd.obj",
               texture_file=(),
               mass=1,
               base_position=(0., 0., 0.),
               base_orientation=(0., 0., 0., 1.),
               mesh_scale=(1, 1, 1),
               COM_shift=(0, 0.0, 0),
               color=(),
               diagonal_inertial=None,
               virtual_links=False,
               constrained=False):
    """
    Adds a textured mesh into the scene.
    Args:
        graphic_file        : Relative or absolute path to the mesh file.
        collision_file      : Relative or absolute path to the mesh file.
        texture_file        : Relative or absolute path to the texture file.
                            If empty, no texture will be loaded.
        mass                : Mass of the object in Kg.
        base_position       : Position of the base of the object.
        base_orientation    : Orientation of the base of the object.
        mesh_scale          : Scale of the mesh.
        COM_shift           : COM position of the object w.r.t the local coordinates.
        color               : Color of the object.
        virtual_links       : If true, defines virtual links for moving the base in the space.
        constrained         : If true, creates a dynamics constraint.

    Returns:
        int         : Unique ID of the object in Pybullet
        (int)       : Constraint ID (only returns if constrained is True).
    """
    shift = [0, 0, 0]
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=str(graphic_file),
                                        rgbaColor=[1, 1, 1, 1],
                                        specularColor=[0.4, .4, 0],
                                        visualFramePosition=[0, 0, 0],
                                        meshScale=mesh_scale)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                              fileName=str(collision_file),
                                              collisionFramePosition=shift,
                                              meshScale=mesh_scale)

    if virtual_links:
        n_links = 6
        link_mass = [0.001] * n_links
        link_col_shape = [-1] * n_links
        link_vis_shape = [-1] * n_links
        link_position = [[0, 0, 0]] * n_links
        link_orientation = [[0, 0, 0, 1]] * n_links
        link_inertial_frame_position = [[0, 0, 0]] * n_links
        link_inertial_frame_orientation = [[0, 0, 0, 1]] * n_links
        link_parent = [0, 1, 2, 3, 4, 5]
        link_joint_type = [p.JOINT_PRISMATIC, p.JOINT_PRISMATIC, p.JOINT_PRISMATIC, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        link_joint_axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        obj_id = p.createMultiBody(baseMass=mass,
                                   baseInertialFramePosition=COM_shift,
                                   baseInertialFrameOrientation=[0, 0, 0, 1],
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=base_position,
                                   baseOrientation=base_orientation,
                                   linkMasses=link_mass,
                                   linkCollisionShapeIndices=link_col_shape,
                                   linkVisualShapeIndices=link_vis_shape,
                                   linkPositions=link_position,
                                   linkOrientations=link_orientation,
                                   linkInertialFramePositions=link_inertial_frame_position,
                                   linkInertialFrameOrientations=link_inertial_frame_orientation,
                                   linkParentIndices=link_parent,
                                   linkJointTypes=link_joint_type,
                                   linkJointAxis=link_joint_axis,
                                   useMaximalCoordinates=False)
        for i in range(n_links):
            p.changeDynamics(bodyUniqueId=obj_id,
                             linkIndex=i,
                             jointDamping=0.1,
                             jointLowerLimit=-100,
                             jointUpperLimit=100,
                             localInertiaDiagonal=[0.001, 0.001, 0.001])

    else:
        obj_id = p.createMultiBody(baseMass=mass,
                                   baseInertialFramePosition=COM_shift,
                                   baseInertialFrameOrientation=[0, 0, 0, 1],
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=base_position,
                                   baseOrientation=base_orientation,
                                   useMaximalCoordinates=False)

    if texture_file:
        textureId = p.loadTexture(textureFilename=str(texture_file))
        p.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, textureUniqueId=textureId)

    if color:
        p.changeVisualShape(objectUniqueId=obj_id, linkIndex=-1, rgbaColor=color)

    if diagonal_inertial:
        assert isinstance(diagonal_inertial, list)
        p.changeDynamics(bodyUniqueId=obj_id, linkIndex=-1, localInertiaDiagonal=diagonal_inertial)

    if constrained:
        constraint_id = p.createConstraint(parentBodyUniqueId=obj_id,
                                           parentLinkIndex=-1,
                                           childBodyUniqueId=-1,
                                           childLinkIndex=-1,
                                           jointType=p.JOINT_FIXED,
                                           jointAxis=[0, 0, 0],
                                           parentFramePosition=[0, 0, 0],
                                           childFramePosition=[0, 0, 0],
                                           childFrameOrientation=[0, 0, 0])
        return obj_id, constraint_id

    return obj_id


def add_objects(graphic_files, collision_files, texture_files, masses,
                base_positions, base_orientations, mesh_scales, COM_shifts):
    """
    Adds multiple objects to the scene.
    Args:
        graphic_files           : Relative or absolute path to the mesh file.
        collision_files         : Relative or absolute path to the mesh file.
        texture_files           : Relative or absolute path to the texture file.
                                If empty, no texture will be loaded.
        masses                  : Mass of each object in Kg.
        base_positions          : Position of the base of the object.
        base_orientations       : Orientation of the base of the object.
        mesh_scales             : Scale of the mesh.
        COM_shifts              : COM position of the object w.r.t the local coordinates.

    Returns:
        list : List of Unique IDs of the added objects
    """
    assert len(graphic_files) == len(collision_files) == len(texture_files) == len(base_positions) == \
           len(base_orientations) == len(mesh_scales) == len(COM_shifts), \
        "All lists must have the same number of elements."
    obj_id = []

    for graphic, collision, texture, mass, position, orientation, mesh_scale, COM_shift in \
            zip(graphic_files, collision_files, texture_files, masses,
                base_positions, base_orientations, mesh_scales, COM_shifts):
        try:
            obj_id.append(add_object(graphic, collision, texture, mass, position, orientation, mesh_scale, COM_shift))
        except p.error:
            print("Cannot load the mesh, will skip this one.")

    return obj_id


def remove_objects(obj_id):
    """
    Remove the object(s) from Pybullet.
    Args:
        obj_id (list or int)        : ID of the object(s).
    """
    if isinstance(obj_id, list):
        for id in obj_id:
            p.removeBody(id)
    else:
        p.removeBody(obj_id)


def create_gui_controller(velocity=False, amp=1):
    """
    Creates a simple velocity or position controller for the sensor.
    Args:
        velocity (bool)         : If true, defines velocity controller. Otherwise, uses position controller.
        amp (float)             : Amplitude of the commands.

    Returns:
        list : List of motor commands
    """
    motor_ids = []
    for param in ['posX', 'posY', 'posZ']:
        motor_ids.append(p.addUserDebugParameter(param, -amp, amp, 0))
    for param in ['rotX', 'rotY', 'rotZ']:
        if velocity:
            motor_ids.append(p.addUserDebugParameter(param, -amp, amp, 0))
        else:
            motor_ids.append(p.addUserDebugParameter(param, -math.pi, math.pi, 0))
    return motor_ids


def fix_object(obj_id, constraint_id, max_force=100000):
    """
    Fixes an object in the space. Note that this should be called in every step
    before the call to p.stepSimulation.
    Args:
        obj_id (int)            : Obj ID.
        constraint_id (int)     : Constraint ID.
        max_force (float)       : Maximum force applied to the object to keep it fixed.

    Returns:

    """
    pos, orn = p.getBasePositionAndOrientation(obj_id)
    p.changeConstraint(constraint_id, pos, orn, maxForce=max_force)
