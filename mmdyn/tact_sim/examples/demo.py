"""
Sample script for collecting single object imprint.
Does not need ShapeNetSem to run.
"""

import pybullet as p
import argparse
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict

from mmdyn.tact_sim.utils.pybullet import add_object, setup_pybullet
from mmdyn.tact_sim.utils.dataset import preload_object
from mmdyn.tact_sim.tactile.sensor import make_sensor
from mmdyn.tact_sim import config
from mmdyn.tact_sim.utils import transformations, geometry

logging.getLogger('trimesh').disabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--object', type=str, default='winebottle')
parser.add_argument('--logdir', type=str, default='sim_logs')
parser.add_argument('--n_timesteps', type=int, default=200,
                    help='Number of timesteps in simulation.')
parser.add_argument('--show_image', default=False, action='store_true',
                    help='If true, shows the sensor images. ')
parser.add_argument('--interval', type=int, default=10,
                    help='Number of timesteps between each snapshot.')
parser.add_argument('--headless', action='store_true', default=False,
                    help='If true, uses headless rendering.')
parser.add_argument('--debug', action='store_true', default=False)


if __name__ == '__main__':
    args = parser.parse_args()

    info = preload_object(args.object)
    print("Preloaded the object.")

    # setup simulator
    setup_pybullet(time_step=config.TIME_STEP,
                   renders=not args.headless,
                   gravity=True)
    sensor = make_sensor(size=[1.5, 1.5, 1],
                         position=[0, 0, 0.5],
                         sensor_vector=[0, 0, 1],
                         thickness=0.01,
                         use_force=False,
                         constrained=False)

    img_counter = 0

    # set initial position and orientation
    position = np.array([0., 0., 1.3])
    orientation = np.array([0, 0, 0, 1])
    base_pose = geometry.list2pose_stamped(list(position) + list(orientation))
    T = transformations.euler_matrix(0, 0, 0)
    pose_transform = geometry.pose_from_matrix(T, frame_id='body')
    object_pose = geometry.transform_body(base_pose, pose_transform)
    object_pose_list = geometry.pose_stamped2list(object_pose)

    # add object
    obj_id = add_object(
        graphic_file=info['obj'],
        collision_file=info['obj'],
        mass=.5,
        base_position=object_pose_list[0:3],
        base_orientation=object_pose_list[3:7],
        mesh_scale=info['scale'],
        color=[1, 0, 0, 1],
    )

    data = defaultdict(lambda: [])

    for t in range(args.n_timesteps):

        if (t + 1) % args.interval == 0:
            rgb_img, rgb_equilibrium, depth_equilibrium, seg_img, seg_equilibrium = sensor.get_sensor_image()
            seg_img = np.where(seg_img != obj_id, -1, obj_id)

            pointcloud = sensor.get_sensor_pointcloud(rgb_equilibrium, depth_equilibrium, mask=False)
            tactile_img = sensor.get_tactile_image(rgb_equilibrium, depth_equilibrium, pointcloud)

            pose = p.getBasePositionAndOrientation(obj_id)
            data['time_step'].append(t)
            data['time'].append(t * config.TIME_STEP)
            data['position'].append(list(pose[0]))
            data['orientation'].append(list(pose[1]))

            path = Path(args.logdir).joinpath('dataset')
            sensor.camera.save_image(rgb_img, path, title='visual_' + str(img_counter).zfill(4), time_stamp=False)
            sensor.camera.save_image(tactile_img, path, title='tactile_' + str(img_counter).zfill(4), time_stamp=False)
            sensor.camera.save_image(seg_img, path, RGB=False, title='seg_' + str(img_counter).zfill(4), time_stamp=False)
            sensor.camera.save_image(depth_equilibrium, path, RGB=False, title='depth_' + str(img_counter).zfill(4), time_stamp=False)
            img_counter += 1

            if args.debug:
                camera_info = p.getDebugVisualizerCamera()
                _, _, debug_rgb, _, _ = p.getCameraImage(640, 480, camera_info[2], camera_info[3],
                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)
                sensor.camera.save_image(debug_rgb, path, RGB=True, title='debug_' + str(img_counter).zfill(4), time_stamp=False)
                # sensor.camera.show_image(debug_rgb, title="Debug", save=False)

            if args.show_image:
                sensor.camera.show_image(rgb_img, title='Raw RGB', save=False)
                sensor.camera.show_image(tactile_img, title='Tactile RGB', save=False)

        p.stepSimulation()

    with open(path.joinpath('data.json'), 'w') as f:
        json.dump(data, f)

    p.resetSimulation()
