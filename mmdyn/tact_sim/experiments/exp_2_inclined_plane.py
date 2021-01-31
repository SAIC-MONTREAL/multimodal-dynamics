"""
Script for collecting data with ShapeNet objects.
This is the the experiment #2 in our AAAI 2021 paper:
''objects freefalling on an inclined surface''.
"""

import pybullet as p
import random
import argparse
import numpy as np
import math
import json
import logging
from pathlib import Path
from collections import defaultdict

from mmdyn.tact_sim.utils.pybullet import add_object, setup_pybullet, remove_objects, fix_object
from mmdyn.tact_sim.utils.dataset import preload_shapenet_sem, parse_shapenet_sem
from mmdyn.tact_sim.utils.sample import sample_pose
from mmdyn.tact_sim.tactile.sensor import make_sensor
from mmdyn.tact_sim import config


logging.getLogger('trimesh').disabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--n_timesteps', type=int, default=500,
                    help='Number of timesteps in each trial.')
parser.add_argument('--dataset_dir', type=str, default='~/datasets/ShapeNetSem',
                    help='Absolute path to the dataset directory.')
parser.add_argument('--logdir', type=str, default='sim_logs',
                    help='Absolute path to log directory.')
parser.add_argument('--category', type=lambda s: [item.replace(" ", "") for item in s.split(',')], default='',
                    help='Category of the ShapeNetSem dataset.')
parser.add_argument('--show_image', default=False, action='store_true',
                    help='If true, shows the sensor images. ')
parser.add_argument('--interval', type=int, default=10,
                    help='Number of timesteps between each snapshot.')
parser.add_argument('--headless', action='store_true', default=False,
                    help='If true, uses headless rendering.')
parser.add_argument('--trial_per_obj', type=int, default=10,
                    help='Number of trials per each object.')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--slope', type=float, default=30, help="Slope in degrees")


if __name__ == '__main__':
    args = parser.parse_args()
    meta_df, root = preload_shapenet_sem(path=args.dataset_dir, category=args.category)
    print("Total number of available objects (before filtering out): {}".format(meta_df.shape))

    total_counter = 0

    for index, row in meta_df.iterrows():
        mass = 1
        slope = args.slope / 180. * math.pi
        info = parse_shapenet_sem(row, root)

        total_counter += 1

        if (info['colors'] or info['textured_material']) and np.linalg.norm(info['center_mass']) < config.COM_THRESHOLD:
            for k in range(args.trial_per_obj):
                setup_pybullet(time_step=config.TIME_STEP, renders=not args.headless, gravity=True)
                sensor = make_sensor(size=[1, 1, 1], position=[0, 0, 1], sensor_vector=[0, 0, 1],
                                     orientation=p.getQuaternionFromEuler([slope, 0, 0]),
                                     thickness=0.005, use_force=False, constrained=True)
                add_object(graphic_file="cube.obj",
                           collision_file="cube.obj",
                           base_position=[0, -1.7, 2],
                           base_orientation=[0, 0, 0, 1],
                           mesh_scale=[2, 2, 4],
                           color=[x / 255 for x in [199, 193, 193, 100]],
                           mass=10000,)

                img_counter = 0
                print("Index {}".format(index))
                print("OBJ #{} - {}: Collecting images from the object {} from category {}".format(
                    total_counter, k + 1, info['obj_name'], info['category']))

                # if the object has no image-based texture, pick one color randomly
                if not info['textured_material']:
                    color = random.choice(info['colors'])
                    color[-1] = 1.0
                else:
                    color = []

                # sample position and orientation
                init_pos = np.array([0., 0., 1.8])

                COM_shift = info['center_mass'] - np.array([0, 0, info['mesh_height'] / 4])
                # COM_shift = info['center_mass']
                print(COM_shift)

                position, orientation = sample_pose(init_pos, random_chance=0.8, random_orn=True, gaussian_mean=0, gaussian_std=0.05)

                obj_id = add_object(
                    graphic_file=info['obj'],
                    collision_file=info['obj'],
                    # mass=info['weight'],
                    mass=mass,
                    base_position=init_pos - info['center_mass'],
                    base_orientation=[0, 0, 0, 1],
                    mesh_scale=[info['scale']] * 3,
                    COM_shift=COM_shift,
                    color=color,
                )

                # quick hack for better falling dynamics
                inertial = np.array((p.getDynamicsInfo(obj_id, -1))[2]) / 5
                p.changeDynamics(obj_id, -1,
                                 # localInertiaDiagonal=inertial.tolist(),
                                 rollingFriction=0.005,
                                 # restitution=1,
                                 contactStiffness=2000,
                                 contactDamping=1,
                                 contactProcessingThreshold=10)

                pos, orn = p.getBasePositionAndOrientation(obj_id)
                p.resetBasePositionAndOrientation(obj_id, pos, orientation)

                # quick hack to prevent generating blank images
                rgb_img, rgb_equilibrium, depth_equilibrium, seg_img, seg_equilibrium = sensor.get_sensor_image()
                if sensor.is_blank(seg_img):
                    p.resetSimulation()
                    p.disconnect()
                    continue

                data = defaultdict(lambda: [])

                for t in range(args.n_timesteps):
                    fix_object(sensor.sensor_id, sensor._sensor_constraint)

                    if (t + 1) % args.interval == 0:
                        rgb_img, rgb_equilibrium, depth_equilibrium, seg_img, seg_equilibrium = sensor.get_sensor_image()

                        # there is an invisible wall at the end of the slope to prevent the object from falling on the
                        # ground, but the wall shows up in the segmentation image and the following line removes it.
                        # BUT, note that np.where is slow and should be replace by a better solution.
                        seg_img = np.where(seg_img != obj_id, -1, obj_id)

                        pointcloud = sensor.get_sensor_pointcloud(rgb_equilibrium, depth_equilibrium, mask=False)
                        tactile_img = sensor.get_tactile_image(rgb_equilibrium, depth_equilibrium, pointcloud)

                        pose = p.getBasePositionAndOrientation(obj_id)
                        data['time_step'].append(t)
                        data['time'].append(t * config.TIME_STEP)
                        data['position'].append(list(pose[0]))
                        data['orientation'].append(list(pose[1]))
                        data['force'].append(sensor.contacts.total_force(obj_id))

                        path = Path(args.logdir).joinpath(info['synset'], info['obj_name'], 'sequence_' + str(k).zfill(4))
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
                p.disconnect()
