"""Utilities for handling datasets"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import trimesh
from trimesh.points import PointCloud

from mmdyn.tact_sim import config
from pywavefront.material import MaterialParser


def preload_object(name='winebottle', n_objects=1):
    """
    Loads the object from `graphics/descriptions/objects`.
    Args:
        name (str)                  : Name of the object.
        n_objects (int)             : Number of the objects. Note that they will be all identical.

    Returns:
        dict                        : Dictionary containing the list of objects, textures, mesh scales and COM positions
    """
    assert name in config.OBJECTS, "The specified object is not valid. Available objects are {}".format(config.OBJECTS)

    path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2].joinpath('graphics', 'objects',
                                                                                 name, 'models',
                                                                                 'model_normalized.obj')
    mesh_scale = [.05] * 3 if name == 'winebottle' else [1.] * 3
    shift = [[0, 0., 0.]]

    if n_objects > 1:
        return {
            'obj': [path] * n_objects,
            'texture': [[]] * n_objects,
            'scale': [mesh_scale] * n_objects,
            'shift': [shift] * n_objects,
        }
    else:
        return {
            'obj': path,
            'texture': [],
            'scale': mesh_scale,
            'shift': shift,
        }


def preload_shapenet_core(path=None, category=''):
    """
    **NOTE: ShapeNetCore has less categories than ShapeNetSem and the objects don't have scale and weight.
            So I recommend using ShapeNetSem over this.


    Gets the list of available objects from the ShapeNetSem objects.
    Default path for dataset is 'sesame/srs/graphics/shapenet_sem'
    Args:
        path (str)          : Path to the dataset
        category (str)      : Category of the object. If not provided, loads all allowable categories.

    Returns:
        dict                : Dictionary containing the list of objects, categories, mesh scales and COM positions
    """
    default_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[1].joinpath('graphics', 'shapenet_core')
    root = default_path if path is None else Path(path)
    obj_list = []

    if category:
        assert category in config.SHAPENET_CORE, \
            "The specified category is not valid. Available categories are {}".format(config.SHAPENET_CORE)
        obj_list = sorted(root.glob(config.SHAPENET_CORE[category] + '/**/*.obj'))
    else:
        for _, v in config.SHAPENET_CORE.items():
            obj_list += (sorted(root.glob(v + '/**/*.obj')))

    # remove objects with color-based materials or multiple texture files
    for i, obj in enumerate(obj_list):
        images_dir = obj.parents[1].joinpath('images')
        texture_list = sorted(images_dir.glob('*.*'))
        if len(texture_list) == 0:
            obj_list.pop(i)

    mesh_scale = [1, 1, 1]
    COM_shift = [0, 0, -0.1]

    assert len(obj_list) > 0, "Cannot load the ShapeNet_Core dataset."

    return {
            'obj': obj_list,
            'scale': [mesh_scale] * len(obj_list),
            'shift': [COM_shift] * len(obj_list),
    }


def preload_shapenet_sem(path=None, category='FoodItem'):
    """
    Gets the list of available objects from the ShapeNetSem objects.
    Default path for dataset is 'sesame/srs/graphics/shapenet_sem'
    Args:
        path                (str) : Path to the dataset.
        category (str)      : Category of the object. If not provided, loads all allowable categories.

    Returns:

    """
    default_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[1].joinpath('graphics', 'ShapeNetSem')
    root = default_path if path is None else Path(path)

    meta_df = pd.read_csv(root.joinpath('metadata.csv'))
    synset_df = pd.read_csv(root.joinpath('categories.synset.csv'))

    if category[0] != '':
        assert set(category).issubset(set([k for k in config.SHAPENET_SEM])), \
            "The specified category is not valid. Available categories are {}".format([k for k in config.SHAPENET_SEM])
        categories = []
        for c in category:
            # categories.append(c)
            categories.append([c] + config.SHAPENET_SEM[c])
    else:
        categories = [[k] + v for k, v in config.SHAPENET_SEM.items()]

    # flatten the categories list
    categories = [item for sublist in categories for item in sublist]

    # convert categories to synset
    synset_df = synset_df.loc[synset_df['category'].isin(categories)]
    synset = synset_df['synset'].tolist()

    # pick the specified categories
    meta_df = meta_df.loc[meta_df['wnsynset'].isin(synset)]
    meta_df['fullId'] = meta_df['fullId'].str.replace('wss.', '')

    # fill NaN values with some default constants
    meta_df = meta_df.fillna(value={
        'weight': config.DEFAULT_WEIGHT,
        'unit': config.DEFAULT_UNIT,
        'up': config.DEFAULT_UP,
        'front': config.DEFAULT_FRONT,
    })

    return meta_df, root.joinpath('models-OBJ', 'models')


def parse_shapenet_sem(row, root):
    """
    Parses a row of ShapeNetSem dataset.
    Args:
        row (row of pd.Dataframe)       : A row of pd dataframe.
        root (Path)                     : Path to the root of the dataset.

    Returns:

    """
    obj_name = row['fullId']
    scale = row['unit']
    weight = row['weight']
    category = row['category']
    synset = row['wnsynset']
    up_vector = np.array([float(s) for s in row['up'].split('\,')])
    front_vector = np.array([float(s) for s in row['front'].split('\,')])
    aligned_dims = np.array([float(s) for s in row['aligned.dims'].split('\,')]) * scale

    obj = root.joinpath(obj_name + '.obj')
    mtl = root.joinpath(obj_name + '.mtl')

    colors = []
    # dirty fix to skip objects without image-based textures or multiple materials
    # and also get a list of non-white colors
    textured_material = False
    materials = MaterialParser(file_name=mtl).materials
    for k, v in materials.items():
        if len(set(v.ambient[:-1])) > 1:
            colors.append(v.ambient)
        if v.texture is not None:
            textured_material = True

    # compute center of mass with trimesh
    # trimesh loads OBJ files with multiple materials into a scene, so we have to concatenate them
    mesh = trimesh.load_mesh(obj, 'obj')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    # center_mass = np.array(mesh.center_mass) * scale
    pcl = PointCloud(mesh.vertices)
    center_mass = np.array(pcl.centroid) * scale
    mesh_height = np.array(mesh.extents[-1]) * scale

    return {
        'obj_name': obj_name,
        'obj': obj,
        'mtl': mtl,
        'weight': weight,
        'scale': scale,
        'category': category,
        'synset': synset,
        'colors': colors,
        'textured_material': textured_material,
        'center_mass': center_mass,
        'mesh_height': mesh_height
    }

