# Learning Intuitive Physics with Multimodal Generative Models
This package provides the PyTorch implementation and the vision-based tactile sensor 
simulator for our [AAAI 2021 paper](https://arxiv.org/pdf/2101.04454.pdf).
The tactile simulator is based on PyBullet and provide the simulation of the
[Semi-transparent Tactile Sensor (STS)](https://openaccess.thecvf.com/content/WACV2021/papers/Hogan_Seeing_Through_Your_Skin_Recognizing_Objects_With_a_Novel_Visuotactile_WACV_2021_paper.pdf). 

## Installation
The recommended way is to install the package and all its dependencies in a
virtual environment using:
```
git clone https://github.com/SAIC-MONTREAL/multimodal-dynamics.git
cd multimodal-dynamics
pip install -e .
```

## Visuotactile Simulation
The sub-package [tact_sim](TODO) provides the components required for visoutactile simulation
of the STS sensor and is implemented in [PyBullet](https://github.com/bulletphysics/bullet3). 
The simulation is vision based and is not meant to be physically
accurate of the contacts and soft body dynamics. 

To run an example script of an object falling on the sensor use:
```
FROM ~/multimodal-dynamics/mmdyn/tact_sim
python examples/demo.py --show_image --object winebottle
```
This loads the object from the [graphics/objects](TODO) and renders 
the resulting visual and tactile images.

The example scripts following the name format `experiments/exp_{ID}_{task}.py` have been 
used to generate the dataset of our [AAAI 2021 paper](https://arxiv.org/pdf/2101.04454.pdf).
In order to run them, you need to have [ShapeNetSem](https://www.shapenet.org/download/shapenetsem)
dataset installed on your machine

### Preparing ShapeNetSem 
Follow the steps below to download and prepare the ShapeNetSem dataset:
1. Register and get access to [ShapeNetSem](https://www.shapenet.org/download/shapenetsem).
1. Only the OBJ and texture files are needed. Download `models-OBJ.zip` and `models-textures.zip`.
1. Download `metadata.csv` and `categories.synset.csv`.
1. Unzip the compressed files and move the contents of `models-textures.zip` to `models-OBJ/models`:
```
.
└── ShapeNetSem
    ├── categories.synset.csv
    ├── metadata.csv
    └── models-OBJ
        └── models
```

### Data Collection
To run the data collection scripts use:
```
python experiments/exp_{ID}_{task}.py --logdir {path_to_logdir} --dataset_dir {path_to_ShapeNetSem} --category "WineBottle, Camera" --show_image
```
To see all available object classes that are suitable for these experiments see 
[tact_sim/config.py](TODO).  

## License 
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">SECA</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

## Citing
If you use this code in your research, please cite:
```
@article{rezaei2021learning,
  title={Learning Intuitive Physics with Multimodal Generative Models},
  author={Rezaei-Shoshtari, Sahand and Hogan, Francois Robert and Jenkin, Michael and Meger, David and Dudek, Gregory},
  journal={arXiv preprint arXiv:2101.04454},
  year={2021}
}
```
```
@inproceedings{hogan2021seeing,
  title={Seeing Through your Skin: Recognizing Objects with a Novel Visuotactile Sensor},
  author={Hogan, Francois R and Jenkin, Michael and Rezaei-Shoshtari, Sahand and Girdhar, Yogesh and Meger, David and Dudek, Gregory},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1218--1227},
  year={2021}
}
```
