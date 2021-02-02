from setuptools import setup

setup(
    name='mmdyn',
    py_modules=['mmdyn'],
    version='0.1.0',
    install_requires=[
        'pybullet',
        'torch',
        'torchvision',
        'tensorboard',
        'numpy',
        'pandas',
        'opencv-python',
        'pywavefront',
        'trimesh',
        'open3d',
        'pillow',
        'networkx',
        'seaborn',
        'pyquaternion',
        'pyyaml'
    ],
    description="Tactile Simulator based on PyBullet",
    author="Sahand Rezaei-Shoshtari"
)