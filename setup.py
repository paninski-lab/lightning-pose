#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "torchvision",
    "pytorch-lightning",
    "pandas",
    "pillow",
    "pytest",
    "h5py",
    "matplotlib",
    "typeguard",
    "torchtyping",
    "imgaug",
    "sklearn",
    "hydra-core",
    "black",
    "fiftyone",
]


setup(
    name="pose-estimation-nets",
    packages=find_packages(),
    version=version,
    description="Convnets for tracking body poses",
    author="Dan Biderman",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/pose-estimation-nets",
    keywords=["machine learning", "deep learning"],
)
