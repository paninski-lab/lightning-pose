#:!/usr/bin/env python
from setuptools import find_packages, setup

version = None

install_requires = [
    "black",
    "fiftyone",
    "h5py",
    "hydra-core",
    "imgaug",
    "kornia",
    "matplotlib",
    "moviepy",
    "pandas",
    "pillow",
    "pytest",
    "pytorch-lightning",
    "lightning-bolts",
    "scikit-image",
    "sklearn",
    "torchtyping",
    "torchvision",
    "typeguard",
]


setup(
    name="lightning-pose",
    packages=find_packages(),
    version=version,
    description="Convnets for tracking body poses",
    author="Dan Biderman",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/lightning-pose",
    keywords=["machine learning", "deep learning"],
)
