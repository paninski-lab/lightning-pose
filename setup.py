#:!/usr/bin/env python
from setuptools import find_packages, setup

VERSION = "0.0.1"  # was previously None

install_requires = [
    "black",
    "fiftyone",
    "h5py",
    "hydra-core",
    "imgaug",
    "kaleido",
    "kornia",
    "matplotlib",
    "moviepy",
    "opencv-python",
    "pandas",
    "pillow",
    "pytest",
    "lightning",
    "tensorboard",
    "lightning-bolts",
    "seaborn",
    "scikit-image",
    "scikit-learn",
    "streamlit",
    "torchtyping",
    "torchvision",
    "typeguard",
    "typing",
]


setup(
    name="lightning-pose",
    packages=find_packages(),
    version=VERSION,
    description="Semi-supervised pose estimation using Lightning.",
    author="Dan Biderman",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/lightning-pose",
    keywords=["machine learning", "deep learning", "computer_vision"],
)
