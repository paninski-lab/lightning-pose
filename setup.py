#:!/usr/bin/env python
## TODO: copied this file from another repo, verify that it's fine
import os
from setuptools import find_packages, setup

version = None
PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name='pose-estimation-nets',
    packages=find_packages(),
    version=version,
    description='Convnets for tracking body poses',
    author='Dan Biderman',
    install_requires=load_requirements(PATH_ROOT),
    author_email='danbider@gmail.com',
    url='https://github.com/danbider/pose-estimation-nets',
    keywords=[ 
        'machine learning',
        'deep learning' 
    ]
)
