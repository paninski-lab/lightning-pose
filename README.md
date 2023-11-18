![](https://github.com/danbider/lightning-pose/raw/main/assets/images/LightningPose_horizontal_light.png)
Pose estimation models implemented in **Pytorch Lightning**, supporting massively accelerated training on _unlabeled_ videos using **NVIDIA DALI**. 
The whole process is orchestrated by **Hydra**. 
Models can be evaluated with **TensorBoard**, **FiftyOne**, and **Streamlit**.

Preprint: [Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling, and cloud-native open-source tools](https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/danbider/lightning-pose)
[![Documentation Status](https://readthedocs.org/projects/lightning-pose/badge/?version=latest)](https://lightning-pose.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)

## Try our demo!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb)

Train a network on an example dataset and visualize the results in Google Colab.

## Getting Started
Please see the [Lightning Pose documentation](TODO) for installation instructions and user guides.
Note that the Lightning Pose package provides tools for training and evaluating models on 
_already labeled data_ and unlabeled video clips. 

We also offer a [browser-based application](https://github.com/Lightning-Universe/Pose-app) that 
supports the full life cycle of a pose estimation project, from data annotation to model training 
(with Lightning Pose) to diagnostics visualizations.

## Community

Lightning Pose is primarily maintained by 
[Dan Biderman](https://dan-biderman.netlify.app) (Columbia University) 
and 
[Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University). 

Lightning Pose is under active development and we welcome community contributions.
Whether you want to implement some of your own ideas or help out with our [development roadmap](docs/roadmap.md), please get in touch with us on Discord (see contributing guidelines [here](CONTRIBUTING.md)). 
