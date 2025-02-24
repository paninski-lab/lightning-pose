![](https://github.com/danbider/lightning-pose/raw/main/docs/images/LightningPose_horizontal_light.png)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/danbider/lightning-pose)
[![Documentation Status](https://readthedocs.org/projects/lightning-pose/badge/?version=latest)](https://lightning-pose.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)
![PyPI Downloads](https://static.pepy.tech/badge/lightning-pose/week)


Pose estimation models implemented in **Pytorch Lightning**, supporting massively accelerated training on _unlabeled_ videos using **NVIDIA DALI**. 
Models can be evaluated with **TensorBoard**, **FiftyOne**, and **Streamlit**.

As of June 2024, Lightning Pose is now [published in Nature Methods](https://rdcu.be/dLP3z)!

## Try our demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danbider/lightning-pose/blob/main/scripts/litpose_training_demo.ipynb)

Train a network on an example dataset and visualize the results in Google Colab.

## Getting Started
Please see the [Lightning Pose documentation](https://lightning-pose.readthedocs.io/) 
for installation instructions and user guides.
Note that the Lightning Pose package provides tools for training and evaluating models on 
_already labeled data_ and unlabeled video clips. 

We also offer a [browser-based application](https://github.com/Lightning-Universe/Pose-app) that 
supports the full life cycle of a pose estimation project, from data annotation to model training 
to diagnostic visualizations.

The Lightning Pose team also actively develops the 
[Ensemble Kalman Smoother (EKS)](https://github.com/paninski-lab/eks), 
a simple and performant post-processor that works with any pose estimation package including 
Lightning Pose, DeepLabCut, and SLEAP.

## Community

Lightning Pose is primarily maintained by 
[Dan Biderman](https://dan-biderman.netlify.app) (Columbia University) 
and 
[Matt Whiteway](https://themattinthehatt.github.io/) (Columbia University). 

Lightning Pose is under active development and we welcome community contributions.
Whether you want to implement some of your own ideas or help out with our [development roadmap](docs/roadmap.md), please get in touch with us on Discord (see contributing guidelines [here](CONTRIBUTING.md)). 
