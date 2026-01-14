![](https://github.com/paninski-lab/lightning-pose/raw/main/docs/images/LightningPose_horizontal_light.png)

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/paninski-lab/lightning-pose)
[![Documentation Status](https://readthedocs.org/projects/lightning-pose/badge/?version=latest)](https://lightning-pose.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/lightning-pose)
![PyPI Downloads](https://static.pepy.tech/badge/lightning-pose/week)


Lightning Pose is an end-to-end toolkit designed for robust multi-view and single-view animal
pose estimation using advanced transformer architectures. It leverages Multi-View Transformers
and patch-masking training to learn geometric relationships between views,
resulting in strong performance on occlusions [Aharon, Lee et al. 2025](https://arxiv.org/abs/2510.09903).
For single-view datasets  it leverages temporal context and learned plausibility constraints for strong performance
in challenging scenarios [Biderman, Whiteway et al. 2024, Nature Methods](https://rdcu.be/dLP3z).
It has a rich GUI that supports the end-to-end workflow: labeling, model management, and evaluation.


## Installation

Lightning-pose requires a Linux or WSL environment with an NVIDIA GPU.

For users without access to a local NVIDIA GPU, it is highly recommended to
use the [Lightning AI](https://lightning.ai/) cloud environment, which provides
persistent, browser-based "Studios" with on-demand access to powerful GPUs
and pre-configured CUDA environments.

Install dependencies:

```shell
sudo apt install ffmpeg

# Verify nvidia-driver with CUDA 12+
nvidia-smi
```

In a clean python virtual environment (conda or other virtual environment manager), run:

```shell
pip install lightning-pose lightning-pose-app
```

That's it! To run the app:

```shell
litpose run_app
```

Please see the [installation guide](https://lightning-pose.readthedocs.io/en/latest/source/installation_guide.html)
for more detailed instructions, and feel free to reach out to us on [Discord](https://discord.gg/tDUPdRj4BM)
in case of any hiccups.

## Getting Started

To get started with Lightning Pose, follow the guides on our documentation:
* [Create your first project](https://lightning-pose.readthedocs.io/en/latest/source/create_first_project.html) using the app
* or follow the CLI User Guides ([Singleview](https://lightning-pose.readthedocs.io/en/latest/source/user_guide_singleview/index.html), [Multiview](https://lightning-pose.readthedocs.io/en/latest/source/user_guide_multiview/index.html)).

## Community

The Lightning Pose team also actively develops the 
[Ensemble Kalman Smoother (EKS)](https://github.com/paninski-lab/eks), 
a simple and performant post-processor that works with any pose estimation package including 
Lightning Pose, DeepLabCut, and SLEAP.

Lightning Pose is primarily maintained by 
[Karan Sikka](https://github.com/ksikka) (Columbia University),
[Matt Whiteway](https://themattinthehatt.github.io) (Columbia University),
and
[Dan Biderman](https://dan-biderman.netlify.app) (Stanford University). 
 
Lightning Pose is under active development and we welcome community contributions.
Whether you want to implement some of your own ideas or help out with our [development roadmap](docs/roadmap.md), please get in touch with us on Discord (see contributing guidelines [here](CONTRIBUTING.md)). 
