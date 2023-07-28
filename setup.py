
import re
import subprocess

from setuptools import find_packages, setup

VERSION = "0.0.4"

# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()


def get_cuda_version():
    nvcc_paths = ["nvcc", "/usr/local/cuda/bin/nvcc"]
    for nvcc in nvcc_paths:
        try:
            output = subprocess.check_output([nvcc, "--version"]).decode()
            match = re.search(r"release (\d+\.\d+)", output)
            if match:
                return float(match.group(1))
        except FileNotFoundError:
            continue
    print("nvcc is not installed.")
    return None


cuda_version = get_cuda_version()


if cuda_version is not None:
    if 11.0 <= cuda_version < 12.0:
        dali = "nvidia-dali-cuda110"
    elif 12.0 <= cuda_version < 13.0:
        dali = "nvidia-dali-cuda120"
    else:
        dali = "nvidia-dali-cuda110"
        print("WARNING! Unsupported CUDA version. Some training/inference features will not work.")
else:
    dali = "nvidia-dali-cuda110"
    print("WARNING! CUDA not found. Some training/inference features will not work.")

print(f"Found CUDA version: {cuda_version}, using DALI: {dali}")


# basic requirements
install_requires = [
    "fiftyone",
    "h5py",
    "hydra-core",
    "imgaug",
    "kaleido",  # export plotly figures as static images
    "kornia",
    "lightning",
    "matplotlib",
    "moviepy",
    "opencv-python",
    "pandas>=2.0.0",
    "pillow",
    "plotly",
    "pytest",
    "scikit-learn",
    "seaborn",
    "streamlit",
    "tensorboard",
    "torchtyping",
    "torchvision",
    "typeguard",
    "typing",
    dali,
]

# additional requirements
extras_require = {
    "dev": {
        "black",
        "flake8",
        "isort",
    },
    "extra_models": {
        "lightning-bolts",  # resnet-50 trained on imagenet using simclr
    },
}


setup(
    name="lightning-pose",
    packages=find_packages(),
    version=VERSION,
    description="Semi-supervised pose estimation using pytorch lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dan Biderman and Matt Whiteway",
    install_requires=install_requires,
    extras_require=extras_require,
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/lightning-pose",
    keywords=["machine learning", "deep learning", "computer_vision"],
)
