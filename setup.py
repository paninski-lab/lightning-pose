
import re
import subprocess

from setuptools import find_packages, setup

VERSION = "0.0.2"  # was previously None

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


install_requires = [
    "black==23.3.0",
    "fiftyone==0.20.1",
    "h5py==3.8.0",
    "hydra-core==1.3.2",
    "imgaug==0.4.0",
    "kaleido==0.2.1",
    "kornia==0.6.12",
    "matplotlib==3.7.1",
    "moviepy==1.0.3",
    "opencv-python==4.7.0.72",
    "pandas==2.0.1",
    "pillow==9.5.0",
    "pytest==7.3.1",
    "lightning",
    dali,
    "tensorboard==2.13.0",
    "lightning-bolts==0.6.0.post1",
    "seaborn==0.12.2",
    "scikit-image==0.20.0",
    "scikit-learn==1.2.2",
    "streamlit==1.22.0",
    "torchtyping==0.1.4",
    "torchvision==0.15.2",
    "typeguard==3.0.2",
    "typing==3.7.4.3",
    "botocore==1.27.59",
    "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git",
]


setup(
    name="lightning-pose",
    packages=find_packages(),
    version=VERSION,
    description="Semi-supervised pose estimation using Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dan Biderman and Matt Whiteway",
    install_requires=install_requires,  # load_requirements(PATH_ROOT),
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/lightning-pose",
    keywords=["machine learning", "deep learning", "computer_vision"],
)
