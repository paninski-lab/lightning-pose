import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup


def read(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


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


# add the README.md file to the long_description
with open("README.md", "r") as fh:
    long_description = fh.read()

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
    "numpy",
    "opencv-python-headless",
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
    # PyPI does not support direct dependencies, so we remove this line before uploading from PyPI
    "segment_anything @ git+https://github.com/facebookresearch/segment-anything.git",
]

# additional requirements
extras_require = {
    "dev": {
        "black",
        "flake8",
        "isort",
        "Sphinx",
        "sphinx_rtd_theme",
        "sphinx-rtd-dark-mode",
        "sphinx-automodapi",
        "sphinx-copybutton",
        "sphinx-design",
    },
    "extra_models": {
        "lightning-bolts",  # resnet-50 trained on imagenet using simclr
    },
}

setup(
    name="lightning-pose",
    packages=find_packages() + ["mirror_mouse_example"],  # include data for wheel packaging
    version=get_version(Path("lightning_pose").joinpath("__init__.py")),
    description="Semi-supervised pose estimation using pytorch lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dan Biderman and Matt Whiteway",
    install_requires=install_requires,
    extras_require=extras_require,
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/lightning-pose",
    keywords=["machine learning", "deep learning", "computer_vision"],
    package_dir={
        "lightning_pose": "lightning_pose",
        "mirror_mouse_example": "data/mirror-mouse-example",  # remap 'data/mirror-mouse-example'
    },
    include_package_data=True,  # required to get the non-.py data files in the wheel
)
