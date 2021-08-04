# pose-estimation-nets
A Pytorch and Pytorch Lightning implementation of convnets for pose estimation.
## Installation
create conda environment (call it however you want)
`conda create --name pose-estimation-nets`
then activate the environment
`conda activate pose-estimation-nets`
now, move into the folder where you want the repository installed, and clone the repo as follows:
`cd <SOME FOLDER>` and when there, run
`git clone https://github.com/danbider/pose-estimation-nets.git`
then `cd` into the new directory called `pose-estimation-nets` like so
`cd pose-estimation-nets`
and install our package and its dependencies like so
`pip install -e .`
If you use CUDA 11.0, please install `nvidia-dali` as follows (for other CUDA versions, check out https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html):

```pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```

then you can verify that all the tests are passing on your machine by running
`pytest`


