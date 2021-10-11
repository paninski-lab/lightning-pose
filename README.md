# pose-estimation-nets
Scalable pose estimation based on Pytorch-Lightning, with support for massive unlabeled datasets using `DALI`.
## Hardware
We assume that you are running on a machine that has at least one GPU and CUDA 11 installed.
## Installation
First create a `conda` environment in which `pose-estimation-nets` and its dependencies will be installed. 
create:

`conda create --name pose-estimation-nets`

activate it:

`conda activate pose-estimation-nets`

now, move into the folder where you want the repository installed:

`cd <SOME FOLDER>`

and within that folder, clone the repository:

`git clone https://github.com/danbider/pose-estimation-nets.git`

then move into the package directory:

`cd pose-estimation-nets`

and install our package and its dependencies like so

`pip install -r requirements.txt`

You should be ready to go! you can verify that all the tests are passing on your machine by running
`pytest`


