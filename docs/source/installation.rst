############
Installation
############

Lightning Pose can be installed as a Python package on Linux or Windows (using WSL, see below). Access to a GPU is required for the
NVIDIA DALI dependency. You must have **CUDA 11.0-12.x** installed.
If you have a Mac OS and would like to install Lightning Pose, please get in contact by
`raising an issue <https://github.com/danbider/lightning-pose/issues>`_.

There are several installation methods:

* :ref:`pip package <pip_package>` provides the basic Lightning Pose package. This option is intended for non-interactive environments, such as remote servers.

* :ref:`conda from source <conda_from_source>` additionally provides example data, a boilerplate training script, and diagnostic visualization scripts. This option is the recommended one for new users, as it provides the full breadth of Lightning Pose capabilities.

* :ref:`Lightning Studio <lightning_studio>` is a cloud-based environment that comes with Lightning Pose already installed. Requires creating a Lightning.ai account.

Optionally, instructions are provided for :ref:`Docker users <docker_users>`.

If you are a **Windows user**, please first read :ref:`Windows Installation with WSL <windows_users>`.

.. _install_ffmpeg:

**Install ffmpeg**

First, check to see if you have ``ffmpeg`` installed by typing the following in the terminal:

.. code-block:: console

    ffmpeg -version

If not, install:

.. code-block:: console

    sudo apt install ffmpeg

**Set up a conda environment**

For both installation methods we recommend using
`conda <https://docs.anaconda.com/free/anaconda/install/index.html>`_
to create a new environment in which this package and its dependencies will be installed:

.. code-block:: console

    conda create --name <YOUR_ENVIRONMENT_NAME> python=3.10

Activate the new environment:

.. code-block:: console

    conda activate <YOUR_ENVIRONMENT_NAME>

Make sure you are in the activated environment during the Lightning Pose installation.

.. _pip_package:

Method 1: pip package
=====================

#. ``pip install`` inside the activated conda environment:

   .. code-block:: console

       pip install lightning-pose

#. Check for successful installation by importing:

   .. code-block:: console

       python -c "import lightning_pose"

   You should not see any error messages.

.. _conda_from_source:

Method 2: conda from source
===========================

#. First, ensure git is installed:

   .. code-block:: console

       git --version

   If 'git' is not recognized, `install git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

#. Move into the directory where you want to place the repository folder, and then download it from GitHub:

   .. code-block:: console

       cd <SOME_FOLDER>
       git clone https://github.com/danbider/lightning-pose.git

#. Then move into the newly-created repository folder:

   .. code-block:: console

       cd lightning-pose

   and install dependencies using one of the lines below that suits your needs best:

   * ``pip install -e .``: basic installation, covers most use-cases (note the period!)
   * ``pip install -e ".[dev]"``: basic install + dev tools
   * ``pip install -e ".[extra_models]"``: basic install + tools for loading resnet-50 simclr weights
   * ``pip install -e ".[dev,extra_models]"``: install all available requirements

   This installation might take between 3-10 minutes, depending on your machine and internet connection.

   If you are using Ubuntu 22.04 or newer, you'll need an additional update for the Fiftyone package:

   .. code-block:: console

       pip install fiftyone-db-ubuntu2204

#. Verify that all the unit tests are passing on your machine by running

   .. code-block:: console

       pytest

   This will take several minutes.

.. _lightning_studio:

Method 3: Lightning Studio
==========================

Follow
`this link <https://lightning.ai/themattinthehatt/studios/lightning-pose?section=all>`_
to the Lightning Pose Studio.
When you click the **Get** button you will be taken to a Lightning Studio environment with access to a command line interface, VSCode IDE, Jupyter IDE, and more.
The ``lightning-pose`` package and all dependencies are already installed.

You will be required to create a Lightning account if you have not already signed up.

.. _docker_users:

Docker users
============

Use the appropriate Dockerfiles in the root directory to build a Docker image:

.. code-block:: console

    docker build -f Dockerfile.cuda11 -t my-image:cuda11 .


.. code-block:: console

    docker build -f Dockerfile.cuda12 -t my-image:cuda12 .

Run code inside a container (following `this tutorial <https://docs.docker.com/get-started/>`_):

.. code-block:: console

    docker run -it --rm --gpus all my-image:cuda11


.. code-block:: console

    docker run -it --rm --gpus all --shm-size 256m my-image:cuda12

For a g4dn.xlarge AWS EC2 instance adding the flag ``--shm-size=256m`` will provide the necessary
memory to execute.
The ``--gpus all`` flag is necessary to allow Docker to access the required drivers for NVIDIA DALI to work properly.


.. _windows_users:

Windows Installation with WSL
===============================

* Windows Subsystem for Linux (WSL) is a Windows feature that enables users to run native Linux applications, containers, and command-line tools directly on Windows 10<. 
* WSL support for GPU allows for these applications to benefit from GPU accelerated computing which is vital for the training of machine learning models like lightning-pose.

*******************************
Preparing for WSL installation
*******************************

1. Before installing WSL, ensure you have an appropriate NVIDIA driver for GPU support. This is the only driver you need to install. If you already utilise an NVIDIA GPU within your Windows system then you do not need to install an additional driver; WSL will automatically detect your existing installation. **do NOT install any Linux display driver within WSL.**

`For further details on NVIDIA driver support and installation see here. <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_  

*******************************
Installing WSL
*******************************

This installation is for WSL2. for the latest on WSL updates, see `here. <https://learn.microsoft.com/en-us/windows/wsl/about>`_

2. Within a Windows PowerShell terminal (with admin privileges) run: 

.. code-block:: console

    wsl.exe --install

3. Ensure you have the latest WSL kernel by running: 

.. code-block:: console

    wsl.exe --update

4. Restart your computer. This is necessary for WSL Ubuntu to take full effect. 

5. Within the Windows terminal, open a Ubuntu terminal. A console will open and you will be asked to wait for files to de-compress and be stored on your machine. All future launches should take less than a second. 

6. Once installed, within the Ubuntu terminal, you will be prompted to create a user account and password for your newly installed Linux distribution. *Note: when typing your password it will be invisible but it will be there!*

**For more information on WSL see:**
`Nvidia guidelines <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`_ , `WSL installation guide by Microsoft <https://learn.microsoft.com/en-us/windows/wsl/install>`_ , `Useful video tutorial <https://youtu.be/_fntjriRe48?si=vie0HJEjzjMucwmq>`_

*******************************
Optimising your WSL setup
*******************************

7. It is recommended that you follow the `Best practices for setting up a WSL development environment <https://learn.microsoft.com/en-us/windows/wsl/setup/environment>`_.

8. Specifically, we advise setting up Windows terminal, Visual Studio code, and management with Git. 

*********************************
Setting up Lightning-pose in WSL
*********************************

9. Activate WSL within a Windows PowerShell terminal by running:

.. code-block:: console

    wsl

10. :ref:`Install ffmpeg<install_ffmpeg>`

11. follow the steps in `Method 2: conda from source`_.

12. Remain inside your Lightning Pose environment and install the following packages:

.. code-block:: console

    sudo apt install python-is-python3 
    sudo apt install libgl1-mesa-glx

*that's it!* - Lightning Pose should now function within WSL. 


Getting help
------------

If you encounter any issues during installation, first check out the
`GitHub Issues <https://github.com/danbider/lightning-pose/issues>`_
page to see if others have had the same problem.

If you do not find a similar issue, please raise an issue or reach out on
`Discord <https://discord.gg/tDUPdRj4BM>`_
to get help from the community.
