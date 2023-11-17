############
Installation
############


++++++++++++

Lightning Pose can be installed as a Python package on Linux. Access to a GPU is required for the
NVIDIA DALI dependency. You must have **CUDA 11.0-12.x** installed.
If you have a Mac OS and would like to install Lightning Pose, please get in contact by
`raising an issue <https://github.com/danbider/lightning-pose/issues>`_.

There are two installation methods:

* :ref:`pip package <pip_package>` provides the basic Lightning Pose package. This option is intended for non-interactive environments, such as remote servers.

* :ref:`conda from source <conda_from_source>` additionally provides example data, a boilerplate training script, and diagnostic visualization scripts. This option is the recommended one for new users, as it provides the full breadth of Lightning Pose capabilities.

Optionally, instructions are provided for :ref:`Docker users <docker_users>`.

**Set up a conda environment**

For both installation methods we recommend using
`conda <https://docs.anaconda.com/free/anaconda/install/index.html>`_
to create a new environment in which this package and its dependencies will be installed:

.. code-block:: console

    conda create --name <YOUR_ENVIRONMENT_NAME> python=3.8

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


Getting help
------------

If you encounter any issues during installation, first check out the
`GitHub Issues <https://github.com/danbider/lightning-pose/issues>`_
page to see if others have had the same problem.

If you do not find a similar issue, please raise an issue or reach out on
`Discord <https://discord.gg/tDUPdRj4BM>`_
to get help from the community.
