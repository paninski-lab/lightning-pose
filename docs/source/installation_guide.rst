===================
Installation
===================

-------------------
System Requirements
-------------------

Before proceeding, ensure your system meets these hardware and software specifications:

* **Operating System**: Linux or Windows Subsystem for Linux (`WSL2 <https://learn.microsoft.com/en-us/windows/wsl/install>`_).
* **Hardware**: An NVIDIA GPU is required for model training and inference.
* **Driver**: NVIDIA Driver supporting CUDA 12.4 or higher.

-----------------------------
Step 1: Install Dependencies
-----------------------------

Lightning Pose requires specific system-level tools to handle video processing and GPU acceleration.

FFmpeg
------

Lightning Pose uses FFmpeg for high-performance video decoding.

.. code-block:: bash

   # Ubuntu/WSL
   sudo apt update && sudo apt install ffmpeg

   # Verify installation
   ffmpeg -version

NVIDIA GPU Setup
----------------

Ensure your GPU drivers are correctly installed and recognized by the system.

.. code-block:: bash

   nvidia-smi

The output should display a table showing your GPU model, Driver Version, and CUDA Version.

--------------------------------
Step 2: Create Conda Environment
--------------------------------

We recommend using **Conda** (or another python environment management tool) to create an isolated python environment.
If you don't already have ``conda`` installed, download and install it from `here <https://conda-forge.org/download/>`_.

.. code-block:: bash

   # Create a new environment named 'lp' with Python 3.12
   conda create -n lp python=3.12

   # Activate the environment
   conda activate lp

.. tip::
   **Check your environment**: Always ensure your terminal prompt is prefixed with ``(lp)`` before running installation or training commands. This confirms you are working within the isolated environment and not your system's global Python.

----------------------------------
Step 3: Install Lightning Pose
----------------------------------

Choose the method that best fits your needs.

Option A: Standard Installation (Standard)
---------------------------------------------

Use this method if you don't know which method to use.

.. code-block:: bash

   pip install lightning-pose lightning-pose-app

Option B: Installation from Source (Development)
-------------------------------------------------

Use this if you plan to modify the source code. This requires cloning the repositories for an **editable** install.

.. code-block:: bash

   # Clone the core repo and install
   git clone https://github.com/paninski-lab/lightning-pose.git
   cd lightning-pose
   pip install -e .
   cd ..

   # Clone the app repo and install
   git clone https://github.com/paninski-lab/lightning-pose-app.git
   cd lightning-pose-app
   pip install -e .

.. note::
   In case of a PyTorch installation issue in either Option A or B, You may need to install the PyTorch matching your CUDA version by following the `PyTorch Installation Guide <https://pytorch.org/get-started/locally/>`_.

----------------------
How to Choose?
----------------------

.. list-table:: Installation Methods
   :widths: 25 25 50
   :header-rows: 1

   * - Goal
     - Method
     - Benefit
   * - Standard Usage
     - **PyPI (Pip)**
     - Simple setup; stable release versions.
   * - Development/Research
     - **Source (Git)**
     - Access to unreleased features; ability to edit code.

You can migrate from one to the other, just ``pip uninstall`` and then re-install using the new method.

----------------------------
Verification
----------------------------

Verify the installation by checking the Command Line Interface (CLI):

.. code-block:: bash

   litpose --help

If successful, you will see a list of available commands for the Lightning Pose suite.
