##########################
Importing labeled data
##########################

You can import labeled data from other software like DLC and Sleap
using scripts in the Lightning Pose git repo.

DLC
====

Run the following script from the git repo:

.. code-block:: console

    python scripts/converters/dlc2lp.py --dlc_dir=/path/to/dlc_dir --lp_dir=/path/to/lp_dir


SLEAP
======

First, export your SLEAP project as a .pkg.slp file (Predict -> Export Labels Package in the SLEAP gui).
Then run the following script from the git repo:

.. code-block:: console

    python scripts/converters/slp2lp.py --slp_file=/path/to/<project>.pkg.slp --lp_dir=/path/to/lp/dir

Request support for more converters
====================================

If you have labeled data from other pose estimation packages (like DPK) and
would like to try out Lightning Pose, please
`raise an issue <https://github.com/paninski-lab/lightning-pose/issues>`_.
