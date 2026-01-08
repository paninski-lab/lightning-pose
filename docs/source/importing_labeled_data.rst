##########################
Importing labeled data
##########################

Converting DLC projects to Lightning Pose format
================================================

Once you have installed Lightning Pose, you can convert previous DLC projects into the proper
Lightning Pose format by running the following script from the git repo:

.. code-block:: console

    python scripts/converters/dlc2lp.py --dlc_dir=/path/to/dlc_dir --lp_dir=/path/to/lp_dir

That's it!
After this you will need to update your config file with the correct paths (see next page).

Converting SLEAP projects to Lightning Pose format
==================================================

First, export your SLEAP project as a .pkg.slp file (Predict -> Export Labels Package in the SLEAP gui).
Then, once you have installed Lightning Pose, you can convert previous SLEAP projects into the proper
Lightning Pose format by running the following script from the git repo:

.. code-block:: console

    python scripts/converters/slp2lp.py --slp_file=/path/to/<project>.pkg.slp --lp_dir=/path/to/lp/dir

That's it!
After this you will need to update your config file with the correct paths (see next page).

Converting other projects to Lightning Pose format
==================================================

If you have labeled data from other pose estimation packages (like DPK) and
would like to try out Lightning Pose, please
`raise an issue <https://github.com/paninski-lab/lightning-pose/issues>`_.
