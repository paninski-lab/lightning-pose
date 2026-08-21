.. _importing_labeled_data:

##########################
Importing labeled data
##########################

You can import labeled data from other software like DLC and SLEAP
using the :ref:`litpose convert <cli-convert>` CLI command.

Importing from DLC
====================

Run the following:

.. code-block:: console

    litpose convert /path/to/dlc_dir --lp_dir=/path/to/lp_dir


Importing from SLEAP
======================

.. note::

    This only works with single-view, single-animal SLEAP projects.
    A multi-view, single-animal SLEAP is in the works.

First, export your SLEAP project as a .pkg.slp file (Predict -> Export Labels Package in the SLEAP gui).
Then run the following:

.. code-block:: console

    litpose convert /path/to/<project>.pkg.slp --lp_dir=/path/to/lp_dir

Importing from the legacy Lightning Pose-app
==============================================

See the :doc:`app migration guide <migrating_to_app>`.

Request support for more converters
====================================

If you have labeled data from other pose estimation packages (like DPK) and
would like to try out Lightning Pose, please
`raise an issue <https://github.com/paninski-lab/lightning-pose/issues>`_.
