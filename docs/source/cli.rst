Lightning Pose command-line interface (CLI)
==========================================

This page documents the primary ``litpose`` CLI commands using sphinx-argparse.

.. _cli-train:

Train
-----

.. argparse::
   :module: lightning_pose.cli.commands.train
   :func: get_parser
   :prog: litpose train

.. _cli-predict:

Predict
-------

.. argparse::
   :module: lightning_pose.cli.commands.predict
   :func: get_parser
   :prog: litpose predict

.. _cli-crop:

Crop
----

.. argparse::
   :module: lightning_pose.cli.commands.crop
   :func: get_parser
   :prog: litpose crop

.. _cli-remap:

Remap
-----

.. argparse::
   :module: lightning_pose.cli.commands.remap
   :func: get_parser
   :prog: litpose remap
