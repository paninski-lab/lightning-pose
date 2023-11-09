.. lightning-pose documentation master file, created by
   sphinx-quickstart on Thu Nov  9 13:15:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lightning Pose's documentation!
==========================================
Lightning Pose models are implemented in Pytorch Lightning,
supporting massively accelerated training on unlabeled videos using NVIDIA DALI.
The whole process is orchestrated by Hydra.
Models can be evaluated with TensorBoard, FiftyOne, and Streamlit.

Please see the `preprint <https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1>`_
for additional details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation
   source/user_guide
   source/developer_guide


Lightning Pose API
==================
.. toctree::
   :maxdepth: 1

   modules/lightning_pose.data
   modules/lightning_pose.losses
   modules/lightning_pose.models
   modules/lightning_pose.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
