.. lightning-pose documentation master file, created by
   sphinx-quickstart on Thu Nov  9 13:15:31 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lightning Pose's documentation!
==========================================
Lightning Pose is an open source deep-learning package for animal pose estimation
(`preprint <https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1>`_).
The framework is based on Pytorch Lightning and supports accelerated training on unlabeled videos
using NVIDIA DALI. Models can be evaluated with TensorBoard, FiftyOne, and Streamlit.

If you would like to try out Lightning Pose, we provide a Google Colab notebook that steps through
the process of training and evaluating a model on an example dataset - no data labeling or
software installation required!

We also provide a browser-based GUI for labeling data, training and evaluating models, and running
inference on new videos.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation
   source/user_guide/index
   source/user_guide_advanced/index
   source/developer_guide/index
   source/faqs


Developer API
=============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
