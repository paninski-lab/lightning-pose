.. _inference:

#########
Inference
#########

The command ``litpose predict`` is used to run model inference on new data.

.. _inference-on-new-videos:

Inference on new videos
=======================

The model_dir argument is the path to the model outputted by ``litpose train``.

To predict on one or more video files:

.. code-block:: shell

    litpose predict <model_dir> <video_file1> <video_file2> ...

To predict on a folder of video files:

.. code-block:: shell

    litpose predict <model_dir> <video_files_dir>

The ``litpose predict`` command saves frame-by-frame predictions and confidences as a CSV file,
unsupervised losses in CSV file per loss type. By default it also generates videos annotated with 
predictions, a feature which can be disabled using the ``--skip_viz`` flag.

For the full list of options, see the CLI reference: :ref:`Predict <cli-predict>`. 

.. note::

  Videos *must* be mp4 files that use the h.264 codec; see more information in the
  :ref:`FAQs<faq_video_formats>`.


.. _inference-on-new-images:

Inference on new images
=======================

Lightning pose also supports inference on images, as well 
as computing pixel error against new labeled images. This is useful
for evaluating a model on out-of-distribution data to see how well the
model generalizes.

Currently it's required to create a CSV file similar to
the one used for training labeled frames. Once you have a CSV file,
run: 

.. code-block:: shell

    litpose predict <model_dir> <csv_file>

Output location
===============

Video predictions are saved to:

.. code-block::

    <model_dir>/
    └── video_preds/
        ├── <video_filename>.csv              (predictions)
        ├── <video_filename>_<metric>.csv     (losses)
        └── labeled_videos/
            └── <video_filename>_labeled.mp4

Image predictions are saved to:

.. code-block::

    <model_dir>/
    └── image_preds/
        └── <image_dirname | csv_filename | timestamp>/
            ├── predictions.csv
            ├── predictions_<metric>.csv      (losses)
            └── <image_filename>_labeled.png

Inference on sample dataset
===========================

The lightning pose repo includes a sample dataset (see :ref:`training-on-sample-dataset`).
The sample video file is located in the git repo at ``data/mirror-mouse-example/videos``.
Thus, to run inference on a model trained on the sample dataset,
run from the ``lightning-pose`` directory
(make sure you have activated your conda environment):

.. code-block:: shell

    litpose predict <model_dir> data/mirror-mouse-example/videos