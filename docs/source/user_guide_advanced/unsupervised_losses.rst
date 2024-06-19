.. _unsupervised_losses:

###################
Unsupervised losses
###################

The application of unsupervised losses to unlabeled video data is a core componenet of the
Lightning Pose algorithm.
This page describes the required video data format, updates you need to make to the config file,
and brief descriptions of some of the available losses.

#. :ref:`Data requirements <unsup_data>`
#. :ref:`The configuration file <unsup_config>`
#. :ref:`Loss options <unsup_loss_options>`
    * :ref:`Temporal difference <unsup_loss_temporal>`
    * :ref:`Pose PCA <unsup_loss_pcasv>`
    * :ref:`Multiview PCA <unsup_loss_pcamv>`

.. _unsup_data:

Data
====
All unlabeled videos must be placed in a single directory.
We recommend including at least a few videos that each have more than 10k frames.
The more diversity of animal behavior in the videos the better.

.. warning::

    The NVIDIA DALI video readers require a specific video codec (h264) and pixel format (yuv420p).

You can check if your videos are in the correct format with the following python function:

.. code-block:: python

    import subprocess

    def check_codec_format(input_file: str) -> bool:
        """Run FFprobe command to get video codec and pixel format."""
        ffmpeg_cmd = f'ffmpeg -i {input_file}'
        output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
        # stderr because the ffmpeg command has no output file, but the stderr still has codec info.
        output_str = output_str.stderr
        # search for correct codec (h264) and pixel format (yuv420p)
        if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
            # print('Video uses H.264 codec')
            is_correct_format = True
        else:
            is_correct_format = False
        return is_correct_format

If you need to re-encode your videos, you can do so with the following python function:

.. code-block:: python

    import os
    import subprocess

    def reencode_video(input_file: str, output_file: str) -> None:
        """reencodes video into H.264 coded format using ffmpeg from a subprocess.

        Args:
            input_file: abspath to existing video
            output_file: abspath to to new video

        """
        # check input file exists
        assert os.path.isfile(input_file), "input video does not exist."
        # check directory for saving outputs exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
        subprocess.run(ffmpeg_cmd, shell=True)


.. _unsup_config:

Config file
===========

.. note::

    Recall that any of the config options can be updated directly from the command line;
    see the :ref:`Training <training>` section.

There are several fields of the config file that must be updated to properly fit a model with
unsupervised losses. First, ``data.video_dir`` should be an **absolute** path that points to the
video directory.

Second, ``model.losses_to_use`` must be non-empty (which indicates a fully supervised model).
You can choose a single loss:

.. code-block:: yaml

    model:
      losses_to_use: [temporal]

or multiple losses:

.. code-block:: yaml

    model:
      losses_to_use: [temporal,pca_singleview,pca_multiview]

In the ``dali`` section of the config file,
the field ``dali.base.train.sequence_length`` defines the unlabeled batch size;
if you encounter out of memory errors try reducing this value.
The field ``dali.base.predict.sequence_length`` defines the batch size during inference;
in general this can be larger than during training because there are no labeled frames or
gradients.

Finally, the ``losses`` section of the config file defines hyperparameters for each of the
individual losses, which are addressed below.

.. _unsup_loss_options:

Loss options
============
For a detailed mathematical description of the losses, see the
`Lightning Pose paper <https://www.biorxiv.org/content/10.1101/2023.04.28.538703v1>`_.
Each loss contains multiple hyperparameters.
The most important is the ``log_weight``; we have found 5.0-7.0 to be a reasonable range for all
losses across multiple datasets, but we encourage users to test out several values on their own
data for best effect. The inverse of this weight is actually used for the final weight, so smaller
values indicate stronger penalties.

We are particularly interested in preventing, and having the network learn from, severe violations
of the different losses.
Therefore, we enforce our losses only when they exceed a tolerance threshold :math:`\epsilon`,
rendering them :math:`\epsilon`-insensitive:

.. math::

    \mathscr{L}(\epsilon) = \textrm{max}(0, \mathscr{L} - \epsilon).

.. _unsup_loss_temporal:

Temporal difference
-------------------
This loss penalizes the difference in predictions between successive timepoints for each keypoint
independently.

.. code-block:: yaml

      temporal:
        log_weight: 5.0
        prob_threshold: 0.05
        epsilon: 20.0


* ``log_weight``: weight of the loss in the final cost function
* ``prob_threshold``: predictions with a probability below this threshold are not included in the loss. This is desirable if, for example, a keypoint is occluded and the prediction has low probability.
* ``epsilon``: in pixels; temporal differences below this threshold are not penalized, which keeps natural movements from being penalized. The value of epsilon will depend on the size of the video frames, framerate (how much does the animal move from one frame to the next), the size of the animal in the frame, etc.

The temporal difference between consecutive frames is a dynamic quantity - when an animal moves
quickly, the temporal differences are large; when an animal moves slowly, they are small.
Furthermore, some keypoints might move a lot while others not at all.
For these reasons, this loss is most effective when epsilon is rather large - for example,
the longest distance any keypoint could travel throughout the video recordings.
In this case the temporal loss is less about ensuring smooth trajectories, and more about making
sure there aren't implausibly large jumps from one frame to the next.

.. _unsup_loss_pcasv:

Pose PCA
-----------------
This loss penalizes deviations away from a low-dimensional subspace of plausible poses computed on
labeled data.
It is possible that the labeled data does not contain the full diversity of poses encountered
in the video data, and will erroneously penalize rare poses.
More and diverse labels will mitigate this potential issue.

It is also necessary to label a minimum number of frames to utilize this loss: since each keypoint
is 2-dimensional (x, y coords), if there are `K` keypoints labeled on each frame then each pose is
described by a `2K`-dimensional vector. Therefore, at least `2K` frames need to be labeled to
compute the PCA subspace.

It is up to the user to select which keypoints are included in the Pose plausibility loss.
Including static keypoints (e.g. those marking a corner of an arena) are generally not helpful.
Also be careful to not include keypoints that are often occluded, like the tongue.
If these keypoints are included the loss will try to localize them even when they are occluded,
which might be unhelpful if you want to use the confidence of the outputs as a lick detector.

Select the keypoints used for this loss with the config field ``data.columns_for_singleview_pca``.
The numbers used should correspond to the order of the keypoints in the labeled csv file.
For example, if the keypoints in the csv file have the order

0. nose
1. L_ear
2. R_ear
3. neck
4. tailbase

and you want to include the nose and ears, the config file will look like

.. code-block:: yaml

    data:
      columns_for_singleview_pca: [0, 1, 2]

If instead you want to include the ears and tailbase:

.. code-block:: yaml

    data:
      columns_for_singleview_pca: [1, 2, 4]

See
`these config files <https://github.com/danbider/lightning-pose/tree/main/scripts/configs>`_
for more examples.

Below are the various hyperparameters and their descriptions.
Besides the ``log_weight`` none of the provided values need to be tested for new datasets.

.. code-block:: yaml

      pca_singleview:
        log_weight: 5.0
        components_to_keep: 0.99
        epsilon: null

* ``log_weight``: weight of the loss in the final cost function
* ``components_to_keep``: predictions should lie within the low-d subspace spanned by components that describe this fraction of variance
* ``epsilon``: if not null, this parameter is automatically computed from the labeled data

.. _unsup_loss_pcamv:

Multiview PCA
---------------------
This loss penalizes deviations of predictions across all available views away from a 3-dimensional
subspace computed on labeled data.

.. warning::

    This loss will not work in the presence of large distortions, for example from fish-eye lenses.

Selecting the keypoints for this loss depends on the data format; here we will assume all views
are fused into a single frame at each time point, for both labeled data and videos.
This can trivially be achieved, for example, when using a mirror to capture different angles with
a single camera
(see the :ref:`Multiview: mirrored or fused frames <multiview_fused>` section for more details).

During labeling each keypoint of the fused data is treated independently, with no explicit
information on which keypoints correspond to the same body part
(see the `example mirror-mouse data <https://github.com/danbider/lightning-pose/tree/main/data/mirror-mouse-example>`_).
We need to record this information for the multiview loss.

Select the keypoints used for this loss with the config field ``data.mirrored_column_matches``,
which will be a list of arrays.
The length of the list corresponds to the number of views.
The length of each array should be the same; the nth element of each array should all correspond
to the same body part.

For example, let's say we have two views (side and bottom) and four keypoints per view.
The full list of keypoints (the order they appear in the labeled data file) is

0. nose_side
1. paw1_side
2. paw2_side
3. tailbase_side
4. nose_bottom
5. paw1_bottom
6. paw2_bottom
7. tailbase_bottom

To include the nose and paws in the multiview consistency loss, the config file will look like

.. code-block:: yaml

    data:
      mirrored_column_matches:
        - [0, 1, 2]  # nose + paws in side view
        - [4, 5, 6]  # nose + paws in bottom view

If instead you want to include the nose and tailbase:

.. code-block:: yaml

    data:
      mirrored_column_matches:
        - [0, 3]  # nose + tailbase in side view
        - [4, 7]  # nose + tailbase in bottom view

Below are the various hyperparameters and their descriptions.
Besides the ``log_weight`` none of the provided values need to be tested for new datasets.

.. code-block:: yaml

      pca_multiview:
        log_weight: 5.0
        components_to_keep: 3
        epsilon: null

* ``log_weight``: weight of the loss in the final cost function
* ``components_to_keep``: should be set to 3 so that predictions lie within a 3D subspace
* ``epsilon``: if not null, this parameter is automatically computed from the labeled data
