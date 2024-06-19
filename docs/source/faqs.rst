#############
FAQs
#############

.. dropdown:: Can I import a pose estimation project in another format?

    We currently support conversion from DLC projects into Lightning Pose projects
    (if you would like support for another format, please
    `open an issue <https://github.com/danbider/lightning-pose/issues>`_).
    You can find more details in the :ref:`Organizing your data <directory_structure>` section.

.. _faq_video_formats:

.. dropdown:: What video formats are supported by Lightning Pose?

    Lightning Pose requires videos that use the h.264 codec.
    AVI files do not use the h.264 codec, but MP4 files typically do (though not always).
    The following function will check for the proper codec using ``ffmpeg``:

    .. code-block:: python

        import subprocess

        def check_codec_format(input_file: str) -> bool:
            """Run FFprobe command to get video codec and pixel format."""

            ffmpeg_cmd = f'ffmpeg -i {input_file}'
            output_str = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True, text=True)
            # stderr still has codec info
            output_str = output_str.stderr

            # search for correct codec (h264) and pixel format (yuv420p)
            if output_str.find('h264') != -1 and output_str.find('yuv420p') != -1:
                is_codec = True
            else:
                is_codec = False
            return is_codec

    If your videos do not use the h.264 codec the following python code will convert them:

    .. code-block:: python

        import os
        import subprocess

        def reencode_video(input_file: str, output_file: str) -> None:
            """Reencodes video into h.264 coded format using ffmpeg from a subprocess.

            Args:
                input_file: abspath to existing video
                output_file: abspath to to new mp4 video using h.264 codec

            """
            # check input file exists
            assert os.path.isfile(input_file), 'input video does not exist.'
            # check directory for saving outputs exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # create ffmpeg command
            ffmpeg_cmd = f'ffmpeg -i {input_file} -c:v libx264 -pix_fmt yuv420p -c:a copy -y {output_file}'
            # run command
            subprocess.run(ffmpeg_cmd, shell=True)

    Note that you can also run the ``ffmpeg`` command directly from the command line.


.. dropdown:: What if I encounter a CUDA out of memory error?

    We recommend a GPU with at least 8GB of memory.
    Note that both semi-supervised and context models will increase memory usage
    (with semi-supervised context models needing the most memory).
    If you encounter this error, reduce batch sizes during training or inference.
    You can find the relevant parameters to adjust in :ref:`The configuration file <config_file>`
    section.

.. dropdown:: Why does the network produce high confidence values for keypoints even when they are occluded?

    Generally, when a keypoint is briefly occluded and its location can be resolved by the network,
    we are fine with high confidence values (this will happen, for example, when using temporal
    context frames).
    However, there may be scenarios where the goal is to explicitly track whether a keypoint is
    visible or hidden using confidence values (e.g., quantifying whether a tongue is in or out of
    the mouth).
    In this case, if the confidence values are too high during occlusions, try the suggestions
    below.

    First, note that including a keypoint in the unsupervised losses - especially the PCA losses -
    will generally increase confidence values even during occlusions (by design).
    If a low confidence value is desired during occlusions, ensure the keypoint in question is not
    included in those losses.

    If this does not fix the issue, another option is to set the following field in the config file:
    ``training.uniform_heatmaps_for_nan_keypoints: true``.
    [This field is not visible in the default config but can be added.]
    This option will force the model to output a uniform heatmap for any keypoint that does not
    have a ground truth label in the training data.
    The model will therefore not try to guess where the occluded keypoint is located.
    This approach requires a set of training frames that include both visible and occluded examples
    of the keypoint in question.
