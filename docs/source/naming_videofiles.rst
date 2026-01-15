.. _naming_videofiles:

Naming your video files
------------------------

Lightning Pose expects Session and view names to be embedded in the filename as follows:

.. code-block:
    
    SESSION_VIEW.mp4

View is present only for multiview projects. For singleview projects, it's just ``SESSION.mp4``.

Session and View are identifiers that are used throughout a Lightning Pose project directory.
Once that happens, they are hard to change. So it's important to name 
them thoughtfully.

Tips: Plan for change
~~~~~~~~~~~~~~~~~~~~~~

A simple and effective starting point for a naming convention is something like this:

.. code-block::

    SUBJECT_DATE_TIME_VIEW.mp4

It's worth thinking about how your project might evolve over time.
For example, if you anticipate cropping or clipping, you might
say to yourself that you'll embed more metadata like:

.. code-block::

    SUBJECT_DATE_TIME_CLIP_T1-T2_VIEW.mp4

If you have multiple methodologies for cropping, you might further extend
the naming convention like so:

.. code-block::

    SUBJECT_DATE_TIME_CLIP_T1-T2_CROP_cropper_model_1_VIEW.mp4

The key is to think about what metadata you might want to incorporate later,
and that your naming strategy is extensible to allow that.

However, as you add metadata, remember that VIEW must always be the
last token in the filename, and VIEW should not contain underscores.

Use valid characters
~~~~~~~~~~~~~~~~~~~~~~
Stick with filesystem and URL-safe characters: alphanumeric, -, _.

Stick with your naming scheme throughout the app
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Whether you upload videos in the Labeler for extracting frames for labeling,
or running model inference on a longer video, both
of these videos will show up in the Viewer next to each other.

Using the same naming scheme across all videos allows
you to identify them easily afterwards in the Viewer,
or anywhere in the Project.
