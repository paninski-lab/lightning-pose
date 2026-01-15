Migrating to the App
=====================

From the old app
-----------------

These instructions convert a singleview project directory from the old app
to make it compatible with the new app.


1. Copy the old app's project directory out into a new folder to work on:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console
 
    # Copies project directory from old path to new location.
    cp -r ~/Pose-app/data/PROJECT_NAME  ~/LPProjects/PROJECT_NAME

2. Fix data directory structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The old app directory structure is as follows:

.. code-block::

    PROJ_DIR/
    ├── labeled-data/
    |   └── session0/
    |       └── selected_frames.csv
    ├── videos/
    ├── videos_infer/
    ├── CollectedData.csv
    ├── label_studio_config.xml
    ├── label_studio_metadata.yaml
    ├── label_studio_tasks.pkl
    ├── model_config_<PROJ_NAME>.yaml
    └── models/
        └── YYYY-MM-DD/
            └── HH-MM-SS_model_name/
                └── video_preds_infer/

The task is to make the directory structure conform to the specification in :doc:`directory_structure_reference/singleview_structure`.

The following changes are required:

1. Create a ``project.yaml`` file in the project data directory per the :doc:`docs <directory_structure_reference/project_yaml_file_format>`. 
2. Add the project to ``~/.lightning-pose/projects.toml``, following the example in :ref:`project_directories`.
3. Copy all videos from the ``videos_infer`` to ``videos`` directory. (Required to see these in the viewer.)
4. Rename the video prediction directory in the model directories from ``video_preds_infer`` to ``video_preds``.

The following are recommended, but not strictly required:

5. Remove ``label_studio`` files.  If you had unlabeled frames in the labeling queue,
   these will be lost and need to be re-extracted. Alternatively, manually migrate these using the
   to unlabeled sidecar format.
6. Remove ``model_config_<PROJ_NAME>.yaml`` file, as its no longer used.
7. Use ffmpeg to re-encode videos such that every frame is an Intra frame. This is required
   in order for the app viewer to be 100% frame-accurate, but not strictly required otherwise.

That's it. Next time you run the app and you should see your new singleview project.

From the CLI
----------------------------------------

The task is to make the directory structure conform to the specification in :doc:`directory_structure_reference/singleview_structure`
or :doc:`directory_structure_reference/multiview_structure`, depending on your project type.

The app `should` work with just the following:

1. Create a ``project.yaml`` file in the project data directory per the :doc:`docs <directory_structure_reference/project_yaml_file_format>`. 
2. Add the project to ``~/.lightning-pose/projects.toml``, following the example in :ref:`project_directories`.

You should consider:

- Moving models into data directory so it's one directory, rather than having models and data be separate.
- If you do this, you can remove the ``model_dir`` attribute from the projects.toml file.

Verify:

- Label files are stored as expected
- Extracted frames are stored as specified
- Multiview: verify calibration file naming

