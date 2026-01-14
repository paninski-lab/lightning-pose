.. _migrating_to_app

Migrating to the App
=====================

In v2.0.5 we launched a new App to replace the
old `Pose app <https://pose-app.readthedocs.io/en/latest/>`_.
It adds Multiview support and better usability.

The directory structure changes are outlined in this <document>.
The latest structure is fully documented in <reference>.

From the old app
-----------------

The old app supported singleview projects only, so these instructions
will create an app-compatible singleview project.

1. Copy the old app's project directory out app ~/Pose-app/data/<PROJ_NAME>
to ~/LPProjects/<PROJ_NAME>

2. Fix data directory structure

- copy videos_infer -> videos
- remove unused label_ pkl files, selected_frames files. If you had unlabeled frames in the labeling queue,
these will be lost and need to be re-extracted. Alternatively, manually migrate these using the
to unlabeled sidecar format.
- rm unused model_config_*

Unused files
~~~~~~~~~~~~~

The following files from the old app are unused and can be deleted:

- label studio pkl files
- selected_frames.csv
- model_config.yaml



3. Add project.yaml file and projects.toml file.

See the section dedicated to this task.

4. Fix model directory structure

video_preds_infer -> video_preds

5. Process videos

run: ffmpeg...

This will make them frame-accurate in the viewer, at the cost of taking up more space.
If you absolutely can't afford to use more space, skip this step with the caveat that the
viewer might not be frame-accurate.

5. Run the app.

That's it. Run the app and you'll see your new singleview project.

Changes from CLI-only usage to the App
----------------------------------------

The app is designed to work with just the following:
- Add project.yaml file to data directory
- Update projects.toml file

But for maximum compatibility, you should:

Consider:
- Moving models into data directory so it's one directory, rather than having models and data be separate.
- If you do this, you can remove the model_dir attribute in projects.toml.

Consider:
- Label files are stored as specified
- Extracted frame names are stored as specified
- Multiview: verify calibration naming

How to add project.yaml file and projects.toml file
----------------------------------------------------

