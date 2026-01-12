Create your first project
==========================

Now that you've installed Lightning Pose and understood the core concepts,
you're ready to create your first project.

Run the app
---------------

From a command line, activate your conda environment and run:

.. code-block:: console

    $ litpose run_app

You should see some output that says the server is running on port 8000.
Now that the server is running, open a web browser and navigate to
``http://localhost:8000`` if you ran the server on your local machine,
or replace ``localhost`` with the IP address of the machine where you ran the app.

.. note::
    In cloud environments, the cloud provider has some mechanism for opening the port
    and providing a link for you to access it. In Lightning Studio, this is done
    using the Port icon on the right hand side.

Once the app is open in the browser, it should look like this:

.. image:: /images/app_screenshots/app_new_project_page.png

Then, click on New Project and begin to fill out the form.

.. image:: /images/app_screenshots/app_new_project_form.png


Data directory
~~~~~~~~~~~~~~

Specify the directory where Lightning Pose will store the project.
Our recommendation is ``HOME_DIR/LPProjects/YourProjectNameHere``. Get your home directory path by
running ``echo ~`` in a terminal.

Model directory can be left as the default.

Keypoints and View names
~~~~~~~~~~~~~~

First name your keypoints, the points you'd like to track. The names
will be used as column headers in label files and prediction output files.

Views are the names of your camera views. **If you have only one view, you can
leave this blank**. For multiple views, **your videos must end in _viewname as
the suffix**. This convention is relied upon for Lightning Pose to
extract the View name and Session name from a video's filename.

That's it! Hit save and your project will be created. You will be redirected
to the project home page.

.. image:: /images/app_screenshots/app_project_home.png

Label data
-----------

This tutorial assumes you do not yet have labels.
If you have labels, see `<importing_labeled_data>`_.

Labeling occurs in the Labeler module.

.. image:: /images/app_screenshots/app_labeler_home.png


Create a label file and extract frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating a Label file left side green button.
This opens the "extract frames" workflow:
- name your label file (leave it as the default)
- upload a session for frame extraction
- specify frame extraction settings

.. image:: /images/app_screenshots/app_extract_frames_name_lblfile.png

Leave the default name and click next. You will see the session selection screen.
On the right, click Upload Session.

.. image:: /images/app_screenshots/app_extract_frames_2_select_session.png

Select videos from your local filesystem and click Import.
Videos will upload and be transcoded. Upon completion, the video import dialog
will close and you will be able to see and select the newly uploaded session on the
left hand side of the session selection screen.

.. image:: /images/app_screenshots/app_session_import_successful.png


Label frames
~~~~~~~~~~~~

Click an unlabeled frame on the left and start labeling.
You need to hit save in order to persist the label file.

.. image:: /images/app_screenshots/app_labeler_with_unlabeled_frame_loaded.png

Explainer: Label files
~~~~~~~~~~~~~~~~~~~~~~~~~

See docs for directory structure of label files.

1. You might want to backup a copy the label files before doing
a round of labeling or label refinement.
Copy the files and suffix with a timestamp or ID.
These will be selectable in the label file dropdown, and serve as a backup
and allow you to compare model performance before and after.

2. Most users will have just one primary label file. However some users might
have niche use cases such as keeping separate "in-distribution" and "out-of-distribution"
label files. This is mostly useful for model research.

Multiview features
~~~~~~~~~~~~~~~~~~~~~

Lighting pose supports utilizing camera calibration to make multiview labeling easier.
Using labels from two views, it can triangulate into 3D space and reproject onto
the remaining views. See the in product documentation here for more information.

Create a model
----------------

In the Models module, click New Model and follow the instructions to create a model.
Training will begin automatically.

.. image:: /images/app_screenshots/app_model_creation_form.png


Visualize predictions
----------------------

This doc is a work in progress.