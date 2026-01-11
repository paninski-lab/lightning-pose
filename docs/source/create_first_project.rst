Create your first project
==========================

Now that you've installed Lightning Pose and understood the core concepts,
you're ready to create your first project.

This tutorial assumes you are starting with only videos and want to label frames
and train models. If you have labels, see `<importing_labeled_data>`_.

Run the app
---------------

From a command line, activate your conda environment and run:

.. code-block:: console

    litpose run_app

You should see some output that says the server is running on port 8000.
Now that the server is running, open a web browser and navigate to
``http://localhost:8000`` if you ran the server on your local machine,
or replace ``localhost`` with the IP address of the machine where you ran the app.

In cloud environments, the cloud provider has some mechanism for opening the port
and providing a link for you to access it. In Lightning Studio, this is done
using the Port icon on the right hand side.

Once the app is open in the browser, it should look like this:

Then, click on New Project and begin to fill out the form.

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

That's it! Hit save and your project will be created.

Label data
-----------

This tutorial assumes you do not yet have labels.
If you have labels, see `<importing_labeled_data>`_.

Labeling occurs in the Labeler module.

Create a label file and extract frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start by creating a Label file left side green button.
This opens the "extract frames" workflow:
- name your label file
- upload a session for frame extraction
- specify frame extraction settings

Label frames
~~~~~~~~~~~~

Click an unlabeled frame on the left and start labeling.
You need to hit save in order to persist the label file.

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

This doc is a work in progress.

Visualize predictions
----------------------

This doc is a work in progress.