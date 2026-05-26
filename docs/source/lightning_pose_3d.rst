.. _lightning_pose_3d:

Lightning Pose 3D
==================

This repo and the Lightning Pose App support multi-camera projects with an arbitrary number of
cameras (tested with up to six).
Before starting, cameras must be synchronized across views, and the resulting video files for a
given session must each contain the same number of frames.

**On this page:**

* :ref:`Camera setup <lp3d_camera_setup>`
* :ref:`Camera calibration (optional) <lp3d_camera_calibration>`
* :ref:`Data organization <lp3d_data_organization>`
* :ref:`Data annotation <lp3d_data_annotation>`
* :ref:`Model training <lp3d_model_training>`
* :ref:`Model inference <lp3d_model_inference>`
* :ref:`3D inference <lp3d_3d_inference>`

--------

.. _lp3d_camera_setup:

Camera setup
------------

We recommend using at least three cameras to maximize the number of views in which each keypoint
is unoccluded.
Cameras should be positioned at relatively orthogonal angles to one another so that each view
provides complementary information.

.. _lp3d_camera_calibration:

Camera calibration (optional)
------------------------------

Camera calibration determines the intrinsic parameters of each camera (focal length, principal
point, distortion coefficients) and the extrinsic parameters that describe how the cameras are
positioned and oriented relative to each other.
Together, these parameters make it possible to map 2D pixel coordinates in any view to a shared
3D world coordinate system.

We recommend using the `Anipose <https://anipose.readthedocs.io/>`_ package for calibration.
If you use a different calibration tool, you will need to convert your files into the
:doc:`expected format <directory_structure_reference/camera_calibration_file_format>`.

**How Lightning Pose uses calibration:**

* **3D data augmentation:** calibration parameters allow geometrically consistent augmentation
  across views during training (see
  :ref:`3D augmentations and loss <patch_masking_3d_loss>` for details).
* **3D reprojection loss:** calibration enables a training loss that penalizes geometrically
  inconsistent 2D predictions across views.

.. note::

    Camera calibration is **not** required to train a multi-view Lightning Pose model.
    However, calibration **is** required to obtain 3D coordinates unified across cameras
    (see :ref:`3D inference <lp3d_3d_inference>` below).

.. _lp3d_data_organization:

Data organization
-----------------

**Using the App**

Create a multi-view project by following the
:doc:`Create your first project <create_first_project>` guide.
The App will store your data in the correct format automatically.

**Without the App (or converting from another format)**

See the :doc:`multi-view directory structure reference <directory_structure_reference/multiview_structure>`
for the expected layout.

**Important for all users**

Calibration files must be saved manually in the correct location, regardless of whether you use
the App.
See the
:doc:`calibration file format reference <directory_structure_reference/camera_calibration_file_format>`
for the required location and format.

.. _lp3d_data_annotation:

Data annotation
---------------

The App provides a multi-view annotation tool that lets you label a keypoint in two views and then
uses the calibration information to automatically project those labels into the remaining views.
In general, we recommend keeping the automatically projected label in each view even when the body
part is occluded; doing so helps the 3D data augmentation and reprojection loss learn the geometric
structure of the scene.

Multi-view annotation is time-consuming even with this assistance.
We recommend the following workflow:

1. Label approximately 100 frames across as many individuals as possible.
2. Train an initial model.
3. Run inference on videos from new individuals (preferred) or new sessions from the same
   individuals to surface difficult frames.
4. Use the Viewer tab to identify those difficult frames and add them to your labeled set.

In general, labeling a smaller number of frames from a larger number of individuals leads to
better generalization.
For example, if your labeling budget is 200 frames, labeling 20 frames from 10 separate
individuals is preferable to labeling 200 frames from a single individual.

.. _lp3d_model_training:

Model training
--------------

Model training in the App is straightforward; see the
:doc:`Create your first project <create_first_project>` guide for a walkthrough.

For training via the CLI, see:

* :doc:`Training and inference (multi-view) <user_guide_multiview/training_inference>` — general
  training procedure for multi-view setups.
* :doc:`Patch masking and 3D loss <user_guide_multiview/patch_masking_3d_loss>` — multi-view
  specific training features including patch masking and the 3D reprojection loss.

.. _lp3d_model_inference:

Model inference
---------------

Inference in the App follows the same workflow as for single-view projects.

For inference via the CLI, see:

* :doc:`Training and inference (multi-view) <user_guide_multiview/training_inference>` — covers
  the multi-view inference procedure and expected file layout.

.. _lp3d_3d_inference:

3D inference
------------

.. note::

    Camera calibration information is required for 3D inference.

Once per-view 2D predictions are available, 3D coordinates can be reconstructed across cameras.
We recommend the
`Ensemble Kalman Smoother (EKS) <https://github.com/paninski-lab/eks>`_ tool for this step.
EKS can operate on predictions from a single model or from an ensemble of models; ensembling
improves accuracy and provides better-calibrated uncertainty estimates than the likelihood outputs
of any single network.
