#############
FAQs
#############

* :ref:`Can I import a pose estimation project in another format? <faq_can_i_import>`
* :ref:`What if I encounter a CUDA out of memory error? <faq_oom>`
* :ref:`Why does the network produce high confidence values for keypoints even when they are occluded? <faq_nan_heatmaps>`

.. _faq_can_i_import:

**Q: Can I import a pose estimation project in another format?**

We currently support conversion from DLC projects into Lightning Pose projects
(if you would like support for another format, please
`open an issue <https://github.com/danbider/lightning-pose/issues>`_).
You can find more details in the :ref:`Organizing your data <directory_structure>` section.

.. _faq_oom:

**Q: What if I encounter a CUDA out of memory error?**

We recommend a GPU with at least 8GB of memory.
Note that both semi-supervised and context models will increase memory usage
(with semi-supervised context models needing the most memory).
If you encounter this error, reduce batch sizes during training or inference.
You can find the relevant parameters to adjust in :ref:`The configuration file <config_file>`
section.

.. _faq_nan_heatmaps:

**Q: Why does the network produce high confidence values for keypoints even when they are occluded?**

Generally, when a keypoint is briefly occluded and its location can be resolved by the network, we are fine with
high confidence values (this will happen, for example, when using temporal context frames).
However, there may be scenarios where the goal is to explicitly track whether a keypoint is visible or hidden using
confidence values (e.g., quantifying whether a tongue is in or out of the mouth).
In this case, if the confidence values are too high during occlusions, try the suggestions below.

First, note that including a keypoint in the unsupervised losses - especially the PCA losses -
will generally increase confidence values even during occlusions (by design).
If a low confidence value is desired during occlusions, ensure the keypoint in question is not
included in those losses.

If this does not fix the issue, another option is to set the following field in the config file:
``training.uniform_heatmaps_for_nan_keypoints: true``.
[This field is not visible in the default config but can be added.]
This option will force the model to output a uniform heatmap for any keypoint that does not have
a ground truth label in the training data.
The model will therefore not try to guess where the occluded keypoint is located.
This approach requires a set of training frames that include both visible and occluded examples
of the keypoint in question.
