#############
FAQs
#############

* :ref:`Can I import a pose estimation project in another format? <faq_can_i_import>`
* :ref:`What if I encounter a CUDA out of memory error? <faq_oom>`

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
