.. rst-class:: home-display-none

Lightning Pose Homepage
========================

.. image:: images/LightningPose_horizontal_light.png


.. meta::
   :description: Accessible documentation for animal pose estimation.

.. raw:: html

    <div style="text-align: center; margin-top: 1em; margin-bottom: 2em;">
        <p style="font-size: 1.2em;">An end-to-end toolkit for robust multi-view animal pose estimation.</p>
    </div>

.. grid:: 1 1 3 3
    :gutter: 3
    :class-container: feature-card-container

    .. grid-item-card:: 🛰️ Multi-View
        :class-card: feature-toggle-card

        Multiview transformers and patch masking for robust 3D tracking.

    .. grid-item-card:: 🎬 Single-View
        :class-card: feature-toggle-card

        Temporal context networks that learn from unlabeled video.

    .. grid-item-card:: ☁️ Cloud Ready GUI
        :class-card: feature-toggle-card

        Browser-based labeling and training on headless GPU servers.

.. rst-class:: multi-view-section

Multi-view Capabilities
------------------------

* **Multi-View Transformer (MVT):** A unified architecture that enables simultaneous processing of information across all camera views through early-feature fusion.
* **Patch Masking:** A novel training scheme that masks random image patches to force the model to learn robust cross-view correspondences.
* **Geometric Consistency:** For calibrated setups, the framework incorporates 3D triangulation losses and geometrically-aware 3D data augmentation.
* **Variance Inflation:** An advanced technique for outlier detection that identifies geometrically inconsistent predictions.

.. rst-class:: single-view-section

Single-view Capabilities
-------------------------

* **Temporal Context Networks:** Utilizes information from surrounding frames to resolve anatomical ambiguities and maintain tracking through brief occlusions.
* **Unsupervised Learning:** Employs training objectives that penalize predictions for violating physical laws.
* **Pretrained Backbone Support:** Optimized to work with generic, off-the-shelf Vision Transformer (ViT) backbones.

.. rst-class:: cloud-section

Cloud Application & Workflow
-----------------------------

* **Cloud & Headless Compatibility:** A browser-based interface designed for local or cloud deployment.
* **Multi-view Labeling:** A specialized annotation tool that streamlines the labeling process by using camera calibration.
* **Unified Multi-view Viewer:** Integrated visualization tools to inspect and compare predictions across all camera views simultaneously.

.. raw:: html

   <script>
   document.addEventListener("DOMContentLoaded", function() {
       const cards = document.querySelectorAll('.feature-toggle-card');
       const detailSections = [
           document.querySelector('.multi-view-section'),
           document.querySelector('.single-view-section'),
           document.querySelector('.cloud-section')
       ];

       function showSection(index) {
           // Hide all sections and remove active styling from cards
           detailSections.forEach((sec, i) => {
               if (sec) sec.style.display = 'none';
               cards[i].style.border = '1px solid var(--sd-color-card-border)';
               cards[i].style.backgroundColor = 'transparent';
           });

           // Show the selected section
           if (detailSections[index]) {
               detailSections[index].style.display = 'block';
               // Add active styling to card
               cards[index].style.border = '2px solid #3498db';
               cards[index].style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
           }
       }

       // Event listeners for clicks
       cards.forEach((card, index) => {
           card.style.cursor = 'pointer';
           card.addEventListener('click', () => showSection(index));
           card.addEventListener('mouseenter', () => showSection(index));
           card.addEventListener('touchstart', () => showSection(index));

       });

       // Select the first card by default
       showSection(0);
   });
   </script>

   <style>
   /* Enhance hover effect */
   .feature-toggle-card:hover {
       transform: translateY(-2px);
       transition: all 0.2s ease;
       box-shadow: 0 4px 12px rgba(0,0,0,0.1);
   }
   </style>


--------

Read the papers
----------------

The original Nature Methods 2024 paper that introduced Lightning Pose for single-view pose estimation using semisupervised learning and ensemble kalman smoothing (EKS):

| **Lightning Pose: improved animal pose estimation via semi-supervised learning, Bayesian ensembling and cloud-native open-source tools**
| Biderman, D., Whiteway, M. R., Hurwitz, C., et al.

.. grid::
   :padding: 0
   :margin: 2 0 0 0

   .. grid-item::
      .. button-link:: https://pmc.ncbi.nlm.nih.gov/articles/PMC12087009/
         :color: primary
         :outline:

         *Nature Methods* 21, 1316–1328 (2024)

The 2025 paper that added robust multiview support using multiview transformers (MVT),
patch masking, 3d image augmentation and losses, and multiview EKS.

| **An Uncertainty-Aware Framework for Data-Efficient Multi-View Animal Pose Estimation**
| Aharon, L., Lee, K., et al.

.. grid::
   :padding: 0
   :margin: 2 0 0 0

   .. grid-item::
      .. button-link:: https://arxiv.org/abs/2510.09903
         :color: primary
         :outline:

         arXiv Preprint (2025)

.. rst-class:: section-multi-view

--------


Get started with the app
-------------------------

The lightning pose app provides an easy-to-use GUI to access most lightning pose features.

To get started, :doc:`install lightning pose <source/installation_guide>`
and follow the :doc:`Create your first project <source/create_first_project>` tutorial.
It covers the end-to-end workflow of labeling, training, and evaluation.

.. toctree::
   :maxdepth: 2
   :hidden:

   self

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting started

   source/installation_guide
   source/core_concepts
   source/create_first_project
   source/example_data
   source/importing_labeled_data

.. toctree::
   :maxdepth: 4
   :hidden:
   :caption: CLI User guide

   source/user_guide_singleview/index
   source/user_guide_multiview/index
   source/user_guide_advanced/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Community

   source/developer_guide/index
   source/faqs
   Release notes <https://github.com/paninski-lab/lightning-pose/releases>
   source/migrating_to_app

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   source/user_guide_singleview/config_file
   source/api
   source/cli_reference/index