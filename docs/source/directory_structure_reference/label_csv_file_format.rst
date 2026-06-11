.. _label_csv_file_format:

Label CSV File Format
======================

See `data/mirror-mouse-example/CollectedData.csv <https://github.com/paninski-lab/lightning-pose/blob/main/data/mirror-mouse-example/CollectedData.csv>`_ in the git repo
for an example of the expected format of our label files.

For multiview projects, each view has its own label file. The label files are
aligned across views: the Nth row represents the same frame across files.

Standard format (x, y only)
-----------------------------

The standard format uses two columns per keypoint — ``x`` and ``y``.
Missing (unlabeled) keypoints are left as empty cells, which pandas reads as ``NaN``.
The training behavior for these unlabeled keypoints is controlled by
:ref:`training.uniform_heatmaps_for_nan_keypoints <config_file>`.

.. code-block:: text

    scorer,    scorer,  scorer,  scorer,  scorer
    bodyparts, kp1,     kp1,     kp2,     kp2
    coords,    x,       y,       x,       y
    img01.png, 77.25,   36.25,   ,
    img02.png, 37.25,   110.75,  12.5,    88.0

Extended format with per-keypoint visibility
---------------------------------------------

An optional third column ``visible`` can be added after each ``x, y`` pair to specify
per-keypoint visibility flags. The flag values follow the
`COCO keypoint format <https://cocodataset.org/#keypoints-eval>`_:

* **0** — keypoint is not labeled; it is excluded from the loss (same as leaving the cell empty
  when ``training.uniform_heatmaps_for_nan_keypoints: false``).
* **1** — keypoint is occluded; a uniform heatmap is used as the training target, which
  encourages the model to output low-confidence predictions (same as
  ``training.uniform_heatmaps_for_nan_keypoints: true``).
* **2** — keypoint is visible; a Gaussian heatmap is used as the training target (standard
  supervised training).

.. code-block:: text

    scorer,    scorer,  scorer,  scorer,  scorer,  scorer,  scorer
    bodyparts, kp1,     kp1,     kp1,     kp2,     kp2,     kp2
    coords,    x,       y,       visible, x,       y,       visible
    img01.png, 77.25,   36.25,   2,       ,        ,        0
    img02.png, 37.25,   110.75,  2,       ,        ,        1

The extended format allows you to mix visibility behaviors within a single dataset —
some keypoints can be excluded from the loss, others can encourage uncertainty, and others
can receive normal supervised targets, all in the same training run.

When the ``visible`` column is present it takes precedence over
``training.uniform_heatmaps_for_nan_keypoints``, which serves only as a fallback for
CSVs that omit the visibility column.

.. note::

   If a keypoint is marked ``visible=1`` (occluded) but ``x, y`` coordinates are also
   provided in the CSV, a warning is logged and the coordinates are ignored — the uniform
   heatmap is used regardless.

Manipulating label files
-------------------------

Label files are intended to be parsed as pandas dataframes like so:

.. code-block:: python

    pd.read_csv(csv_file, header=[0,1,2], index_col=0)
