.. _bounding_box_file_format:

Bounding box file format
------------------------

Each bounding box CSV file must have exactly five columns:

* **First column** (no header): The relative path to each labeled image file
* **x**: Upper-left x-coordinate of the bounding box
* **y**: Upper-left y-coordinate of the bounding box
* **h**: Height of the bounding box
* **w**: Width of the bounding box

Example ``bboxes_view0.csv`` format:

.. code-block::

    ,x,y,h,w
    labeled-data/session0_view0/img00000005.png,1230,117,391,391
    labeled-data/session0_view0/img00000010.png,482,138,425,425
    labeled-data/session0_view0/img00000230.png,1230,117,391,391
    labeled-data/session1_view0/img00000151.png,625,125,405,405
    labeled-data/session1_view0/img00000201.png,1186,118,343,344

The image paths in the first column should match exactly with the paths used in your
labeled data CSV files.