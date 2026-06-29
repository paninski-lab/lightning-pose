.. _bounding_box_file_format:

Bounding box file format
------------------------

Bbox CSV files share the same five-column layout but differ in how rows are indexed
depending on whether the input is a video or a set of labeled frames.

Columns (both modes)
^^^^^^^^^^^^^^^^^^^^

* **First column** (no header): image path (used as a key in labeled-frames mode;
  ignored in video mode — see below)
* **x**: upper-left x-coordinate of the bounding box in pixels
* **y**: upper-left y-coordinate of the bounding box in pixels
* **h**: height of the bounding box in pixels
* **w**: width of the bounding box in pixels

.. _bbox_video_mode:

Video mode (``<video_stem>_bbox.csv``) — dense, positional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When cropping or predicting on a **video**, Lightning Pose reads bbox rows
*positionally*: row ``i`` is matched to video frame ``i`` by index, and the path
column is ignored.

.. warning::

   The CSV must be **dense**: it must contain exactly one row per video frame with
   no gaps.  If your tracking skips frames (e.g. the animal is absent or occluded),
   carry the last known bbox forward to fill the gap.  A sparse CSV will silently
   misalign every row after the first gap — ``litpose crop`` will raise a
   ``ValueError`` if the row count does not match the frame count.

File naming: ``<video_stem>_bbox.csv`` in the bbox directory.

.. code-block::

    ,x,y,h,w
    0,640,256,205,128
    1,642,258,205,128
    2,641,257,205,128

(Row 0 → frame 0, row 1 → frame 1, … — first column is unused.)

Labeled-frames mode (``bbox.csv``) — sparse, path-keyed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When cropping **labeled frames**, rows are matched by the path in the first column.
Rows may be sparse (only labeled frames need an entry).  The path must match exactly
the image paths used in your labeled data CSV.

File naming: ``bbox.csv`` in the bbox directory.

.. code-block::

    ,x,y,h,w
    labeled-data/session0_view0/img00000005.png,1230,117,391,391
    labeled-data/session0_view0/img00000010.png,482,138,425,425
    labeled-data/session0_view0/img00000230.png,1230,117,391,391
    labeled-data/session1_view0/img00000151.png,625,125,405,405
    labeled-data/session1_view0/img00000201.png,1186,118,343,344