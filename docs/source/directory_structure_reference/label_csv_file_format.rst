.. _label_csv_file_format:

Label CSV File Format
======================

See `data/mirror-mouse-example/CollectedData.csv <https://github.com/paninski-lab/lightning-pose/blob/main/data/mirror-mouse-example/CollectedData.csv>`_ in the git repo
for an example of the expected format of our label files.

For multiview projects, each view has its own label file. The label files are
aligned across views: the Nth row represents the same frame across files.

Manipulating label files
-------------------------

Label files are intended to be parsed as pandas dataframes like so:

.. code-block:: python

    pd.read_csv(csv_file, header=[0,1,2], index_col=0)
