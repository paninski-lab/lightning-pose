======================================
Singleview Data Directory Structure
======================================

The structure is the same as the multiview structure, except wherever
there is a _VIEW suffix, it is omitted. 

.. code-block:: text

    /path/to/project/
      ├── project.yaml
      ├── labeled-data/
      │   ├── session0/
      │   │   └── frame000001.png
      │   └── session1/
      │       └── frame000123.png
      ├── videos/
      │   ├── session0.mp4
      |   └── session1.mp4
      └-─ CollectedData.csv

For more information on the files above and their formats, see :doc:`directory_structure_reference/multiview_structure`.