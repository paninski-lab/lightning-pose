.. _project_yaml_file_format:

Project Yaml File Format
=========================

The file is stored at ``DATA_DIR/project.yaml``.

Example (copy and modify to your needs):

.. code-block:: yaml

    # List of strings or [] for single view
    # view_names: []
    view_names:
        - view1
        - view2

    # List of strings, used by app to init label files.
    keypoint_names:
        - nose
        - tail
