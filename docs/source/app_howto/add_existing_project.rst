.. _add_existing_project:

Add an existing project directory
==================================

As briefly mentioned in :ref:`Core Concepts <core_concepts>`, Lightning Pose finds projects in ``~/.lightning-pose/projects.toml``
Should you want to add an existing project directory, simply edit the file in a text editor.

The file can contain multiple projects, in the following format:

.. code-block:: toml

    [project_name_here]
    data_dir = "/home/username/LPProjects/data"
    # model_dir omitted, defaults to "/home/username/LPProjects/models"

    [yet_another_project]
    data_dir = "/home/username/LPProjects/data"
    model_dir = "/home/username/LPModels/models"


That's it. The app fetches the project list from this file. You don't even
need to restart the app server when changing this file, it is sufficient
to refresh the browser.