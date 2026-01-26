.. _transfer_project:

Transfer a project directory
=============================

A lightning pose project directory is portable. You can zip it up in the remote machine, download it, extract it
on the local machine, and :ref:`add it to your local projects.toml<add_existing_project>`.

The rest of the how-to guide is for users who want to know the details of how to do that.
We give copy-pastable commands, and recommend using tools like ChatGPT to further explore these commands.

1. Find the project directory
------------------------------

The locations of projects are stored in the ``~/.lightning_pose/projects.toml`` file.

.. code-block:: bash

    cat ~/.lightning_pose/projects.toml

Copy the data_dir of your project. If you also see a model_dir, you will want to zip and download that too using the instructions in the rest of the tutorial.

2. Create a zip or tar file of the project directory
------------------------------------------------------

To upload/download a directory, you first convert it to a single file using zip or tar. For this example, let’s say your project directory is

.. code-block:: bash

    /home/lp-projects/mouse-tracking

To zip the directory, first cd into its parent directory:

.. code-block:: bash

    cd /home/lp-projects


Now run a zip command to output a file called mouse-tracking.zip:

.. code-block:: bash

    zip mouse-tracking.zip mouse-tracking

This compresses the directory mouse-tracking into the specified zip.

3. Download the directory
---------------------------

If your machine is a physical computer you have access to, you can simply transfer the zip onto a USB stick,
or upload it into Google Drive, or similar service. Skip to the next section.

If your machine is a cloud server, there are a few ways ways to access the files there.

A simple way is to use python’s simple http.server. The idea is that we’ll run a file-hosting server that is built-in to python, expose a port, and access that from the local PC we wish to download from.

Change to the directory containing the zip archive and run the server.

.. code-block:: bash

    cd /home/lp-projects
    python -m http.server


It will say that it’s running on 0.0.0.0 on port 8000. The 0.0.0.0 IP means that any external IP can connect to the server. And the port 8000 tells us what port we need to open in the studio.

If you're using Lightning AI studio as your cloud environment, then use the studio port plugin to open the port 8000.
If you're using another cloud provider, refer to their instructions for exposing a port to the internet.

When you navigate to the port, you’ll see a basic webpage with a download link for archive.zip, (as well as all other files in the directory). If you click that, the archive.zip should download onto your computer. Hooray!

**Alternative techniques**

Consider using ssh, scp, or sftp. These topics are well covered on the internet.

4. Add the project to the registry
------------------------------------

Once you've downloaded and extracted the project directory onto the new machine, follow this guide: :ref:`add_existing_project`

Appendix: Transferring a subset of the project directory
----------------------------------------------------------

A project directory can get very large due to excessive models and videos.

You can be selective about what you transfer by:

* omitting the models directory entirely if you do not need to do any model operations
* selectively including just the models you need
* omitting videos if you do not need to view or extract frames from them

When in doubt, consult the :doc:`Directory Structure Reference </source/directory_structure_reference/index>` to see what the purpose of each file is.
