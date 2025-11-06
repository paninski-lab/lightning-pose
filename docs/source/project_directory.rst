.. _lightning_pose_api:

###################################
Project directory & ProjectSchema
###################################

A **Project** is the root container of your all your lightning pose data and models.
All data is contained in the project directory.

The **Project Schema** bidirectionally maps resources
to paths based on naming conventions. For example, the video for session
``Session1`` and view ``Cam-A`` is always stored in ``videos/Session1_Cam-A.mp4``.

If you write scripts that rely on schema assumptions,
we recommend using the ``ProjectSchema`` utility
described in the rest of this doc.

.. contents:: On this page
   :local:
   :depth: 2


Step 1: Construct a ProjectSchema object
========================================
You can construct a schema by manually specifying the version,
or by providing a project and the version will be inferred automatically.

From project (recommended)
---------------------------
If your project is registered via ``projects.toml`` and has a ``project.toml``
under the data directory, use ``for_project``. This automatically configures
single-vs-multiview, schema version, and the base directory used for resource
enumeration.

.. code-block:: python

   from lightning_pose.utils.paths.project_schema import ProjectSchema

   schema = ProjectSchema.for_project("my_project")

Or skip the ``projects.toml`` lookup by providing project directories manually:

.. code-block:: python

   from lightning_pose.data.datatypes import ProjectDirs
   from lightning_pose.utils.paths.project_schema import ProjectSchema

   schema = ProjectSchema.for_project(ProjectDirs(data_dir="/absolute/path/to/my/data"))

From scratch / by version
-------------------------
Under the hood, the ``for_project`` method unpacks a project config and creates
the correct version of schema. You can do this manually as follows. ``base_dir`` is
required to use enumeration methods (like list all videos).

.. code-block:: python

   from pathlib import Path
   from lightning_pose.utils.paths.project_schema import ProjectSchema

   # Single-view example
   base = Path("/absolute/path/to/my/data")
   schema = ProjectSchema.for_version(schema_version=1, is_multiview=False, base_dir=base)

   # Multiview example
   mv_schema = ProjectSchema.for_version(schema_version=1, is_multiview=True, base_dir=base)

Working with resources
======================

Getting the path for a resource: keys -> path
---------------------------------------------
Use ``get_path(key)`` to build a relative path from a typed key.

.. code-block:: python

   from lightning_pose.data.keys import VideoFileKey, FrameKey

   # videos
   vkey = VideoFileKey(session_key="sessionA", view=None)  # single-view
   rel_video = schema.videos.get_path(vkey)        # Path('videos/sessionA.mp4')

   # frames (with zero-padded index)
   fkey = FrameKey(session_key="sessionA", view=None, frame_index=42)
   rel_frame = schema.frames.get_path(fkey)        # Path('labeled-data/frames/sessionA/frame_00000042.png')

Parsing the path for a resource: path → keys
---------------------------
Use ``parse_path(rel_path)`` to parse a relative path back into a key.

.. code-block:: python

   k = schema.videos.parse_path("videos/sessionA.mp4")
   assert isinstance(k, type(vkey))
   assert k == vkey

.. note::
   ``parse_path`` expects a relative path (project-root relative). It raises
   ``ValueError`` for absolute inputs and ``PathParseException`` when the path
   does not match the resource pattern.

Listing out resources (enumeration)
-----------------------------------
The following methods list out resources currently present in the project directory.

- ``iter_paths()`` → yields ``Path`` objects relative to ``schema.base_dir``
- ``iter_keys()`` → yields parsed keys; by default enumeration is strict and will raise if it encounters a non-matching file. To bypass, pass ``strict=False``.
- ``list_keys(sort=True)`` → collect and optionally sort keys into a list

Example: listing out video files

.. code-block:: pycon

   >>> video_keys = schema.videos.list_keys()

   >>> for v_key in video_keys:
   ...     print(v_key.session_key, v_key.view)
   ...
   05272019_fly1_0_R1C24_rot-ccw-006_sec, Cam-A
   05272019_fly1_0_R1C24_rot-ccw-006_sec, Cam-B
   05272019_fly1_0_R3C1_str-cw-0_sec, Cam-A
   05272019_fly1_0_R3C1_str-cw-0_sec, Cam-B
   ...

Example: listing out label files and their paths

.. code-block:: pycon

   >>> key_and_path = [
   ...     (schema.label_files.parse_path(p), p)
   ...     for p in schema.label_files.iter_paths()
   ... ]
   >>> for (labelfilekey, view), path in key_and_path:
   ...     print(f"{labelfilekey}, {view}, {path}")
   ...
   CollectedData, Cam-A, labeled-data/labels/CollectedData_Cam-A.csv
   CollectedData_new, Cam-A, labeled-data/labels/CollectedData_new_Cam-A.csv
   CollectedData, Cam-B, labeled-data/labels/CollectedData_Cam-B.csv
   CollectedData_new, Cam-B, labeled-data/labels/CollectedData_new_Cam-B.csv
   CollectedData, Cam-C, labeled-data/labels/CollectedData_Cam-C.csv
   ...


Strictness during enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, enumeration is strict and will raise a ``PathParseException`` if a
non-matching file is encountered. To bypass strict behavior and skip
non-matching files, pass ``strict=False``.

.. code-block:: python

   # skip files that do not match the expected pattern
   keys = list(schema.videos.iter_keys(strict=False))


Single-view vs multiview
========================
In multiview projects, view-specific placeholders appear in templates and keys.

.. code-block:: python

   from lightning_pose.data.keys import VideoFileKey

   mv_schema = ProjectSchema.for_version(1, is_multiview=True, base_dir=schema.base_dir)

   vA = VideoFileKey(session_key="S1", view="camA")
   vB = VideoFileKey(session_key="S1", view="camB")

   pA = mv_schema.videos.get_path(vA)  # Path('videos/S1_camA.mp4')
   pB = mv_schema.videos.get_path(vB)  # Path('videos/S1_camB.mp4')

   kA = mv_schema.videos.parse_path("videos/S1_camA.mp4")
   assert kA == vA

Recipes
=======
List all videos and their label files
-------------------------------------

.. code-block:: python

   from lightning_pose.utils.paths import ResourceType

   videos = schema.videos.list_keys()                     # list[VideoFileKey]
   labels = [
       (schema.label_files.parse_path(p.relative_to(schema.base_dir)), p)
       for p in schema.label_files.iter_paths()
   ]  # list[((LabelFileKey, View|None), Path)]

   # Create a map from video session to label file path(s)
   from collections import defaultdict
   by_session = defaultdict(list)
   for (label_key, path) in labels:
       lfile_key, view = label_key  # unpack tuple
       by_session[lfile_key].append(path)

   # Use resource map generically
   frames_util = schema.for_(ResourceType.frames)
   frame_keys = frames_util.list_keys()  # list[FrameKey]

Find all frames for a specific session
--------------------------------------

.. code-block:: python

   # Filter in memory after enumeration
   all_frames = schema.frames.list_keys()
   s1_frames = [fk for fk in all_frames if fk.session_key == "sessionA"]

Validate key ↔ path round-trip
------------------------------

.. code-block:: python

   fk = FrameKey(session_key="S2", view=None, frame_index=123)
   path = schema.frames.get_path(fk)
   assert schema.frames.parse_path(path) == fk


API reference
=============

.. autoclass:: lightning_pose.utils.paths.project_schema_v1.ProjectSchemaV1
    :special-members: __init__
    :inherited-members:
    :members:
    :member-order: bysource
    :undoc-members:

.. autoclass:: lightning_pose.utils.paths.AbstractResourceUtil
    :inherited-members:
    :members:
    :member-order: bysource
    :undoc-members: