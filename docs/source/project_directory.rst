.. _lightning_pose_api:

###################################
Project directory & ProjectSchema
###################################

`ProjectSchemaV1` provides a standardized way to build and parse
project-relative paths (videos, frames, labels, calibrations, etc.).
It supports both forward mapping (keys → paths), reverse parsing (paths → keys),
and filesystem enumeration. This interface is preferred to manipulating paths
directly because it makes code robust to changes in LP's file structure
down the line.

This page shows how to use it by example. For full API details, see the
reference at the end of the page.

.. contents:: On this page
   :local:
   :depth: 2

What is a schema?
=================
A schema defines the layout of project files. In v1, each resource (e.g.,
``videos``, ``frames``, ``label-files``) has:

- a template for building paths, and
- a pattern for parsing paths back into lookup keys.

Creating a schema
=================
You can construct a schema in two ways.

From a registered project (recommended)
---------------------------------------
If your project is registered via ``projects.toml`` and has a ``project.toml``
under the data directory, use ``for_project``. This automatically configures
single-vs-multiview, schema version, and the base directory used for filesystem
enumeration.

.. code-block:: python

   from lightning_pose.utils.paths.project_schema import ProjectSchema

   # project can be a string key, ProjectKey, ProjectDirs, or ProjectConfig
   schema = ProjectSchema.for_project("my_project")
   # schema.base_dir is set to your project's data directory
   print(schema.is_multiview, schema.base_dir)

From scratch / by version
-------------------------
If you’re not using a registered project, construct by version and pass the
``base_dir`` explicitly (recommended when you want to use enumeration helpers).

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
Forward mapping: keys → paths
----------------------------
Use ``get(key)`` to build a relative path from a typed key.

.. code-block:: python

   from lightning_pose.data.keys import VideoFileKey, FrameKey

   # videos
   vkey = VideoFileKey(session_key="sessionA", view=None)  # single-view
   rel_video = schema.videos.get(vkey)        # Path('videos/sessionA.mp4')

   # frames (with zero-padded index)
   fkey = FrameKey(session_key="sessionA", view=None, frame_index=42)
   rel_frame = schema.frames.get(fkey)        # Path('labeled-data/frames/sessionA/frame_00000042.png')

Reverse parsing: paths → keys
-----------------------------
Use ``reverse(rel_path)`` to parse a relative path back into a key.

.. code-block:: python

   k = schema.videos.reverse("videos/sessionA.mp4")
   assert isinstance(k, type(vkey))
   assert k == vkey

.. note::
   ``reverse`` expects a relative path (project-root relative). It raises
   ``ValueError`` for absolute inputs and ``PathParseException`` when the path
   does not match the resource pattern.

Filesystem enumeration
----------------------
If your schema was constructed with a ``base_dir``, you can enumerate existing
files and their parsed keys from disk. Methods raise a clear ``RuntimeError`` if
``schema.base_dir`` is ``None``.

- ``iter_paths()`` → yields absolute ``Path`` objects
- ``iter_keys(strict=False)`` → yields parsed keys (skips unparsable files unless ``strict=True``)
- ``get_all(return_='keys' | 'paths' | 'both', sort=True, strict=False)`` → convenience wrapper

Examples:

.. code-block:: python

   # list all videos on disk (keys)
   video_keys = schema.videos.get_all()  # default return_='keys'

   # absolute paths to label CSVs
   label_paths = schema.label_files.get_all(return_='paths')

   # pairs of (key, absolute path)
   key_and_path = schema.frames.get_all(return_='both')

Strict parsing during enumeration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``strict=True`` to fail fast on stray files that don’t match the resource
pattern.

.. code-block:: python

   # raises PathParseException if a non-matching file is encountered
   keys = list(schema.videos.iter_keys(strict=True))

Predicate resources
-------------------
Some resources represent the presence of a specific file rather than a
collection (e.g., ``project-calibration``). For such resources,
``iter_paths``, ``iter_keys``, and ``get_all`` are not supported and will raise
``TypeError``.

Single-view vs multiview
========================
In multiview projects, view-specific placeholders appear in templates and keys.

.. code-block:: python

   from lightning_pose.data.keys import VideoFileKey

   mv_schema = ProjectSchema.for_version(1, is_multiview=True, base_dir=schema.base_dir)

   vA = VideoFileKey(session_key="S1", view="camA")
   vB = VideoFileKey(session_key="S1", view="camB")

   pA = mv_schema.videos.get(vA)  # Path('videos/S1_camA.mp4')
   pB = mv_schema.videos.get(vB)  # Path('videos/S1_camB.mp4')

   kA = mv_schema.videos.reverse("videos/S1_camA.mp4")
   assert kA == vA

Recipes
=======
List all videos and their label files
-------------------------------------

.. code-block:: python

   from lightning_pose.utils.paths import ResourceType

   videos = schema.videos.get_all()                     # list[VideoFileKey]
   labels = schema.label_files.get_all(return_='both')  # list[(LabelFileKey|View, Path)]

   # Create a map from video session to label file path(s)
   from collections import defaultdict
   by_session = defaultdict(list)
   for (label_key, path) in labels:
       lfile_key, view = label_key  # unpack tuple
       by_session[lfile_key].append(path)

   # Use resource map generically
   frames_util = schema.for_(ResourceType.frames)
   frame_keys = frames_util.get_all()  # list[FrameKey]

Find all frames for a specific session
--------------------------------------

.. code-block:: python

   # Filter in memory after enumeration
   all_frames = schema.frames.get_all()
   s1_frames = [fk for fk in all_frames if fk.session_key == "sessionA"]

Validate key ↔ path round-trip
------------------------------

.. code-block:: python

   fk = FrameKey(session_key="S2", view=None, frame_index=123)
   path = schema.frames.get(fk)
   assert schema.frames.reverse(path) == fk

Troubleshooting
===============
- ``RuntimeError: ... base_dir is None``

  Construct the schema with a base directory (either via
  ``ProjectSchema.for_project(...)`` or by passing ``base_dir=...`` to
  ``ProjectSchema.for_version(...)``) before using filesystem enumeration.

- ``PathParseException: Could not parse ...``

  The file does not match the resource’s expected pattern. Ensure you’re passing
  a project-relative path to ``reverse`` and that the path layout matches the
  schema templates.

- ``ValueError: Argument must be relative path`` (in ``reverse``)

  ``reverse`` only accepts project-relative paths. Convert absolute paths to
  relative using ``p.relative_to(schema.base_dir)`` first.

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