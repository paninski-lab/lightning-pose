"""Convert labeled datasets from other pose estimation tools into Lightning Pose projects.

Two converters, one per source format:

- :func:`lightning_pose.converters.dlc.convert` -- reads a DeepLabCut project directory
  (``labeled-data/<video>/CollectedData*.{csv,h5}``) and writes a Lightning Pose project
  directory (``CollectedData.csv`` plus copied ``labeled-data/`` and ``videos/``).
- :func:`lightning_pose.converters.sleap.convert` -- reads a SLEAP ``.pkg.slp`` package
  (frames and labels bundled in one HDF5 file) and writes a Lightning Pose project directory.

Both converters are dispatched from ``litpose convert <dataset_path> --lp_dir <lp_dir>``
(:mod:`lightning_pose.cli.commands.convert`), which decides which one to call based on
whether ``dataset_path`` is a directory (DLC) or a ``.slp`` file (SLEAP).

**Adding a new converter**: create ``<name>.py`` here with a ``convert(<source>: Path, lp_dir:
Path) -> None`` function, then add a dispatch branch in
``lightning_pose.cli.commands.convert.handle``.
"""
