"""Command modules for the lightning-pose CLI.

Each module in :data:`COMMANDS` is a plain Python module, not a class -- there is no
``Protocol`` or base class enforcing this, so the contract is documented here instead. A
command module must define three top-level functions:

- ``register_parser(subparsers) -> ArgumentParser`` -- attaches this command's subparser to
  the shared top-level parser. Called by ``cli/main.py`` at CLI startup for every module.
- ``handle(args) -> None`` -- runs the command given the parsed ``argparse.Namespace``. Called
  by ``cli/main.py`` after argument parsing.
- ``get_parser() -> ArgumentParser`` -- builds a standalone top-level parser for *this command
  alone* (by constructing its own throwaway ``subparsers`` and delegating to
  ``register_parser``). **Not called by ``main.py``** -- it exists for two other consumers:
  ``docs/source/cli_reference/<command>.rst``'s ``sphinx-argparse`` directive (``:func:
  get_parser``), which needs a zero-argument parser-returning function to auto-generate the
  CLI reference docs page, and ``tests/cli/test_<command>.py``, which calls it directly as a
  basic sanity check. A command missing or mis-signaturing ``get_parser`` won't break the CLI
  at all -- it silently breaks the docs build or a test instead, which is easy to miss.

**Adding a new command**: create ``<name>.py`` here with all three functions (copy an existing
module, e.g. ``train.py``, as a template), add it to :data:`COMMANDS` below, and add a matching
``docs/source/cli_reference/<name>.rst`` page (with a toctree entry in that directory's
``index.rst``). ``tests/cli/test_main.py::TestCliReferenceDocs`` fails if either is missing, so
a forgotten doc page is caught by CI rather than relying on this docstring being read.
"""

from lightning_pose.cli.commands import (
    convert,
    create_bbox,
    crop,
    export,
    predict,
    recommend,
    remap,
    run_app,
    smooth_bbox,
    train,
)

# List of all available commands
COMMANDS = {
    'train': train,
    'predict': predict,
    'export': export,
    'create_bbox': create_bbox,
    'smooth_bbox': smooth_bbox,
    'crop': crop,
    'remap': remap,
    'run_app': run_app,
    'recommend': recommend,
    'convert': convert,
}
