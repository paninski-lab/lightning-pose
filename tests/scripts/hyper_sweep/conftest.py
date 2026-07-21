"""Fixtures for scripts/hyper-sweep tests.

`scripts/hyper-sweep` uses a hyphen in its directory name, so it is not a valid Python
package and its modules cannot be reached with a normal `import` statement. Load them
from their file path instead.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import lightning_pose as lp

HYPER_SWEEP_DIR = lp.LP_ROOT_PATH / 'scripts' / 'hyper-sweep'


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def run_sweep() -> ModuleType:
    return _load_module('run_sweep', HYPER_SWEEP_DIR / 'run_sweep.py')


@pytest.fixture(scope='module')
def run_single_job() -> ModuleType:
    return _load_module('run_single_job', HYPER_SWEEP_DIR / 'run_single_job.py')
