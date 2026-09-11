"""Fixtures that load the scripts/inference-timing/*.py modules by file path.

scripts/inference-timing is not a valid Python package name (it has a
hyphen), so its modules can't be imported with a normal `import` statement.
Mirrors the pattern used in tests/scripts/hyper_sweep/conftest.py for
scripts/hyper-sweep.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "inference-timing"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    assert spec is not None and spec.loader is not None, f"could not load spec for {filename}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark_single_config():
    return _load_module("benchmark_single_config", "benchmark_single_config.py")


@pytest.fixture
def plot_results():
    return _load_module("plot_results", "plot_results.py")


@pytest.fixture
def load_config():
    return _load_module("load_config", "load_config.py")
