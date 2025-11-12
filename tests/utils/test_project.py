import pytest
import tomli_w
from pathlib import Path

from lightning_pose.utils.project import ProjectUtil
from lightning_pose.data.datatypes import ProjectPaths
from lightning_pose.rootconfig import RootConfig


@pytest.fixture
def mock_config(tmp_path: Path) -> RootConfig:
    """Pytest fixture to create a mock config with a temporary projects.toml."""
    config = RootConfig(LP_SYSTEM_DIR=tmp_path / ".lightning-pose")

    return config


@pytest.fixture
def project_util(mock_config: RootConfig) -> ProjectUtil:
    """Pytest fixture to create a ProjectUtil instance."""
    return ProjectUtil(config=mock_config)


def test_project_model():
    # Case: model_dir omitted, direct constructor usage
    p = ProjectPaths(data_dir=Path("/test"))

    # Default applied
    assert p.model_dir == Path("/test/models")
    # Unset fields excluded
    assert "model_dir" not in p.model_fields_set
    out = p.model_dump(mode="json", exclude_unset=True)
    assert set(out.keys()) == set(["data_dir"])
    assert out["data_dir"] == "/test"

    # Case: model_dir omitted, model_validate usage
    p = ProjectPaths.model_validate({"data_dir": Path("/test")})

    # Default applied
    assert p.model_dir == Path("/test/models")
    # Unset fields excluded
    assert "model_dir" not in p.model_fields_set
    out = p.model_dump(mode="json", exclude_unset=True)
    assert set(out.keys()) == set(["data_dir"])
    assert out["data_dir"] == "/test"

    # Case: model_dir explicitly specified
    p = ProjectPaths.model_validate(dict(data_dir="/test", model_dir="/nottest"))

    # Default not applied
    assert p.model_dir == Path("/nottest")
    # model_dir included in serialization.
    assert "model_dir" in p.model_fields_set
    out = p.model_dump(mode="json", exclude_unset=True)
    assert set(out.keys()) == set(["data_dir", "model_dir"])
    assert out["data_dir"] == "/test"
    assert out["model_dir"] == "/nottest"


def test_get_all_project_paths(project_util: ProjectUtil):
    """Test retrieving all project paths from the TOML file."""
    # Setup: write some data to the toml file
    projects_data = {
        "proj1": {
            "data_dir": "/path/to/data1",
            "model_dir": "/path/to/models1",
        },
        "proj2": {
            "data_dir": "/path/to/data2",
        },
    }
    with open(project_util.config.PROJECTS_TOML_PATH, "wb") as f:
        tomli_w.dump(projects_data, f)

    # Action
    all_projects = project_util.get_all_project_paths()

    # Assert
    assert len(all_projects) == 2
    assert "proj1" in all_projects
    assert "proj2" in all_projects
    assert isinstance(all_projects["proj1"], ProjectPaths)
    assert all_projects["proj1"].data_dir == Path("/path/to/data1")
    assert all_projects["proj1"].model_dir == Path("/path/to/models1")
    assert isinstance(all_projects["proj2"], ProjectPaths)
    assert all_projects["proj2"].data_dir == Path("/path/to/data2")
    assert all_projects["proj2"].model_dir == Path("/path/to/data2/models")


def test_update_project_paths(project_util: ProjectUtil):
    """Test updating a project's paths in the TOML file."""
    # Setup
    proj_key = "proj_new"
    paths = ProjectPaths(data_dir=Path("/new/data"), model_dir=Path("/new/models"))

    # Action
    project_util.update_project_paths(proj_key, paths)

    # Assert
    all_projects = project_util.get_all_project_paths()
    assert len(all_projects) == 1
    assert proj_key in all_projects
    assert all_projects[proj_key].data_dir == Path("/new/data")
    assert all_projects[proj_key].model_dir == Path("/new/models")


def test_get_project_paths_for_model(project_util: ProjectUtil):
    """Test finding project paths for a given model directory."""
    # Setup: write some data to the toml file
    model_dir1 = Path("/path/to/models1/some_model")
    projects_data = {
        "proj1": {
            "data_dir": "/path/to/data1",
            "model_dir": "/path/to/models1",
        },
        "proj2": {
            "data_dir": "/path/to/data2",
            "model_dir": "/path/to/models2",
        },
    }
    with open(project_util.config.PROJECTS_TOML_PATH, "wb") as f:
        tomli_w.dump(projects_data, f)

    # Action
    project_paths = project_util.get_project_paths_for_model(model_dir=model_dir1)

    # Assert
    assert project_paths.data_dir == Path("/path/to/data1")

    # Test for model not in any project
    with pytest.raises(RuntimeError):
        project_util.get_project_paths_for_model(model_dir=Path("/not/a/project/model"))
