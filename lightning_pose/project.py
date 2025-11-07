from pathlib import Path

import tomli, tomli_w

from lightning_pose import LP_DIR
from lightning_pose.data.datatypes import ProjectConfig, ProjectDirs
from lightning_pose.data.keys import ProjectKey


def get_all_projects() -> list[ProjectDirs]:
    """Return a list of all project keys."""
    projects_toml = LP_DIR / "projects.toml"
    with open(projects_toml, 'rb') as f:
        config = tomli.load(f)
        projects: list[ProjectDirs] = []
        for key in config:
            p = ProjectDirs(project_key=ProjectKey(key),
                            data_dir=Path(config[key]["data_dir"]),
                            model_dir=Path(config[key].get("model_dir", Path(config[key]["data_dir"]) / "models")))
            projects.append(p)
        return projects


def get_project_config(project: str | ProjectKey | ProjectDirs | ProjectConfig) -> ProjectConfig:
    """Resolve multitype project arg into a ProjectConfig."""
    if isinstance(project, str) or isinstance(project, ProjectKey):
        project = ProjectKey(project)
        project = get_project_dirs(project)

    if isinstance(project, ProjectDirs):
        project = ProjectConfig.model_validate({
            "dirs": project,
            **tomli.loads((project.data_dir / "project.toml").read_text())
        })
    assert isinstance(project, ProjectConfig)
    return project


def get_project_dirs(project_key: ProjectKey) -> ProjectDirs:
    """Find the project location for a project key."""
    projects_toml = LP_DIR / "projects.toml"
    for project in get_all_projects():
        if project.project_key == project_key:
            return project
    raise FileNotFoundError(f"Project not found in {projects_toml}: {project_key}")

def set_project_dirs(project_key: str | ProjectKey, data_dir: Path, model_dir: Path | None = None):
    """Update a project's entry in projects.toml."""
    if not data_dir.is_absolute():
        raise ValueError("data_dir must be an absolute path.")
    if model_dir is not None and not model_dir.is_absolute():
        raise ValueError("model_dir must be an absolute path.")

    projects_toml = LP_DIR / "projects.toml"
    config = tomli.load(projects_toml)
    config[str(project_key)] = ProjectDirs(data_dir=data_dir, model_dir=model_dir)._asdict()
    tomli_w.dump(config, projects_toml)

def set_project_config(project: str | ProjectKey | ProjectDirs, **kwargs):
    """Update a project's config file."""
    project = get_project_config(project)
    project.update(**kwargs)
    project_toml = project.dirs.data_dir / "project.toml"
    with open(project_toml, 'wb') as f:
        tomli_w.dumps(project.model_dump(), f)