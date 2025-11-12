from pathlib import Path

import tomli, tomli_w
from lightning_pose.rootconfig import RootConfig
from lightning_pose.data.datatypes import ProjectPaths


class ProjectUtil:
    config: RootConfig

    def __init__(self, config: RootConfig):
        self.config = config

    def _read_projects_toml(self) -> dict:
        """Read the projects.toml file and return its contents."""
        with open(self.config.PROJECTS_TOML_PATH, "rb") as f:
            return tomli.load(f)

    def _write_projects_toml(self, data: dict):
        """Write data to the projects.toml file."""
        with open(self.config.PROJECTS_TOML_PATH, "wb") as f:
            tomli_w.dump(data, f)

    ####################################
    # Functions for project management
    ####################################
    def get_all_project_paths(self) -> dict[str, ProjectPaths]:
        """Return a dictionary containing all registered project configurations."""
        config = self._read_projects_toml()
        return {key: ProjectPaths.model_validate(config[key]) for key in config}

    def get_project_paths_for_model(self, model_dir: Path) -> ProjectPaths:
        """Finds the project paths for a given model directory by searching through
        all project paths in projects.toml."""
        for key, project_paths in self.get_all_project_paths().items():
            if model_dir.is_relative_to(project_paths.model_dir):
                return project_paths
        raise RuntimeError(f"Model dir {model_dir} not part of any project in projects.toml.")

    def update_project_paths(
        self,
        project_key: str,
        projectpaths: ProjectPaths | None,
    ):
        """Update a project's entry in projects.toml.

        Args:
            project_key (str): Key identifier for the project in the TOML file
            projectpaths (ProjectPaths | None): Project paths to update (or add if not yet in config).
                If None, the project entry will be deleted
        """
        projects_toml_dict = self._read_projects_toml()
        if projectpaths is None:
            del projects_toml_dict[str(project_key)]
        else:
            projects_toml_dict[str(project_key)] = projectpaths.model_dump(
                mode="json", exclude_unset=True
            )
        self._write_projects_toml(projects_toml_dict)

    ####################################
    # Functions for individual projects
    ####################################
    def get_project_yaml_path(self, data_dir: Path) -> Path:
        """Gets the path to project's yaml config for a project."""
        return data_dir / "project.yaml"
