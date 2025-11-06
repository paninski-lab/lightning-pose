from pathlib import Path
from lightning_pose.data.datatypes import ProjectConfig, ProjectDirs
from lightning_pose.data.keys import ProjectKey
from lightning_pose.project import get_project_config
from lightning_pose.utils.paths.base_project_schema_v1 import BaseProjectSchemaV1


class ProjectSchema:
    """
    Utility class for path management.
    """


    @staticmethod
    def for_version(schema_version: int, is_multiview: bool, base_dir: Path | str | None = None) -> BaseProjectSchemaV1:
        if schema_version == 1:
            from lightning_pose.utils.paths.project_schema_v1 import ProjectSchemaV1
            return ProjectSchemaV1(is_multiview=is_multiview, base_dir=Path(base_dir) if base_dir is not None else None)
        elif schema_version == 0:
            raise NotImplementedError("Migrate the directory to a project to use this util")
        else:
            raise ValueError(f"Unrecognized version: {schema_version}")


    @staticmethod
    def for_project(project: str | ProjectKey | ProjectDirs | ProjectConfig) -> BaseProjectSchemaV1:
        """Creates a PathResolver instance for a given project."""
        project_config: ProjectConfig = get_project_config(project)

        version = project_config.schema_version
        view_names = project_config.view_names or []
        is_multiview = len(view_names) > 0

        if version == 0:
            from lightning_pose.utils.paths.project_schema_legacy import ProjectSchemaLegacy
            return ProjectSchemaLegacy(view_names=view_names)
        else:
            base_dir = project_config.dirs.data_dir if project_config.dirs and project_config.dirs.data_dir else None
            return ProjectSchema.for_version(schema_version=version, is_multiview=is_multiview, base_dir=base_dir)
