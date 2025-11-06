from lightning_pose.data.datatypes import ProjectConfig, ProjectDirs
from lightning_pose.data.keys import ProjectKey
from lightning_pose.project import get_project_config
from lightning_pose.utils.paths.base_project_schema_v1 import BaseProjectSchemaV1


class ProjectSchema:
    """
    Utility class for path management.
    """


    @staticmethod
    def for_version(schema_version: int, is_multiview: bool) -> BaseProjectSchemaV1:
        if schema_version == 1:
            from lightning_pose.utils.paths.project_schema_v1 import ProjectSchemaV1
            return ProjectSchemaV1(is_multiview=is_multiview)
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
            return ProjectSchema.for_version(schema_version=version, is_multiview=is_multiview)
