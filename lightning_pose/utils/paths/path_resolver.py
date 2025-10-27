from lightning_pose.data.datatypes import ProjectConfig, ProjectDirs
from lightning_pose.data.keys import ProjectKey
from lightning_pose.project import get_project_config
from lightning_pose.utils.paths.base_path_resolver_v1 import BasePathResolverV1


class PathResolver:
    """
    Utility class for path management.
    """

    @staticmethod
    def for_project(project: str | ProjectKey | ProjectDirs | ProjectConfig) -> BasePathResolverV1:
        project_config: ProjectConfig = get_project_config(project)

        version = project_config.schema_version
        view_names = project_config.view_names or []
        is_multiview = len(view_names) > 0

        if version == 1:
            from lightning_pose.utils.paths.path_resolver_v1 import PathResolverV1
            return PathResolverV1(is_multiview=is_multiview)
        elif version == 0:
            from lightning_pose.utils.paths.path_resolver_legacy import PathResolverLegacy
            return PathResolverLegacy(view_names=view_names)
        else:
            raise ValueError(f"Unrecognized version: {version}")
