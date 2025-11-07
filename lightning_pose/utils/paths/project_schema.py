from __future__ import annotations
from pathlib import Path
from lightning_pose.data.datatypes import ProjectConfig, ProjectDirs
from lightning_pose.data.keys import ProjectKey
from lightning_pose.project import get_project_config
from typing import Literal
from abc import abstractmethod, ABC
from typing import Any, TYPE_CHECKING, overload

from lightning_pose.utils.paths import ResourceType, ResourceUtil

if TYPE_CHECKING:
    from typing import Optional
    from lightning_pose.data.keys import (
        VideoFileKey,
        FrameKey,
        SessionKey,
        LabelFileKey,
        ViewName,
    )


class ProjectSchema(ABC):
    """Base class for resource path management utils."""

    @staticmethod
    def for_version(schema_version: int, is_multiview: bool, base_dir: Path | str | None = None) -> ProjectSchema:
        """Gets a ProjectSchema instance for the manually specified arguments."""
        if schema_version == 1:
            from lightning_pose.utils.paths.project_schema_v1 import ProjectSchemaV1
            return ProjectSchemaV1(is_multiview=is_multiview, base_dir=Path(base_dir) if base_dir is not None else None)
        elif schema_version == 0:
            raise NotImplementedError("Migrate the directory to a project to use this util")
        else:
            raise ValueError(f"Unrecognized version: {schema_version}")


    @staticmethod
    def for_project(project: str | ProjectKey | ProjectDirs | ProjectConfig) -> ProjectSchema:
        """Gets a ProjectSchema instance for a project and sets up `base_dir` for enumerating resources in the project."""
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
    is_multiview: bool
    base_dir: Path | None

    def __init__(self, is_multiview: bool, base_dir: Path | None = None):
        self.is_multiview = is_multiview
        self.base_dir = base_dir

    # Precise overloads for for_(), allowing static narrowing when literals are used.
    @overload
    def for_(self, resource_type: Literal[ResourceType.VIDEO]) -> ResourceUtil[VideoFileKey]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.VIDEO_BBOX]) -> ResourceUtil[VideoFileKey]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.FRAME]) -> ResourceUtil[FrameKey]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.LABEL_FILE]) -> ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.LABEL_FILE_BBOX]) -> ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.CENTER_FRAME_LIST]) -> ResourceUtil[VideoFileKey]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.SESSION_CALIBRATION]) -> ResourceUtil[SessionKey]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.PROJECT_CALIBRATION]) -> ResourceUtil[None]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.CALIBRATION_BACKUP]) -> ResourceUtil[tuple[SessionKey, int]]: ...

    @abstractmethod
    def for_(self, resource_type: ResourceType) -> ResourceUtil[Any]:
        """Return the resource util for the given type."""
        raise NotImplementedError

    def _require_base_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError(
                "Schema.base_dir is None; filesystem enumeration requires a base_dir. "
                "Construct the schema via ProjectSchema.for_project(...) or pass base_dir to for_version(...)."
            )
        return self.base_dir

    videos: ResourceUtil[VideoFileKey]
    video_boxes: ResourceUtil[VideoFileKey]
    frames: ResourceUtil[FrameKey]
    label_files: ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]
    label_file_bboxes: ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]
    center_frames: ResourceUtil[VideoFileKey]
    session_calibrations: ResourceUtil[SessionKey]
    project_calibration: ResourceUtil[None]
    calibration_backups: ResourceUtil[tuple[SessionKey, int]]
