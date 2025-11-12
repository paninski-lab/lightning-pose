from __future__ import annotations
from pathlib import Path
from abc import abstractmethod, ABC
from typing import Any, TYPE_CHECKING, overload

from lightning_pose.utils.paths import ResourceType, ResourceUtil

if TYPE_CHECKING:
    from typing import Optional
    from lightning_pose.data.datatypes import (
        VideoFileKey,
        FrameKey,
        SessionKey,
        LabelFileKey,
        ViewName,
    )


class PathUtil(ABC):
    """Base class for resource path management utils."""

    @staticmethod
    def for_version(
        schema_version: int, is_multiview: bool, base_dir: Path | str | None = None
    ) -> PathUtil:
        if schema_version == 1:
            raise NotImplementedError("Not yet implemented")
        elif schema_version == 0:
            from lightning_pose.utils.paths.path_util_legacy import PathUtilLegacy

            return PathUtilLegacy(is_multiview=is_multiview, base_dir=base_dir)
        else:
            raise ValueError(f"Unrecognized version: {schema_version}")

    is_multiview: bool
    base_dir: Path | None

    def __init__(self, is_multiview: bool, base_dir: Path | None = None):
        self.is_multiview = is_multiview
        self.base_dir = base_dir

    videos: ResourceUtil[VideoFileKey]
    video_boxes: ResourceUtil[VideoFileKey]
    frames: ResourceUtil[FrameKey]
    label_files: ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]
    label_file_bboxes: ResourceUtil[tuple[LabelFileKey, Optional[ViewName]]]
    center_frames: ResourceUtil[VideoFileKey]
    calibrations: ResourceUtil[SessionKey]
    project_calibration: ResourceUtil[None]
    calibration_backups: ResourceUtil[tuple[SessionKey, int]]
