from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, TYPE_CHECKING, overload, Optional
from typing import Literal

from lightning_pose.utils.paths import ResourceType, BaseResourceUtil

if TYPE_CHECKING:
    from typing import Optional
    from lightning_pose.data.keys import (
        VideoFileKey,
        FrameKey,
        SessionKey,
        LabelFileKey,
        ViewName,
    )


class BaseProjectSchemaV1(ABC):
    is_multiview: bool
    base_dir: Path | None

    def __init__(self, is_multiview: bool, base_dir: Path | None = None):
        self.is_multiview = is_multiview
        self.base_dir = base_dir

    # Precise overloads for for_(), allowing static narrowing when literals are used.
    @overload
    def for_(self, resource_type: Literal[ResourceType.videos]) -> BaseResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.video_boxes]) -> BaseResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.frames]) -> BaseResourceUtil["FrameKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.label_files]) -> BaseResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.label_file_bboxes]) -> BaseResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.center_frames]) -> BaseResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.session_calibrations]) -> BaseResourceUtil["SessionKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.project_calibration]) -> BaseResourceUtil[bool]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.calibration_backups]) -> BaseResourceUtil[tuple["SessionKey", int]]: ...

    @abstractmethod
    def for_(self, resource_type: ResourceType) -> BaseResourceUtil[Any]:
        """Return the resource util for the given type."""
        raise NotImplementedError

    def _require_base_dir(self) -> Path:
        if self.base_dir is None:
            raise RuntimeError(
                "Schema.base_dir is None; filesystem enumeration requires a base_dir. "
                "Construct the schema via ProjectSchema.for_project(...) or pass base_dir to for_version(...)."
            )
        return self.base_dir

    videos: BaseResourceUtil["VideoFileKey"]
    video_boxes: BaseResourceUtil["VideoFileKey"]
    frames: BaseResourceUtil["FrameKey"]
    label_files: BaseResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]
    label_file_bboxes: BaseResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]
    center_frames: BaseResourceUtil["VideoFileKey"]
    session_calibrations: BaseResourceUtil["SessionKey"]
    project_calibration: BaseResourceUtil[bool]
    calibration_backups: BaseResourceUtil[tuple["SessionKey", int]]
