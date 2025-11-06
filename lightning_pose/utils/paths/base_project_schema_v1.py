from abc import abstractmethod, ABC
from typing import Any, TYPE_CHECKING, overload, Optional
from typing import Literal

from lightning_pose.utils.paths import ResourceType, AbstractResourceUtil

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

    def __init__(self, is_multiview: bool):
        self.is_multiview = is_multiview

    # Precise overloads for for_(), allowing static narrowing when literals are used.
    @overload
    def for_(self, resource_type: Literal[ResourceType.videos]) -> AbstractResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.video_boxes]) -> AbstractResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.frames]) -> AbstractResourceUtil["FrameKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.label_files]) -> AbstractResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.label_file_bboxes]) -> AbstractResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.center_frames]) -> AbstractResourceUtil["VideoFileKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.session_calibrations]) -> AbstractResourceUtil["SessionKey"]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.project_calibration]) -> AbstractResourceUtil[bool]: ...
    @overload
    def for_(self, resource_type: Literal[ResourceType.calibration_backups]) -> AbstractResourceUtil[tuple["SessionKey", int]]: ...

    @abstractmethod
    def for_(self, resource_type: ResourceType) -> AbstractResourceUtil[Any]:
        """Return the resource util for the given type."""
        raise NotImplementedError

    videos: AbstractResourceUtil["VideoFileKey"]
    video_boxes: AbstractResourceUtil["VideoFileKey"]
    frames: AbstractResourceUtil["FrameKey"]
    label_files: AbstractResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]
    label_file_bboxes: AbstractResourceUtil[tuple["LabelFileKey", Optional["ViewName"]]]
    center_frames: AbstractResourceUtil["VideoFileKey"]
    session_calibrations: AbstractResourceUtil["SessionKey"]
    project_calibration: AbstractResourceUtil[bool]
    calibration_backups: AbstractResourceUtil[tuple["SessionKey", int]]
