from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable, Dict, Any, Union # Add Union to imports

from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    SessionKey,
    LabelFileKey,
    ViewName,
)
# Import PathTypes and PathTypeHandler from the same package
from lightning_pose.utils.paths import PathType, PathTypeHandler


class BasePathResolverV1(ABC):
    _GET_PATH_MAP_TEMPL: Dict[PathType, str] = {
        PathType.videos: "get_video_file_path",
        PathType.video_boxes: "get_video_bbox_path",
        PathType.label_files: "get_label_file_path",
        PathType.label_file_bboxes: "get_label_file_bbox_path",
        PathType.frames: "get_frame_path",
        PathType.session_calibrations: "get_session_calibration_path",
        PathType.calibration_backups: "get_calibration_backup_path",
        PathType.center_frames: "get_center_frames_path",
        PathType.project_calibration: "get_project_calibration_path",
    }

    _PARSE_PATH_MAP_TEMPL: Dict[PathType, str] = {
        PathType.videos: "parse_video_file_path",
        PathType.video_boxes: "parse_video_bbox_path",
        PathType.label_files: "parse_label_file_path",
        PathType.label_file_bboxes: "parse_label_file_bbox_path",
        PathType.frames: "parse_frame_path",
        PathType.session_calibrations: "parse_session_calibration_path",
        PathType.calibration_backups: "parse_calibration_backup_path",
        PathType.center_frames: "parse_center_frames_path",
        PathType.project_calibration: "is_project_calibration_path", # Special case for project calibration
    }

    is_multiview: bool

    def __init__(self, is_multiview: bool):
        self.is_multiview = is_multiview
        # Maps PathType to the actual bound get_ method
        self.get_path_map: Dict[PathType, Callable[..., Path]] = {}
        # Maps PathType to the actual bound parse_ method
        self.parse_path_map: Dict[PathType, Callable[[Union[Path, str]], Any]] = {}
        self._build_path_maps()

    def _build_path_maps(self) -> None:
        """Populates the get_path_map and parse_path_map with bound methods."""
        for path_type, method_name in self._GET_PATH_MAP_TEMPL.items():
            # Use getattr to get the actual method from the instance (self)
            # This ensures that if a subclass overrides a method, the map will
            # point to the subclass's implementation.
            self.get_path_map[path_type] = getattr(self, method_name)
        for path_type, method_name in self._PARSE_PATH_MAP_TEMPL.items():
            self.parse_path_map[path_type] = getattr(self, method_name)

    def for_(self, path_type: PathType) -> PathTypeHandler[Any]:
        """
        Returns a PathTypeHandler for a specific PathType, providing .get() and .reverse() methods.
        """
        if path_type not in self.get_path_map or path_type not in self.parse_path_map:
            raise ValueError(f"PathType {path_type.value} not supported by this parser.")
        return PathTypeHandler(
            get_func=self.get_path_map[path_type],
            parse_func=self.parse_path_map[path_type]
        )

    @abstractmethod
    def get_video_file_path(self, video_file_key: VideoFileKey) -> Path:
        pass

    @abstractmethod
    def parse_video_file_path(self, path: Path | str) -> VideoFileKey:
        pass

    @abstractmethod
    def get_video_bbox_path(self, video_file_key: VideoFileKey) -> Path:
        pass

    @abstractmethod
    def parse_video_bbox_path(self, path: Path | str) -> VideoFileKey:
        pass

    @abstractmethod
    def get_frame_path(self, key: FrameKey) -> Path:
        pass

    @abstractmethod
    def parse_frame_path(self, path: Path | str) -> FrameKey:
        pass

    @abstractmethod
    def get_label_file_path(self, key: LabelFileKey | str, view: ViewName | str) -> Path:
        pass

    @abstractmethod
    def parse_label_file_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        pass

    @abstractmethod
    def get_label_file_bbox_path(self, key: LabelFileKey |str, view: ViewName | str) -> Path:
        pass

    @abstractmethod
    def parse_label_file_bbox_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        pass

    @abstractmethod
    def get_center_frames_path(self, key: VideoFileKey) -> Path:
        pass

    @abstractmethod
    def parse_center_frames_path(self, path: Path | str) -> VideoFileKey:
        pass

    @abstractmethod
    def get_session_calibration_path(self, session_key: str) -> Path:
        pass

    @abstractmethod
    def parse_session_calibration_path(self, path: Path | str) -> SessionKey:
        pass

    @abstractmethod
    def get_project_calibration_path(self) -> Path:
        pass

    @abstractmethod
    def get_calibration_backup_path(
        self, session_key: SessionKey | str, time_ns: int
    ) -> Path:
        pass

    @abstractmethod
    def is_project_calibration_path(self, path: Path | str) -> bool:
        pass

    @abstractmethod
    def parse_calibration_backup_path(self, path: Path | str) -> tuple[SessionKey, int]:
        pass
