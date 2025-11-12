import re
from pathlib import Path

from lightning_pose.data.datatypes import (
    VideoFileKey,
    FrameKey,
    SessionKey,
    ViewName,
    LabelFileKey,
)
from lightning_pose.utils.paths import _check_relative_and_normalize, PathParseException
from lightning_pose.utils.paths.path_util import PathUtil


class PathUtilLegacy(PathUtil):
    """Parser for paths in projects before V1 strict schema."""

    # Legacy parser needs access to view names for parsing.
    view_names: list[str]

    def __init__(self, view_names, *args, **kwargs):
        self.view_names = view_names
        super().__init__(is_multiview=len(view_names) > 0, *args, **kwargs)

        # Explicit resource utils for legacy behavior (public attributes)
        self.videos = _LegacyVideoUtil(self)
        self.video_boxes = _LegacyVideoBBoxUtil(self)
        self.frames = _LegacyFrameUtil(self)
        self.label_files = _LegacyLabelFileUtil(self)
        self.label_file_bboxes = _LegacyLabelFileBBoxUtil(self)
        self.center_frames = _LegacyCenterFramesUtil(self)
        self.calibrations = _LegacyCalibrationUtil(self)
        self.project_calibration = _LegacyProjectCalibrationUtil(self)
        self.calibration_backups = _LegacyCalibrationBackupUtil(self)

        # Lazy import to avoid circulars at module import time
        from lightning_pose.utils.paths import ResourceType, ResourceUtil  # type: ignore

        self._resource_map: dict[ResourceType, ResourceUtil] = {
            ResourceType.VIDEO: self.videos,
            ResourceType.VIDEO_BBOX: self.video_boxes,
            ResourceType.FRAME: self.frames,
            ResourceType.LABEL_FILE: self.label_files,
            ResourceType.LABEL_FILE_BBOX: self.label_file_bboxes,
            ResourceType.CENTER_FRAME_LIST: self.center_frames,
            ResourceType.CALIBRATION: self.calibrations,
            ResourceType.PROJECT_CALIBRATION: self.project_calibration,
            ResourceType.CALIBRATION_BACKUP: self.calibration_backups,
        }

    # Inline for_ returning resource utils
    def for_(self, resource_type):
        return self._resource_map[resource_type]

    def _parse_session_name_and_view(self, potential_session_view_str: str) -> VideoFileKey:
        """Parses a string (like a filename stem or directory name) to extract session key and view.

        Args:
            potential_session_view_str: A string that might contain both the session key and view name,
                                        e.g., "mouse_session_top" or "mouse_session".

        Returns:
            A VideoFileKey containing the extracted session_key and view.
        """
        potential_session_view_str = str(potential_session_view_str)
        if not self.is_multiview:
            return VideoFileKey(session_key=SessionKey(potential_session_view_str), view=None)

        # In multiview projects, try to find the matching view name
        for view_name in self.view_names:
            if f"_{view_name}" in potential_session_view_str:
                # Replace the view name suffix to get the session key
                session_key = potential_session_view_str.replace(f"_{view_name}", "")
                return VideoFileKey(session_key=SessionKey(session_key), view=ViewName(view_name))

        raise PathParseException()


# ---------------------------------------------------------------------------
# Resource util classes implementing get()/reverse()
# ---------------------------------------------------------------------------
from lightning_pose.utils.paths import ResourceUtil, ResourceType  # type: ignore


class _LegacyVideoUtil(ResourceUtil[VideoFileKey]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if not path.suffix == ".mp4":
            raise PathParseException()
        # Throws away vids-from-labeled-frames
        if len(path.parent.parts) > 1:
            raise PathParseException()
        return self._schema._parse_session_name_and_view(path.stem)


class _LegacyVideoBBoxUtil(ResourceUtil[VideoFileKey]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if not path.suffix == ".csv":
            raise PathParseException()
        if not path.stem.endswith("_bbox"):
            raise PathParseException()
        stem_without_bbox = path.stem.removesuffix("_bbox")
        return self._schema._parse_session_name_and_view(stem_without_bbox)


class _LegacyFrameUtil(ResourceUtil[FrameKey]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: FrameKey) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> FrameKey:
        path = _check_relative_and_normalize(path)
        # Regex to capture the directory segment containing session and view
        # e.g., "labeled-data/sessionkey_view/" or "labeled-data/sessionkey/"
        pattern = r"[^/]*/(?P<session_view_str>[^/]+)/[a-zA-Z_-]+(?P<frameindex>\d+)\.(png|jpg)"
        m = re.search(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse label frame path: {path.as_posix()}, multiview={self._schema.is_multiview}"
            )
        session_view_str = m.group("session_view_str")
        video_file_key = self._schema._parse_session_name_and_view(session_view_str)
        frame_index = int(m.group("frameindex"))
        return FrameKey(
            session_key=video_file_key.session_key,
            frame_index=frame_index,
            view=video_file_key.view,
        )


class _LegacyLabelFileUtil(ResourceUtil[tuple[LabelFileKey, ViewName | None]]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key_view: tuple[LabelFileKey, ViewName | None]) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if path.suffix != ".csv":
            raise PathParseException()
        if "calibration" in str(path):
            raise PathParseException()
        if "bbox" in str(path):
            raise PathParseException()
        video_file_key = self._schema._parse_session_name_and_view(path.stem)
        prefix = "_".join(path.parent.parts)
        labelfilekey = "_".join(token for token in (prefix, video_file_key.session_key) if token)
        return LabelFileKey(labelfilekey), video_file_key.view


class _LegacyLabelFileBBoxUtil(ResourceUtil[tuple[LabelFileKey, ViewName | None]]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key_view: tuple[LabelFileKey, ViewName | None]) -> Path:
        label_file_key, view = key_view
        # bboxes_lTop.csv, bboxes_new.csv
        tokens = ("bboxes", label_file_key.replace("CollectedData", ""), view)
        return Path("_".join(tokens) + ".csv")

    def parse_path(self, path: Path | str) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if not Path(path.as_posix()).name.startswith("bboxes_"):
            raise PathParseException()
        video_file_key = self._schema._parse_session_name_and_view(Path(path.as_posix()).stem)
        labelfilekey = LabelFileKey(
            str(video_file_key.session_key).replace("bboxes", "CollectedData")
        )
        return labelfilekey, video_file_key.view


class _LegacyCenterFramesUtil(ResourceUtil[VideoFileKey]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        pattern = r"labeled-data/(?P<session_view_str>[^/]+)/center_frames\.txt"
        m = re.search(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse center frames path: {path.as_posix()}, multiview={self._schema.is_multiview}"
            )
        session_view_str = m.group("session_view_str")
        return self._schema._parse_session_name_and_view(session_view_str)


class _LegacyCalibrationUtil(ResourceUtil[SessionKey]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: SessionKey) -> Path:
        return Path("calibrations") / f"{key}.toml"

    def parse_path(self, path: Path | str) -> SessionKey:
        path = _check_relative_and_normalize(path)
        pattern = r"calibrations/(?P<session>[^/]+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse session calibration path: {path.as_posix()}"
            )
        return SessionKey(m.group("session"))


class _LegacyProjectCalibrationUtil(ResourceUtil[None]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self) -> Path:  # type: ignore[override]
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> None:
        path = _check_relative_and_normalize(path)
        pattern = r"calibration\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException()
        return None


class _LegacyCalibrationBackupUtil(ResourceUtil[tuple[SessionKey, int]]):
    def __init__(self, schema: "PathUtilLegacy"):
        self._schema = schema

    def get_path(self, key: tuple[SessionKey, int]) -> Path:
        raise NotImplementedError()

    def parse_path(self, path: Path | str) -> tuple[SessionKey, int]:
        path = _check_relative_and_normalize(path)
        pattern = r"calibration_backups/(?P<session>[^/]+)\.(?P<time>\d+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(f"Could not parse calibration backup path: {path.as_posix()}")
        return SessionKey(m.group("session")), int(m.group("time"))
