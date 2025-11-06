import re
import sys
from pathlib import Path
from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    SessionKey,
    ViewName,
    LabelFileKey,
)
from lightning_pose.utils.paths.base_project_schema_v1 import BaseProjectSchemaV1
from lightning_pose.utils.paths import _check_relative_and_normalize, PathParseException


class ProjectSchemaLegacy(BaseProjectSchemaV1):
    # Legacy parser needs access to view names for parsing.
    view_names: list[str]

    def __init__(self, view_names, *args, **kwargs):
        self.view_names = view_names
        super().__init__(is_multiview=len(view_names) > 0, *args, **kwargs)

        # Explicit resource utils for legacy behavior
        self._videos = _LegacyVideoUtil(self)
        self._video_boxes = _LegacyVideoBBoxUtil(self)
        self._frames = _LegacyFrameUtil(self)
        self._label_files = _LegacyLabelFileUtil(self)
        self._label_file_bboxes = _LegacyLabelFileBBoxUtil(self)
        self._center_frames = _LegacyCenterFramesUtil(self)
        self._session_calibrations = _LegacySessionCalibrationUtil(self)
        self._project_calibration = _LegacyProjectCalibrationUtil(self)
        self._calibration_backups = _LegacyCalibrationBackupUtil(self)

        # Lazy import to avoid circulars at module import time
        from lightning_pose.utils.paths import ResourceType, AbstractResourceUtil  # type: ignore
        self._resource_map: dict[ResourceType, AbstractResourceUtil] = {
            ResourceType.videos: self._videos,
            ResourceType.video_boxes: self._video_boxes,
            ResourceType.frames: self._frames,
            ResourceType.label_files: self._label_files,
            ResourceType.label_file_bboxes: self._label_file_bboxes,
            ResourceType.center_frames: self._center_frames,
            ResourceType.session_calibrations: self._session_calibrations,
            ResourceType.project_calibration: self._project_calibration,
            ResourceType.calibration_backups: self._calibration_backups,
        }

    # Inline for_ returning resource utils
    def for_(self, resource_type):
        return self._resource_map[resource_type]

    def _parse_session_name_and_view(
        self, potential_session_view_str: str
    ) -> VideoFileKey:
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
# Explicit legacy resource util classes implementing get()/reverse()
# ---------------------------------------------------------------------------
from lightning_pose.utils.paths import AbstractResourceUtil, ResourceType  # type: ignore


class _LegacyVideoUtil(AbstractResourceUtil[VideoFileKey]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: VideoFileKey) -> Path:
        if self._schema.is_multiview:
            return Path(f"{key.session_key}_{key.view}.mp4")
        else:
            return Path(f"{key.session_key}.mp4")

    def reverse(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if not path.suffix == ".mp4":
            raise PathParseException()
        # Throws away vids-from-labeled-frames
        if len(path.parent.parts) > 1:
            raise PathParseException()
        return self._schema._parse_session_name_and_view(path.stem)


class _LegacyVideoBBoxUtil(AbstractResourceUtil[VideoFileKey]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: VideoFileKey) -> Path:
        if self._schema.is_multiview:
            return Path(f"{key.session_key}_{key.view}_bbox.csv")
        else:
            return Path(f"{key.session_key}_bbox.csv")

    def reverse(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if not path.suffix == ".csv":
            raise PathParseException()
        if not path.stem.endswith("_bbox"):
            raise PathParseException()
        stem_without_bbox = path.stem.removesuffix("_bbox")
        return self._schema._parse_session_name_and_view(stem_without_bbox)


class _LegacyFrameUtil(AbstractResourceUtil[FrameKey]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: FrameKey) -> Path:
        # Choose a conservative legacy-friendly location
        if self._schema.is_multiview:
            dir_name = f"{key.session_key}_{key.view}"
        else:
            dir_name = f"{key.session_key}"
        # Use a generic filename that matches the legacy parser pattern
        return Path("labeled-data") / dir_name / f"frame_{key.frame_index}.png"

    def reverse(self, path: Path | str) -> FrameKey:
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


class _LegacyLabelFileUtil(AbstractResourceUtil[tuple[LabelFileKey, ViewName | None]]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key_view: tuple[LabelFileKey, ViewName | None]) -> Path:
        # Legacy label file naming is dataset-specific; generation is not supported.
        raise NotImplementedError("Legacy label file path generation is not supported.")

    def reverse(self, path: Path | str) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if path.suffix != ".csv":
            raise PathParseException()
        if "calibration" in str(path):
            raise PathParseException()
        if "bbox" in str(path):
            raise PathParseException()
        video_file_key = self._schema._parse_session_name_and_view(path.stem)
        prefix = "_".join(path.parent.parts)
        labelfilekey = "_".join(
            token for token in (prefix, video_file_key.session_key) if token
        )
        return LabelFileKey(labelfilekey), video_file_key.view


class _LegacyLabelFileBBoxUtil(AbstractResourceUtil[tuple[LabelFileKey, ViewName | None]]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key_view: tuple[LabelFileKey, ViewName | None]) -> Path:
        # Legacy label bbox naming is dataset-specific; generation is not supported.
        raise NotImplementedError("Legacy label bbox path generation is not supported.")

    def reverse(self, path: Path | str) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if not Path(path.as_posix()).name.startswith("bboxes_"):
            raise PathParseException()
        video_file_key = self._schema._parse_session_name_and_view(Path(path.as_posix()).stem)
        labelfilekey = LabelFileKey(str(video_file_key.session_key).replace("bboxes", "CollectedData"))
        return labelfilekey, video_file_key.view


class _LegacyCenterFramesUtil(AbstractResourceUtil[VideoFileKey]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: VideoFileKey) -> Path:
        if self._schema.is_multiview:
            dir_name = f"{key.session_key}_{key.view}"
        else:
            dir_name = f"{key.session_key}"
        return Path("labeled-data") / dir_name / "center_frames.txt"

    def reverse(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        pattern = r"labeled-data/(?P<session_view_str>[^/]+)/center_frames\.txt"
        m = re.search(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse center frames path: {path.as_posix()}, multiview={self._schema.is_multiview}"
            )
        session_view_str = m.group("session_view_str")
        return self._schema._parse_session_name_and_view(session_view_str)


class _LegacySessionCalibrationUtil(AbstractResourceUtil[SessionKey]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: SessionKey) -> Path:
        return Path("calibrations") / f"{key}.toml"

    def reverse(self, path: Path | str) -> SessionKey:
        path = _check_relative_and_normalize(path)
        pattern = r"calibrations/(?P<session>[^/]+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse session calibration path: {path.as_posix()}"
            )
        return SessionKey(m.group("session"))


class _LegacyProjectCalibrationUtil(AbstractResourceUtil[bool]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self) -> Path:  # type: ignore[override]
        # Legacy project calibration file name
        return Path("calibration.toml")

    def reverse(self, path: Path | str) -> bool:
        path = _check_relative_and_normalize(path)
        pattern = r"calibration\.toml"
        return bool(re.match(pattern, path.as_posix()))


class _LegacyCalibrationBackupUtil(AbstractResourceUtil[tuple[SessionKey, int]]):
    def __init__(self, schema: "ProjectSchemaLegacy"):
        self._schema = schema

    def get(self, key: tuple[SessionKey, int]) -> Path:
        session_key, time_ns = key
        return Path("calibration_backups") / f"{session_key}.{time_ns}.toml"

    def reverse(self, path: Path | str) -> tuple[SessionKey, int]:
        path = _check_relative_and_normalize(path)
        pattern = r"calibration_backups/(?P<session>[^/]+)\.(?P<time>\d+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse calibration backup path: {path.as_posix()}"
            )
        return SessionKey(m.group("session")), int(m.group("time"))
