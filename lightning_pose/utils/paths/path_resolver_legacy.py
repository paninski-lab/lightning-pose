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
from lightning_pose.utils.paths.base_path_resolver_v1 import BasePathResolverV1
from lightning_pose.utils.paths import _check_relative_and_normalize, PathParseException


class PathResolverLegacy(BasePathResolverV1):
    # Legacy parser needs access to view names for parsing.
    view_names: list[str]

    def __init__(self, view_names, *args, **kwargs):
        self.view_names = view_names
        super().__init__(is_multiview=len(view_names) > 0, *args, **kwargs)

    def get_video_file_path(self, video_file_key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def get_video_bbox_path(self, video_file_key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def get_frame_path(self, key: FrameKey) -> Path:
        raise NotImplementedError()

    def get_label_file_path(self, key: LabelFileKey, view: ViewName) -> Path:
        raise NotImplementedError()

    def get_label_file_bbox_path(self, key: LabelFileKey, view: ViewName) -> Path:
        raise NotImplementedError()

    def get_center_frames_path(self, key: VideoFileKey) -> Path:
        raise NotImplementedError()

    def get_session_calibration_path(self, session_key: SessionKey | str) -> Path:
        raise NotImplementedError()

    def get_project_calibration_path(self) -> Path:
        raise NotImplementedError()

    def get_calibration_backup_path(
        self, session_key: SessionKey | str, time_ns: int
    ) -> Path:
        raise NotImplementedError()

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

    def parse_video_file_path(self, path: Path | str) -> VideoFileKey:
        """Extracts VideoFileKey."""
        path = _check_relative_and_normalize(path)

        if not path.suffix == ".mp4":
            raise PathParseException()

        # Throws away vids-from-labeled-frames
        if len(path.parent.parts) > 1:
            raise PathParseException()

        return self._parse_session_name_and_view(path.stem)

    def parse_video_bbox_path(self, path: Path | str) -> VideoFileKey:
        """Extracts VideoFileKey."""
        path = _check_relative_and_normalize(path)

        if not path.suffix == ".csv":
            raise PathParseException()
        if not path.stem.endswith("_bbox"):
            raise PathParseException()

        # First strip the _bbox suffix
        stem_without_bbox = path.stem.removesuffix("_bbox")
        return self._parse_session_name_and_view(stem_without_bbox)

    def parse_frame_path(self, path: Path | str) -> FrameKey:
        """Extracts FrameKey."""
        path = _check_relative_and_normalize(path)

        # Regex to capture the directory segment containing session and view
        # e.g., "labeled-data/sessionkey_view/" or "labeled-data/sessionkey/"
        pattern = r"[^/]*/(?P<session_view_str>[^/]+)/[a-zA-Z_-]+(?P<frameindex>\d+)\.(png|jpg)"
        m = re.search(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse label frame path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        session_view_str = m.group("session_view_str")
        video_file_key = self._parse_session_name_and_view(session_view_str)
        frame_index = int(m.group("frameindex"))

        return FrameKey(
            session_key=video_file_key.session_key,
            frame_index=frame_index,
            view=ViewName(video_file_key.view),
        )

    def parse_label_file_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        """Extracts tuple[LabelFileKey, ViewName | None]."""
        path = _check_relative_and_normalize(path)
        if path.suffix != ".csv":
            raise PathParseException()
        if "calibration" in str(path):
            raise PathParseException()
        if "bbox" in str(path):
            raise PathParseException()

        video_file_key = self._parse_session_name_and_view(path.stem)

        prefix = "_".join(path.parent.parts)
        labelfilekey = "_".join(
            token for token in (prefix, video_file_key.session_key) if token
        )
        return LabelFileKey(labelfilekey), video_file_key.view

    def parse_label_file_bbox_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if not path.name.startswith("bboxes_"):
            raise PathParseException()

        video_file_key = self._parse_session_name_and_view(path.stem)

        labelfilekey = LabelFileKey(video_file_key.session_key.replace("bboxes", "CollectedData"))
        return labelfilekey, video_file_key.view

    def parse_center_frames_path(self, path: Path | str) -> VideoFileKey:
        """Extracts VideoFileKey."""
        path = _check_relative_and_normalize(path)
        # Regex to capture the directory segment containing session and view
        # e.g., "labeled-data/sessionkey_view/" or "labeled-data/sessionkey/"
        pattern = r"labeled-data/(?P<session_view_str>[^/]+)/center_frames\.txt"

        m = re.search(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse center frames path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        session_view_str = m.group("session_view_str")
        return self._parse_session_name_and_view(session_view_str)

    def parse_session_calibration_path(self, path: Path | str) -> SessionKey:
        """Extracts SessionKey."""
        path = _check_relative_and_normalize(path)
        pattern = r"calibrations/(?P<session>[^/]+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse session calibration path: {path.as_posix()}"
            )

        return SessionKey(m.group("session"))

    def is_project_calibration_path(self, path: Path | str) -> bool:
        """Checks if path matches calibration.toml"""
        path = _check_relative_and_normalize(path)
        pattern = r"calibration\.toml"
        return bool(re.match(pattern, path.as_posix()))

    def parse_calibration_backup_path(self, path: Path | str) -> tuple[SessionKey, int]:
        """Extracts tuple[SessionKey, int]."""
        path = _check_relative_and_normalize(path)
        pattern = r"calibration_backups/(?P<session>[^/]+)\.(?P<time>\d+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse calibration backup path: {path.as_posix()}"
            )
        return SessionKey(m.group("session")), int(m.group("time"))
