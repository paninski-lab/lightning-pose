import re
from pathlib import Path

from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    SessionKey,
    LabelFileKey,
    ViewName,
)
from lightning_pose.utils.paths import PathParseException, _check_relative_and_normalize
from lightning_pose.utils.paths.base_path_resolver_v1 import BasePathResolverV1


class PathResolverV1(BasePathResolverV1):
    def get_video_file_path(self, video_file_key: VideoFileKey) -> Path:
        """videos/<SessionKey>_<View>.mp4"""
        if self.is_multiview:
            return (
                Path("videos")
                / f"{video_file_key.session_key}_{video_file_key.view}.mp4"
            )
        else:
            return Path("videos") / f"{video_file_key.session_key}.mp4"

    def parse_video_file_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            pattern = r"videos/(?P<session>[^/]+)_(?P<view>[^/_]+)\.mp4"
        else:
            pattern = r"videos/(?P<session>[^/]+)\.mp4"

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse video file path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        if self.is_multiview:
            return VideoFileKey(
                session_key=m.group("session"),
                view=m.group("view"),
            )
        else:
            return VideoFileKey(
                session_key=m.group("session"),
                view=None,
            )

    def get_video_bbox_path(self, video_file_key: VideoFileKey) -> Path:
        """videos/<SessionKey>_<View>_bbox.csv"""
        if self.is_multiview:
            return (
                Path("videos")
                / f"{video_file_key.session_key}_{video_file_key.view}_bbox.csv"
            )
        else:
            return Path("videos") / f"{video_file_key.session_key}_bbox.csv"

    def parse_video_bbox_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            pattern = r"videos/(?P<session>[^/]+)_(?P<view>[^/_]+)_bbox\.csv"
        else:
            pattern = r"videos/(?P<session>[^/]+)_bbox\.csv"

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse video bbox path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        if self.is_multiview:
            return VideoFileKey(
                session_key=m.group("session"),
                view=m.group("view"),
            )
        else:
            return VideoFileKey(
                session_key=m.group("session"),
                view=None,
            )

    def get_frame_path(self, key: FrameKey) -> Path:
        """labeled-data/frames/<SessionKey>_<View>/frame_<FrameIndex>.png"""
        if self.is_multiview:
            # Example: labeled-data/frames/sessionkey_view/frame_123.png
            if len(str(key.frame_index)) > 8:
                raise ValueError("Frame index requires more than 8 digits")
            return (
                Path("labeled-data")
                / "frames"
                / f"{key.session_key}_{key.view}"
                / f"frame_{key.frame_index:08d}.png"
            )
        else:
            # Example: labeled-data/frames/sessionkey/frame_123.png
            if len(str(key.frame_index)) > 8:
                raise PathParseException("Frame index requires more than 8 digits")
            return (
                Path("labeled-data")
                / "frames"
                / f"{key.session_key}"
                / f"frame_{key.frame_index:08d}.png"
            )

    def parse_frame_path(self, path: Path | str) -> FrameKey:
        """Extracts FrameKey."""
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            # Example: labeled-data/frames/sessionkey_view/frame_123.png
            pattern = r"labeled-data/frames/(?P<session>[^/]+)_(?P<view>[^/_]+)/frame_(?P<frameindex>\d+)\.png"
        else:
            # Example: labeled-data/sessionkey/frame_123.png
            pattern = (
                r"labeled-data/frames/(?P<session>[^/]+)/frame_(?P<frameindex>\d+)\.png"
            )

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse label frame path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        session_key = m.group("session")
        frame_index = int(m.group("frameindex"))  # Convert frame index to integer

        if self.is_multiview:
            view = m.group("view")
            return FrameKey(session_key=session_key, frame_index=frame_index, view=view)
        else:
            return FrameKey(session_key=session_key, frame_index=frame_index, view=None)

    def get_label_file_path(self, key: LabelFileKey, view: ViewName) -> Path:
        """labeled-data/labels/<LabelFileKey>_<ViewName>.csv"""
        if key.endswith("_bbox"):
            raise ValueError("Invalid key")
        if self.is_multiview:
            return Path("labeled-data/labels") / f"{key}_{view}.csv"
        else:
            return Path("labeled-data/labels") / f"{key}.csv"

    def parse_label_file_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            pattern = (
                r"labeled-data/labels/(?P<labelfilekey>[^/]+)_(?P<view>[^/_]+)\.csv"
            )
        else:
            pattern = r"labeled-data/labels/(?P<labelfilekey>[^/]+)\.csv"

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse label file path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        if self.is_multiview:
            return m.group("labelfilekey"), m.group("view")
        else:
            return m.group("labelfilekey"), None

    def get_label_file_bbox_path(self, key: LabelFileKey, view: ViewName) -> Path:
        """labeled-data/labels/<LabelFileKey>_<ViewName>_bbox.csv"""
        if self.is_multiview:
            return Path("labeled-data") / "labels" / f"{key}_{view}_bbox.csv"
        else:
            return Path("labeled-data") / "labels" / f"{key}_bbox.csv"

    def parse_label_file_bbox_path(
        self, path: Path | str
    ) -> tuple[LabelFileKey, ViewName | None]:
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            pattern = r"labeled-data/labels/(?P<labelfilekey>[^/]+)_(?P<view>[^/_]+)_bbox\.csv"
        else:
            pattern = r"labeled-data/labels/(?P<labelfilekey>[^/]+)_bbox\.csv"

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse label file bbox path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        return (
            (m.group("labelfilekey"), m.group("view"))
            if self.is_multiview
            else (m.group("labelfilekey"), None)
        )

    def get_center_frames_path(self, key: VideoFileKey) -> Path:
        """labeled-data/frames/<SessionKey>_<View>/center_frames.txt"""
        if self.is_multiview:
            return (
                Path("labeled-data")
                / "frames"
                / f"{key.session_key}_{key.view}"
                / "center_frames.txt"
            )
        else:
            return Path("labeled-data") / f"{key.session_key}" / "center_frames.txt"

    def parse_center_frames_path(self, path: Path | str) -> VideoFileKey:
        path = _check_relative_and_normalize(path)
        if self.is_multiview:
            pattern = r"labeled-data/frames/(?P<session>[^/]+)_(?P<view>[^/_]+)/center_frames\.txt"
        else:
            pattern = r"labeled-data/frames/(?P<session>[^/]+)/center_frames\.txt"

        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse center frames path: {path.as_posix()}, multiview={self.is_multiview}"
            )

        if self.is_multiview:
            return VideoFileKey(session_key=m.group("session"), view=m.group("view"))
        else:
            return VideoFileKey(session_key=m.group("session"), view=None)

    def get_session_calibration_path(self, session_key: SessionKey | str) -> Path:
        """calibrations/<SessionKey>.toml"""
        return Path("calibrations") / f"{session_key}.toml"

    def parse_session_calibration_path(self, path: Path | str) -> SessionKey:
        path = _check_relative_and_normalize(path)
        pattern = r"calibrations/(?P<session>[^/]+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse session calibration path: {path.as_posix()}"
            )
        return m.group("session")

    def get_project_calibration_path(self) -> Path:
        return Path("calibrations/default.toml")

    def is_project_calibration_path(self, path: Path | str) -> bool:
        path = _check_relative_and_normalize(path)
        pattern = r"calibrations/default.toml"
        return bool(re.match(pattern, path.as_posix()))

    def get_calibration_backup_path(
        self, session_key: SessionKey, time_ns: int
    ) -> Path:
        """calibration_backups/<SessionKey>.<TimeNS>.toml"""
        return Path("calibration_backups") / f"{session_key}.{time_ns}.toml"

    def parse_calibration_backup_path(self, path: Path | str) -> tuple[SessionKey, int]:
        path = _check_relative_and_normalize(path)
        pattern = r"calibration_backups/(?P<session>[^/]+)\.(?P<time>\d+)\.toml"
        m = re.match(pattern, path.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse calibration backup path: {path.as_posix()}"
            )
        return m.group("session"), int(m.group("time"))
