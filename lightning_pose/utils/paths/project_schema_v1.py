from typing import Any
from pathlib import Path

from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    SessionKey,
    LabelFileKey,
    ViewName,
)
from lightning_pose.utils.paths import (
    ResourceSpec,
    DefaultResourceUtil,
    ResourceType,
    AbstractResourceUtil,
)
from lightning_pose.utils.paths.base_project_schema_v1 import BaseProjectSchemaV1


class ProjectSchemaV1(BaseProjectSchemaV1):
    def __init__(self, is_multiview: bool, base_dir: Path | None = None):
        super().__init__(is_multiview, base_dir)

        # Define specs for each resource
        self._video_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.videos,
            template_single="videos/{session_key}.mp4",
            template_multi="videos/{session_key}_{view}.mp4",
            pattern_single=r"videos/(?P<session_key>[^/]+)\.mp4",
            pattern_multi=r"videos/(?P<session_key>[^/]+)_(?P<view>[^/_]+)\.mp4",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._video_bbox_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.video_boxes,
            template_single="videos/{session_key}_bbox.csv",
            template_multi="videos/{session_key}_{view}_bbox.csv",
            pattern_single=r"videos/(?P<session_key>[^/]+)_bbox\.csv",
            pattern_multi=r"videos/(?P<session_key>[^/]+)_(?P<view>[^/_]+)_bbox\.csv",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._frame_spec = ResourceSpec[FrameKey](
            name=ResourceType.frames,
            template_single="labeled-data/frames/{session_key}/frame_{frame_index:08d}.png",
            template_multi="labeled-data/frames/{session_key}_{view}/frame_{frame_index:08d}.png",
            pattern_single=r"labeled-data/frames/(?P<session_key>[^/]+)/frame_(?P<frame_index>\d{8})\.png",
            pattern_multi=r"labeled-data/frames/(?P<session_key>[^/]+)_(?P<view>[^/_]+)/frame_(?P<frame_index>\d{8})\.png",
            to_key=lambda g: FrameKey(session_key=g["session_key"], view=g.get("view"), frame_index=int(g["frame_index"])),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view, "frame_index": k.frame_index},
        )
        self._label_file_spec = ResourceSpec[tuple[LabelFileKey, ViewName | None]](
            name=ResourceType.label_files,
            template_single="labeled-data/labels/{label_file_key}.csv",
            template_multi="labeled-data/labels/{label_file_key}_{view}.csv",
            pattern_single=r"labeled-data/labels/(?P<label_file_key>[^/]+)\.csv",
            pattern_multi=r"labeled-data/labels/(?P<label_file_key>[^/]+)_(?P<view>[^/_]+)\.csv",
            to_key=lambda g: (g["label_file_key"], g.get("view")),
            from_key=lambda kv: {"label_file_key": kv[0], "view": kv[1]},
        )
        self._label_file_bbox_spec = ResourceSpec[tuple[LabelFileKey, ViewName | None]](
            name=ResourceType.label_file_bboxes,
            template_single="labeled-data/labels/{label_file_key}_bbox.csv",
            template_multi="labeled-data/labels/{label_file_key}_{view}_bbox.csv",
            pattern_single=r"labeled-data/labels/(?P<label_file_key>[^/]+)_bbox\.csv",
            pattern_multi=r"labeled-data/labels/(?P<label_file_key>[^/]+)_(?P<view>[^/_]+)_bbox\.csv",
            to_key=lambda g: (g["label_file_key"], g.get("view")),
            from_key=lambda kv: {"label_file_key": kv[0], "view": kv[1]},
        )
        self._center_frames_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.center_frames,
            template_single="labeled-data/{session_key}/center_frames.txt",
            template_multi="labeled-data/frames/{session_key}_{view}/center_frames.txt",
            pattern_single=r"labeled-data/(?P<session_key>[^/]+)/center_frames\.txt",
            pattern_multi=r"labeled-data/frames/(?P<session_key>[^/]+)_(?P<view>[^/_]+)/center_frames\.txt",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._session_calib_spec = ResourceSpec[SessionKey](
            name=ResourceType.session_calibrations,
            template_multi="calibrations/{session_key}.toml",
            pattern_multi=r"calibrations/(?P<session_key>[^/]+)\.toml",
            to_key=lambda g: g["session_key"],
            from_key=lambda k: {"session_key": k},
        )
        self._project_calib_spec = ResourceSpec[bool](
            name=ResourceType.project_calibration,
            template_multi="calibrations/default.toml",
            pattern_multi=r"calibrations/default\.toml",
            is_predicate=True,
        )
        self._calib_backup_spec = ResourceSpec[tuple[SessionKey, int]](
            name=ResourceType.calibration_backups,
            template_multi="calibration_backups/{session_key}.{time_ns}.toml",
            pattern_multi=r"calibration_backups/(?P<session_key>[^/]+)\.(?P<time_ns>\d+)\.toml",
            to_key=lambda g: (g["session_key"], int(g["time_ns"])),
            from_key=lambda kv: {"session_key": kv[0], "time_ns": kv[1]},
        )

        # Create resource utils
        base_dir_getter = lambda: self.base_dir
        self.videos = DefaultResourceUtil(self._video_spec, self.is_multiview, base_dir_getter)
        self.video_boxes = DefaultResourceUtil(self._video_bbox_spec, self.is_multiview, base_dir_getter)
        self.frames = DefaultResourceUtil(self._frame_spec, self.is_multiview, base_dir_getter)
        self.label_files = DefaultResourceUtil(self._label_file_spec, self.is_multiview, base_dir_getter)
        self.label_file_bboxes = DefaultResourceUtil(self._label_file_bbox_spec, self.is_multiview, base_dir_getter)
        self.center_frames = DefaultResourceUtil(self._center_frames_spec, self.is_multiview, base_dir_getter)
        self.session_calibrations = DefaultResourceUtil(self._session_calib_spec, self.is_multiview, base_dir_getter)
        self.project_calibration = DefaultResourceUtil(self._project_calib_spec, self.is_multiview, base_dir_getter)
        self.calibration_backups = DefaultResourceUtil(self._calib_backup_spec, self.is_multiview, base_dir_getter)

        self._resource_map: dict[ResourceType, AbstractResourceUtil[Any]] = {
            ResourceType.videos: self.videos,
            ResourceType.video_boxes: self.video_boxes,
            ResourceType.frames: self.frames,
            ResourceType.label_files: self.label_files,
            ResourceType.label_file_bboxes: self.label_file_bboxes,
            ResourceType.center_frames: self.center_frames,
            ResourceType.session_calibrations: self.session_calibrations,
            ResourceType.project_calibration: self.project_calibration,
            ResourceType.calibration_backups: self.calibration_backups,
        }

    # Return spec-driven resource utils only
    def for_(self, path_type: ResourceType) -> AbstractResourceUtil[Any]:
        return self._resource_map[path_type]
