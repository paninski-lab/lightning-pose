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
    ResourceUtilImpl,
    ResourceType,
    ResourceUtil,
)
from lightning_pose.utils.paths.project_schema import ProjectSchema


class ProjectSchemaV1(ProjectSchema):
    """
    Contains ResourceUtils for all types of project resources.
    """
    def __init__(self, is_multiview: bool, base_dir: Path | None = None):
        super().__init__(is_multiview, base_dir)

        # Define specs for each resource
        self._video_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.VIDEO,
            template_single="videos/{session_key}.mp4",
            template_multi="videos/{session_key}_{view}.mp4",
            pattern_single=r"videos/(?P<session_key>[^/]+)\.mp4",
            pattern_multi=r"videos/(?P<session_key>[^/]+)_(?P<view>[^/_]+)\.mp4",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._video_bbox_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.VIDEO_BBOX,
            template_single="videos/{session_key}_bbox.csv",
            template_multi="videos/{session_key}_{view}_bbox.csv",
            pattern_single=r"videos/(?P<session_key>[^/]+)_bbox\.csv",
            pattern_multi=r"videos/(?P<session_key>[^/]+)_(?P<view>[^/_]+)_bbox\.csv",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._frame_spec = ResourceSpec[FrameKey](
            name=ResourceType.FRAME,
            template_single="labeled-data/frames/{session_key}/frame_{frame_index:08d}.png",
            template_multi="labeled-data/frames/{session_key}_{view}/frame_{frame_index:08d}.png",
            pattern_single=r"labeled-data/frames/(?P<session_key>[^/]+)/frame_(?P<frame_index>\d{8})\.png",
            pattern_multi=r"labeled-data/frames/(?P<session_key>[^/]+)_(?P<view>[^/_]+)/frame_(?P<frame_index>\d{8})\.png",
            to_key=lambda g: FrameKey(session_key=g["session_key"], view=g.get("view"), frame_index=int(g["frame_index"])),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view, "frame_index": k.frame_index},
        )
        self._label_file_spec = ResourceSpec[tuple[LabelFileKey, ViewName | None]](
            name=ResourceType.LABEL_FILE,
            template_single="labeled-data/labels/{label_file_key}.csv",
            template_multi="labeled-data/labels/{label_file_key}_{view}.csv",
            pattern_single=r"labeled-data/labels/(?P<label_file_key>[^/]+)\.csv",
            pattern_multi=r"labeled-data/labels/(?P<label_file_key>[^/]+)_(?P<view>[^/_]+)\.csv",
            to_key=lambda g: (g["label_file_key"], g.get("view")),
            from_key=lambda kv: {"label_file_key": kv[0], "view": kv[1]},
        )
        self._label_file_bbox_spec = ResourceSpec[tuple[LabelFileKey, ViewName | None]](
            name=ResourceType.LABEL_FILE_BBOX,
            template_single="labeled-data/labels/{label_file_key}_bbox.csv",
            template_multi="labeled-data/labels/{label_file_key}_{view}_bbox.csv",
            pattern_single=r"labeled-data/labels/(?P<label_file_key>[^/]+)_bbox\.csv",
            pattern_multi=r"labeled-data/labels/(?P<label_file_key>[^/]+)_(?P<view>[^/_]+)_bbox\.csv",
            to_key=lambda g: (g["label_file_key"], g.get("view")),
            from_key=lambda kv: {"label_file_key": kv[0], "view": kv[1]},
        )
        self._center_frames_spec = ResourceSpec[VideoFileKey](
            name=ResourceType.CENTER_FRAME_LIST,
            template_single="labeled-data/{session_key}/center_frames.txt",
            template_multi="labeled-data/frames/{session_key}_{view}/center_frames.txt",
            pattern_single=r"labeled-data/(?P<session_key>[^/]+)/center_frames\.txt",
            pattern_multi=r"labeled-data/frames/(?P<session_key>[^/]+)_(?P<view>[^/_]+)/center_frames\.txt",
            to_key=lambda g: VideoFileKey(session_key=g["session_key"], view=g.get("view")),
            from_key=lambda k: {"session_key": k.session_key, "view": k.view},
        )
        self._session_calib_spec = ResourceSpec[SessionKey](
            name=ResourceType.SESSION_CALIBRATION,
            template_multi="calibrations/{session_key}.toml",
            pattern_multi=r"calibrations/(?P<session_key>[^/]+)\.toml",
            to_key=lambda g: g["session_key"],
            from_key=lambda k: {"session_key": k},
        )
        self._project_calib_spec = ResourceSpec[None](
            name=ResourceType.PROJECT_CALIBRATION,
            template_multi="calibrations/default.toml",
            pattern_multi=r"calibrations/default\.toml",
        )
        self._calib_backup_spec = ResourceSpec[tuple[SessionKey, int]](
            name=ResourceType.CALIBRATION_BACKUP,
            template_multi="calibration_backups/{session_key}.{time_ns}.toml",
            pattern_multi=r"calibration_backups/(?P<session_key>[^/]+)\.(?P<time_ns>\d+)\.toml",
            to_key=lambda g: (g["session_key"], int(g["time_ns"])),
            from_key=lambda kv: {"session_key": kv[0], "time_ns": kv[1]},
        )

        # Create resource utils
        base_dir_getter = lambda: self.base_dir
        self.videos = ResourceUtilImpl(self._video_spec, self.is_multiview, base_dir_getter)
        self.video_boxes = ResourceUtilImpl(self._video_bbox_spec, self.is_multiview, base_dir_getter)
        self.frames = ResourceUtilImpl(self._frame_spec, self.is_multiview, base_dir_getter)
        self.label_files = ResourceUtilImpl(self._label_file_spec, self.is_multiview, base_dir_getter)
        self.label_file_bboxes = ResourceUtilImpl(self._label_file_bbox_spec, self.is_multiview, base_dir_getter)
        self.center_frames = ResourceUtilImpl(self._center_frames_spec, self.is_multiview, base_dir_getter)
        self.session_calibrations = ResourceUtilImpl(self._session_calib_spec, self.is_multiview, base_dir_getter)
        self.project_calibration = ResourceUtilImpl(self._project_calib_spec, self.is_multiview, base_dir_getter)
        self.calibration_backups = ResourceUtilImpl(self._calib_backup_spec, self.is_multiview, base_dir_getter)

        self._resource_map: dict[ResourceType, ResourceUtil[Any]] = {
            ResourceType.VIDEO: self.videos,
            ResourceType.VIDEO_BBOX: self.video_boxes,
            ResourceType.FRAME: self.frames,
            ResourceType.LABEL_FILE: self.label_files,
            ResourceType.LABEL_FILE_BBOX: self.label_file_bboxes,
            ResourceType.CENTER_FRAME_LIST: self.center_frames,
            ResourceType.SESSION_CALIBRATION: self.session_calibrations,
            ResourceType.PROJECT_CALIBRATION: self.project_calibration,
            ResourceType.CALIBRATION_BACKUP: self.calibration_backups,
        }


    def for_(self, resource_type: ResourceType) -> ResourceUtil[Any]:
        return self._resource_map[resource_type]
