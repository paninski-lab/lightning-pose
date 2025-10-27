import pytest
from pathlib import Path, PureWindowsPath, PurePosixPath

from lightning_pose.data.datatypes import Project
from lightning_pose.data.keys import VideoFileKey, FrameKey
from lightning_pose.utils.paths.base_path_resolver_v1 import (
    ProjectPathUtil,
    PathParseException,
)
from lightning_pose.utils.paths import _check_relative_and_normalize


def test_check_relative_and_normalize_relative(tmp_path):
    relative_path_posix = PurePosixPath("path/to/file")
    assert _check_relative_and_normalize(relative_path_posix) == relative_path_posix

    relative_path_windows = PureWindowsPath("path\\to\\file")
    assert _check_relative_and_normalize(relative_path_windows) == PurePosixPath(
        "path/to/file"
    )


def test_check_relative_and_normalize_absolute():
    absolute_path_posix = PurePosixPath("/path/to/file")
    with pytest.raises(ValueError, match="Argument must be relative path"):
        _check_relative_and_normalize(absolute_path_posix)

    absolute_path_posix = PureWindowsPath("C:\\path\\to\\file")
    with pytest.raises(ValueError, match="Argument must be relative path"):
        _check_relative_and_normalize(absolute_path_posix)


# --------------------
# video key/path tests
# --------------------

VIDEO_TEST_CASES = [
    # (key, path, is_multiview)
    # single view
    (VideoFileKey("session1"), "videos/session1.mp4", False),
    (VideoFileKey("singlesession"), "videos/singlesession.mp4", False),
    (
        VideoFileKey("another_single_session"),
        "videos/another_single_session.mp4",
        False,
    ),
    # multi view
    (
        VideoFileKey("session_abc", "view1"),
        "videos/session_abc_view1.mp4",
        True,
    ),
    (VideoFileKey("mysession", "viewA"), "videos/mysession_viewA.mp4", True),
    (
        VideoFileKey("another_session", "viewB"),
        "videos/another_session_viewB.mp4",
        True,
    ),
    (
        VideoFileKey("session_name_cam", "a"),
        "videos/session_name_cam_a.mp4",
        True,
    ),
]


@pytest.mark.parametrize("key, path, is_multiview", VIDEO_TEST_CASES)
def test_video_key_path_conversions(key, path, is_multiview):
    """Test bidirectional conversion between video key and path."""
    project = Project(
        base_dir="unused", views=["view1", "view2"] if is_multiview else []
    )
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_video_file_path(key) == Path(path)
    # test path -> key for string, PurePosixPath and windows path string
    assert path_util.parse_video_file_path(path) == key
    assert path_util.parse_video_file_path(PurePosixPath(path)) == key
    win_path_str = str(path).replace("/", "\\")
    assert path_util.parse_video_file_path(win_path_str) == key


VIDEO_INVALID_TEST_CASES = [
    # (path, is_multiview)
    ("images/not_a_video.jpg", True),
    ("videos/malformed.txt", True),
    ("images/not_a_video.jpg", False),
    ("videos/malformed.txt", False),
]


@pytest.mark.parametrize("path, is_multiview", VIDEO_INVALID_TEST_CASES)
def test_extract_video_file_key_invalid(path, is_multiview):
    """Test that invalid video paths raise PathParseException."""
    project = Project(base_dir="unused", views=["view1"] if is_multiview else [])
    path_util = ProjectPathUtil(project)
    with pytest.raises(PathParseException, match="Could not parse video file path"):
        path_util.parse_video_file_path(path)
    with pytest.raises(PathParseException, match="Could not parse video file path"):
        path_util.parse_video_file_path(PurePosixPath(path))
    win_path_str = str(path).replace("/", "\\")
    with pytest.raises(PathParseException, match="Could not parse video file path"):
        path_util.parse_video_file_path(win_path_str)


# --------------------------
# label frame key/path tests
# --------------------------

LABEL_FRAME_TEST_CASES = [
    # (key, path, is_multiview)
    # single view
    (
        FrameKey("session1", 123),
        "labeled-data/session1/frame_123.png",
        False,
    ),
    (
        FrameKey("singlesession", 7),
        "labeled-data/singlesession/frame_7.png",
        False,
    ),
    (
        FrameKey("another_single_session", 1000),
        "labeled-data/another_single_session/frame_1000.png",
        False,
    ),
    # multi view
    (
        FrameKey("session_abc", 456, "view1"),
        "labeled-data/session_abc_view1/frame_456.png",
        True,
    ),
    (
        FrameKey("mysession", 5, "viewA"),
        "labeled-data/mysession_viewA/frame_5.png",
        True,
    ),
    (
        FrameKey("another_session", 200, "viewB"),
        "labeled-data/another_session_viewB/frame_200.png",
        True,
    ),
]


@pytest.mark.parametrize("key, path, is_multiview", LABEL_FRAME_TEST_CASES)
def test_label_frame_key_path_conversions(key, path, is_multiview):
    """Test bidirectional conversion between label key and path."""
    project = Project(
        base_dir="unused", views=["view1", "view2"] if is_multiview else []
    )
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_frame_path(key) == Path(path)
    # test path -> key for string, PurePosixPath and windows path string
    assert path_util.parse_frame_path(path) == key
    assert path_util.parse_frame_path(PurePosixPath(path)) == key
    win_path_str = str(path).replace("/", "\\")
    assert path_util.parse_frame_path(win_path_str) == key


LABEL_FRAME_INVALID_TEST_CASES = [
    # (path, is_multiview)
    ("videos/not_a_label.mp4", True),
    ("labeled-data/malformed.txt", True),
    ("labeled-data/session_no_frame.png", True),
    ("labeled-data/session_view/frame_abc.png", True),
    ("videos/not_a_label.mp4", False),
    ("labeled-data/malformed.txt", False),
    ("labeled-data/session_no_frame.png", False),
    ("labeled-data/session/frame_abc.png", False),
]


@pytest.mark.parametrize("path, is_multiview", LABEL_FRAME_INVALID_TEST_CASES)
def test_extract_label_frame_key_invalid(path, is_multiview):
    """Test that invalid label paths raise PathParseException."""
    project = Project(base_dir="unused", views=["view1"] if is_multiview else [])
    path_util = ProjectPathUtil(project)
    with pytest.raises(PathParseException, match="Could not parse label frame path"):
        path_util.parse_frame_path(path)
    with pytest.raises(PathParseException, match="Could not parse label frame path"):
        path_util.parse_frame_path(PurePosixPath(path))
    win_path_str = str(path).replace("/", "\\")
    with pytest.raises(PathParseException, match="Could not parse label frame path"):
        path_util.parse_frame_path(win_path_str)


# -------------------------
# label file key/path tests
# -------------------------


LABEL_FILE_TEST_CASES = [
    # (key, view, path, is_multiview)
    # single view
    ("session1", None, "labeled-data/label-files/session1.csv", False),
    ("singlesession", None, "labeled-data/label-files/singlesession.csv", False),
    # multi view
    (
        "session_abc",
        "view1",
        "labeled-data/label-files/session_abc_view1.csv",
        True,
    ),
    (
        "mysession",
        "viewA",
        "labeled-data/label-files/mysession_viewA.csv",
        True,
    ),
]


@pytest.mark.parametrize("key, view, path, is_multiview", LABEL_FILE_TEST_CASES)
def test_label_file_key_path_conversions(key, view, path, is_multiview):
    """Test bidirectional conversion between label file key and path."""
    project = Project(
        base_dir="unused", views=["view1", "view2"] if is_multiview else []
    )
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_label_file_path(key, view) == Path(path)
    # test path -> key for string, PurePosixPath and windows path string
    assert path_util.parse_label_file_path(path) == (key, view)
    assert path_util.parse_label_file_path(PurePosixPath(path)) == (key, view)
    win_path_str = str(path).replace("/", "\\")
    assert path_util.parse_label_file_path(win_path_str) == (key, view)


# ----------------------------
# center frames key/path tests
# ----------------------------


CENTER_FRAMES_TEST_CASES = [
    # (key, path, is_multiview)
    # single view
    (VideoFileKey("session1"), "labeled-data/session1/center_frames.txt", False),
    (
        VideoFileKey("singlesession"),
        "labeled-data/singlesession/center_frames.txt",
        False,
    ),
    # multi view
    (
        VideoFileKey("session_abc", "view1"),
        "labeled-data/session_abc_view1/center_frames.txt",
        True,
    ),
    (
        VideoFileKey("mysession", "viewA"),
        "labeled-data/mysession_viewA/center_frames.txt",
        True,
    ),
]


@pytest.mark.parametrize("key, path, is_multiview", CENTER_FRAMES_TEST_CASES)
def test_center_frames_key_path_conversions(key, path, is_multiview):
    """Test bidirectional conversion between center frames key and path."""
    project = Project(
        base_dir="unused", views=["view1", "view2"] if is_multiview else []
    )
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_center_frames_path(key) == Path(path)
    # test path -> key for string, PurePosixPath and windows path string
    assert path_util.parse_center_frames_path(path) == key
    assert path_util.parse_center_frames_path(PurePosixPath(path)) == key
    win_path_str = str(path).replace("/", "\\")
    assert path_util.parse_center_frames_path(win_path_str) == key


# ----------------------------------
# session calibration key/path tests
# ----------------------------------


SESSION_CALIBRATION_TEST_CASES = [
    # (key, path)
    ("session1", "calibrations/session1.csv"),
    ("my_session", "calibrations/my_session.csv"),
]


@pytest.mark.parametrize("key, path", SESSION_CALIBRATION_TEST_CASES)
def test_session_calibration_key_path_conversions(key, path):
    project = Project(base_dir="unused", views=[])
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_session_calibration_path(key) == Path(path)
    # test path -> key
    assert path_util.parse_session_calibration_path(path) == key


def test_get_project_calibration_path():
    project = Project(base_dir="unused", views=[])
    path_util = ProjectPathUtil(project)
    assert path_util.get_project_calibration_path() == Path("calibration.csv")


# -----------------------------------
# calibration backup key/path tests
# -----------------------------------


CALIBRATION_BACKUP_TEST_CASES = [
    # (session_key, time_ns, path)
    ("session1", 12345, "calibration_backups/session1.12345.csv"),
    ("my_session", 98765, "calibration_backups/my_session.98765.csv"),
]


@pytest.mark.parametrize("session_key, time_ns, path", CALIBRATION_BACKUP_TEST_CASES)
def test_calibration_backup_key_path_conversions(session_key, time_ns, path):
    project = Project(base_dir="unused", views=[])
    path_util = ProjectPathUtil(project)
    # test key -> path
    assert path_util.get_calibration_backup_path(session_key, time_ns) == Path(path)
    # test path -> key
    extracted_session_key, extracted_time_ns = (
        path_util.parse_calibration_backup_path(path)
    )
    assert extracted_session_key == session_key
    assert extracted_time_ns == time_ns

