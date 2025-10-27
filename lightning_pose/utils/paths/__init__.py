from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from enum import Enum
from typing import Callable, Any, TypeVar, Generic, Union # Add these imports


class PathParseException(Exception):
    pass


def _check_relative_and_normalize(path: Path | str) -> PurePath:
    path = PureWindowsPath(path) if "\\" in str(path) else PurePosixPath(path)
    if path.is_absolute():
        raise ValueError("Argument must be relative path: " + str(path))

    return PurePosixPath(str(path).replace("\\", "/"))


# Define a TypeVar for the key type that get_ and parse_ methods operate on
KeyType = TypeVar('KeyType')

class PathTypeHandler(Generic[KeyType]):
    """
    Helper class to provide .get() and .reverse() methods for a specific PathType.
    """
    def __init__(self, get_func: Callable[..., Path], parse_func: Callable[[Union[Path, str]], KeyType]):
        self._get_func = get_func
        self._parse_func = parse_func

    def get(self, *args: Any, **kwargs: Any) -> Path:
        """
        Calls the underlying get_ method.
        """
        return self._get_func(*args, **kwargs)

    def reverse(self, path: Union[Path, str]) -> KeyType:
        """
        Calls the underlying parse_ method.
        """
        return self._parse_func(path)

class PathType(str, Enum):
    videos = 'videos'
    video_boxes = 'video-bboxes'
    label_files = 'label-files'
    label_file_bboxes = 'label-file-bboxes'
    frames = 'frames'
    session_calibrations = 'session-calibrations'
    calibration_backups = 'calibration-backups'
    center_frames = 'center-frames'
    project_calibration = 'project-calibration'

