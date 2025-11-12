from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from enum import Enum
from dataclasses import dataclass
from typing import (
    Callable,
    Any,
    TypeVar,
    Generic,
    Union,
    Mapping,
    Optional,
    Iterator,
    Literal,
)
from abc import ABC
import re


class PathParseException(Exception):
    pass


def _check_relative_and_normalize(path: Path | str) -> PurePath:
    path = PureWindowsPath(path) if "\\" in str(path) else PurePosixPath(path)
    if path.is_absolute():
        raise ValueError("Argument must be relative path: " + str(path))

    return PurePosixPath(str(path).replace("\\", "/"))


KeyType = TypeVar("KeyType")


class ResourceUtil(Generic[KeyType], ABC):
    """Base class resource-specific path utilities.
    The type of key for this resource is a generic type parameter.
    To find the type of the key, see the signature of the `PathUtil`
    attribute for this resource.
    """

    def get_path(self, key: KeyType, **kwargs: Any) -> Path:
        """Return the path for a resource."""
        raise NotImplementedError()

    def parse_path(self, path: Union[Path, str]) -> KeyType:
        """
        Return the resource key for a path, or raise `PathParseException` if the
        path is not valid for this resource type.
        """
        raise NotImplementedError()

    def iter_paths(self) -> Iterator[Path]:
        """
        Iterate over this resource's paths relative to base directory.
        """
        raise NotImplementedError()

    def list_paths(self, *, sort: bool = True) -> list[Path]:
        """
        Returns a list of paths relative to base directory.
        """
        return sorted(self.iter_paths(), key=lambda p: str(p)) if sort else list(self.iter_paths())

    def iter_keys(self, *, strict: bool = False) -> Iterator[KeyType]:
        """
        Iterate over this resource's keys.
        """
        raise NotImplementedError()

    def list_keys(self, *, sort: bool = True, strict: bool = False) -> list[KeyType]:
        """
        Returns a list of this resource's keys.
        """
        keys = list(self.iter_keys(strict=strict))
        if sort:
            try:
                return sorted(keys)  # type: ignore
            except Exception:
                # Fallback: sort by path strings derived from keys
                return sorted(keys, key=lambda k: str(self.get_path(k)))
        return keys


# ---------------------------------------------------------------------------
# Resource type enum (new name), with temporary alias for PathType
# ---------------------------------------------------------------------------
class ResourceType(str, Enum):
    VIDEO = "videos"
    VIDEO_BBOX = "video-bboxes"
    LABEL_FILE = "label-files"
    LABEL_FILE_BBOX = "label-file-bboxes"
    FRAME = "frames"
    CALIBRATION = "calibrations"
    CENTER_FRAME_LIST = "center-frames"
    PROJECT_CALIBRATION = "project-calibration"
    CALIBRATION_BACKUP = "calibration-backups"
