from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any, TypeVar, Generic, Union, Mapping, Optional
from abc import ABC, abstractmethod
import re


class PathParseException(Exception):
    pass


def _check_relative_and_normalize(path: Path | str) -> PurePath:
    path = PureWindowsPath(path) if "\\" in str(path) else PurePosixPath(path)
    if path.is_absolute():
        raise ValueError("Argument must be relative path: " + str(path))

    return PurePosixPath(str(path).replace("\\", "/"))


# Define a TypeVar for the key type that get_ and parse_ methods operate on
KeyType = TypeVar('KeyType')



# ---------------------------------------------------------------------------
# New abstract resource util interface (target design)
# ---------------------------------------------------------------------------
class AbstractResourceUtil(Generic[KeyType], ABC):
    @abstractmethod
    def get(self, *args: Any, **kwargs: Any) -> Path:
        ...

    @abstractmethod
    def reverse(self, path: Union[Path, str]) -> KeyType:
        ...

    def get_all_keys(self) -> list[KeyType]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Resource type enum (new name), with temporary alias for PathType
# ---------------------------------------------------------------------------
class ResourceType(str, Enum):
    videos = 'videos'
    video_boxes = 'video-bboxes'
    label_files = 'label-files'
    label_file_bboxes = 'label-file-bboxes'
    frames = 'frames'
    session_calibrations = 'session-calibrations'
    calibration_backups = 'calibration-backups'
    center_frames = 'center-frames'
    project_calibration = 'project-calibration'



# ---------------------------------------------------------------------------
# Spec and default implementation for spec-driven resources
# ---------------------------------------------------------------------------
@dataclass
class ResourceSpec(Generic[KeyType]):
    name: ResourceType
    template_single: Optional[str] = None
    template_multi: Optional[str] = None
    pattern_single: Optional[str] = None
    pattern_multi: Optional[str] = None
    to_key: Optional[Callable[[Mapping[str, str]], KeyType]] = None
    from_key: Optional[Callable[[KeyType], Mapping[str, Any]]] = None
    is_predicate: bool = False
    list_keys: Optional[Callable[[], list[KeyType]]] = None


class DefaultResourceUtil(AbstractResourceUtil[KeyType]):
    def __init__(self, spec: ResourceSpec[KeyType], is_multiview: bool):
        self._spec = spec
        self._is_multiview = is_multiview

    def _select_template(self) -> Optional[str]:
        return self._spec.template_multi if self._is_multiview else self._spec.template_single

    def _select_pattern(self) -> Optional[str]:
        return self._spec.pattern_multi if self._is_multiview else self._spec.pattern_single

    def get(self, *args: Any, **kwargs: Any) -> Path:
        # For spec-driven utils, a single positional key is expected by default
        if self._spec.is_predicate:
            # Predicate resources typically have a fixed path and no key args
            template = self._select_template()
            if not template:
                raise ValueError(f"No template defined for predicate resource {self._spec.name}.")
            # Allow kwargs to fill placeholders if any
            return Path(template.format(**kwargs))

        if self._spec.from_key is None:
            raise ValueError(f"from_key not defined for resource {self._spec.name}.")
        if len(args) != 1:
            raise TypeError("DefaultResourceUtil.get() expects a single key argument.")
        key = args[0]
        data = dict(self._spec.from_key(key))
        template = self._select_template()
        if not template:
            raise ValueError(f"No template defined for resource {self._spec.name} (is_multiview={self._is_multiview}).")
        # Support standard formatting, including zero-padded numbers via preformatted values in from_key
        return Path(template.format(**data))

    def reverse(self, path: Union[Path, str]) -> KeyType:
        normalized = _check_relative_and_normalize(path)
        pattern = self._select_pattern()
        if not pattern:
            raise PathParseException(
                f"No pattern defined for resource {self._spec.name} (is_multiview={self._is_multiview})."
            )
        m = re.match(pattern, normalized.as_posix())
        if not m:
            raise PathParseException(
                f"Could not parse {self._spec.name} path: {normalized.as_posix()}, multiview={self._is_multiview}"
            )
        if self._spec.is_predicate:
            # type: ignore[return-value]
            return True  # type: ignore
        if self._spec.to_key is None:
            raise ValueError(f"to_key not defined for resource {self._spec.name}.")
        return self._spec.to_key(m.groupdict())  # type: ignore[return-value]

    def get_all_keys(self) -> list[KeyType]:
        if self._spec.list_keys is None:
            raise NotImplementedError()
        return self._spec.list_keys()


