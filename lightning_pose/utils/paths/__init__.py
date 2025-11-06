from pathlib import Path, PurePath, PureWindowsPath, PurePosixPath
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any, TypeVar, Generic, Union, Mapping, Optional, Iterator, Literal
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


class AbstractResourceUtil(Generic[KeyType], ABC):
    @abstractmethod
    def get_path(self, *args: Any, **kwargs: Any) -> Path:
        ...

    @abstractmethod
    def parse_path(self, path: Union[Path, str]) -> KeyType:
        ...

    def iter_paths(self) -> Iterator[Path]:
        raise NotImplementedError()

    def list_paths(self, *, sort: bool = True) -> list[Path]:
        return sorted(self.iter_paths(), key=lambda p: str(p)) if sort else list(self.iter_paths())

    def iter_keys(self, *, strict: bool = False) -> Iterator[KeyType]:
        raise NotImplementedError()

    def list_keys(self, *, sort: bool = True, strict: bool = False) -> list[KeyType]:
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
    def __init__(self, spec: ResourceSpec[KeyType], is_multiview: bool, get_base_dir: Optional[Callable[[], Optional[Path]]] = None):
        self._spec = spec
        self._is_multiview = is_multiview
        self._get_base_dir = get_base_dir or (lambda: None)

    def _select_template(self) -> Optional[str]:
        return self._spec.template_multi if self._is_multiview else self._spec.template_single

    def _select_pattern(self) -> Optional[str]:
        return self._spec.pattern_multi if self._is_multiview else self._spec.pattern_single

    def _require_base_dir(self) -> Path:
        base_dir = self._get_base_dir()
        if base_dir is None:
            raise RuntimeError(
                f"{self._spec.name}: filesystem enumeration requires schema.base_dir; got None. "
                "Construct schema via ProjectSchema.for_project(...) or pass base_dir to for_version(...)."
            )
        return Path(base_dir)

    def _derive_glob(self) -> tuple[Path, str]:
        template = self._select_template()
        if not template:
            raise ValueError(f"No template defined for resource {self._spec.name} (is_multiview={self._is_multiview}).")
        # Find directory root up to the last '/' before the first placeholder
        s = template
        parts = []
        i = 0
        first_placeholder_index = s.find('{')
        literal_prefix = s[:first_placeholder_index] if first_placeholder_index != -1 else s
        # directory to start globbing
        dir_end = literal_prefix.rfind('/')
        start_dir = literal_prefix[:dir_end+1] if dir_end != -1 else ''
        # Build glob pattern from scratch
        glob_pattern = ''
        while i < len(s):
            if s[i] == '{':
                j = s.find('}', i+1)
                if j == -1:
                    raise ValueError(f"Unclosed placeholder in template for {self._spec.name}: {template}")
                placeholder = s[i+1:j]
                # check for format spec like name:08d
                if ':' in placeholder and placeholder.split(':', 1)[1].endswith('d'):
                    width_spec = placeholder.split(':', 1)[1]
                    # extract width digits before 'd'
                    try:
                        width = int(width_spec[:-1])
                    except ValueError:
                        width = 0
                    glob_pattern += ('?' * width) if width > 0 else '*'
                else:
                    glob_pattern += '*'
                i = j + 1
            else:
                glob_pattern += s[i]
                i += 1
        # Make pattern relative to start_dir
        pattern_rel = glob_pattern[len(start_dir):] if start_dir and glob_pattern.startswith(start_dir) else glob_pattern
        return Path(start_dir), pattern_rel

    def get_path(self, *args: Any, **kwargs: Any) -> Path:
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

    def parse_path(self, path: Union[Path, str]) -> KeyType:
        normalized = _check_relative_and_normalize(path)
        pattern = self._select_pattern()
        if not pattern:
            raise PathParseException(
                f"No pattern defined for resource {self._spec.name} (is_multiview={self._is_multiview})."
            )
        m = re.fullmatch(pattern, normalized.as_posix())
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

    def list_keys(self) -> list[KeyType]:
        if self._spec.list_keys is None:
            raise NotImplementedError()
        return self._spec.list_keys()

    # Filesystem-backed enumeration
    def iter_paths(self) -> Iterator[Path]:
        if self._spec.is_predicate:
            raise TypeError(f"Enumeration not supported for predicate resource {self._spec.name}.")
        base_dir = self._require_base_dir()
        start_dir_rel, pattern = self._derive_glob()
        start_dir_abs = (base_dir / start_dir_rel).resolve()
        yield from (p for p in start_dir_abs.glob(pattern) if p.is_file())

    def iter_keys(self, *, strict: bool = False) -> Iterator[KeyType]:
        base_dir = self._require_base_dir()
        for p in self.iter_paths():
            try:
                rel = p.relative_to(base_dir)
            except Exception:
                rel = Path(str(p).replace(str(base_dir) + '/', ''))
            try:
                yield self.parse_path(rel)
            except PathParseException:
                if strict:
                    # Re-raise with context
                    raise
                # else skip
                continue


