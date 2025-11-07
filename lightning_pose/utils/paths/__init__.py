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


KeyType = TypeVar('KeyType')


class ResourceUtil(Generic[KeyType], ABC):
    """Base class for resource schema utilities.
    The type of key for this resource is a generic type parameter.
    To find the type of the key, see the signature of the `ProjectSchema`
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
    VIDEO = 'videos'
    VIDEO_BBOX = 'video-bboxes'
    LABEL_FILE = 'label-files'
    LABEL_FILE_BBOX = 'label-file-bboxes'
    FRAME = 'frames'
    SESSION_CALIBRATION = 'session-calibrations'
    CALIBRATION_BACKUP = 'calibration-backups'
    CENTER_FRAME_LIST = 'center-frames'
    PROJECT_CALIBRATION = 'project-calibration'



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

    def __repr__(self):
        return f"{self.name} sphinx hello world "


class ResourceUtilImpl(ResourceUtil[KeyType]):
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

    def get_path(self, key: KeyType, **kwargs: Any) -> Path:
        # For spec-driven utils, a single positional key is expected by default
        template = self._select_template()
        if not template:
            raise ValueError(f"No template defined for resource {self._spec.name} (is_multiview={self._is_multiview}).")

        # If a formatter is provided, use it; otherwise allow keyless singleton resources
        if self._spec.from_key is not None and key is not None:
            data: dict[str, Any] = dict(self._spec.from_key(key))
        else:
            data = {}
        # kwargs can always be used to fill placeholders (e.g., time_ns)
        data.update(kwargs)
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
        if self._spec.to_key is None:
            # Keyless resource: return None as key
            return None  # type: ignore[return-value]
        return self._spec.to_key(m.groupdict())

    # Filesystem-backed enumeration
    def iter_paths(self) -> Iterator[Path]:
        """Return paths relative to base directory"""
        base_dir = self._require_base_dir()
        start_dir_rel, pattern = self._derive_glob()
        start_dir_abs = (base_dir / start_dir_rel).resolve()
        yield from (p.relative_to(base_dir) for p in start_dir_abs.glob(pattern) if p.is_file())

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


