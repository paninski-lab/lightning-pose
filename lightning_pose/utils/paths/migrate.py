from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict, List, Union, Generic, TypeVar, Callable, Any
import os
import re
import yaml
from enum import Enum

from lightning_pose.data.datatypes import ProjectConfig
from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    ViewName,
    SessionKey,
    LabelFileKey,
)
from lightning_pose.utils.paths.base_path_resolver_v1 import BasePathResolverV1
from lightning_pose.utils.paths.path_resolver import PathResolver
from lightning_pose.utils.paths import PathParseException, PathType


TInputPath = Tuple[str, str]  # (path_str, "file"|"directory")


# --- Result Monad Implementation ---
class MigrationError(Exception):
    pass


class ParsingError(MigrationError):
    def __init__(self, message, original_path: str):
        super().__init__(message)
        self.original_path = original_path


class SanitizationError(MigrationError):
    def __init__(self, message, parsed_key: Any, path_type: PathType):
        super().__init__(message)
        self.parsed_key = parsed_key
        self.path_type = path_type


class SerializationError(MigrationError):
    def __init__(self, message, sanitized_key: Any, path_type: PathType):
        super().__init__(message)
        self.sanitized_key = sanitized_key
        self.path_type = path_type


T = TypeVar("T")
E = TypeVar("E", bound=MigrationError)


class Result(Generic[T, E]):
    pass


class Ok(Result[T, E]):
    def __init__(self, value: T):
        self._value = value

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> E:
        raise RuntimeError("Called unwrap_err on an Ok value")

    def and_then(self, func: Callable[[T], "Result[U, E_new]"]) -> "Result[U, E_new]":
        return func(self._value)


U = TypeVar("U")  # For and_then
E_new = TypeVar("E_new", bound=MigrationError)  # For and_then


class Err(Result[T, E]):
    def __init__(self, error: E):
        self._error = error

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise self._error

    def unwrap_err(self) -> E:
        return self._error

    def and_then(self, func: Callable[[T], "Result[U, E_new]"]) -> "Result[T, E_new]":
        # If it's an Err, just pass the Err along, func is not called
        return self  # type: ignore


# --- Original _sanitize_key function (remains mostly unchanged) ---
def _sanitize_key(parsed_key: Any) -> Any:
    """
    Sanitize keys returned by a resolver's reverse() to ensure they can be fed
    into the destination resolver's get().
    Mirrors the helper previously defined in scripts/migrate.py.
    """
    if parsed_key is None:
        return None

    if isinstance(parsed_key, str):
        # ViewName is a special case of str. Underscores are not allowed in ViewName.
        if isinstance(parsed_key, ViewName):
            return type(parsed_key)(re.sub(r"[^a-zA-Z0-9-]", "", parsed_key))
        else:
            return type(parsed_key)(re.sub(r"[^a-zA-Z0-9_-]", "", parsed_key))

    if isinstance(parsed_key, VideoFileKey):
        return VideoFileKey(
            session_key=_sanitize_key(parsed_key.session_key),
            view=_sanitize_key(parsed_key.view),
        )

    if isinstance(parsed_key, FrameKey):
        return FrameKey(
            session_key=_sanitize_key(parsed_key.session_key),
            view=_sanitize_key(parsed_key.view),
            frame_index=parsed_key.frame_index,
        )

    if isinstance(parsed_key, tuple):
        return tuple(_sanitize_key(x) for x in parsed_key)

    return parsed_key


ParsedInfo = Tuple[Any, PathType]  # (parsed_key, path_type_enum)
SanitizedInfo = Tuple[Any, PathType]  # (sanitized_key, path_type_enum)


def parse_path(
    path_str: str, source_resolver: BasePathResolverV1
) -> Result[ParsedInfo, ParsingError]:
    """
    Attempts to parse a path string using all available PathTypes in the source resolver.
    Returns Ok((parsed_key, path_type)) on success, or Err(ParsingError) on failure.
    """
    for path_type_enum in PathType:
        try:
            parsed_key = source_resolver.for_(path_type_enum).reverse(path_str)
            if isinstance(parsed_key, bool) and not parsed_key:
                # Specific error condition from original code
                raise PathParseException("Parsed key is boolean False")
            return Ok((parsed_key, path_type_enum))
        except PathParseException:
            # Continue to next PathType if this one fails to parse
            continue
        except Exception as e:
            # Catch other unexpected errors during parsing
            return Err(
                ParsingError(
                    f"Unexpected error during parsing '{path_str}': {e}",
                    original_path=path_str,
                )
            )
    return Err(
        ParsingError(
            f"No PathType could parse the path: '{path_str}'", original_path=path_str
        )
    )


def sanitize_key(parsed_info: ParsedInfo) -> Result[SanitizedInfo, SanitizationError]:
    """
    Sanitizes a parsed key. Returns Ok((sanitized_key, path_type)) on success,
    or Err(SanitizationError) if _sanitize_key were to fail (currently it doesn't raise).
    """
    parsed_key, path_type_enum = parsed_info
    try:
        sanitized_key = _sanitize_key(parsed_key)
        return Ok((sanitized_key, path_type_enum))
    except Exception as e:
        # In the current _sanitize_key implementation, this block might not be hit,
        # but it's good practice for future modifications.
        return Err(
            SanitizationError(
                f"Error sanitizing key '{parsed_key}' for PathType '{path_type_enum}': {e}",
                parsed_key=parsed_key,
                path_type=path_type_enum,
            )
        )


def serialize_key(
    sanitized_info: SanitizedInfo, dest_resolver: BasePathResolverV1
) -> Result[Path, SerializationError]:
    """
    Serializes a sanitized key into a new Path object using the destination resolver.
    Returns Ok(new_path) on success, or Err(SerializationError) on failure.
    """
    sanitized_key, path_type_enum = sanitized_info
    try:
        if isinstance(sanitized_key, tuple) and type(sanitized_key).__name__ == "tuple":
            # Handles regular tuples vs. potential named tuples or other objects
            new_path = dest_resolver.for_(path_type_enum).get(*sanitized_key)
        elif isinstance(sanitized_key, bool):
            # Original code asserted `sanitized_key`, implying it must be True here
            assert sanitized_key
            new_path = dest_resolver.for_(path_type_enum).get()
        else:
            new_path = dest_resolver.for_(path_type_enum).get(sanitized_key)
        return Ok(new_path)
    except Exception as e:
        return Err(
            SerializationError(
                f"Error serializing key '{sanitized_key}' for PathType '{path_type_enum}': {e}",
                sanitized_key=sanitized_key,
                path_type=path_type_enum,
            )
        )


def duplicate_original_video_structure(
    input_path_tuple: TInputPath,
    source_resolver: BasePathResolverV1,
    dest_resolver: BasePathResolverV1,
) -> Result[Path, MigrationError]:
    """
    Additionally maps video dir structure
        # Map videos => videos_orig to backup ind videos
        # Map videos* => videos* to backup other video dir structure

    """
    path_str, file_type = input_path_tuple

    if path_str == "." or file_type != "file":
        return Err(
            ParsingError(
                f"Skipped (non-file or '.' entry): '{path_str}'", original_path=path_str
            )
        )

    if not (path_str.startswith("video") and path_str.endswith(".mp4")):
        return Err(
            ParsingError(
                f"Skipped non-video file: '{path_str}'", original_path=path_str
            )
        )

    return (
        parse_path(path_str, source_resolver)
        .and_then(lambda parsed_info: sanitize_key(parsed_info))
        .and_then(lambda sanitized_info: serialize_key(sanitized_info, dest_resolver))
        .and_then(
            lambda path: Ok(
                (
                    Path("videos_orig")
                    if Path(path_str).parent.name == "videos"
                    else Path(path_str).parent
                )
                / path.name
            )
        )
    )


def migrate_single_path(
    input_path_tuple: TInputPath,
    source_resolver: BasePathResolverV1,
    dest_resolver: BasePathResolverV1,
) -> Result[Path, MigrationError]:
    """
    Composes parse, sanitize, and serialize operations for a single input path.
    Returns Ok(new_path) if migration is successful, or Err(MigrationError) if any step fails.
    """
    path_str, file_type = input_path_tuple

    if path_str == "." or file_type != "file":
        return Err(
            ParsingError(
                f"Skipped (non-file or '.' entry): '{path_str}'", original_path=path_str
            )
        )

    # Compose the functions using the Result monad's and_then method
    # The dest_resolver needs to be 'carried through' the chain.
    # Lambdas are used here to create closures over dest_resolver.
    return (
        parse_path(path_str, source_resolver)
        .and_then(lambda parsed_info: sanitize_key(parsed_info))
        .and_then(lambda sanitized_info: serialize_key(sanitized_info, dest_resolver))
    )


def build_resolvers_from_config(model_dir: str | Path):
    """
    Build source/destination path resolvers for a project by reading its config YAML
    from the given model directory (e.g., ``<project>_configs/``).

    Returns a tuple: (source_resolver_v0, dest_resolver_v1)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir not found or not a directory: {model_dir}")

    config_file_path: Path | None = None
    for fname in os.listdir(model_dir):
        if fname.lower().endswith((".yaml", ".yml")):
            config_file_path = model_dir / fname
            break

    if config_file_path is None:
        raise FileNotFoundError(f"No YAML config found in model dir: {model_dir}")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "data" not in config:
        raise KeyError(f"'data' section not found in config: {config_file_path}")

    project_view_names = config["data"].get("view_names", []) or []
    project_keypoint_names = config["data"].get("keypoint_names", []) or []

    # Source: legacy (schema 0)
    src_cfg = ProjectConfig(
        schema_version=0,
        view_names=project_view_names,
        keypoint_names=project_keypoint_names,
    )
    source_resolver = PathResolver.for_project(src_cfg)

    # Destination: v1
    dst_cfg = ProjectConfig(
        schema_version=1,
        view_names=project_view_names,
        keypoint_names=project_keypoint_names,
    )
    dest_resolver = PathResolver.for_project(dst_cfg)

    return source_resolver, dest_resolver


def migrate_directory_structure_core(
    input_paths: List[TInputPath],
    source_resolver: BasePathResolverV1,
    dest_resolver: BasePathResolverV1,
) -> Tuple[List[Tuple[str, Path]], List[str]]:
    """
    Pure core migration: maps input file paths to new paths using resolvers
    via a functional composition of parse, sanitize, and serialize.

    - Skips directory entries and '.' paths before processing.
    - Uses a Result monad to handle success/failure of each step.
    - Returns a tuple: ([(original_path_str, Path(new_path)), ...], [unparsed_file_paths]).
    """
    project_mapping: List[Tuple[str, Path]] = []
    unparsed_files: list[str] = []

    for path_str, file_type in input_paths:
        result = migrate_single_path(
            (path_str, file_type), source_resolver, dest_resolver
        )
        if result.is_ok():
            project_mapping.append((path_str, result.unwrap()))
        else:
            # All failures (skipped, parsing, sanitization, serialization)
            # are collected as unparsed for now.
            unparsed_files.append(path_str)
            # Optionally, you could log result.unwrap_err() for more detailed reporting

    return project_mapping, unparsed_files
