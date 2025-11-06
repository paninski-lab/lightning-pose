from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple, List, Generic, TypeVar, Callable, Any

import yaml

from lightning_pose.data.datatypes import ProjectConfig
from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    ViewName,
)
from lightning_pose.utils.paths import PathParseException, ResourceType
from lightning_pose.utils.paths.base_project_schema_v1 import BaseProjectSchemaV1
from lightning_pose.utils.paths.project_schema import ProjectSchema

TInputPath = Tuple[str, str]  # (path_str, "file"|"directory")



def _sanitize_key(parsed_key: Any) -> Any:
    """
    Sanitize keys returned by a resolver's reverse() to ensure they can be fed
    into the destination resolver's get().
    Mirrors the helper previously defined in scripts/old_migrate.py.
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


ParsedKeys = Tuple[Any, ResourceType]  # (parsed_key, path_type_enum)
SanitizedInfo = Tuple[Any, ResourceType]  # (sanitized_key, path_type_enum)


def parse_path(
    path_str: str, source_resolver: BaseProjectSchemaV1
) -> ParsedKeys:
    """
    Attempts to parse a path string using all available PathTypes in the source resolver.
    Returns Ok((parsed_key, path_type)) on success, or Err(ParsingError) on failure.
    """
    for path_type_enum in ResourceType:
        try:
            parsed_key = source_resolver.for_(path_type_enum).reverse(path_str)
            if isinstance(parsed_key, bool) and not parsed_key:
                # Specific error condition from original code
                raise PathParseException("Parsed key is boolean False")
            return parsed_key, path_type_enum
        except PathParseException:
            # Continue to next PathType if this one fails to parse
            continue
    raise PathParseException(f"No PathType could parse the path: '{path_str}'")


def sanitize_key(parsed_info: ParsedKeys) -> SanitizedInfo:
    """
    Sanitizes a parsed key. Returns Ok((sanitized_key, path_type)) on success,
    or Err(SanitizationError) if _sanitize_key were to fail (currently it doesn't raise).
    """
    parsed_key, path_type_enum = parsed_info
    sanitized_key = _sanitize_key(parsed_key)
    return sanitized_key, path_type_enum


def get_path(
    sanitized_key: Any, path_type_enum: ResourceType, dest_resolver: BaseProjectSchemaV1
) -> Path:
    """
    Serializes a sanitized key into a new Path object using the destination resolver.
    Returns Ok(new_path) on success, or Err(SerializationError) on failure.
    """

    if isinstance(sanitized_key, bool):
        # Predicate resources: no key args
        assert sanitized_key
        return dest_resolver.for_(path_type_enum).get()
    # Always pass the key as a single argument, even if it is a tuple
    return dest_resolver.for_(path_type_enum).get(sanitized_key)


def build_resolvers_from_config(search_dir: str | Path):
    """
    Build source/destination path resolvers for a project by reading its config YAML
    from the given model directory (e.g., ``<project>_configs/``).

    Returns a tuple: (source_resolver_v0, dest_resolver_v1)
    """
    search_dir = Path(search_dir)
    if not search_dir.exists() or not search_dir.is_dir():
        raise FileNotFoundError(f"Search dir not found or not a directory: {search_dir}")

    config_file_path: Path | None = None
    for fname in os.listdir(search_dir):
        if fname.lower().endswith((".yaml", ".yml")):
            config_file_path = search_dir / fname
            break

    if config_file_path is None:
        raise FileNotFoundError(f"No YAML config found in model dir: {search_dir}")

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
    source_resolver = ProjectSchema.for_project(src_cfg)

    # Destination: v1
    dst_cfg = ProjectConfig(
        schema_version=1,
        view_names=project_view_names,
        keypoint_names=project_keypoint_names,
    )
    dest_resolver = ProjectSchema.for_project(dst_cfg)

    return source_resolver, dest_resolver


