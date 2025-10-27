from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict, List
import os
import re
import yaml

from lightning_pose.data.datatypes import ProjectConfig
from lightning_pose.data.keys import (
    VideoFileKey,
    FrameKey,
    ViewName,
    SessionKey,
    LabelFileKey,
)
from lightning_pose.utils.paths.path_resolver import PathResolver
from lightning_pose.utils.paths import PathParseException, PathType


TInputPath = Tuple[str, str]  # (path_str, "file"|"directory")


def _sanitize_key(parsed_key):
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
        raise FileNotFoundError(
            f"No YAML config found in model dir: {model_dir}")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f) or {}

    if "data" not in config:
        raise KeyError(
            f"'data' section not found in config: {config_file_path}")

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
    source_resolver,
    dest_resolver,
) -> Tuple[Dict[str, Path], List[str]]:
    """
    Pure core migration: map input file paths to new paths using resolvers.

    - Skips directory entries.
    - Tries all PathType parsers until one reverse() succeeds.
    - Handles tuple/boolean/None keys and sanitizes them before dest get().
    - Returns a tuple: ({original_path_str: Path(new_path)}, [unparsed_file_paths]).
    """
    output_paths_map: dict[str, Path] = {}
    unparsed_files: list[str] = []

    for path_str, file_type in input_paths:
        if path_str == ".":
            continue
        if file_type != "file":
            continue

        parsed_successfully = False
        parsed_key = None
        for path_type_enum in PathType:
            try:
                parsed_key = source_resolver.for_(path_type_enum).reverse(path_str)
                if isinstance(parsed_key, bool) and not parsed_key:
                    raise PathParseException()

                sanitized_key = _sanitize_key(parsed_key)

                if isinstance(sanitized_key, tuple) and type(sanitized_key).__name__ == "tuple":
                    new_path = dest_resolver.for_(path_type_enum).get(*sanitized_key)
                elif isinstance(sanitized_key, bool):
                    assert sanitized_key
                    new_path = dest_resolver.for_(path_type_enum).get()
                else:
                    new_path = dest_resolver.for_(path_type_enum).get(sanitized_key)

                output_paths_map[path_str] = new_path
                parsed_successfully = True
                break
            except PathParseException:
                continue
            except Exception:
                # fail this file, move on
                parsed_successfully = False
                break
        if not parsed_successfully:
            unparsed_files.append(path_str)

    return output_paths_map, unparsed_files


def migrate_directory_structure(
    data_dir: str | Path,
    model_dir: str | Path,
    dry_run: bool = True,
) -> Dict[str, Path]:
    """
    High-level wrapper for future direct filesystem migrations.

    For now, only validates arguments and constructs resolvers. It returns an empty
    mapping; the CSV-driven workflow should call `migrate_directory_structure_core`.
    """
    if not dry_run:
        raise NotImplementedError("dry_run=False is not implemented yet")

    _ = Path(data_dir)  # reserved for future use
    source_resolver, dest_resolver = build_resolvers_from_config(model_dir)
    # Without a list of input files, there's nothing to migrate here.
    return {}
