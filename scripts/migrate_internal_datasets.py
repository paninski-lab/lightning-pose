#!/usr/bin/env python3
from __future__ import annotations
import os
import shutil
import argparse
import csv
from collections import namedtuple
from pathlib import Path
from typing import List, Callable, TypeVar, Generic, Any

from lightning_pose.utils.paths import PathParseException
from lightning_pose.utils.paths.base_path_resolver_v1 import BasePathResolverV1
from lightning_pose.utils.paths.migrate import (
    build_resolvers_from_config,
    parse_path,
    get_path,
    _sanitize_key,
)

# These will be set by run_migration()
source_resolver: BasePathResolverV1
dest_resolver: BasePathResolverV1


def process_file(source_path: Path, input_dir: Path, output_dir: Path) -> List[FileOp]:
    """
    Args:
        source_path: The full path to the original file in the input directory.
        input_dir: The root of the input directory being scanned.
        output_dir: The root of the output directory where new files will be placed.

    Returns:
        A list of FileOp objects representing the planned actions.
    """

    relative_path_from_input = source_path.relative_to(input_dir)
    path_str = str(relative_path_from_input)

    try:
        global source_resolver, dest_resolver

        # Uncomment to include videos-for-each-labeled-frame.
        # It's excluded to reduce file size.
        """
        # Rewrite videos-for-each-labeled-frame so parser can handle that.
        if "videos-for-each-labeled-frame" in path_str:
            new_file_name = relative_path_from_input.parts[1] + "_" + relative_path_from_input.name
            relative_path_from_input = Path("videos-for-each-labeled-frame") / new_file_name
            path_str = str(relative_path_from_input)
        """
        if "config" in path_str and path_str.endswith(".yaml"):
            new_path = Path("configs") / "default.yaml"
        else:
            # Compute new path
            new_path = (
                Pipeline(parse_path(path_str, source_resolver))
                .then(lambda x: Object(keys=_sanitize_key(x[0]), path_type=x[1]))
                .then(lambda x: get_path(x.keys, x.path_type, dest_resolver))
                .get()
            )

        # Depending on the filetype, handle it differently
        if new_path.suffix == ".mp4":
            # Store all videos in /videos.
            ops = [hardlink(source_path, output_dir / new_path, input_dir, output_dir)]
            # Additionally recreate original directory structure using hardlinks.
            # Map videos from `videos` to `videos_ind`.
            assert new_path.parts[0].startswith("videos")
            assert len(relative_path_from_input.parts) == 2
            additional_parent_dir = relative_path_from_input.parent.name
            additional_parent_dir = (
                "videos_ind"
                if additional_parent_dir == "videos"
                else additional_parent_dir
            )
            ops.append(
                hardlink(
                    source_path,
                    output_dir / additional_parent_dir / new_path.name,
                    input_dir,
                    output_dir,
                )
            )
            # Return either 1 or 2 ops.
            return ops
        elif new_path.suffix == ".csv":
            return [
                process_csv(source_path, output_dir / new_path, input_dir, output_dir)
            ]
        elif new_path.suffix == ".yaml":
            return [
                process_yaml(source_path, output_dir / new_path, input_dir, output_dir)
            ]
        elif new_path.suffix == ".png":
            return [hardlink(source_path, output_dir / new_path, input_dir, output_dir)]
        elif new_path.suffix == ".jpg":
            raise NotImplementedError("Need to implement png to jpg conversion")
        else:
            return [copy(source_path, output_dir / new_path, input_dir, output_dir)]

    except PathParseException:
        pass

    return [drop(source_path, input_dir)]


T = TypeVar("T")
U = TypeVar("U")


class Pipeline(Generic[T]):
    def __init__(self, value: T):
        self._value: T = value

    @staticmethod
    def lift(value: U) -> "Pipeline[U]":
        return Pipeline(value)

    def then(self, func: Callable[[T], U]) -> "Pipeline[U]":
        try:
            new_value = func(self._value)
            return Pipeline(new_value)
        except Exception as e:
            raise RuntimeError(f"Error during chain step: {e}") from e

    def get(self) -> T:
        return self._value


class Object:

    def __new__(cls, **kwargs: Any):
        field_names = tuple(kwargs.keys())
        field_values = tuple(kwargs.values())
        TypeName = "_Anon_" + "_".join(field_names)
        DynamicType = namedtuple(TypeName, field_names)
        return DynamicType(*field_values)


class FileOp:
    """
    Represents a single planned file system operation (e.g., copy, link, modify).
    This design pattern separates the "what to do" (description) from the "how to do it" (action).
    """

    def __init__(self, description: str, action: Callable[[], None]):
        self._description = description
        self._action = action

    def describe(self) -> str:
        """Returns a human-readable description of the operation for dry runs."""
        return self._description

    def execute(self) -> None:
        """Performs the actual file system operation."""
        return self._action()


def hardlink(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a hard-link operation.
    Hardlinks create a new directory entry for existing file content. They must be on the same filesystem.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'HARDLINK':<12}: {relative_source} -> {relative_dest}"

    def action():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            # Overwrite: remove existing file/link before creating a new one
            destination_path.unlink()
        os.link(source_path, destination_path)

    return FileOp(description, action)


def copy(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a copy operation (copies file content and metadata).
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'COPY':<12}: {relative_source} -> {relative_dest}"

    def action():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy2 overwrites existing files, so no explicit unlink is needed for files.
        # If destination_path refers to an existing directory, shutil.copy2 will raise an error.
        shutil.copy2(source_path, destination_path)

    return FileOp(description, action)


def drop(source_path: Path, input_dir: Path) -> FileOp:
    """
    Builds a copy operation (copies file content and metadata).
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    description = f"{'DROP':<12}: {relative_source} -> None"

    def action():
        pass

    return FileOp(description, action)


def softlink(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a symbolic link (softlink) operation.
    Softlinks are pointers to another file or directory and can span filesystems.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'SOFTLINK':<12}: {relative_source} -> {relative_dest}"

    def action():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            # Overwrite: remove existing file/link before creating a new one
            destination_path.unlink()
        os.symlink(source_path, destination_path)

    return FileOp(description, action)


def process_video_file(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a ProcessVideo operation.
    This function demonstrates how to read, modify, and write a CSV file.
    Customize the 'modified_rows = all_rows' line for your specific CSV transformations.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'PROCESS_VIDEO':<12}: {relative_source} -> {relative_dest}"

    def action():
        # Use FFMPEG to transcode the video to h264, YUV, every frame is a keyframe.
        # TODO implement. For now, softlink.
        hardlink(source_path, destination_path, input_dir, output_dir).execute()

    return FileOp(description, action)


def process_csv(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a ProcessCSV operation.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)

    description = f"{'PROCESS_CSV':<12}: {relative_source} -> {relative_dest}"

    def value_map_fn(value: str) -> str:
        if value.isnumeric():
            return value
        global source_resolver, dest_resolver
        try:
            new_path = (
                Pipeline(parse_path(value, source_resolver))
                .then(lambda x: Object(keys=_sanitize_key(x[0]), path_type=x[1]))
                .then(lambda x: get_path(x.keys, x.path_type, dest_resolver))
                .get()
            )
            return str(new_path)
        except PathParseException:
            return value

    def action():
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(source_path, "r", newline="") as infile,
            open(destination_path, "w", newline="") as outfile,
        ):
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                new_row = [value_map_fn(value) for value in row]
                writer.writerow(new_row)

    return FileOp(description, action)


def process_yaml(
    source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path
) -> FileOp:
    """
    Builds a ProcessYAML operation.
    This function demonstrates how to read, modify, and write a YAML file.
    Customize the 'modified_data = data' line for your specific YAML transformations.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'PROCESS_YAML':<12}: {relative_source} -> {relative_dest}"

    def action():
        # Import pyyaml here, making it an optional dependency.
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "YAML processing requires the 'pyyaml' library. Please install it (`pip install pyyaml`)."
            )

        # 1. Read YAML content
        with open(source_path, "r") as infile:
            # Use FullLoader if available (better safety and compatibility)
            loader = getattr(
                yaml, "FullLoader", yaml.SafeLoader
            )  # Fallback to SafeLoader
            data = yaml.load(infile, Loader=loader)

        # 2. Apply custom processing
        # ------------------------------------------------------------------
        # !!! CUSTOMIZE THIS LINE FOR YOUR YAML TRANSFORMATION LOGIC !!!
        modified_data = data  # Placeholder: currently no modification
        # Example: if 'old_key' in data: data['new_key'] = data.pop('old_key')
        # ------------------------------------------------------------------

        # 3. Write modified data to the destination YAML file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # Using 'w' mode overwrites existing files.
        with open(destination_path, "w") as outfile:
            dumper = getattr(yaml, "SafeDumper", yaml.Dumper)  # Fallback to Dumper
            yaml.dump(modified_data, outfile, Dumper=dumper)

    return FileOp(description, action)


# ==============================================================================
# 🧩 USER CUSTOMIZATION AREA END
#
# Below this line are the core functions for planning and executing the migration.
# Generally, you should not need to modify these unless you're extending
# the script's fundamental behavior (e.g., changing how file scanning works).
# ==============================================================================


def build_fileops(input_dir: Path, output_dir: Path) -> List[FileOp]:
    """
    Walks the input directory and generates a list of all FileOp objects.
    This function uses the `process_file` logic defined by the user.
    """
    all_ops: List[FileOp] = []

    for root, _, files in os.walk(input_dir):
        root_path = Path(root)

        for file_name in files:
            source_path = root_path / file_name

            # Call the user-defined planning function (process_file)
            ops = process_file(source_path, input_dir, output_dir)
            all_ops.extend(ops)

    return all_ops


def run_migration(input_dir: Path, output_dir: Path, dry_run: bool):
    """
    Executes the migration based on the `dry_run` flag.
    In dry-run mode, it only prints the planned operations.
        In actual run mode, it executes the operations and reports success/failure.
    """

    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Mode:             {'Dry Run' if dry_run else 'Execution'}")
    print("Building Migration Plan...")
    global source_resolver, dest_resolver
    source_resolver, dest_resolver = build_resolvers_from_config(input_dir)

    fileops = build_fileops(input_dir, output_dir)
    print(f"Planned {len(fileops)} operations.")

    if dry_run:
        for op in fileops:
            print(op.describe())
        print("--- DRY RUN COMPLETE. Review the planned operations above. ---")

    else:
        print(f"--- EXECUTING MIGRATION to: {output_dir} ---")

        if output_dir.exists() and output_dir.is_dir() and any(output_dir.iterdir()):
            print(
                f"Error: Output directory '{output_dir}' already exists and is not empty."
            )
            raise SystemExit(1)

        # Ensure output directory exists before execution starts
        # This will create parent directories if they don't exist.
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured to exist.")

        success_count = 0
        failure_count = 0

        for op in fileops:
            # Execute the operation
            try:
                op.execute()
                # If execute() completes without raising an exception
                print(f"[SUCCESS] {op.describe()}")
                success_count += 1
            except Exception as e:
                # Catch any exception raised by the action
                print(f"[FAILURE] {op.describe()} | ERROR: {e}")
                failure_count += 1

        print("-" * 30)
        print(
            f"MIGRATION COMPLETE: {success_count} successful, {failure_count} failed."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A generic and customizable script for migrating, reorganizing, and transforming files.\n"
            "Download this script, modify the `process_file` function (and optionally other `FileOp` builders)\n"
            "to define your specific migration logic, and then run it.\n\n"
            "Example Usage:\n"
            "  # Preview changes:\n"
            "  python YOUR_SCRIPT_NAME.py /path/to/old_project /path/to/new_project --dry-run\n"
            "  # Execute changes:\n"
            "  python YOUR_SCRIPT_NAME.py /path/to/old_project /path/to/new_project\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="The source directory containing the files to be processed.",
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="The destination directory where the migrated/transformed files will be placed. "
        "Existing files at the destination will be overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the script will only plan and describe the operations without executing them. "
        "No files will be modified on disk. Always use this first!",
    )

    args = parser.parse_args()

    # Use Path objects and resolve for absolute paths for robustness
    input_path = Path(args.input_directory).resolve()
    output_path = Path(args.output_directory).resolve()

    if not input_path.is_dir():
        print(f"Error: Input directory not found or is not a directory: {input_path}")
        raise SystemExit(1)
    if input_path == output_path:
        print(f"Error: Input and output directories cannot be the same: {input_path}")
        print("Please specify distinct directories to prevent data loss.")
        raise SystemExit(1)

    # Run the main process
    try:
        run_migration(input_path, output_path, args.dry_run)
    except Exception as e:
        print(f"\nAn unexpected error occurred during migration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
