#!/usr/bin/env python3
from __future__ import annotations
import os
import shutil
import argparse
import csv
from pathlib import Path
from typing import List, Callable


"""
Generic File Migration and Transformation Script Template


How to Use This Template:

1. Download this file and save it under a new name (e.g., `my_project_migration.py`)
2. Implement `process_file` to define how each file should be processed.
3. Run the Script with --dry-run:
        `python my_project_migration.py /path/to/source_data /path/to/new_structure --dry-run`
4. Run the script without --dry-run
"""


def process_file(source_path: Path, input_dir: Path, output_dir: Path) -> List[FileOp]:
    """
    Args:
        source_path: The full path to the original file in the input directory.
        input_dir: The root of the input directory being scanned.
        output_dir: The root of the output directory where new files will be placed.

    Returns:
        A list of FileOp objects representing the planned actions.
    """
    # ==============================================================================
    # This is the primary section to modify the script
    # ==============================================================================

    # Example: Copies all files to output dir without any changes.
    relative_path_from_input = source_path.relative_to(input_dir)
    destination_path = output_dir / relative_path_from_input

    if source_path.name.startswith('.'):
        return []
    # Other than `copy` the template provides built-in FileOp builders:
    # - `hardlink`
    # - `softlink`
    # - `process_csv`
    # - `process_yaml`
    return [copy(source_path, destination_path, input_dir, output_dir)]


# --- Core FileOp Command Pattern (no need to modify) ---

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


# --- Common FileOp Builder Functions (customize if default behavior is not sufficient) ---

def hardlink(source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path) -> FileOp:
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


def copy(source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path) -> FileOp:
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


def softlink(source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path) -> FileOp:
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


def process_csv(source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path) -> FileOp:
    """
    Builds a ProcessCSV operation.
    This function demonstrates how to read, modify, and write a CSV file.
    Customize the 'modified_rows = all_rows' line for your specific CSV transformations.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'PROCESS_CSV':<12}: {relative_source} -> {relative_dest} (Custom Rewrite)"

    def action():
        # 1. Read all rows from the source CSV
        with open(source_path, 'r', newline='') as infile:
            reader = csv.reader(infile)
            all_rows = list(reader)

        # 2. Apply custom processing
        # ------------------------------------------------------------------
        # !!! CUSTOMIZE THIS LINE FOR YOUR CSV TRANSFORMATION LOGIC !!!
        modified_rows = all_rows # Placeholder: currently no modification
        # Example: modified_rows = [row for row in all_rows if row[0] != 'skip_this_row']
        # Example: modified_rows = [['new_header_1', 'new_header_2']] + all_rows[1:]
        # ------------------------------------------------------------------

        # 3. Write the modified rows to the destination CSV
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # Using 'w' mode overwrites existing files.
        with open(destination_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerows(modified_rows)

    return FileOp(description, action)


def process_yaml(source_path: Path, destination_path: Path, input_dir: Path, output_dir: Path) -> FileOp:
    """
    Builds a ProcessYAML operation.
    This function demonstrates how to read, modify, and write a YAML file.
    Customize the 'modified_data = data' line for your specific YAML transformations.
    If the destination already exists, it will be overwritten.
    """
    relative_source = source_path.relative_to(input_dir)
    relative_dest = destination_path.relative_to(output_dir)
    description = f"{'PROCESS_YAML':<12}: {relative_source} -> {relative_dest} (Custom Rewrite)"

    def action():
        # Import pyyaml here, making it an optional dependency.
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "YAML processing requires the 'pyyaml' library. Please install it (`pip install pyyaml`).")

        # 1. Read YAML content
        with open(source_path, 'r') as infile:
            # Use FullLoader if available (better safety and compatibility)
            loader = getattr(yaml, 'FullLoader', yaml.SafeLoader) # Fallback to SafeLoader
            data = yaml.load(infile, Loader=loader)

        # 2. Apply custom processing
        # ------------------------------------------------------------------
        # !!! CUSTOMIZE THIS LINE FOR YOUR YAML TRANSFORMATION LOGIC !!!
        modified_data = data # Placeholder: currently no modification
        # Example: if 'old_key' in data: data['new_key'] = data.pop('old_key')
        # ------------------------------------------------------------------

        # 3. Write modified data to the destination YAML file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        # Using 'w' mode overwrites existing files.
        with open(destination_path, 'w') as outfile:
            dumper = getattr(yaml, 'SafeDumper', yaml.Dumper) # Fallback to Dumper
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
    fileops = build_fileops(input_dir, output_dir)
    print(f"Planned {len(fileops)} operations.")

    if dry_run:
        for op in fileops:
            print(op.describe())
        print("--- DRY RUN COMPLETE. Review the planned operations above. ---")

    else:
        print(f"--- EXECUTING MIGRATION to: {output_dir} ---")

        if output_dir.exists() and output_dir.is_dir() and any(output_dir.iterdir()):
            print(f"Error: Output directory '{output_dir}' already exists and is not empty.")
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
        print(f"MIGRATION COMPLETE: {success_count} successful, {failure_count} failed.")


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
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_directory",
        type=str,
        help="The source directory containing the files to be processed."
    )
    parser.add_argument(
        "output_directory",
        type=str,
        help="The destination directory where the migrated/transformed files will be placed. "
             "Existing files at the destination will be overwritten."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, the script will only plan and describe the operations without executing them. "
             "No files will be modified on disk. Always use this first!"
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
