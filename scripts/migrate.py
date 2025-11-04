import argparse
import os
import csv
import shutil
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from lightning_pose.utils.paths.migrate import (
    build_resolvers_from_config,
    migrate_directory_structure_core,
    duplicate_original_video_structure,
)


def read_directory_structure_from_disk(base_dir: str) -> list[tuple[str, str]]:
    """
    Traverses a directory and returns a list of file paths.

    Args:
        base_dir: The root directory to scan.

    Returns:
        A list of tuples, where each tuple contains a relative path string
        and its type ("file"), sorted by path.
    """
    input_paths = []
    base_path_obj = Path(base_dir)
    for root, _, files in os.walk(base_dir):
        for f_name in files:
            full_path = Path(root) / f_name
            rel_path = full_path.relative_to(base_path_obj)
            input_paths.append((str(rel_path), "file"))
    return sorted(input_paths, key=lambda x: x[0])


def apply_migration(
    source_directory: str,
    destination_directory: str,
    mapping: List[Tuple[str, Path]],  # Changed from Dict[str, Path]
    dry_run: bool,
) -> Tuple[int, int, List[str]]:
    """
    Applies the generated mapping by hardlinking/copying files from source to a new destination.

    Args:
        source_directory: The root of the original project directory.
        destination_directory: The root of the new, migrated directory structure.
        mapping: A list of tuples where each tuple is (original_relative_path_str, new_relative_path_obj).
        dry_run: If True, prints actions without performing them.

    Returns:
        A tuple: (hardlinks_created, files_copied, failed_files_list)
    """
    print(
        f"Applying migration from '{source_directory}' to '{destination_directory}' (dry_run={dry_run})"
    )

    source_base_path_obj = Path(source_directory)
    dest_base_path_obj = Path(destination_directory)

    # Ensure destination directory exists (or create it)
    if not dest_base_path_obj.exists():
        if dry_run:
            print(f"  Would create destination directory: {dest_base_path_obj}")
        else:
            dest_base_path_obj.mkdir(parents=True, exist_ok=True)
            print(f"  Created destination directory: {dest_base_path_obj}")
    elif not dest_base_path_obj.is_dir():
        raise SystemExit(
            f"Destination path exists but is not a directory: {dest_base_path_obj}"
        )

    hardlinks_created = 0
    files_copied = 0
    failed_files = []

    for original_rel_path_str, new_path_obj in mapping:  # Changed iteration
        original_full_path = source_base_path_obj / original_rel_path_str
        new_full_path = dest_base_path_obj / new_path_obj

        if not original_full_path.exists():
            print(
                f"WARNING: Original source path does not exist, skipping: {original_full_path}"
            )
            continue
        if not original_full_path.is_file():
            print(
                f"WARNING: Original source path is not a file, skipping: {original_full_path}"
            )
            continue

        # Ensure parent directory for the new file in the destination exists
        if not new_full_path.parent.exists():
            if dry_run:
                print(
                    f"  Would create destination parent directory: {new_full_path.parent}"
                )
            else:
                new_full_path.parent.mkdir(parents=True, exist_ok=True)

        action_type = ""
        if original_full_path.suffix.lower() in (".mp4", ".png"):
            action_type = "hardlink"
            action_func = os.link
            print_action = "Hardlinking"
        else:
            action_type = "copy"
            action_func = shutil.copy2
            print_action = "Copying"

        action_msg = (
            f"  Would {action_type}: {original_full_path} -> {new_full_path}"
            if dry_run
            else f"  {print_action}: {original_full_path} -> {new_full_path}"
        )
        print(action_msg)

        if not dry_run:
            try:
                action_func(str(original_full_path), str(new_full_path))
                print(f"    SUCCESS: {print_action} {original_full_path.name}")
                if action_type == "hardlink":
                    hardlinks_created += 1
                else:
                    files_copied += 1
            except OSError as e:
                print(
                    f"    ERROR {action_type} {original_full_path} to {new_full_path}: {e}"
                )
                failed_files.append(original_rel_path_str)

    print("Migration application finished.")
    return hardlinks_created, files_copied, failed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate directory structures based on resolvers."
    )
    parser.add_argument(
        "--source-directory",
        required=True,
        help="The root of the original project directory to migrate.",
    )
    parser.add_argument(
        "--destination-directory",
        help="The root of the new directory where migrated files will be hardlinked/copied (required if not --dry-run).",
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Directory containing resolver configurations for a single project (e.g., 'my_project_configs').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run mode (default: False). If True, no file system changes occur, and mapping CSVs are outputted. If False, actual file system changes will occur.",
    )
    args = parser.parse_args()

    # Validate arguments
    if not Path(args.source_directory).is_dir():
        raise SystemExit(
            f"Provided --source-directory is not a directory or does not exist: {args.source_directory}"
        )
    if not Path(args.config_dir).is_dir():
        raise SystemExit(f"Provided --config-dir is not a directory: {args.config_dir}")

    if not args.dry_run:
        if not args.destination_directory:
            raise SystemExit(
                "--destination-directory is required when not running in --dry-run mode."
            )
        dest_path_obj = Path(args.destination_directory)
        if dest_path_obj.exists() and not dest_path_obj.is_dir():
            raise SystemExit(
                f"Destination path exists but is not a directory: {args.destination_directory}"
            )
        if dest_path_obj.exists() and any(dest_path_obj.iterdir()):
            print(
                f"WAR: Destination directorNINGy '{args.destination_directory}' exists and is not empty."
            )
            response = input(
                "This could overwrite existing files. Continue anyway? (y/N): "
            ).lower()
            if response != "y":
                raise SystemExit("Migration aborted by user.")

    # --- Generate Mapping ---
    print(
        f"Generating mapping from source directory: {args.source_directory} and config: {args.config_dir}"
    )
    try:
        source_resolver, dest_resolver = build_resolvers_from_config(args.config_dir)
    except Exception as e:
        raise SystemExit(
            f"ERROR: Failed to build resolvers for '{args.config_dir}': {e}"
        )

    input_paths = read_directory_structure_from_disk(args.source_directory)
    if not input_paths:
        print(f"No files found in source directory: {args.source_directory}")
        raise SystemExit(0)

    project_mapping, unparsed_files_for_report = migrate_directory_structure_core(
        input_paths, source_resolver, dest_resolver
    )
    print(
        f"Generated {len(project_mapping)} mappings. {len(unparsed_files_for_report)} unparsed files."
    )

    # Additional mapping for video files and unparsed files.
    # map_files("videos/*", "videos_orig/*")
    # map_files("videos*", "videos*")
    output_paths_map2: List[Tuple[str, Path]] = []  # Changed to List[Tuple[str, Path]]
    for path_str, file_type in input_paths:
        result = duplicate_original_video_structure(
            (path_str, file_type), source_resolver, dest_resolver
        )
        if result.is_ok():
            output_paths_map2.append((path_str, result.unwrap()))  # Appending tuple
    project_mapping.extend(output_paths_map2)

    # Additionally store unparsed files in "misc/*"
    output_paths_map3: List[Tuple[str, Path]] = []  # Changed to List[Tuple[str, Path]]
    for path_str in unparsed_files_for_report:
        output_paths_map3.append((path_str, Path("misc") / path_str))  # Appending tuple
    project_mapping.extend(output_paths_map3)

    # --- Save Mapping CSVs ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_dir_name = (
        Path(args.source_directory).name.replace(os.sep, "_").replace(":", "")
    )

    if args.dry_run:
        out_root = Path("outputs") / "dry_run_mappings" / timestamp / safe_dir_name
    else:
        out_root = Path("outputs") / "migration_reports" / timestamp / safe_dir_name

    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / "generated.mapping.csv"
    with open(out_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "new_path"])
        for k, v in sorted(project_mapping, key=lambda x: x[0]):  # Changed iteration
            writer.writerow([k, str(v)])
    print(f"Saved generated mapping to {out_csv}")

    if unparsed_files_for_report:
        out_unparsed_csv = out_root / "generated.unparsed.csv"
        with open(out_unparsed_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["original_path"])
            for p in sorted(unparsed_files_for_report):
                writer.writerow([p])
        print(f"Saved unparsed files report to {out_unparsed_csv}")

    # --- Apply Migration or Report Dry Run ---
    if args.dry_run:
        print(f"\nDry Run Summary:")
        print(f"  Mapping generated for {len(project_mapping)} files.")
        print(f"  {len(unparsed_files_for_report)} files were unparsed.")
        print(f"  No actual file system changes were made.")
    else:
        hardlinks, copied, failed_ops = apply_migration(
            args.source_directory,
            args.destination_directory,
            project_mapping,
            args.dry_run,
        )
        print(f"\nMigration Summary (dry_run={args.dry_run}):")
        print(f"  Hardlinks created: {hardlinks}")
        print(f"  Files copied: {copied}")
        if failed_ops:
            print(f"  Failed operations: {len(failed_ops)} (see above for details)")
