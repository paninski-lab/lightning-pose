import argparse
import os
import csv
import shutil  # New import for file operations
import datetime
from pathlib import Path
from typing import List, Dict, Tuple  # Updated import for type hints

from lightning_pose.utils.paths.migrate import (
    build_resolvers_from_config,
    migrate_directory_structure_core,
)


def read_directory_structure_from_csv(
    ksikka_base_dir: str, project_name: str
) -> list[tuple[str, str]]:
    """
    Reads the directory structure from a CSV file for a given project.

    Args:
        ksikka_base_dir: Base directory containing project CSV files.
        project_name: The name of the project.

    Returns:
        A list of tuples, where each tuple contains a path string
        and its type ("file" or "directory"), sorted by path.
    """
    file_list = []
    csv_file_path = os.path.join(ksikka_base_dir, f"{project_name}.csv")

    if not os.path.exists(csv_file_path):
        print(
            f"ERROR: CSV file for project '{project_name}' not found at '{csv_file_path}'"
        )
        return []

    with open(csv_file_path, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                path = row[0].strip()
                file_type_abbr = row[1].strip()
                if file_type_abbr == "f":
                    file_list.append((path, "file"))
                elif file_type_abbr == "d":
                    file_list.append((path, "directory"))
    return sorted(file_list, key=lambda x: x[0])


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
    mapping: Dict[str, Path],
    dry_run: bool,
) -> Tuple[int, int, List[str]]:
    """
    Applies the generated mapping by hardlinking/copying files from source to a new destination.

    Args:
        source_directory: The root of the original project directory.
        destination_directory: The root of the new, migrated directory structure.
        mapping: A dictionary where keys are original relative paths (str)
                 and values are new relative paths (Path object).
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

    for original_rel_path_str, new_path_obj in mapping.items():
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
        if original_full_path.suffix.lower() == ".mp4":
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
        "--mode",
        choices=["generate_mapping", "execute_migration"],
        default="generate_mapping",
        help="Operation mode: 'generate_mapping' (default) creates CSVs, 'execute_migration' applies mapping to a directory.",
    )
    # Arguments for generate_mapping mode
    parser.add_argument(
        "--projects-csv-dir",
        help="Directory containing per-project CSVs and corresponding <project>_configs/ directories (for generate_mapping mode).",
    )
    # Arguments for execute_migration mode
    parser.add_argument(
        "--source-directory",
        help="The root of the original project directory to migrate (for execute_migration mode).",
    )
    parser.add_argument(
        "--destination-directory",
        help="The root of the new directory where migrated files will be hardlinked/copied (for execute_migration mode).",
    )
    parser.add_argument(
        "--config-dir",
        help="Directory containing resolver configurations for a single project (e.g., 'my_project_configs') (for execute_migration mode).",
    )
    parser.add_argument(
        "--mapping-csv",
        help="Optional: Pre-generated mapping CSV file (original_path,new_path) to use for execution. If not provided, mapping will be generated from --source-directory and --config-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True). If False, actual file system changes will occur for execute_migration mode.",
    )
    args = parser.parse_args()

    dry_run = args.dry_run

    if args.mode == "generate_mapping":
        if not args.projects_csv_dir:
            raise SystemExit(
                "--projects-csv-dir is required for 'generate_mapping' mode."
            )
        if not os.path.isdir(args.projects_csv_dir):
            raise SystemExit(
                f"Provided --projects-csv-dir is not a directory: {args.projects_csv_dir}"
            )

        base_dir = args.projects_csv_dir
        csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
        if not csv_files:
            print("No project CSV files found in the specified directory.")
            raise SystemExit(0)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        out_root = Path("outputs") / "migrations" / timestamp
        out_root.mkdir(parents=True, exist_ok=True)

        total_projects = 0
        for csv_name in sorted(csv_files):
            project_name = csv_name[:-4]
            model_dir = os.path.join(base_dir, f"{project_name}_configs")
            if not os.path.isdir(model_dir):
                print(
                    f"WARNING: Config dir not found for project '{project_name}': {model_dir}. Skipping."
                )
                continue

            try:
                source_resolver, dest_resolver = build_resolvers_from_config(model_dir)
            except Exception as e:
                print(
                    f"WARNING: Failed to build resolvers for '{project_name}': {e}. Skipping."
                )
                continue

            input_paths = read_directory_structure_from_csv(base_dir, project_name)
            if not input_paths:
                print(f"WARNING: No paths found in CSV for '{project_name}'. Skipping.")
                continue

            mapping, unparsed_files = migrate_directory_structure_core(
                input_paths, source_resolver, dest_resolver
            )

            out_csv = out_root / f"{project_name}.mapping.csv"
            with open(out_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["original_path", "new_path"])
                for k, v in sorted(mapping.items(), key=lambda x: x[0]):
                    writer.writerow([k, str(v)])

            out_unparsed_csv = out_root / f"{project_name}.unparsed.csv"
            with open(out_unparsed_csv, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["original_path"])
                for p in sorted(unparsed_files):
                    writer.writerow([p])

            files_in = sum(1 for p, t in input_paths if t == "file")
            print(
                f"Project '{project_name}': input files={files_in}, mapped={len(mapping)}, unparsed={len(unparsed_files)} -> {out_csv} | {out_unparsed_csv}"
            )
            total_projects += 1
        print(f"Done. Wrote mappings for {total_projects} project(s) to: {out_root}")

    elif args.mode == "execute_migration":
        if not args.source_directory:
            raise SystemExit(
                "--source-directory is required for 'execute_migration' mode."
            )
        if not Path(args.source_directory).is_dir():
            raise SystemExit(
                f"Provided --source-directory is not a directory or does not exist: {args.source_directory}"
            )
        if not args.destination_directory:
            raise SystemExit(
                "--destination-directory is required for 'execute_migration' mode."
            )

        dest_path_obj = Path(args.destination_directory)
        if dest_path_obj.exists() and not dest_path_obj.is_dir():
            raise SystemExit(
                f"Destination path exists but is not a directory: {args.destination_directory}"
            )
        if dest_path_obj.exists() and any(dest_path_obj.iterdir()):
            print(
                f"WARNING: Destination directory '{args.destination_directory}' exists and is not empty."
            )
            if not dry_run:  # Only exit if not dry-run and directory is not empty
                response = input(
                    "This could overwrite existing files. Continue anyway? (y/N): "
                ).lower()
                if response != "y":
                    raise SystemExit("Migration aborted by user.")

        if not args.config_dir and not args.mapping_csv:
            raise SystemExit(
                "Either --config-dir or --mapping-csv must be provided for 'execute_migration' mode."
            )
        if args.config_dir and not Path(args.config_dir).is_dir():
            raise SystemExit(
                f"Provided --config-dir is not a directory: {args.config_dir}"
            )
        if args.mapping_csv and not Path(args.mapping_csv).is_file():
            raise SystemExit(
                f"Provided --mapping-csv is not a file: {args.mapping_csv}"
            )

        project_mapping: Dict[str, Path] = {}
        unparsed_files_for_report: List[str] = []

        if args.mapping_csv:
            print(f"Reading mapping from provided CSV: {args.mapping_csv}")
            with open(args.mapping_csv, mode="r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                if header != ["original_path", "new_path"]:
                    raise ValueError(
                        f"Invalid mapping CSV header: {header}. Expected ['original_path', 'new_path']"
                    )
                for row in reader:
                    if len(row) == 2:
                        project_mapping[row[0]] = Path(row[1])
            print(f"Loaded {len(project_mapping)} mappings from {args.mapping_csv}")

        else:  # Generate mapping on the fly
            if not args.config_dir:
                raise SystemExit(
                    "--config-dir is required when --mapping-csv is not provided for 'execute_migration' mode."
                )
            print(
                f"Generating mapping from source directory: {args.source_directory} and config: {args.config_dir}"
            )
            try:
                source_resolver, dest_resolver = build_resolvers_from_config(
                    args.config_dir
                )
            except Exception as e:
                raise SystemExit(
                    f"ERROR: Failed to build resolvers for '{args.config_dir}': {e}"
                )

            input_paths_from_disk = read_directory_structure_from_disk(
                args.source_directory
            )
            if not input_paths_from_disk:
                print(f"No files found in source directory: {args.source_directory}")
                raise SystemExit(0)

            project_mapping, unparsed_files_for_report = (
                migrate_directory_structure_core(
                    input_paths_from_disk, source_resolver, dest_resolver
                )
            )
            print(
                f"Generated {len(project_mapping)} mappings. {len(unparsed_files_for_report)} unparsed files."
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            safe_dir_name = (
                Path(args.source_directory).name.replace(os.sep, "_").replace(":", "")
            )
            out_root = (
                Path("outputs")
                / "migration_execution_reports"
                / timestamp
                / safe_dir_name
            )
            if not dry_run:
                out_root.mkdir(parents=True, exist_ok=True)

                out_csv = out_root / "generated.mapping.csv"
                with open(out_csv, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["original_path", "new_path"])
                    for k, v in sorted(project_mapping.items(), key=lambda x: x[0]):
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

        hardlinks, copied, failed_ops = apply_migration(
            args.source_directory, args.destination_directory, project_mapping, dry_run
        )
        print(f"\nMigration Summary (dry_run={dry_run}):")
        print(f"  Hardlinks created: {hardlinks}")
        print(f"  Files copied: {copied}")
        if failed_ops:
            print(f"  Failed operations: {len(failed_ops)} (see above for details)")

        if unparsed_files_for_report:
            print(
                "\nWARNING: The following files were not parsed and were not included in the migration:"
            )
            for f in unparsed_files_for_report:
                print(f"  - {Path(args.source_directory) / f}")
            print("Please review these files manually.")

    else:
        raise SystemExit(
            "Invalid mode specified. Choose 'generate_mapping' or 'execute_migration'."
        )
