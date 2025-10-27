import argparse
import os
import csv
from pathlib import Path

from lightning_pose.utils.paths.migrate import build_resolvers_from_config, migrate_directory_structure_core


def read_directory_structure_from_csv(
    ksikka_base_dir: str,
    project_name: str
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


if __name__ == "__main__":
    import datetime

    parser = argparse.ArgumentParser(description="Generate migration mapping CSVs for all projects in a directory.")
    parser.add_argument(
        "--projects-csv-dir",
        required=True,
        help="Directory containing per-project CSVs and corresponding <project>_configs/ directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: True). Non-dry-run is not implemented yet.",
    )
    args = parser.parse_args()

    base_dir = args.projects_csv_dir
    dry_run = args.dry_run
    if not dry_run:
        raise NotImplementedError("dry_run=False is not implemented yet")

    if not os.path.isdir(base_dir):
        raise SystemExit(f"Provided projects-csv-dir is not a directory: {base_dir}")

    # Discover all CSVs in base_dir
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
            print(f"WARNING: Config dir not found for project '{project_name}': {model_dir}. Skipping.")
            continue

        # Build resolvers
        try:
            source_resolver, dest_resolver = build_resolvers_from_config(model_dir)
        except Exception as e:
            print(f"WARNING: Failed to build resolvers for '{project_name}': {e}. Skipping.")
            continue

        # Read input CSV
        input_paths = read_directory_structure_from_csv(base_dir, project_name)
        if not input_paths:
            print(f"WARNING: No paths found in CSV for '{project_name}'. Skipping.")
            continue

        # Run core migration logic
        mapping, unparsed_files = migrate_directory_structure_core(input_paths, source_resolver, dest_resolver)

        # Write mapping CSV
        out_csv = out_root / f"{project_name}.mapping.csv"
        with open(out_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["original_path", "new_path"])
            for k, v in sorted(mapping.items(), key=lambda x: x[0]):
                writer.writerow([k, str(v)])

        # Write unparsed CSV
        out_unparsed_csv = out_root / f"{project_name}.unparsed.csv"
        with open(out_unparsed_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["original_path"])  # header
            for p in sorted(unparsed_files):
                writer.writerow([p])

        # Stats
        files_in = sum(1 for p, t in input_paths if t == "file")
        dirs_in = sum(1 for p, t in input_paths if t == "directory")
        print(
            f"Project '{project_name}': input files={files_in}, directories={dirs_in}, mapped={len(mapping)}, unparsed={len(unparsed_files)} -> {out_csv} | {out_unparsed_csv}"
        )
        total_projects += 1

    print(f"Done. Wrote mappings for {total_projects} project(s) to: {out_root}")
