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
