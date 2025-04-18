import os
import re
from pathlib import Path


def needs_migration():
    """Checks if the time directory rename migration is needed."""
    outputs_path = Path("outputs")
    if not outputs_path.is_dir():
        return
    for date_dir in outputs_path.iterdir():
        if date_dir.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
            for time_dir in date_dir.iterdir():
                if time_dir.is_dir() and re.match(r"\d{2}:\d{2}:\d{2}", time_dir.name):
                    return True
    return False


def migrate():
    """Renames time directories."""
    print(
        "Fixing directory names (https://github.com/paninski-lab/lightning-pose/issues/278)..."
    )
    outputs_path = Path("outputs")
    migration_applied = False
    for date_dir in outputs_path.iterdir():
        if re.match(r"\d{4}-\d{2}-\d{2}", date_dir.name):
            for time_dir in date_dir.iterdir():
                if re.match(r"\d{2}:\d{2}:\d{2}", time_dir.name):
                    new_name = time_dir.name.replace(":", "-")
                    new_path = time_dir.parent / new_name
                    try:
                        os.rename(time_dir, new_path)
                        print(f"Renamed: {time_dir} -> {new_path}")
                        migration_applied = True
                    except OSError as e:
                        print(f"Error renaming {time_dir}: {e}")
    return migration_applied
