from . import rename_time_directories

MIGRATIONS = [
    rename_time_directories,
    # Add other migration modules here as needed
]


def run_migrations() -> None:
    """Runs all pending migrations."""

    for migration_module in MIGRATIONS:
        if migration_module.needs_migration():
            migration_module.migrate()
