from lightning_pose.utils.paths.project_schema import ProjectSchema
from pathlib import Path
from lightning_pose.data.datatypes import ProjectDirs
base = Path("/media/ksikka/data/untar_datasets/fly_anipose_migrations/fly-anipose_new_migrated")
schema = ProjectSchema.for_project(ProjectDirs(data_dir=base))
video_keys = schema.videos.list_keys()

for v_key in video_keys:
  print(f"{v_key.session_key}, {v_key.view}")

# keys and relative paths to label files
key_and_path = [
    (schema.label_files.parse_path(p), p)
    for p in schema.label_files.iter_paths()
]  # list[tuple[LabelFileKey, ViewName], Path]

for (labelfilekey, view), path in key_and_path:
    print(f"{labelfilekey}, {view}, {path}")