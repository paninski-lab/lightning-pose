"""
Script to convert SLEAP project to LP project

Usage:
$ python slp2lp.py --slp_file /path/to/<project>.pkg.slp --lp_dir /path/to/lp/dir

Arguments:
--slp_file    Path to the SLEAP project file (.pkg.slp)
--lp_dir      Path to the output LP project directory

"""

import argparse
import io
import json
import os

import h5py
import numpy as np
import pandas as pd
from PIL import Image


def extract_video_names_from_pkg_slp(file_path: str) -> dict:
    """Identify video names from .pkg.slp file."""
    video_names = {}
    with h5py.File(file_path, 'r') as hdf_file:
        for video_group_name in hdf_file.keys():
            if video_group_name.startswith('video'):
                source_video_path = f'{video_group_name}/source_video'
                if source_video_path in hdf_file:
                    source_video_json = hdf_file[source_video_path].attrs['json']
                    source_video_dict = json.loads(source_video_json)
                    video_filename = source_video_dict['backend']['filename']
                    video_names[video_group_name] = video_filename
    return video_names


def extract_frames_from_pkg_slp(file_path: str, base_output_dir: str) -> None:
    """Extract frame data from .pkg.slp file and save as png files in output directory."""

    video_names = extract_video_names_from_pkg_slp(file_path)
    if len(video_names) == 0:
        raise RuntimeError("Could not find image data in .pkg.slp file!")

    with h5py.File(file_path, 'r') as hdf_file:

        # Extract and save images for each video
        for video_group, video_filename in video_names.items():
            output_dir = os.path.join(
                base_output_dir, "labeled-data", os.path.basename(video_filename).split('.')[0]
            )
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if video_group in hdf_file and 'video' in hdf_file[video_group]:
                video_data = hdf_file[f'{video_group}/video'][:]
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_names = []
                for i, (img_bytes, frame_number) in enumerate(zip(video_data, frame_numbers)):
                    img = Image.open(io.BytesIO(np.array(img_bytes, dtype=np.uint8)))
                    frame_name = f"img{str(frame_number).zfill(8)}.png"
                    img.save(f"{output_dir}/{frame_name}")
                    frame_names.append(frame_name)
                    print(f"Saved frame {frame_number} as {frame_name}")


def extract_labels_from_pkg_slp(file_path: str, base_output_dir: str | None) -> pd.DataFrame:
    """Extract label data from .pkg.slp file and save as csv file in output directory."""

    video_names = extract_video_names_from_pkg_slp(file_path)
    if len(video_names) == 0:
        raise RuntimeError("Could not find image data in .pkg.slp file!")

    data_frames = []

    with h5py.File(file_path, 'r') as hdf_file:

        # Extract data for each video
        for video_group, video_filename in video_names.items():
            if video_group in hdf_file and 'frames' in hdf_file:
                frames_dataset = hdf_file['frames']
                frame_references = {
                    frame['frame_id']: frame['frame_idx']
                    for frame in frames_dataset
                    if frame['video'] == int(video_group.replace('video', ''))
                }

                # Correct frame references for the current video group
                frame_numbers = hdf_file[f'{video_group}/frame_numbers'][:]
                frame_id_to_number = {
                    frame_id: frame_numbers[idx]
                    for idx, frame_id in enumerate(frame_references.keys())
                }

                # Extract instances and points
                points_dataset = hdf_file['points']
                instances_dataset = hdf_file['instances']

                data = []
                for idx, instance in enumerate(instances_dataset):
                    try:
                        frame_id = instance['frame_id']
                        if frame_id not in frame_id_to_number:
                            continue
                        frame_idx = frame_id_to_number[frame_id]
                        point_id_start = instance['point_id_start']
                        point_id_end = instance['point_id_end']

                        points = points_dataset[point_id_start:point_id_end]

                        keypoints_flat = []
                        for kp in points:
                            x, y = kp['x'], kp['y']
                            if np.isnan(x) or np.isnan(y):
                                x, y = None, None
                            keypoints_flat.extend([x, y])

                        data.append([frame_idx] + keypoints_flat)
                    except Exception as e:
                        print(f"Skipping invalid instance {idx}: {e}")

                if data:
                    metadata_json = hdf_file['metadata'].attrs['json']
                    metadata_dict = json.loads(metadata_json)
                    nodes = metadata_dict['nodes']
                    instance_names = [node['name'] for node in nodes]
                    keypoints = [f'{name}' for name in instance_names]
                    columns = pd.MultiIndex.from_product(
                        [["lightning_tracker"], keypoints, ["x", "y"]],
                        names=["scorer", "bodyparts", "coords"],
                    )
                    video_base_name = os.path.basename(video_filename).split('.')[0]
                    index = [
                        f"labeled-data/{video_base_name}/img{str(int(x[0])).zfill(8)}.png"
                        for x in data
                    ]
                    labels_df = pd.DataFrame([d[1:] for d in data], columns=columns, index=index)
                    data_frames.append(labels_df)

    if data_frames:

        # combine all dataframes
        final_df_ = pd.concat(data_frames)
        final_df = final_df_[~final_df_.index.duplicated(keep='first')]

        # save
        if base_output_dir is not None:
            final_df.to_csv(os.path.join(base_output_dir, "CollectedData.csv"))

        return final_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--slp_file", type=str)
    parser.add_argument("--lp_dir", type=str)
    args = parser.parse_args()
    slp_file = args.slp_file
    lp_dir = args.lp_dir

    print(f"Converting SLEAP project located at {slp_file} to LP project located at {lp_dir}")

    # Check provided SLEAP path exists
    if not os.path.exists(slp_file):
        raise FileNotFoundError(f"did not find the file {slp_file}")

    # Check paths are not the same
    if slp_file == lp_dir:
        raise NameError("slp_file and lp_dir cannot be the same")

    # Extract and save labeled data from SLEAP project
    extract_frames_from_pkg_slp(slp_file, lp_dir)

    # Extract labels and create the required DataFrame
    extract_labels_from_pkg_slp(slp_file, lp_dir)
