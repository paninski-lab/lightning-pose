"""Convert a SLEAP ``.pkg.slp`` package into a Lightning Pose project directory.

Only single-view, single-animal SLEAP projects are supported.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def _decode_json_attr(raw: bytes | str) -> dict:
    """Decode an h5py ``json`` attribute, which may come back as ``str`` or ``bytes``.

    h5py returns ``numpy.bytes_`` for fixed-length HDF5 string attributes and native ``str``
    for variable-length UTF-8 ones, depending on how the writer encoded them. Calling ``str()``
    on the former yields its Python repr (e.g. ``"b'{...}'"``), not the decoded text, so bytes
    must be decoded explicitly rather than passed through ``str()``.
    """
    if isinstance(raw, bytes):
        return json.loads(raw.decode('utf-8'))
    return json.loads(str(raw))


def _extract_video_names(pkg_slp_file: Path) -> dict[str, str]:
    """Map each ``video<N>`` group in a ``.pkg.slp`` file to its source video filename."""
    video_names = {}
    with h5py.File(pkg_slp_file, 'r') as hdf_file:
        for video_group_name in hdf_file.keys():
            if video_group_name.startswith('video'):
                source_video_path = f'{video_group_name}/source_video'
                if source_video_path in hdf_file:
                    source_video_ds = cast(h5py.Dataset, hdf_file[source_video_path])
                    source_video_dict = _decode_json_attr(source_video_ds.attrs['json'])
                    video_names[video_group_name] = source_video_dict['backend']['filename']
    return video_names


def _extract_frames(pkg_slp_file: Path, lp_dir: Path) -> None:
    """Extract embedded frame images from a ``.pkg.slp`` file and save them as PNGs.

    Args:
        pkg_slp_file: path to the ``.pkg.slp`` file.
        lp_dir: destination Lightning Pose project directory; frames are written to
            ``lp_dir/labeled-data/<video_stem>/``.

    Raises:
        RuntimeError: if no embedded video groups are found in the file.
    """
    video_names = _extract_video_names(pkg_slp_file)
    if len(video_names) == 0:
        raise RuntimeError('could not find image data in .pkg.slp file!')

    with h5py.File(pkg_slp_file, 'r') as hdf_file:
        for video_group, video_filename in video_names.items():
            output_dir = lp_dir / 'labeled-data' / Path(video_filename).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            has_pixels = video_group in hdf_file and 'video' in cast(
                h5py.Group, hdf_file[video_group],
            )
            if has_pixels:
                video_ds = cast(h5py.Dataset, hdf_file[f'{video_group}/video'])
                frame_numbers_ds = cast(h5py.Dataset, hdf_file[f'{video_group}/frame_numbers'])
                zipped = zip(video_ds[:], frame_numbers_ds[:], strict=True)
                for img_bytes, frame_number in zipped:
                    img_buffer = np.array(img_bytes, dtype=np.uint8).tobytes()
                    img = Image.open(io.BytesIO(img_buffer))
                    frame_name = f'img{str(frame_number).zfill(8)}.png'
                    img.save(output_dir / frame_name)
                    logger.info(f'saved frame {frame_number} as {frame_name}')


def _extract_labels(pkg_slp_file: Path) -> pd.DataFrame | None:
    """Extract keypoint labels from a ``.pkg.slp`` file as a Lightning Pose label DataFrame.

    Args:
        pkg_slp_file: path to the ``.pkg.slp`` file.

    Returns:
        the labels as a DataFrame with a 3-level ``(scorer, bodyparts, coords)`` column
        MultiIndex, indexed by ``labeled-data/<video_stem>/<image>.png``; or None if no
        instances were found.

    Raises:
        RuntimeError: if no embedded video groups are found in the file.
    """
    video_names = _extract_video_names(pkg_slp_file)
    if len(video_names) == 0:
        raise RuntimeError('could not find image data in .pkg.slp file!')

    data_frames = []
    with h5py.File(pkg_slp_file, 'r') as hdf_file:
        for video_group, video_filename in video_names.items():
            if video_group not in hdf_file or 'frames' not in hdf_file:
                continue

            frames_dataset = cast(h5py.Dataset, hdf_file['frames'])
            frame_references = {
                frame['frame_id']: frame['frame_idx']
                for frame in frames_dataset
                if frame['video'] == int(video_group.replace('video', ''))
            }
            frame_numbers_ds = cast(h5py.Dataset, hdf_file[f'{video_group}/frame_numbers'])
            frame_numbers = frame_numbers_ds[:]
            frame_id_to_number = {
                frame_id: frame_numbers[idx]
                for idx, frame_id in enumerate(frame_references.keys())
            }

            points_dataset = cast(h5py.Dataset, hdf_file['points'])
            instances_dataset = cast(h5py.Dataset, hdf_file['instances'])

            data = []
            for idx, instance in enumerate(instances_dataset):
                try:
                    frame_id = instance['frame_id']
                    if frame_id not in frame_id_to_number:
                        continue
                    frame_idx = frame_id_to_number[frame_id]
                    points = points_dataset[instance['point_id_start']:instance['point_id_end']]

                    keypoints_flat = []
                    for kp in points:
                        x, y = kp['x'], kp['y']
                        if np.isnan(x) or np.isnan(y) or not kp['visible'] or not kp['complete']:
                            x, y = None, None
                        keypoints_flat.extend([x, y])

                    data.append([frame_idx] + keypoints_flat)
                except Exception as e:
                    logger.warning(f'skipping invalid instance {idx}: {e}')

            if data:
                metadata_ds = cast(h5py.Dataset, hdf_file['metadata'])
                metadata_dict = _decode_json_attr(metadata_ds.attrs['json'])
                keypoints = [node['name'] for node in metadata_dict['nodes']]
                columns = pd.MultiIndex.from_product(
                    [['lightning_tracker'], keypoints, ['x', 'y']],
                    names=['scorer', 'bodyparts', 'coords'],
                )
                video_stem = Path(video_filename).stem
                index = pd.Index(
                    f'labeled-data/{video_stem}/img{str(int(row[0])).zfill(8)}.png'
                    for row in data
                )
                data_frames.append(
                    pd.DataFrame([row[1:] for row in data], columns=columns, index=index)
                )

    if not data_frames:
        return None

    labels_df = pd.concat(data_frames)
    return cast(pd.DataFrame, labels_df[~labels_df.index.duplicated(keep='first')])


def convert(slp_file: Path, lp_dir: Path) -> None:
    """Convert a SLEAP ``.pkg.slp`` package into a Lightning Pose project directory.

    Args:
        slp_file: path to the ``.pkg.slp`` file.
        lp_dir: destination Lightning Pose project directory; created if it does not exist.

    Raises:
        FileNotFoundError: if ``slp_file`` does not exist.
        ValueError: if ``slp_file`` and ``lp_dir`` are the same path.
        RuntimeError: if no embedded frame data is found in ``slp_file``.
    """
    logger.info(f'converting SLEAP project at {slp_file} to LP project at {lp_dir}')

    if not slp_file.is_file():
        raise FileNotFoundError(f'did not find the file {slp_file}')
    if slp_file.resolve() == lp_dir.resolve():
        raise ValueError('slp_file and lp_dir cannot be the same')

    lp_dir.mkdir(parents=True, exist_ok=True)

    _extract_frames(slp_file, lp_dir)

    labels_df = _extract_labels(slp_file)
    if labels_df is not None:
        labels_df.to_csv(lp_dir / 'CollectedData.csv')
