"""Convert a DeepLabCut project directory into a Lightning Pose project directory."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

import pandas as pd

from lightning_pose.utils.io import fix_empty_first_row

logger = logging.getLogger(__name__)


def _load_labels_for_video_dir(dlc_dir: Path, video_dir_name: str) -> pd.DataFrame | None:
    """Load the ``CollectedData`` labels for a single ``labeled-data/<video_dir_name>`` dir.

    Tries the CSV export first, falling back to the HDF5 export. Both DLC labeling schemes
    are supported: the older scheme indexes rows by ``labeled-data/<video>/<image>.png``
    directly, and the newer scheme splits the video and image name into separate columns
    (or separate MultiIndex levels for the HDF5 export), which are recombined into that same
    index format here.

    Args:
        dlc_dir: root of the DLC project.
        video_dir_name: name of the ``labeled-data`` subdirectory to load labels for.

    Returns:
        the labels as a DataFrame with a 3-level ``(scorer, bodyparts, coords)`` column
        MultiIndex, or None if no ``CollectedData*`` file was found for this directory.
    """
    video_dir = dlc_dir / 'labeled-data' / video_dir_name
    csv_files = sorted(video_dir.glob('CollectedData*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0], header=[0, 1, 2], index_col=0)
        df = fix_empty_first_row(df)
        if len(df.index.unique()) != df.shape[0]:
            # new DLC labeling scheme that splits video/image in different cells
            vids = df.loc[:, ('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2')]
            imgs = df.loc[:, ('Unnamed: 2_level_0', 'Unnamed: 2_level_1', 'Unnamed: 2_level_2')]
            new_index = [f'labeled-data/{v}/{i}' for v, i in zip(vids, imgs, strict=True)]
            df = df.drop(
                columns=[
                    ('Unnamed: 1_level_0', 'Unnamed: 1_level_1', 'Unnamed: 1_level_2'),
                    ('Unnamed: 2_level_0', 'Unnamed: 2_level_1', 'Unnamed: 2_level_2'),
                ],
            )
            df.index = new_index
        return df

    h5_files = sorted(video_dir.glob('CollectedData*.h5'))
    if h5_files:
        df = cast(pd.DataFrame, pd.read_hdf(h5_files[0]))
        if isinstance(df.index, pd.MultiIndex):
            # new DLC labeling scheme that splits video/image in different cells
            imgs = [i[2] for i in df.index]
            vids = [df.index[0][1] for _ in imgs]
            new_index = [f'labeled-data/{v}/{i}' for v, i in zip(vids, imgs, strict=True)]
            df = df.reset_index(drop=True)
            df.index = new_index
        return df

    logger.warning(f'could not find labels for {video_dir_name}; skipping')
    return None


def convert(dlc_dir: Path, lp_dir: Path) -> None:
    """Convert a DeepLabCut project directory into a Lightning Pose project directory.

    Reads every ``labeled-data/<video>/CollectedData*.{csv,h5}`` file under ``dlc_dir``,
    concatenates them into a single ``CollectedData.csv``, and copies the ``labeled-data/``
    and ``videos/`` directories over to ``lp_dir``.

    Args:
        dlc_dir: root of the DLC project (must contain a ``labeled-data`` subdirectory).
        lp_dir: destination Lightning Pose project directory; created if it does not exist.
            Must not already contain a ``labeled-data`` or ``videos`` subdirectory.

    Raises:
        NotADirectoryError: if ``dlc_dir`` does not exist.
        ValueError: if ``dlc_dir`` and ``lp_dir`` are the same path.
    """
    logger.info(f'converting DLC project at {dlc_dir} to LP project at {lp_dir}')

    if not dlc_dir.is_dir():
        raise NotADirectoryError(f'did not find the directory {dlc_dir}')
    if dlc_dir.resolve() == lp_dir.resolve():
        raise ValueError('dlc_dir and lp_dir cannot be the same')

    video_dir_names = sorted(
        p.name for p in (dlc_dir / 'labeled-data').iterdir()
        if p.is_dir() and not p.name.startswith('.') and not p.name.endswith('_labeled')
    )

    dfs = []
    for video_dir_name in video_dir_names:
        logger.info(video_dir_name)
        df = _load_labels_for_video_dir(dlc_dir, video_dir_name)
        if df is not None:
            dfs.append(df)
    df_all = pd.concat(dfs)

    lp_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(lp_dir / 'CollectedData.csv')

    shutil.copytree(dlc_dir / 'labeled-data', lp_dir / 'labeled-data')

    src_videos = dlc_dir / 'videos'
    dst_videos = lp_dir / 'videos'
    if src_videos.exists():
        logger.info('copying video files')
        shutil.copytree(src_videos, dst_videos)
    else:
        logger.info('DLC video directory does not exist; creating empty video directory')
        dst_videos.mkdir(parents=True, exist_ok=True)

    for image_path in df_all.index:
        if not (lp_dir / image_path).exists():
            raise FileNotFoundError(
                f'expected image file not found after conversion: {image_path}'
            )
