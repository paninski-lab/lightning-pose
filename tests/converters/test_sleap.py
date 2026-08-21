"""Test the sleap converter's convert function and its private extraction helpers."""

import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from lightning_pose.converters.sleap import (
    _extract_frames,
    _extract_labels,
    _extract_video_names,
    convert,
)


def _make_pkg_slp(
    path,
    video_filename='vid1.mp4',
    keypoints=('nose', 'tail'),
    include_frames=True,
    include_labels=True,
):
    """Build a minimal single-video, single-instance .pkg.slp file for testing."""
    png_bytes = io.BytesIO()
    Image.new('RGB', (2, 2)).save(png_bytes, format='PNG')
    png_array = np.frombuffer(png_bytes.getvalue(), dtype=np.uint8)

    with h5py.File(path, 'w') as f:
        source_video = f.create_dataset('video0/source_video', data=[0])
        source_video.attrs['json'] = json.dumps({'backend': {'filename': video_filename}})
        f.create_dataset('video0/frame_numbers', data=np.array([0], dtype='i8'))

        if include_frames:
            video_ds = f.create_dataset(
                'video0/video', shape=(1,), dtype=h5py.special_dtype(vlen=np.uint8),
            )
            video_ds[0] = png_array

        if include_labels:
            frames_dtype = np.dtype([('frame_id', 'i8'), ('video', 'i8'), ('frame_idx', 'i8')])
            f.create_dataset('frames', data=np.array([(0, 0, 0)], dtype=frames_dtype))

            points_dtype = np.dtype(
                [('x', 'f8'), ('y', 'f8'), ('visible', 'bool'), ('complete', 'bool')],
            )
            points = np.array(
                [(10.0, 20.0, True, True), (30.0, 40.0, True, True)], dtype=points_dtype,
            )
            f.create_dataset('points', data=points)

            instances_dtype = np.dtype(
                [('frame_id', 'i8'), ('point_id_start', 'i8'), ('point_id_end', 'i8')],
            )
            f.create_dataset('instances', data=np.array([(0, 0, 2)], dtype=instances_dtype))

            metadata = f.create_dataset('metadata', data=[0])
            metadata.attrs['json'] = json.dumps(
                {'nodes': [{'name': kp} for kp in keypoints]},
            )


class TestExtractVideoNames:
    """Test the function _extract_video_names."""

    def test_extracts_source_video_filename(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file)
        assert _extract_video_names(slp_file) == {'video0': 'vid1.mp4'}

    def test_no_videos_returns_empty_dict(self, tmp_path):
        slp_file = tmp_path / 'empty.pkg.slp'
        with h5py.File(slp_file, 'w'):
            pass
        assert _extract_video_names(slp_file) == {}


class TestExtractFrames:
    """Test the function _extract_frames."""

    def test_writes_frame_images(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file)
        lp_dir = tmp_path / 'lp_dir'

        _extract_frames(slp_file, lp_dir)

        assert (lp_dir / 'labeled-data' / 'vid1' / 'img00000000.png').exists()

    def test_no_video_groups_raises(self, tmp_path):
        slp_file = tmp_path / 'empty.pkg.slp'
        with h5py.File(slp_file, 'w'):
            pass

        with pytest.raises(RuntimeError, match='could not find image data'):
            _extract_frames(slp_file, tmp_path / 'lp_dir')

    def test_no_embedded_pixels_writes_nothing(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file, include_frames=False, include_labels=False)
        lp_dir = tmp_path / 'lp_dir'

        _extract_frames(slp_file, lp_dir)

        assert list((lp_dir / 'labeled-data' / 'vid1').iterdir()) == []


class TestExtractLabels:
    """Test the function _extract_labels."""

    def test_extracts_keypoints(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file, include_frames=False)

        df = _extract_labels(slp_file)

        assert df is not None
        assert list(df.index) == ['labeled-data/vid1/img00000000.png']
        assert df.loc[
            'labeled-data/vid1/img00000000.png', ('lightning_tracker', 'nose', 'x')
        ] == 10.0
        assert df.loc[
            'labeled-data/vid1/img00000000.png', ('lightning_tracker', 'tail', 'y')
        ] == 40.0

    def test_no_instances_returns_none(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file, include_frames=False, include_labels=False)
        frames_dtype = np.dtype([('frame_id', 'i8'), ('video', 'i8'), ('frame_idx', 'i8')])
        points_dtype = np.dtype(
            [('x', 'f8'), ('y', 'f8'), ('visible', 'bool'), ('complete', 'bool')],
        )
        instances_dtype = np.dtype(
            [('frame_id', 'i8'), ('point_id_start', 'i8'), ('point_id_end', 'i8')],
        )
        with h5py.File(slp_file, 'a') as f:
            f.create_dataset('frames', data=np.zeros(0, dtype=frames_dtype))
            f.create_dataset('points', data=np.zeros(0, dtype=points_dtype))
            f.create_dataset('instances', data=np.zeros(0, dtype=instances_dtype))

        assert _extract_labels(slp_file) is None


class TestConvert:
    """Test the function convert."""

    def test_convert_writes_frames_and_labels(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file)
        lp_dir = tmp_path / 'lp_dir'

        convert(slp_file, lp_dir)

        assert (lp_dir / 'labeled-data' / 'vid1' / 'img00000000.png').exists()
        assert (lp_dir / 'CollectedData.csv').exists()

    def test_missing_slp_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert(tmp_path / 'does_not_exist.pkg.slp', tmp_path / 'lp_dir')

    def test_same_slp_file_and_lp_dir_raises(self, tmp_path):
        slp_file = tmp_path / 'project.pkg.slp'
        _make_pkg_slp(slp_file)
        with pytest.raises(ValueError, match='cannot be the same'):
            convert(slp_file, slp_file)
