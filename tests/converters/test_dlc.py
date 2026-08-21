"""Test the function convert (and its helper _load_labels_for_video_dir)."""

from unittest.mock import patch

import pandas as pd
import pytest

from lightning_pose.converters.dlc import _load_labels_for_video_dir, convert


def _make_dlc_project(dlc_dir, video_dir_name='vid1', keypoints=('nose', 'tail')):
    """Build a minimal DLC project directory with one labeled video and one video file."""
    video_dir = dlc_dir / 'labeled-data' / video_dir_name
    video_dir.mkdir(parents=True)

    image_path = f'labeled-data/{video_dir_name}/img00000000.png'
    columns = pd.MultiIndex.from_product(
        [['scorer'], keypoints, ['x', 'y']], names=['scorer', 'bodyparts', 'coords'],
    )
    df = pd.DataFrame([[10, 20, 30, 40]], columns=columns, index=pd.Index([image_path]))
    df.index.name = None
    df.to_csv(video_dir / 'CollectedData_scorer.csv')

    (dlc_dir / image_path).touch()

    videos_dir = dlc_dir / 'videos'
    videos_dir.mkdir()
    (videos_dir / f'{video_dir_name}.mp4').touch()

    return image_path


class TestConvert:
    """Test the function convert."""

    def test_convert_writes_labels_and_copies_data(self, tmp_path):
        dlc_dir = tmp_path / 'dlc_proj'
        lp_dir = tmp_path / 'lp_proj'
        image_path = _make_dlc_project(dlc_dir)

        convert(dlc_dir, lp_dir)

        labels_csv = lp_dir / 'CollectedData.csv'
        assert labels_csv.exists()
        df = pd.read_csv(labels_csv, header=[0, 1, 2], index_col=0)
        assert list(df.index) == [image_path]
        assert (lp_dir / image_path).exists()
        assert (lp_dir / 'videos' / 'vid1.mp4').exists()

    def test_convert_creates_empty_videos_dir_when_source_has_none(self, tmp_path):
        dlc_dir = tmp_path / 'dlc_proj'
        lp_dir = tmp_path / 'lp_proj'
        _make_dlc_project(dlc_dir)
        (dlc_dir / 'videos' / 'vid1.mp4').unlink()
        (dlc_dir / 'videos').rmdir()

        convert(dlc_dir, lp_dir)

        assert (lp_dir / 'videos').is_dir()
        assert list((lp_dir / 'videos').iterdir()) == []

    def test_skips_dot_and_labeled_dirs(self, tmp_path):
        dlc_dir = tmp_path / 'dlc_proj'
        lp_dir = tmp_path / 'lp_proj'
        _make_dlc_project(dlc_dir)
        (dlc_dir / 'labeled-data' / '.DS_Store').mkdir()
        (dlc_dir / 'labeled-data' / 'vid1_labeled').mkdir()

        convert(dlc_dir, lp_dir)

        df = pd.read_csv(lp_dir / 'CollectedData.csv', header=[0, 1, 2], index_col=0)
        assert len(df) == 1

    def test_missing_dlc_dir_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            convert(tmp_path / 'does_not_exist', tmp_path / 'lp_dir')

    def test_same_dlc_dir_and_lp_dir_raises(self, tmp_path):
        dlc_dir = tmp_path / 'dlc_proj'
        _make_dlc_project(dlc_dir)
        with pytest.raises(ValueError, match='cannot be the same'):
            convert(dlc_dir, dlc_dir)

    def test_missing_image_after_copy_raises(self, tmp_path):
        dlc_dir = tmp_path / 'dlc_proj'
        lp_dir = tmp_path / 'lp_proj'
        image_path = _make_dlc_project(dlc_dir)
        (dlc_dir / image_path).unlink()

        with pytest.raises(FileNotFoundError):
            convert(dlc_dir, lp_dir)


class TestLoadLabelsForVideoDir:
    """Test the function _load_labels_for_video_dir."""

    def test_returns_none_when_no_collected_data_found(self, tmp_path):
        video_dir = tmp_path / 'labeled-data' / 'vid1'
        video_dir.mkdir(parents=True)
        assert _load_labels_for_video_dir(tmp_path, 'vid1') is None

    def test_falls_back_to_h5(self, tmp_path):
        video_dir = tmp_path / 'labeled-data' / 'vid1'
        video_dir.mkdir(parents=True)
        (video_dir / 'CollectedData_scorer.h5').touch()

        columns = pd.MultiIndex.from_product(
            [['scorer'], ['nose'], ['x', 'y']], names=['scorer', 'bodyparts', 'coords'],
        )
        fake_df = pd.DataFrame(
            [[1, 2]], columns=columns,
            index=pd.Index(['labeled-data/vid1/img00000000.png']),
        )

        with patch('pandas.read_hdf', return_value=fake_df) as mock_read_hdf:
            result = _load_labels_for_video_dir(tmp_path, 'vid1')

        mock_read_hdf.assert_called_once()
        assert result is not None
        assert list(result.index) == ['labeled-data/vid1/img00000000.png']

    def test_h5_multiindex_scheme_rewrites_index(self, tmp_path):
        video_dir = tmp_path / 'labeled-data' / 'vid1'
        video_dir.mkdir(parents=True)
        (video_dir / 'CollectedData_scorer.h5').touch()

        columns = pd.MultiIndex.from_product(
            [['scorer'], ['nose'], ['x', 'y']], names=['scorer', 'bodyparts', 'coords'],
        )
        multi_index = pd.MultiIndex.from_tuples(
            [('labeled-data', 'vid1', 'img00000000.png')],
        )
        fake_df = pd.DataFrame([[1, 2]], columns=columns, index=multi_index)

        with patch('pandas.read_hdf', return_value=fake_df):
            result = _load_labels_for_video_dir(tmp_path, 'vid1')

        assert result is not None
        assert list(result.index) == ['labeled-data/vid1/img00000000.png']
