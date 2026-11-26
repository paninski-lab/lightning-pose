"""Test the io module."""

import logging
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import DictConfig

from lightning_pose.utils import io as io_utils
from lightning_pose.utils.io import (
    ckpt_path_from_base_path,
    get_context_img_paths,
    resolve_path,
    return_absolute_data_paths,
    return_absolute_path,
)


class TestReturnAbsoluteDataPaths:
    """Test the function return_absolute_data_paths."""

    def test_return_absolute_data_paths_missing_video_dir_ok(self, tmp_path):
        cfg = DictConfig({'data_dir': str(tmp_path), 'video_dir': 'videos'})
        data_dir, video_dir = return_absolute_data_paths(cfg)
        assert data_dir == str(tmp_path)
        assert video_dir == str(tmp_path / 'videos')

    def test_return_absolute_data_paths_null_video_dir(self, tmp_path):
        cfg = DictConfig({'data_dir': str(tmp_path), 'video_dir': None})
        assert return_absolute_data_paths(cfg) == (str(tmp_path), None)

    def test_return_absolute_data_paths_absolute_video_dir(self, tmp_path):
        cfg = DictConfig({'data_dir': str(tmp_path), 'video_dir': str(tmp_path / 'elsewhere')})
        assert return_absolute_data_paths(cfg)[1] == str(tmp_path / 'elsewhere')

    def test_return_absolute_data_paths_missing_data_dir_fails(self, tmp_path):
        cfg = DictConfig({'data_dir': str(tmp_path / 'missing'), 'video_dir': 'videos'})
        with pytest.raises(OSError):
            return_absolute_data_paths(cfg)


class TestResolvePath:
    """Test the function resolve_path."""

    def test_resolve_path_relative_to_root(self, tmp_path):
        (tmp_path / 'labels.csv').touch()
        assert resolve_path('labels.csv', tmp_path) == tmp_path / 'labels.csv'

    def test_resolve_path_absolute(self, tmp_path):
        root = tmp_path / 'root'
        root.mkdir()
        (tmp_path / 'labels.csv').touch()
        assert resolve_path(tmp_path / 'labels.csv', root) == tmp_path / 'labels.csv'

    def test_resolve_path_prefers_root_over_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / 'root'
        root.mkdir()
        (root / 'labels.csv').touch()
        (tmp_path / 'labels.csv').touch()
        monkeypatch.chdir(tmp_path)
        assert resolve_path('labels.csv', root) == root / 'labels.csv'

    def test_resolve_path_falls_back_to_cwd(self, tmp_path, monkeypatch):
        root = tmp_path / 'root'
        root.mkdir()
        (tmp_path / 'labels.csv').touch()
        monkeypatch.chdir(tmp_path)
        assert resolve_path('labels.csv', root) == Path('labels.csv')

    def test_resolve_path_missing_lists_locations(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='labels.csv.*looked in'):
            resolve_path('labels.csv', tmp_path)


class TestReturnAbsolutePath:
    """Test the return_absolute_path function."""

    def test_returns_absolute_path_unchanged(self, tmp_path):
        """Absolute paths that exist are returned as-is."""
        result = return_absolute_path(str(tmp_path))
        assert result == str(tmp_path)

    def test_raises_for_nonexistent_absolute_path(self, tmp_path):
        """Raises OSError when an absolute path does not exist."""
        missing = str(tmp_path / 'does_not_exist')
        with pytest.raises(OSError, match='is not a valid path'):
            return_absolute_path(missing)

    def test_relative_path_resolved_from_cwd(self, tmp_path, monkeypatch):
        """Relative path is resolved n_dirs_back from cwd and returned when it exists."""
        # build a fake cwd that is n_dirs_back=1 below tmp_path
        fake_cwd = tmp_path / 'a'
        fake_cwd.mkdir()
        target = tmp_path / 'data'
        target.mkdir()
        monkeypatch.chdir(fake_cwd)

        result = return_absolute_path('data', n_dirs_back=1)
        assert result == str(target)

    def test_relative_path_raises_when_resolved_path_missing(self, tmp_path, monkeypatch):
        """Raises OSError when the resolved relative path does not exist."""
        fake_cwd = tmp_path / 'a'
        fake_cwd.mkdir()
        monkeypatch.chdir(fake_cwd)

        with pytest.raises(OSError, match='is not a valid path'):
            return_absolute_path('nonexistent_dir', n_dirs_back=1)

    def test_multirun_strips_extra_dir(self, tmp_path, monkeypatch):
        """When the resolved parent directory is named 'multirun', one extra level is removed."""
        # structure: tmp_path/multirun/run1  →  cwd
        # with n_dirs_back=1, naive resolution → tmp_path/multirun
        # but multirun stripping should give → tmp_path
        multirun_dir = tmp_path / 'multirun'
        run_dir = multirun_dir / 'run1'
        run_dir.mkdir(parents=True)
        target = tmp_path / 'data'
        target.mkdir()
        monkeypatch.chdir(run_dir)

        result = return_absolute_path('data', n_dirs_back=1)
        assert result == str(target)


def test_ckpt_path_from_base_path_no_checkpoints(tmp_path: Path):
    """
    Test Case 1: No checkpoint files found.
    Expects None.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    (base_path / logging_dir_name / model_name / "version_0" / "checkpoints").mkdir(parents=True)

    result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert result is None


def test_ckpt_path_from_base_path_one_best_checkpoint(tmp_path: Path):
    """
    Test Case 2: Exactly one "best" checkpoint found in the latest version.
    Expects the path to the best checkpoint.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    # Create older version checkpoints
    (base_path / logging_dir_name / model_name / "version_0" / "checkpoints").mkdir(parents=True)
    (base_path / logging_dir_name / model_name / "version_0"
     / "checkpoints" / "epoch=10-step=100.ckpt").touch()

    # Create latest version checkpoints with one best
    latest_version_path = base_path / logging_dir_name / model_name / "version_1" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    (latest_version_path / "epoch=5-step=50.ckpt").touch()
    expected_ckpt = latest_version_path / "epoch=12-step=120-best.ckpt"
    expected_ckpt.touch()

    result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert result == str(expected_ckpt)


def test_ckpt_path_from_base_path_multiple_best_checkpoints(tmp_path: Path, caplog):
    """
    Test Case 3: Multiple "best" checkpoints found in the latest version.
    Expects the checkpoint with the highest step count to be returned.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    latest_version_path = base_path / logging_dir_name / model_name / "version_1" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    (latest_version_path / "epoch=5-step=50-best.ckpt").touch()
    expected_ckpt = latest_version_path / "epoch=12-step=120-best.ckpt"
    expected_ckpt.touch()

    with caplog.at_level(logging.WARNING):
        result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)

    assert result == str(expected_ckpt)
    assert any("Multiple 'best' checkpoint files found" in r.message for r in caplog.records)


def test_ckpt_path_from_base_path_highest_step_count(tmp_path: Path, caplog):
    """
    Test Case 4: No "best" checkpoint, but multiple checkpoints, pick the one with highest step
    count in latest version.
    Expects the path to the checkpoint with the highest step.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    # Create older version checkpoints
    (base_path / logging_dir_name / model_name / "version_0" / "checkpoints").mkdir(parents=True)
    (base_path / logging_dir_name / model_name / "version_0"
     / "checkpoints" / "epoch=10-step=100.ckpt").touch()

    # Create latest version checkpoints without a best, but with varying steps
    latest_version_path = base_path / logging_dir_name / model_name / "version_1" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    (latest_version_path / "epoch=5-step=50.ckpt").touch()
    expected_ckpt = latest_version_path / "epoch=12-step=120.ckpt"
    (latest_version_path / "epoch=12-step=120.ckpt").touch()
    (latest_version_path / "epoch=11-step=110.ckpt").touch()

    with caplog.at_level(logging.WARNING):
        result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert result == str(expected_ckpt)
    assert any("No 'best' checkpoint found" in r.message for r in caplog.records)


def test_ckpt_path_from_base_path_single_checkpoint_no_best(tmp_path: Path, caplog):
    """
    Test Case 5: No "best" checkpoint, and only one checkpoint file in the latest version.
    Expects the path to that single checkpoint.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    # Create older version checkpoints
    (base_path / logging_dir_name / model_name / "version_0" / "checkpoints").mkdir(parents=True)
    (base_path / logging_dir_name / model_name / "version_0"
     / "checkpoints" / "epoch=10-step=100.ckpt").touch()

    # Create latest version with only one checkpoint, no best
    latest_version_path = base_path / logging_dir_name / model_name / "version_1" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    expected_ckpt = latest_version_path / "epoch=5-step=50.ckpt"
    expected_ckpt.touch()

    with caplog.at_level(logging.WARNING):
        result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert result == str(expected_ckpt)
    assert any("No 'best' checkpoint found" in r.message for r in caplog.records)


def test_ckpt_path_from_base_path_cannot_parse_step(tmp_path: Path, caplog):
    """
    Test Case 6: No "best" checkpoint, multiple checkpoints, but step count cannot be parsed.
    Expects ValueError.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "tb_logs"

    # Create latest version with multiple checkpoints, no best, and unparsable steps
    latest_version_path = base_path / logging_dir_name / model_name / "version_1" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    (latest_version_path / "model_v1_A.ckpt").touch()
    (latest_version_path / "model_v1_B.ckpt").touch()

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError, match="cannot parse step counts to determine latest"):
            ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert any("No 'best' checkpoint found" in r.message for r in caplog.records)


def test_ckpt_path_from_base_path_custom_logging_dir_name(tmp_path: Path):
    """
    Test Case 7: Different `logging_dir_name`.
    Expects the correct path using the custom logging directory.
    """
    base_path = tmp_path / "project"
    model_name = "my_model"
    logging_dir_name = "my_custom_logs"

    # Create directory structure with custom logging_dir_name
    latest_version_path = base_path / logging_dir_name / model_name / "version_0" / "checkpoints"
    latest_version_path.mkdir(parents=True)
    expected_ckpt = latest_version_path / "epoch=1-step=10-best.ckpt"
    expected_ckpt.touch()

    result = ckpt_path_from_base_path(str(base_path), model_name, logging_dir_name)
    assert result == str(expected_ckpt)










def test_get_context_img_paths():
    assert get_context_img_paths(Path("a/b/c/img_2.png")) == [
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_1.png"),
        Path("a/b/c/img_2.png"),
        Path("a/b/c/img_3.png"),
        Path("a/b/c/img_4.png"),
    ]

    assert get_context_img_paths(Path("a/b/c/img_0200.png")) == [
        Path("a/b/c/img_0198.png"),
        Path("a/b/c/img_0199.png"),
        Path("a/b/c/img_0200.png"),
        Path("a/b/c/img_0201.png"),
        Path("a/b/c/img_0202.png"),
    ]

    # Test negative indices floored to 0.
    assert get_context_img_paths(Path("a/b/c/img_1.png")) == [
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_0.png"),
        Path("a/b/c/img_1.png"),
        Path("a/b/c/img_2.png"),
        Path("a/b/c/img_3.png"),
    ]


def test_fix_empty_first_row(tmp_path):
    # Set up a df that we're trying to fix
    test_csv = textwrap.dedent(
        """
        scorer,test,test,test,test
        bodyparts,bp_1,bp_1,bp_2,bp_2
        coords,x,y,x,y
        labeled-data/test1.png,,,,
        labeled-data/test2.png,,,,
        labeled-data/test3.png,1.0,2.0,3.0,4.0
        """
    )
    test_csv_path = tmp_path / "test.csv"
    test_csv_path.write_text(test_csv)

    df = pd.read_csv(test_csv_path, header=[0, 1, 2], index_col=0)
    # Confirm this df has the problem.
    assert len(df) == 2
    assert df.index.name == "labeled-data/test1.png"
    assert df.index[0] == "labeled-data/test2.png"

    # Test the fix
    fixed_df = io_utils.fix_empty_first_row(df)
    assert len(fixed_df) == 3
    assert fixed_df.index.name is None
    assert fixed_df.index[0] == "labeled-data/test1.png"



class TestParseLabelCsv:
    """Test the parse_label_csv function."""

    @pytest.fixture
    def standard_csv(self, tmp_path) -> Path:
        """Two-keypoint standard DLC CSV without visibility; frame 1 has one unlabeled kp."""
        content = (
            'scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp2,kp2\n'
            'coords,x,y,x,y\n'
            'labeled-data/img01.png,10.0,20.0,30.0,40.0\n'
            'labeled-data/img02.png,50.0,60.0,,\n'
        )
        p = tmp_path / 'standard.csv'
        p.write_text(content)
        return p

    @pytest.fixture
    def visibility_csv(self, tmp_path) -> Path:
        """Two-keypoint CSV with visible column; kp1 visible, kp2 occluded/unlabeled."""
        content = (
            'scorer,scorer,scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1,kp2,kp2,kp2\n'
            'coords,x,y,visible,x,y,visible\n'
            'labeled-data/img01.png,10.0,20.0,2,30.0,40.0,1\n'
            'labeled-data/img02.png,50.0,60.0,2,,,0\n'
        )
        p = tmp_path / 'visibility.csv'
        p.write_text(content)
        return p

    def test_parse_label_csv_standard_keypoint_names(self, standard_csv):
        """Keypoint names are extracted from the bodyparts row."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert result.keypoint_names == ['kp1', 'kp2']

    def test_parse_label_csv_standard_image_names(self, standard_csv):
        """Image names are read from the index column in order."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert result.image_names == ['labeled-data/img01.png', 'labeled-data/img02.png']

    def test_parse_label_csv_standard_keypoints_shape(self, standard_csv):
        """Keypoints tensor has shape (N, K, 2) for a standard CSV."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert result.keypoints.shape == (2, 2, 2)

    def test_parse_label_csv_standard_keypoint_values(self, standard_csv):
        """Labeled coordinates are parsed correctly; unlabeled become NaN."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert result.keypoints[0, 0, 0] == 10.0
        assert result.keypoints[0, 0, 1] == 20.0
        assert torch.isnan(result.keypoints[1, 1, 0])
        assert torch.isnan(result.keypoints[1, 1, 1])

    def test_parse_label_csv_standard_no_visibility(self, standard_csv):
        """visibility is None for a standard CSV without a visible column."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert result.visibility is None

    def test_parse_label_csv_visibility_tensor_shape(self, visibility_csv):
        """visibility tensor has shape (N, K) and dtype int64 when column is present."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(visibility_csv))
        assert result.visibility is not None
        assert result.visibility.shape == (2, 2)
        assert result.visibility.dtype == torch.long

    def test_parse_label_csv_visibility_values(self, visibility_csv):
        """visibility values match the CSV contents."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(visibility_csv))
        assert result.visibility is not None
        assert result.visibility[0, 0] == 2  # kp1 frame 0: visible
        assert result.visibility[0, 1] == 1  # kp2 frame 0: occluded
        assert result.visibility[1, 0] == 2  # kp1 frame 1: visible
        assert result.visibility[1, 1] == 0  # kp2 frame 1: not labeled

    def test_parse_label_csv_visibility_keypoints_shape(self, visibility_csv):
        """keypoints tensor is (N, K, 2) — visibility column is stripped."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(visibility_csv))
        assert result.keypoints.shape == (2, 2, 2)

    def test_parse_label_csv_nonexistent_file_raises(self, tmp_path):
        """FileNotFoundError is raised for a path that does not exist."""
        from lightning_pose.utils.io import parse_label_csv
        with pytest.raises(FileNotFoundError):
            parse_label_csv(str(tmp_path / 'nonexistent.csv'))

    def test_parse_label_csv_invalid_visibility_raises(self, tmp_path):
        """ValueError is raised when the visible column contains a value outside {0, 1, 2}."""
        content = (
            'scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1\n'
            'coords,x,y,visible\n'
            'labeled-data/img01.png,10.0,20.0,9\n'
        )
        p = tmp_path / 'bad.csv'
        p.write_text(content)
        from lightning_pose.utils.io import parse_label_csv
        with pytest.raises(ValueError, match='invalid values'):
            parse_label_csv(str(p))

    @pytest.mark.parametrize('row', ['50.0,,30.0,40.0', ',60.0,30.0,40.0'])
    def test_parse_label_csv_half_nan_keypoint_raises(self, tmp_path, row):
        """ValueError names the file, image and keypoint when only one of x, y is NaN."""
        content = (
            'scorer,scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp2,kp2\n'
            'coords,x,y,x,y\n'
            f'labeled-data/img01.png,{row}\n'
        )
        p = tmp_path / 'half_nan.csv'
        p.write_text(content)
        from lightning_pose.utils.io import parse_label_csv
        with pytest.raises(ValueError, match=r'img01\.png / kp1'):
            parse_label_csv(str(p))

    def test_parse_label_csv_half_nan_keypoint_with_visibility_raises(self, tmp_path):
        """The half-NaN check also applies to CSVs with a visible column."""
        content = (
            'scorer,scorer,scorer,scorer\n'
            'bodyparts,kp1,kp1,kp1\n'
            'coords,x,y,visible\n'
            'labeled-data/img01.png,10.0,,2\n'
        )
        p = tmp_path / 'half_nan_vis.csv'
        p.write_text(content)
        from lightning_pose.utils.io import parse_label_csv
        with pytest.raises(ValueError, match='only one of x, y'):
            parse_label_csv(str(p))

    def test_parse_label_csv_full_nan_keypoint_ok(self, standard_csv):
        """A keypoint with both coordinates NaN is valid (unlabeled)."""
        from lightning_pose.utils.io import parse_label_csv
        result = parse_label_csv(str(standard_csv))
        assert torch.isnan(result.keypoints[1, 1]).all()

    def test_parse_label_csv_single_read(self, standard_csv, monkeypatch):
        """The CSV file is read exactly once."""
        import pandas as pd

        from lightning_pose.utils.io import parse_label_csv
        read_count = 0
        original_read_csv = pd.read_csv

        def counting_read_csv(*args, **kwargs):
            nonlocal read_count
            read_count += 1
            return original_read_csv(*args, **kwargs)

        monkeypatch.setattr(pd, 'read_csv', counting_read_csv)
        parse_label_csv(str(standard_csv))
        assert read_count == 1
