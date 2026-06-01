import copy
import filecmp
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from lightning_pose.utils.cropzoom import (
    _compute_bbox_df,
    _crop_image,
    crop_labeled_frames,
    crop_video,
    generate_bbox,
    smooth_bbox,
)

from ..fetch_test_data import fetch_test_data_if_needed


class TestCropImage:
    """Test the _crop_image function."""

    def _make_image(self, tmp_path: Path, size: tuple[int, int] = (100, 80)) -> Path:
        """Save a solid-color RGB image and return its path."""
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[:, :] = [10, 20, 30]
        img_path = tmp_path / 'source.png'
        Image.fromarray(arr).save(img_path)
        return img_path

    def test_saves_cropped_image(self, tmp_path):
        """Cropped image is written to the specified output path."""
        img_path = self._make_image(tmp_path)
        out_path = tmp_path / 'out' / 'cropped.png'
        _crop_image(img_path, (10, 10, 50, 40), out_path)
        assert out_path.exists()

    def test_output_has_correct_size(self, tmp_path):
        """Saved image dimensions match the bounding box."""
        img_path = self._make_image(tmp_path)
        bbox = (10, 5, 60, 45)  # width=50, height=40
        out_path = tmp_path / 'cropped.png'
        _crop_image(img_path, bbox, out_path)
        result = Image.open(out_path)
        assert result.size == (50, 40)

    def test_creates_parent_directories(self, tmp_path):
        """Parent directories of the output path are created automatically."""
        img_path = self._make_image(tmp_path)
        out_path = tmp_path / 'a' / 'b' / 'c' / 'cropped.png'
        _crop_image(img_path, (0, 0, 10, 10), out_path)
        assert out_path.exists()

    def test_pixel_values_preserved(self, tmp_path):
        """Pixels in the cropped region match the original image."""
        img_path = self._make_image(tmp_path, size=(100, 100))
        bbox = (20, 30, 70, 80)
        out_path = tmp_path / 'cropped.png'
        _crop_image(img_path, bbox, out_path)
        original = np.array(Image.open(img_path).crop(bbox))
        cropped = np.array(Image.open(out_path))
        assert np.array_equal(original, cropped)


class TestComputeBboxDf:
    """Test the _compute_bbox_df function."""

    @pytest.fixture
    def pred_df(self) -> pd.DataFrame:
        """Minimal two-keypoint, three-frame prediction DataFrame."""
        columns = pd.MultiIndex.from_tuples(
            [
                ('scorer', 'kp0', 'x'),
                ('scorer', 'kp0', 'y'),
                ('scorer', 'kp0', 'likelihood'),
                ('scorer', 'kp1', 'x'),
                ('scorer', 'kp1', 'y'),
                ('scorer', 'kp1', 'likelihood'),
            ],
            names=['scorer', 'bodyparts', 'coords'],
        )
        data = [
            [10.0, 20.0, 0.9, 30.0, 40.0, 0.8],
            [15.0, 25.0, 0.9, 35.0, 45.0, 0.8],
            [20.0, 30.0, 0.9, 40.0, 50.0, 0.8],
        ]
        return pd.DataFrame(data, columns=columns)

    def test_raises_when_both_modes_provided(self, pred_df):
        """Raises ValueError when crop_ratio and crop_height/width are both given."""
        with pytest.raises(ValueError, match='not both'):
            _compute_bbox_df(pred_df, [], crop_ratio=2.0, crop_height=100, crop_width=100)

    def test_raises_when_neither_mode_provided(self, pred_df):
        """Raises ValueError when neither crop_ratio nor crop_height/width are given."""
        with pytest.raises(ValueError, match='must be provided'):
            _compute_bbox_df(pred_df, [])


# TODO: Move to utils.
def compare_directories(dir1: Path, dir2: Path) -> int | dict:
    """
    Compares files in two directories recursively.

    Args:
      dir1: Path to the first directory.
      dir2: Path to the second directory.

    Returns:
      Number of matched files if the directories are identical, or
      a dictionary with the following keys if the directories are different:
        'mismatch': List of files that differ between the directories.
        'errors': List of files that couldn't be compared (e.g., due to permissions).
        'left_only': List of files that exist only in the first directory.
        'right_only': List of files that exist only in the second directory.
    """
    num_matches = 0
    empty_results = {
        "mismatch": [],
        "errors": [],
        "left_only": [],
        "right_only": [],
    }
    results = copy.deepcopy(empty_results)

    dircmp_result = filecmp.dircmp(dir1, dir2)

    # Compare common files
    for common_file in dircmp_result.common_files:
        # fails on mp4s because metadata is different, no quick fix
        if common_file.find('.mp4') > -1:
            continue
        if filecmp.cmp(dir1 / common_file, dir2 / common_file, shallow=False):
            num_matches += 1
        else:
            results["mismatch"].append(common_file)

    # Recursively compare subdirectories
    for sub_dir in dircmp_result.common_dirs:
        sub_results = compare_directories(dir1 / sub_dir, dir2 / sub_dir)
        if isinstance(sub_results, int):
            num_matches += sub_results
        else:
            results["mismatch"].extend(sub_results["mismatch"])
            results["errors"].extend(sub_results["errors"])
            results["left_only"].extend(sub_results["left_only"])
            results["right_only"].extend(sub_results["right_only"])

    # Add files unique to each directory and any errors
    results["errors"].extend(dircmp_result.common_funny)
    results["left_only"].extend(dircmp_result.left_only)
    results["right_only"].extend(dircmp_result.right_only)

    if results == empty_results:
        return num_matches
    return results


class TestGenerateBbox:
    """Test the generate_bbox function."""

    @pytest.fixture
    def setup(self, tmp_path, request):
        fetch_test_data_if_needed(request.path.parent, 'test_cropzoom_data')
        tmp_model_dir = tmp_path / 'test_model'
        shutil.copytree(
            request.path.parent / 'test_cropzoom_data' / 'test_model_output',
            tmp_model_dir,
        )
        return tmp_model_dir, request.path.parent / 'test_cropzoom_data'

    def test_generate_bbox_crop_ratio(self, setup):
        """generate_bbox with crop_ratio writes a valid bbox CSV."""
        tmp_model_dir, _ = setup
        detector_cfg = OmegaConf.create({
            'crop_ratio': 1.5,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        output_bbox_file = tmp_model_dir / 'bbox.csv'
        generate_bbox(
            input_preds_file=tmp_model_dir / 'predictions.csv',
            detector_cfg=detector_cfg,
            output_bbox_file=output_bbox_file,
        )
        assert output_bbox_file.exists()
        bbox_df = pd.read_csv(output_bbox_file, index_col=0)
        assert list(bbox_df.columns) == ['x', 'y', 'h', 'w']

    def test_generate_bbox_crop_size(self, setup):
        """generate_bbox with crop_size produces constant h/w values."""
        tmp_model_dir, _ = setup
        crop_size = 100
        detector_cfg = OmegaConf.create({
            'crop_height': crop_size,
            'crop_width': crop_size,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        output_bbox_file = tmp_model_dir / 'bbox.csv'
        generate_bbox(
            input_preds_file=tmp_model_dir / 'predictions.csv',
            detector_cfg=detector_cfg,
            output_bbox_file=output_bbox_file,
        )
        bbox_df = pd.read_csv(output_bbox_file, index_col=0)
        assert (bbox_df['h'] == crop_size).all()
        assert (bbox_df['w'] == crop_size).all()

    def test_generate_bbox_creates_parent_dirs(self, setup):
        """generate_bbox creates missing parent directories."""
        tmp_model_dir, _ = setup
        detector_cfg = OmegaConf.create({
            'crop_ratio': 2.0,
            'anchor_keypoints': [],
        })
        output_bbox_file = tmp_model_dir / 'new_dir' / 'bbox.csv'
        generate_bbox(
            input_preds_file=tmp_model_dir / 'predictions.csv',
            detector_cfg=detector_cfg,
            output_bbox_file=output_bbox_file,
        )
        assert output_bbox_file.exists()


class TestCropLabeledFrames:
    """Test the crop_labeled_frames function."""

    @pytest.fixture
    def setup(self, tmp_path, request):
        fetch_test_data_if_needed(request.path.parent, 'test_cropzoom_data')
        tmp_model_dir = tmp_path / 'test_model'
        shutil.copytree(
            request.path.parent / 'test_cropzoom_data' / 'test_model_output',
            tmp_model_dir,
        )
        return tmp_model_dir, request.path.parent / 'test_cropzoom_data'

    def test_crop_labeled_frames_crop_ratio(self, setup):
        """crop_labeled_frames output matches expected directory (crop_ratio mode)."""
        tmp_model_dir, test_data_dir = setup
        root_dir = test_data_dir / 'test_data'
        detector_cfg = OmegaConf.create({
            'crop_ratio': 1.5,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        bbox_file = tmp_model_dir / 'cropped_images' / 'bbox.csv'
        generate_bbox(
            input_preds_file=tmp_model_dir / 'predictions.csv',
            detector_cfg=detector_cfg,
            output_bbox_file=bbox_file,
        )
        crop_labeled_frames(
            input_data_dir=root_dir,
            input_csv_file=root_dir / 'CollectedData.csv',
            input_bbox_file=bbox_file,
            output_data_dir=tmp_model_dir / 'cropped_images',
            output_csv_file=tmp_model_dir / 'cropped_images' / 'cropped_labels.csv',
        )
        comparison = compare_directories(
            tmp_model_dir / 'cropped_images',
            test_data_dir / 'expected_model_output' / 'cropped_images',
        )
        assert comparison == 24  # 24 files compared successfully

    def test_crop_labeled_frames_crop_size(self, setup):
        """crop_labeled_frames with crop_size: h/w in bbox match requested size."""
        tmp_model_dir, test_data_dir = setup
        root_dir = test_data_dir / 'test_data'
        crop_size = 100
        detector_cfg = OmegaConf.create({
            'crop_height': crop_size,
            'crop_width': crop_size,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        bbox_file = tmp_model_dir / 'cropped_images' / 'bbox.csv'
        generate_bbox(
            input_preds_file=tmp_model_dir / 'predictions.csv',
            detector_cfg=detector_cfg,
            output_bbox_file=bbox_file,
        )
        crop_labeled_frames(
            input_data_dir=root_dir,
            input_csv_file=root_dir / 'CollectedData.csv',
            input_bbox_file=bbox_file,
            output_data_dir=tmp_model_dir / 'cropped_images',
            output_csv_file=tmp_model_dir / 'cropped_images' / 'cropped_labels.csv',
        )
        bbox_df = pd.read_csv(bbox_file, index_col=0)
        assert (bbox_df['h'] == crop_size).all()
        assert (bbox_df['w'] == crop_size).all()


class TestCropVideo:
    """Test the crop_video function."""

    @pytest.fixture
    def setup(self, tmp_path, request):
        fetch_test_data_if_needed(request.path.parent, 'test_cropzoom_data')
        tmp_model_dir = tmp_path / 'test_model'
        shutil.copytree(
            request.path.parent / 'test_cropzoom_data' / 'test_model_output',
            tmp_model_dir,
        )
        video_dir = request.path.parent / 'test_cropzoom_data' / 'test_data' / 'videos'
        return tmp_model_dir, request.path.parent / 'test_cropzoom_data', video_dir

    def test_crop_video_crop_ratio(self, setup):
        """crop_video bbox CSVs match expected output (crop_ratio mode)."""
        tmp_model_dir, test_data_dir, video_dir = setup
        detector_cfg = OmegaConf.create({
            'crop_ratio': 1.5,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        for video_path in video_dir.iterdir():
            bbox_file = tmp_model_dir / 'cropped_videos' / (video_path.stem + '_bbox.csv')
            generate_bbox(
                input_preds_file=tmp_model_dir / 'video_preds' / (video_path.stem + '.csv'),
                detector_cfg=detector_cfg,
                output_bbox_file=bbox_file,
            )
            crop_video(
                input_video_file=video_path,
                input_bbox_file=bbox_file,
                output_file=tmp_model_dir / 'cropped_videos' / video_path.name,
            )
        comparison = compare_directories(
            tmp_model_dir / 'cropped_videos',
            test_data_dir / 'expected_model_output' / 'cropped_videos',
        )
        assert comparison == 2  # 2 bbox CSVs compared successfully (mp4s skipped)

    def test_crop_video_crop_size(self, setup):
        """crop_video with crop_size: h/w in bbox match requested size."""
        tmp_model_dir, test_data_dir, video_dir = setup
        crop_size = 100
        detector_cfg = OmegaConf.create({
            'crop_height': crop_size,
            'crop_width': crop_size,
            'anchor_keypoints': ['A_head', 'D_tailtip'],
        })
        for video_path in video_dir.iterdir():
            bbox_file = tmp_model_dir / 'cropped_videos' / (video_path.stem + '_bbox.csv')
            generate_bbox(
                input_preds_file=tmp_model_dir / 'video_preds' / (video_path.stem + '.csv'),
                detector_cfg=detector_cfg,
                output_bbox_file=bbox_file,
            )
            crop_video(
                input_video_file=video_path,
                input_bbox_file=bbox_file,
                output_file=tmp_model_dir / 'cropped_videos' / video_path.name,
            )
            bbox_df = pd.read_csv(bbox_file, index_col=0)
            assert (bbox_df['h'] == crop_size).all()
            assert (bbox_df['w'] == crop_size).all()


class TestSmoothBbox:
    """Test the smooth_bbox function."""

    def _make_bbox_file(self, path: Path, n_frames: int = 10) -> None:
        """Write a simple bbox CSV at ``path``."""
        pd.DataFrame({
            'x': range(n_frames),
            'y': range(n_frames),
            'h': [50] * n_frames,
            'w': [50] * n_frames,
        }).to_csv(path)

    def test_creates_output_files(self, tmp_path):
        """Smoothed bbox files are written for every input file."""
        input_dir = tmp_path / 'raw'
        input_dir.mkdir()
        for i in range(3):
            self._make_bbox_file(input_dir / f'vid{i}_bbox.csv')
        output_dir = tmp_path / 'smooth'
        smooth_bbox(input_dir, output_dir, method='median', window=3)
        for i in range(3):
            assert (output_dir / f'vid{i}_bbox.csv').exists()

    def test_writes_metadata(self, tmp_path):
        """metadata.json records method, window, and source."""
        input_dir = tmp_path / 'raw'
        input_dir.mkdir()
        self._make_bbox_file(input_dir / 'vid_bbox.csv')
        output_dir = tmp_path / 'smooth'
        smooth_bbox(input_dir, output_dir, method='median', window=7)
        meta = json.loads((output_dir / 'metadata.json').read_text())
        assert meta['method'] == 'median'
        assert meta['window'] == 7
        assert str(input_dir.resolve()) == meta['source']

    def test_output_has_integer_values(self, tmp_path):
        """Smoothed values are cast to integers."""
        input_dir = tmp_path / 'raw'
        input_dir.mkdir()
        self._make_bbox_file(input_dir / 'vid_bbox.csv', n_frames=9)
        output_dir = tmp_path / 'smooth'
        smooth_bbox(input_dir, output_dir, method='median', window=3)
        result = pd.read_csv(output_dir / 'vid_bbox.csv', index_col=0)
        for col in result.columns:
            assert result[col].dtype == int

    def test_raises_on_unknown_method(self, tmp_path):
        """Raises ValueError for an unsupported smoothing method."""
        input_dir = tmp_path / 'raw'
        input_dir.mkdir()
        output_dir = tmp_path / 'smooth'
        with pytest.raises(ValueError, match='unsupported method'):
            smooth_bbox(input_dir, output_dir, method='foo', window=5)

    def test_raises_when_no_bbox_files(self, tmp_path):
        """Raises ValueError when no *_bbox.csv files are found."""
        input_dir = tmp_path / 'empty'
        input_dir.mkdir()
        output_dir = tmp_path / 'smooth'
        with pytest.raises(ValueError, match=r'no \*_bbox\.csv files found'):
            smooth_bbox(input_dir, output_dir, method='median', window=5)
