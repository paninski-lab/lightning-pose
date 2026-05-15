import copy
import filecmp
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from lightning_pose.utils.cropzoom import (
    _crop_image,
    generate_cropped_labeled_frames,
    generate_cropped_video,
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


def test_generate_cropped_labeled_frames(tmp_path, request):
    # Fetch a dataset and a fully trained model's predictions on it.
    fetch_test_data_if_needed(request.path.parent, "test_cropzoom_data")

    # Copy the model predictions to a temporary model directory.
    tmp_model_directory = tmp_path / "test_model"
    shutil.copytree(
        request.path.parent / "test_cropzoom_data" / "test_model_output",
        tmp_model_directory,
    )

    # Run cropzoom on the test data in the temporary model directory.
    root_directory = request.path.parent / "test_cropzoom_data" / "test_data"
    detector_cfg = OmegaConf.create(
        {"crop_ratio": 1.5, "anchor_keypoints": ["A_head", "D_tailtip"]}
    )
    generate_cropped_labeled_frames(
        input_data_dir=root_directory,
        input_csv_file=root_directory / "CollectedData.csv",
        input_preds_file=tmp_model_directory / "predictions.csv",
        detector_cfg=detector_cfg,
        output_data_dir=tmp_model_directory / "cropped_images",
        output_bbox_file=tmp_model_directory / "cropped_images" / "bbox.csv",
        output_csv_file=tmp_model_directory / "cropped_images" / "cropped_labels.csv",
    )

    # Assert cropzoom output matches expected output.
    comparison = compare_directories(
        tmp_model_directory / "cropped_images",
        request.path.parent
        / "test_cropzoom_data"
        / "expected_model_output"
        / "cropped_images",
    )
    # Successfully compared 24 objects.
    assert comparison == 24


def test_generate_cropped_video(tmp_path, request):
    # Fetch a dataset and a fully trained model's predictions on it.
    fetch_test_data_if_needed(request.path.parent, "test_cropzoom_data")

    # Copy the model predictions to a temporary model directory.
    tmp_model_directory = tmp_path / "test_model"
    shutil.copytree(
        request.path.parent / "test_cropzoom_data" / "test_model_output",
        tmp_model_directory,
    )

    # Run cropzoom on the test data in the temporary model directory.
    video_directory = (
        request.path.parent / "test_cropzoom_data" / "test_data" / "videos"
    )
    detector_cfg = OmegaConf.create(
        {"crop_ratio": 1.5, "anchor_keypoints": ["A_head", "D_tailtip"]}
    )
    for video_path in video_directory.iterdir():
        generate_cropped_video(
            input_video_file=video_path,
            input_preds_file=tmp_model_directory
            / "video_preds"
            / (video_path.stem + ".csv"),
            detector_cfg=detector_cfg,
            output_bbox_file=tmp_model_directory
            / "cropped_videos"
            / (video_path.stem + "_bbox.csv"),
            output_file=tmp_model_directory / "cropped_videos" / video_path.name,
        )

    # Assert cropzoom output matches expected output.
    comparison = compare_directories(
        tmp_model_directory / "cropped_videos",
        request.path.parent
        / "test_cropzoom_data"
        / "expected_model_output"
        / "cropped_videos",
    )
    # Successfully compared 2 objects (need to skip mp4s)
    assert comparison == 2
