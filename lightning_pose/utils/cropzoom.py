import multiprocessing
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig
from PIL import Image
from typeguard import typechecked

from lightning_pose.utils import io

__all__ = [
    "generate_cropped_labeled_frames",
    "generate_cropped_video",
    "generate_cropped_csv_file",
]


@typechecked
def _calculate_bbox_size(
    keypoints_per_frame: np.ndarray, crop_ratio: float = 1.0
) -> np.ndarray:
    """Computes bounding box size for each frame.

    Arguments:
        keypoints_per_frame: Numpy array, shape of (frame, keypoint, x|y).
        crop_ratio: ratio to multiply max difference between x, y to get

    Returns:
        numpy array:  Shape of (frame, 2 (h|w))
    """
    # Extract x and y coordinates
    x_coords = keypoints_per_frame[:, :, 0]  # All rows, all columns, first element (x)
    y_coords = keypoints_per_frame[:, :, 1]  # All rows, all columns, second element (y)
    max_x_diff_per_frame = np.max(x_coords, axis=1) - np.min(x_coords, axis=1)
    max_y_diff_per_frame = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)

    # Max of x_diff and y_diff for each frame. Shape of (frames,).
    max_bbox_size_per_frame = np.max(
        [max_x_diff_per_frame, max_y_diff_per_frame], axis=0
    )

    # Scale by crop_ratio, and take ceiling.
    bbox_size_per_frame = np.ceil(max_bbox_size_per_frame * crop_ratio).astype(int)

    # Many video players don't like odd dimensions.
    # Make sure the bbox has even dimensions.
    bbox_size_per_frame = np.where(
        bbox_size_per_frame % 2 == 0, bbox_size_per_frame, bbox_size_per_frame + 1
    )

    # Change shape from (frames,) to (frames, 2), aka (frame, h|w)
    bbox_sizes = np.column_stack((bbox_size_per_frame, bbox_size_per_frame))

    return bbox_sizes


@typechecked
def _compute_bbox_df(
    pred_df: pd.DataFrame, anchor_keypoints: list[str], crop_ratio: float = 1.0
) -> pd.DataFrame:
    # Get x,y columns for anchor_keypoints (or all keypoints if anchor_keypoints is empty)
    coord_mask = pred_df.columns.get_level_values("coords").isin(["x", "y"])
    if len(anchor_keypoints) > 0:
        # Validate anchor keypoints.
        invalid_keypoints = set(anchor_keypoints) - set(
            pred_df.columns.get_level_values("bodyparts")
        )
        assert (
            not invalid_keypoints
        ), f"Anchor keypoints not found in DataFrame: {invalid_keypoints}"

        coord_mask &= pred_df.columns.get_level_values("bodyparts").isin(
            anchor_keypoints
        )

    # Shape: (frames, keypoints, x|y)
    keypoints_per_frame = (
        pred_df.loc[:, coord_mask].to_numpy().reshape(pred_df.shape[0], -1, 2)
    )

    bbox_sizes = _calculate_bbox_size(keypoints_per_frame, crop_ratio=crop_ratio)

    # Shape: (frames, keypoints, x|y) -> (frames, x|y)
    centroids = keypoints_per_frame.mean(axis=1)

    # Instead of storing centroid, we'll store bbox top-left.
    # Shape: (frames, x|y)
    bbox_toplefts = centroids - bbox_sizes // 2
    # Floor and store ints.
    bbox_toplefts = np.int64(bbox_toplefts)

    # Shape: (frames, x|y) -> (frames, x|y|h|w)
    bboxes = np.concatenate([bbox_toplefts, bbox_sizes], axis=1)

    index = pred_df.index

    return pd.DataFrame(bboxes, index=index, columns=["x", "y", "h", "w"])


def _crop_image(img_path, bbox, cropped_img_path):
    img = Image.open(img_path)
    img = img.crop(bbox)
    cropped_img_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(cropped_img_path)


def _star_crop_image(args):
    return _crop_image(*args)


@typechecked
def _crop_images(
    bbox_df: pd.DataFrame, root_directory: Path, output_directory: Path
) -> None:
    """Crops images according to their bboxes in `bbox_df`.

    root_directory: root of img paths in bbox_df.
    output_directory: where to save cropped images."""

    _file_cache: dict[Path, bool] = {}

    def _file_exists(path):
        # Cache path.exists() as an easy way to speed up.
        # TODO: This is still slow. Get all files in the dir and check if file is in the list.
        if path in _file_cache:
            return _file_cache[path]
        exists = (root_directory / path).exists()
        _file_cache[path] = exists
        return exists

    # img_path -> (abs_img_path, bbox, output_img_path)
    crop_calls: dict[Path, tuple[Path, tuple[int, int, int, int], Path]] = {}

    for center_img_path, row in tqdm.tqdm(
        bbox_df.iterrows(), total=len(bbox_df), desc="Building crop tasks"
    ):
        # TODO Add unit tests for this logic.
        center_img_path = Path(center_img_path)
        for img_path in io.get_context_img_paths(center_img_path):
            # If context frame:
            if img_path != center_img_path:
                # There is no context frame. Continue.
                if not _file_exists(root_directory / img_path):
                    continue
                # The context frame is already in bbox_df as a center frame. Continue.
                if str(img_path) in bbox_df.index:
                    continue
                # There is already a crop task for the context frame. Continue.
                if img_path in crop_calls:
                    continue
            abs_img_path = root_directory / img_path
            bbox = (row.x, row.y, row.x + row.w, row.y + row.h)
            cropped_img_path = output_directory / img_path

            crop_calls[img_path] = (abs_img_path, bbox, cropped_img_path)

    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(
            pool.imap(_star_crop_image, crop_calls.values()),
            total=len(crop_calls),
            desc="Cropping images",
        ):
            pass


@typechecked
def _crop_video_moviepy(video_file: Path, bbox_df: pd.DataFrame, output_file: Path):
    clip = VideoFileClip(str(video_file))

    h = bbox_df["h"].median()
    w = bbox_df["w"].median()

    # Convert to nearest even integer
    h = round(h / 2) * 2
    w = round(w / 2) * 2

    def crop_frame(get_frame, t):
        frame = get_frame(t)

        frame_index = int(t * clip.fps)  # Calculate frame index based on time
        if frame_index >= len(bbox_df):
            print(f"crop_frame: Skipped frame {frame_index}")
            return np.zeros((h, w, frame.shape[2]), dtype=np.uint8)

        b = bbox_df.iloc[frame_index]
        x1, x2 = b.x, b.x + b.w
        y1, y2 = b.y, b.y + b.h
        cropped_frame = np.zeros((b.h, b.w, frame.shape[2]), dtype=np.uint8)

        # Calculate valid crop boundaries within the original frame
        x1_valid = max(0, x1)
        x2_valid = min(clip.w - 1, x2)
        y1_valid = max(0, y1)
        y2_valid = min(clip.h - 1, y2)

        # Calculate corresponding coordinates in the cropped frame
        crop_x1 = abs(min(0, x1))  # Offset in the cropped frame if x1 is negative
        crop_x2 = crop_x1 + (x2_valid - x1_valid)
        crop_y1 = abs(min(0, y1))  # Offset in the cropped frame if y1 is negative
        crop_y2 = crop_y1 + (y2_valid - y1_valid)

        # Copy the valid region to the cropped frame
        cropped_frame[crop_y1:crop_y2, crop_x1:crop_x2] = frame[
            y1_valid:y2_valid, x1_valid:x2_valid
        ]

        return cv2.resize(cropped_frame, (w, h))

    # renamed image_transform in 2.0.0
    cropped_clip = clip.fl(crop_frame, apply_to="mask")

    cropped_clip.write_videofile(str(output_file), codec="libx264")


@typechecked
def generate_cropped_labeled_frames(
    input_data_dir: Path,
    input_csv_file: Path,
    input_preds_file: Path,
    detector_cfg: DictConfig,
    output_data_dir: Path,
    output_bbox_file: Path,
    output_csv_file: Path,
) -> None:
    """Given model predictions, generates a bbox.csv, crops frames,
    and a cropped csv file."""
    # Use predictions rather than CollectedData.csv because collected data can sometimes have NaNs.
    # load predictions
    pred_df = pd.read_csv(input_preds_file, header=[0, 1, 2], index_col=0)
    pred_df = io.fix_empty_first_row(pred_df)

    # compute and save bbox_df
    bbox_df = _compute_bbox_df(
        pred_df, list(detector_cfg.anchor_keypoints), crop_ratio=detector_cfg.crop_ratio
    )

    output_bbox_file.parent.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(output_bbox_file)

    _crop_images(bbox_df, input_data_dir, output_data_dir)

    generate_cropped_csv_file(
        input_csv_file=input_csv_file,
        input_bbox_file=output_bbox_file,
        output_csv_file=output_csv_file,
    )


@typechecked
def generate_cropped_video(
    input_video_file: Path,
    input_preds_file: Path,
    detector_cfg: DictConfig,
    output_bbox_file: Path,
    output_file: Path,
) -> None:
    """TODO make consistent with generate_cropped_labeled_frames"""

    # Given the predictions, compute cropping bboxes
    pred_df = pd.read_csv(input_preds_file, header=[0, 1, 2], index_col=0)
    pred_df = io.fix_empty_first_row(pred_df)

    # Save cropping bboxes
    bbox_df = _compute_bbox_df(
        pred_df, list(detector_cfg.anchor_keypoints), crop_ratio=detector_cfg.crop_ratio
    )
    output_bbox_file.parent.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(output_bbox_file)

    # Generate a cropped video for debugging purposes.
    _crop_video_moviepy(input_video_file, bbox_df, output_file)


def generate_cropped_csv_file(
    input_csv_file: str | Path,
    input_bbox_file: str | Path,
    output_csv_file: str | Path,
    mode: str = "subtract",
):
    """Translate a CSV file by bbox file.
    Requires the files have the same index.

    Defaults to subtraction. Can use mode='add' to map from cropped to original space.
    """
    if mode not in ("add", "subtract"):
        raise ValueError(f"{mode} is not a valid mode")
    # Read csv file from pose_model.cfg.data.csv_file
    # TODO: reuse header_rows logic from datasets.py
    csv_data = pd.read_csv(input_csv_file, header=[0, 1, 2], index_col=0)
    csv_data = io.fix_empty_first_row(csv_data)

    bbox_data = pd.read_csv(input_bbox_file, index_col=0)

    for col in csv_data.columns:
        if col[-1] in ("x", "y"):
            if mode == "subtract":
                csv_data[col] = csv_data[col] - bbox_data[col[-1]]
            else:
                csv_data[col] = csv_data[col] + bbox_data[col[-1]]

    output_csv_file = Path(output_csv_file)
    output_csv_file.parent.mkdir(parents=True, exist_ok=True)
    csv_data.to_csv(output_csv_file)
