from collections.abc import Collection
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig
from PIL import Image
from typeguard import typechecked

from lightning_pose.utils.io import get_context_img_paths


@typechecked
def _calculate_bbox_size(
    keypoints_per_frame: np.ndarray, crop_ratio: float = 1.0
) -> int:
    """Given all labeled keypoints, computes the bounding box square size as
    the length of one side in pixels.
    First we compute the maximum bounding box that would always encompass
    the animal (maximum difference of all x's and y's per frame, max over all frames).
    Then we take the larger dimension of the bbox (x or y). Finally we scale by `crop_ratio`.

    Arguments:
        keypoints_per_frame: np array of all labeled frames. Shape of (frames, keypoints, x|y).
        crop_ratio:
    """
    # Extract x and y coordinates
    x_coords = keypoints_per_frame[:, :, 0]  # All rows, all columns, first element (x)
    y_coords = keypoints_per_frame[:, :, 1]  # All rows, all columns, second element (y)
    max_x_diff_per_frame = np.max(x_coords, axis=1) - np.min(x_coords, axis=1)
    max_y_diff_per_frame = np.max(y_coords, axis=1) - np.min(y_coords, axis=1)

    # Max of all x_diff and y_diff over all frames. A scalar.
    max_bbox_size = np.max([max_x_diff_per_frame, max_y_diff_per_frame], axis=(0, 1))

    # Scale by crop_ratio, and take ceiling.
    bbox_size = int(np.ceil(max_bbox_size * crop_ratio))

    # Many video players don't like odd dimensions.
    # Make sure the bbox has even dimensions.
    bbox_size = bbox_size if bbox_size % 2 == 0 else bbox_size + 1

    return bbox_size


@typechecked
def _compute_bbox_df(
    pred_df: pd.DataFrame, anchor_keypoints: Collection[str], crop_ratio: float = 1.0
) -> pd.DataFrame:
    # Get x,y columns for anchor_keypoints (or all keypoints if anchor_keypoints is empty)
    coord_mask = pred_df.columns.get_level_values("coords").isin(["x", "y"])
    if len(anchor_keypoints) > 0:
        coord_mask &= pred_df.columns.get_level_values("bodyparts").isin(
            anchor_keypoints
        )

    # Shape: (frames, keypoints, x|y)
    keypoints_per_frame = (
        pred_df.loc[:, coord_mask].to_numpy().reshape(pred_df.shape[0], -1, 2)
    )

    bbox_size = _calculate_bbox_size(keypoints_per_frame, crop_ratio=crop_ratio)

    # Shape: (frames, keypoints, x|y) -> (frames, x|y)
    centroids = keypoints_per_frame.mean(axis=1)

    # Instead of storing centroid, we'll store bbox top-left.
    # Shape: (frames, x|y)
    bbox_toplefts = centroids - bbox_size // 2
    # Floor and store ints.
    bbox_toplefts = np.int64(bbox_toplefts)

    # Store bbox size as (h,w). Note that h=w since bbox is a square for now.
    # Shape: (frames, h|w)
    bbox_hws = np.full_like(bbox_toplefts, bbox_size)

    # Shape: (frames, x|y) -> (frames, x|y|h|w)
    bboxes = np.concatenate([bbox_toplefts, bbox_hws], axis=1)

    index = pred_df.index

    return pd.DataFrame(bboxes, index=index, columns=["x", "y", "h", "w"])


@typechecked
def _crop_images(
    bbox_df: pd.DataFrame, root_directory: Path, output_directory: Path
) -> None:
    """Crops images according to their bboxes in `bbox_df`.

    root_directory: root of img paths in bbox_df.
    output_directory: where to save cropped images."""
    for center_img_path, row in bbox_df.iterrows():
        for img_path in get_context_img_paths(Path(center_img_path)):
            # Silently skip non-existent context frames.
            if not (root_directory / img_path).exists() and img_path != center_img_path:
                continue
            img = Image.open(root_directory / img_path)
            img = img.crop((row.x, row.y, row.x + row.h, row.y + row.w))

            # preserve directory structure of img_path
            # (e.g. labeled-data/x.png will create a labeled-data dir)
            cropped_img_path = output_directory / img_path
            cropped_img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(cropped_img_path)


@typechecked
def _crop_video_moviepy(
    video_file: Path, bbox_df: pd.DataFrame, output_directory: Path
):
    clip = VideoFileClip(str(video_file))

    b = bbox_df.iloc[0]
    h = b.h
    w = b.w

    def crop_frame(get_frame, t):
        frame = get_frame(t)

        frame_index = int(t * clip.fps)  # Calculate frame index based on time
        if frame_index >= len(bbox_df):
            print(f"crop_frame: Skipped frame {frame_index}")
            return np.zeros((h, w, frame.shape[2]), dtype=np.uint8)

        b = bbox_df.iloc[frame_index]
        x1, x2 = b.x, b.x + b.w
        y1, y2 = b.y, b.y + b.h
        assert b.h == h
        assert b.w == w
        cropped_frame = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)

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

        return cropped_frame

    # renamed image_transform in 2.0.0
    cropped_clip = clip.fl(crop_frame, apply_to="mask")

    cropped_clip.write_videofile(
        str(output_directory / video_file.name), codec="libx264"
    )


@typechecked
def generate_cropped_labeled_frames(
    root_directory: Path,
    output_directory: Path,
    detector_cfg: DictConfig,
    preds_file: Optional[Path] = None,
) -> None:
    # Use predictions rather than CollectedData.csv because collected data can sometimes have NaNs.
    preds_file = preds_file or output_directory / "predictions.csv"
    # load predictions
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # compute and save bbox_df
    bbox_df = _compute_bbox_df(
        pred_df, detector_cfg.anchor_keypoints, crop_ratio=detector_cfg.crop_ratio
    )

    bbox_csv_path = output_directory / "cropped_images" / "bbox.csv"
    bbox_csv_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(bbox_csv_path)

    _crop_images(bbox_df, root_directory, output_directory / "cropped_images")


@typechecked
def generate_cropped_video(
    video_path: Path, detector_model_dir: Path, detector_cfg: DictConfig
) -> None:
    video_path = Path(video_path)

    # Given the predictions, compute cropping bboxes
    preds_file = detector_model_dir / "video_preds" / (video_path.stem + ".csv")

    # load predictions
    # TODO If predictions do not exist, predict with detector model
    pred_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # Save cropping bboxes
    bbox_df = _compute_bbox_df(
        pred_df, detector_cfg.anchor_keypoints, crop_ratio=detector_cfg.crop_ratio
    )
    output_bbox_path = (
        detector_model_dir / "cropped_videos" / (video_path.stem + "_bbox.csv")
    )
    output_bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(output_bbox_path)

    # Generate a cropped video for debugging purposes.
    _crop_video_moviepy(video_path, bbox_df, detector_model_dir / "cropped_videos")
