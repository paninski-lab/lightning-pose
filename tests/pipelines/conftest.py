"""Fixtures for the pipeline tests: a synthetic video and CPU-only fake models."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lightning_pose.data.bboxes import model_to_frame_batch

NUM_FRAMES = 20
FRAME_HEIGHT = 64
FRAME_WIDTH = 96
RESIZE_DETECTOR = (32, 48)  # (height, width): half the frame size
RESIZE_POSE = (16, 16)


class FakeNet:
    """Stand-in for a LightningModule: ``predict_step`` returns scripted keypoints.

    Keypoints are specified in model-input pixel coordinates and mapped to original-frame
    coordinates with the real ``model_to_frame_batch``, exactly as the real ``predict_step``
    does, so the tests exercise the bbox bookkeeping rather than a mock of it.
    """

    def __init__(
        self,
        keypoints_model: torch.Tensor,
        drift_x: float = 0.0,
        nan_frames: tuple[int, ...] = (),
    ) -> None:
        self.keypoints_model = keypoints_model  # (num_keypoints, 2)
        self.drift_x = drift_x  # added to x per frame, so every frame gets a different bbox
        self.nan_frames = nan_frames
        self.num_frames_seen = 0
        self.batches: list[dict] = []

    def to(self, device) -> 'FakeNet':
        return self

    def eval(self) -> 'FakeNet':
        return self

    def predict_step(self, batch_dict, batch_idx: int):
        self.batches.append(batch_dict)
        batch = len(batch_dict['frames'])
        kp = self.keypoints_model.flatten().repeat(batch, 1).clone()
        kp = model_to_frame_batch(batch_dict, kp)
        conf = torch.ones(batch, len(self.keypoints_model))
        for idx in range(batch):
            idx_global = self.num_frames_seen + idx
            kp[idx, 0::2] += self.drift_x * idx_global
            if idx_global in self.nan_frames:
                kp[idx] = float('nan')
        self.num_frames_seen += batch
        return kp, conf


def make_fake_model(
    net: FakeNet, resize_dims: tuple[int, int], num_keypoints: int, model_dir: Path,
) -> MagicMock:
    """Build a ``Model``-shaped mock around ``net`` (single-view, non-context, fp32)."""
    model = MagicMock()
    model.model = net
    model.precision = 'fp32'
    model.model_dir = model_dir
    model.config.is_multi_view.return_value = False
    model.cfg = OmegaConf.create({
        'model': {'model_type': 'heatmap'},
        'data': {
            'keypoint_names': [f'kp{i}' for i in range(num_keypoints)],
            'image_resize_dims': {'height': resize_dims[0], 'width': resize_dims[1]},
        },
        'dali': {
            'base': {'train': {'sequence_length': 4}, 'predict': {'sequence_length': 96}},
            'context': {'train': {'batch_size': 4}, 'predict': {'sequence_length': 96}},
        },
    })
    return model


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """A small synthetic mp4 with ``NUM_FRAMES`` frames."""
    path = tmp_path / 'video.mp4'
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (FRAME_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        pytest.skip('cv2 cannot write mp4 on this machine')
    rng = np.random.default_rng(0)
    for _ in range(NUM_FRAMES):
        writer.write(rng.integers(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
    writer.release()
    return path
