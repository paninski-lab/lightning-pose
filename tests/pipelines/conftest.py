"""Fixtures for the pipeline tests: a synthetic video and CPU-only fake models."""

import subprocess
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import lightning_pose as lp
from lightning_pose.data.bboxes import model_to_frame_batch
from lightning_pose.data.utils import count_frames
from tests.fetch_test_data import fetch_test_data_if_needed

NUM_FRAMES = 20
FRAME_HEIGHT = 64
FRAME_WIDTH = 96
RESIZE_DETECTOR = (32, 48)  # (height, width): half the frame size
RESIZE_POSE = (16, 16)
NUM_CLIP_FRAMES = 20  # toy-model clip; with batch_size 8 the last batch is padded


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
        'eval': {'confidence_thresh_for_vid': 0.3, 'colormap': 'viridis'},
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


@pytest.fixture(scope='module')
def toy_model_dir() -> Generator[Path, None, None]:
    """The small trained mirror-mouse checkpoint shared with ``tests/api``.

    Used read-only (every test passes explicit output dirs), so it is not copied; the
    teardown fails the run if anything was written into it, since ``tests/api`` asserts that
    this directory has no prediction outputs.
    """
    api_test_dir = Path(__file__).parent.parent / 'api'
    fetch_test_data_if_needed(api_test_dir, 'test_model_mirror_mouse')
    model_dir = api_test_dir / 'test_model_mirror_mouse'
    yield model_dir
    stray = [d for d in ('video_preds', 'image_preds') if (model_dir / d).exists()]
    assert not stray, f'tests wrote {stray} into the shared model dir {model_dir}'


@pytest.fixture(scope='module')
def toy_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """First ``NUM_CLIP_FRAMES`` frames of the toy video, re-encoded as H.264.

    H.264 (not cv2's mp4v) so the GPU decoders (DALI, PyNvVideoCodec) can read it.
    """
    src = lp.LP_ROOT_PATH / 'data' / 'mirror-mouse-example' / 'videos' / 'test_vid.mp4'
    dst = tmp_path_factory.mktemp('toy_clip') / 'toy_clip.mp4'
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', str(src),
            '-frames:v', str(NUM_CLIP_FRAMES), '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(dst),
        ],
        check=True,
    )
    assert count_frames(str(dst)) == NUM_CLIP_FRAMES
    return dst
