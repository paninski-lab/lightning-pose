"""Tests for lightning_pose.pipelines.stages."""

import numpy as np
import pytest
import torch

from lightning_pose.pipelines.stages import KeypointBboxStage

NAMES = ['a', 'b', 'c']


def _kp(*frames: list[list[float]]) -> torch.Tensor:
    return torch.tensor(frames, dtype=torch.float32)


class TestKeypointBboxStageInit:
    """Test KeypointBboxStage construction."""

    def test_requires_size_spec(self):
        with pytest.raises(ValueError, match='one of crop_ratio'):
            KeypointBboxStage(NAMES)

    def test_rejects_both_size_specs(self):
        with pytest.raises(ValueError, match='not both'):
            KeypointBboxStage(NAMES, crop_ratio=1.0, crop_height=10, crop_width=10)

    def test_rejects_half_fixed_size(self):
        with pytest.raises(ValueError, match='together'):
            KeypointBboxStage(NAMES, crop_height=10)

    def test_rejects_unknown_anchor(self):
        with pytest.raises(ValueError, match='zzz'):
            KeypointBboxStage(NAMES, anchor_keypoints=['zzz'], crop_ratio=1.0)

    def test_empty_anchors_use_all_keypoints(self):
        stage = KeypointBboxStage(NAMES, anchor_keypoints=[], crop_ratio=1.0)
        assert stage.idx_anchor == [0, 1, 2]


class TestKeypointBboxStageCall:
    """Test KeypointBboxStage.__call__."""

    def test_matches_offline_computation(self):
        # span 10 in x and 4 in y -> size 10, centred on mean (5, 2) -> top-left (0, -3)
        kp = _kp([[0, 0], [10, 4]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        out = stage(kp, torch.ones(1, 2), 100, 200)
        assert out.dtype == torch.float32
        assert out.tolist() == [[0.0, -3.0, 10.0, 10.0]]

    def test_anchor_subset(self):
        kp = _kp([[0, 0], [10, 10], [1000, 1000]])
        stage = KeypointBboxStage(NAMES, anchor_keypoints=['a', 'b'], crop_height=8, crop_width=8)
        out = stage(kp, torch.ones(1, 3), 100, 100)
        assert out.tolist() == [[1.0, 1.0, 8.0, 8.0]]

    def test_nan_reuses_previous_bbox(self):
        kp = _kp([[0, 0], [10, 10]], [[np.nan, np.nan], [np.nan, np.nan]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        out = stage(kp, torch.ones(2, 2), 100, 100)
        assert out[1].tolist() == out[0].tolist()
        assert stage.num_fallback == 1

    def test_previous_bbox_carries_across_batches(self):
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        first = stage(_kp([[0, 0], [10, 10]]), torch.ones(1, 2), 100, 100)
        second = stage(_kp([[np.nan, 0], [10, 10]]), torch.ones(1, 2), 100, 100)
        assert second[0].tolist() == first[0].tolist()

    def test_leading_missing_frames_get_full_frame(self):
        kp = _kp([[np.nan, 0], [10, 10]], [[0, 0], [10, 10]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        out = stage(kp, torch.ones(2, 2), 120, 160)
        assert out[0].tolist() == [0.0, 0.0, 120.0, 160.0]
        assert out[1].tolist() != out[0].tolist()
        assert stage.num_fallback == 1

    def test_low_confidence_triggers_fallback(self):
        kp = _kp([[0, 0], [10, 10]], [[50, 50], [60, 60]])
        conf = torch.tensor([[0.9, 0.9], [0.9, 0.1]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0, confidence_thresh=0.5)
        out = stage(kp, conf, 100, 100)
        assert out[1].tolist() == out[0].tolist()

    def test_confidence_ignored_without_thresh(self):
        kp = _kp([[0, 0], [10, 10]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0, confidence_thresh=None)
        stage(kp, torch.zeros(1, 2), 100, 100)
        assert stage.num_fallback == 0

    def test_default_confidence_thresh_is_half(self):
        kp = _kp([[0, 0], [10, 10]], [[50, 50], [60, 60]])
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        out = stage(kp, torch.tensor([[0.9, 0.9], [0.9, 0.49]]), 100, 100)
        assert out[1].tolist() == out[0].tolist()

    def test_reset_clears_state(self):
        stage = KeypointBboxStage(['a', 'b'], crop_ratio=1.0)
        stage(_kp([[0, 0], [10, 10]]), torch.ones(1, 2), 100, 100)
        stage.reset()
        out = stage(_kp([[np.nan, 0], [10, 10]]), torch.ones(1, 2), 50, 60)
        assert out[0].tolist() == [0.0, 0.0, 50.0, 60.0]
        assert stage.num_fallback == 1
