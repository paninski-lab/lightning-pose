"""Test the helpers shared by the video readers."""

import pandas as pd
import pytest
import torch

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.video.utils import (
    LitPredictWrapperBase,
    PreparePredictBase,
    build_batch_dict,
    normalize_frames,
)


class TestNormalizeFrames:
    """Test the function normalize_frames."""

    def test_normalize_frames_no_resize_when_dims_none(self):
        out = normalize_frames(torch.zeros(2, 3, 20, 30, dtype=torch.uint8), None)
        assert out.shape == (2, 3, 20, 30)

    def test_normalize_frames_resizes_when_dims_set(self):
        out = normalize_frames(torch.zeros(2, 3, 20, 30, dtype=torch.uint8), [8, 8])
        assert out.shape == (2, 3, 8, 8)

    def test_normalize_frames_applies_imagenet_normalization(self):
        frames = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)
        out = normalize_frames(frames, None)
        for channel in range(3):
            expected = (1.0 - _IMAGENET_MEAN[channel]) / _IMAGENET_STD[channel]
            assert torch.allclose(out[0, channel], torch.tensor(expected))


class TestBuildBatchDict:
    """Test the function build_batch_dict."""

    def test_build_batch_dict_single_view(self):
        frames = [torch.rand(4, 3, 8, 8)]
        bboxes = [torch.zeros(4, 4)]
        transforms = torch.tensor([-1.0])
        batch = build_batch_dict(frames, bboxes, transforms, is_multiview=False)
        assert batch['is_multiview'] is False
        assert batch['frames'].shape == (4, 3, 8, 8)
        assert batch['bbox'].shape == (4, 4)
        assert torch.equal(batch['transforms'], transforms)

    def test_build_batch_dict_multiview_stacks_views(self):
        frames = [torch.rand(4, 3, 8, 8) for _ in range(2)]
        bboxes = [torch.zeros(4, 4) for _ in range(2)]
        batch = build_batch_dict(frames, bboxes, torch.zeros(2, 1, 1), is_multiview=True)
        assert batch['is_multiview'] is True
        assert batch['frames'].shape == (4, 2, 3, 8, 8)
        assert batch['bbox'].shape == (4, 8)


class TestAbstractBases:
    """The shared base classes cannot be used without a decoder."""

    def test_wrapper_base_is_abstract(self):
        with pytest.raises(TypeError, match='abstract'):
            LitPredictWrapperBase(  # type: ignore[abstract]
                resize_dims=[8, 8], decode_resize_dims=None, sequence_length=4, step=4,
                do_context=False, num_iters=1, multiview=False,
            )

    def test_prepare_base_is_abstract(self):
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [10], 'w': [10]})
        with pytest.raises(TypeError, match='abstract'):
            PreparePredictBase(  # type: ignore[abstract]
                'base', ['x.mp4'], [8, 8], {}, bbox_df=bbox_df,
            )
