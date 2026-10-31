"""Test dali dataloading functionality."""

import os
import shutil
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip('nvidia.dali', reason='nvidia-dali not installed')

from lightning_pose.data.video import dali as dali_module
from lightning_pose.data.video.dali import LitDaliWrapper, PrepareDALI, video_pipe


class TestVideoPipe:
    """Test the video_pipe function."""

    @pytest.mark.gpu
    def test_single_view(self, video_list):
        """Single-view pipeline yields (frames, transforms, frame_size) tuples of correct shape."""
        batch_size = 2
        seq_len = 7
        im_height = 256
        im_width = 256
        n_iter = 2

        pipe = video_pipe(
            filenames=video_list,
            resize_dims=[im_height, im_width],
            sequence_length=seq_len,
            batch_size=batch_size,
            device_id=0,
            num_threads=2,
        )
        pipe.build()
        for _ in range(n_iter):
            pipe_out = pipe.run()
            assert len(pipe_out) == 3  # frames, transforms, orig_frame_size
            sequences_out = pipe_out[0].as_cpu().as_array()
            assert sequences_out.shape == (batch_size, seq_len, 3, im_height, im_width)

        del pipe

    @pytest.mark.gpu
    def test_multiview(self, video_list):
        """Multi-view pipeline yields per-view tuples; identical videos produce matching frames."""
        batch_size = 2
        seq_len = 7
        im_height = 256
        im_width = 256
        n_iter = 2

        pipe = video_pipe(
            filenames=[video_list, video_list],
            resize_dims=[im_height, im_width],
            sequence_length=seq_len,
            batch_size=batch_size,
            device_id=0,
            num_threads=2,
        )
        pipe.build()
        for _ in range(n_iter):
            pipe_out = pipe.run()
            assert len(pipe_out) == 2 * 3  # num_views * (frames, transforms, orig_frame_size)
            sequences_out_0 = pipe_out[0].as_cpu().as_array()
            assert sequences_out_0.shape == (batch_size, seq_len, 3, im_height, im_width)
            sequences_out_1 = pipe_out[1].as_cpu().as_array()
            assert sequences_out_1.shape == (batch_size, seq_len, 3, im_height, im_width)
            # same source video → same frames across views
            assert np.allclose(sequences_out_0[0, 0, 0], sequences_out_1[0, 0, 0])
            # different time indices produce different frames
            assert not np.allclose(sequences_out_0[0, 0, 0], sequences_out_0[0, seq_len - 1, 0])

        del pipe


class TestPrepareDALI:
    """Test the PrepareDALI class."""

    @pytest.mark.gpu
    def test_single_view_base_predict(self, cfg, video_list):
        """Base model predict loader yields batches of the configured sequence length and dims."""
        im_height = 256
        im_width = 256

        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=[im_height, im_width],
        )
        loader = vid_pred_class()
        num_iters = vid_pred_class.num_iters

        batch_idx = -1
        for _, batch in enumerate(loader):
            assert batch['frames'].shape == (
                cfg.dali.base.predict.sequence_length,
                3,
                im_height,
                im_width,
            )
            batch_idx += 1
        assert batch_idx == num_iters - 1

    @pytest.mark.gpu
    def test_single_view_context_predict(self, cfg, video_list):
        """Context model predict loader yields batches of the context sequence length and dims."""
        im_height = 256
        im_width = 256

        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='context',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=[im_height, im_width],
        )
        loader = vid_pred_class()
        num_iters = vid_pred_class.num_iters

        batch_idx = -1
        for _, batch in enumerate(loader):
            assert batch['frames'].shape == (
                cfg.dali.context.predict.sequence_length,
                3,
                im_height,
                im_width,
            )
            batch_idx += 1
        assert batch_idx == num_iters - 1

    def test_single_view_raises_on_nonexistent_file(self, cfg, video_list):
        """FileNotFoundError is raised when a video path does not exist."""
        with pytest.raises(FileNotFoundError):
            PrepareDALI(
                train_stage='predict',
                model_type='base',
                filenames=[video_list[0] + '_bad-id.mp4'],
                dali_config=cfg.dali,
                resize_dims=[256, 256],
            )

    def test_single_view_raises_when_path_is_directory(self, cfg, video_list):
        """FileNotFoundError is raised when a video path points to a directory."""
        with pytest.raises(FileNotFoundError):
            PrepareDALI(
                train_stage='predict',
                model_type='base',
                filenames=[os.path.dirname(video_list[0])],
                dali_config=cfg.dali,
                resize_dims=[256, 256],
            )

    @pytest.mark.gpu
    def test_multiview_base_train_and_predict(self, cfg_multiview, video_list):
        """Multi-view base model batches have correct shape for train and predict stages."""
        im_height = 256
        im_width = 256
        num_views = 2
        filenames = [video_list] * num_views

        for train_stage in ['train', 'predict']:
            vid_pred_class = PrepareDALI(
                train_stage=train_stage,  # type: ignore[arg-type]
                model_type='base',
                filenames=filenames,
                dali_config=cfg_multiview.dali,
                resize_dims=[im_height, im_width],
            )
            loader = vid_pred_class()
            batch_size = (
                cfg_multiview.dali.base.train.sequence_length
                if train_stage == 'train'
                else cfg_multiview.dali.base.predict.sequence_length
            )
            batch = loader.__next__()
            assert batch['frames'].shape == (batch_size, num_views, 3, im_height, im_width)
            assert batch['transforms'].shape == (num_views, 1, 1)
            assert batch['bbox'].shape == (batch_size, num_views * 4)

    @pytest.mark.gpu
    def test_multiview_context_train_and_predict(self, cfg_multiview, video_list):
        """Multi-view context model batches have correct shape for train and predict stages."""
        im_height = 256
        im_width = 256
        num_views = 2
        filenames = [video_list] * num_views

        for train_stage in ['train', 'predict']:
            vid_pred_class = PrepareDALI(
                train_stage=train_stage,  # type: ignore[arg-type]
                model_type='context',
                filenames=filenames,
                dali_config=cfg_multiview.dali,
                resize_dims=[im_height, im_width],
            )
            loader = vid_pred_class()
            batch_size = (
                cfg_multiview.dali.context.train.batch_size
                if train_stage == 'train'
                else cfg_multiview.dali.context.predict.sequence_length
            )
            batch = loader.__next__()
            assert batch['frames'].shape == (batch_size, num_views, 3, im_height, im_width)
            assert batch['transforms'].shape == (num_views, 1, 1)
            assert batch['bbox'].shape == (batch_size, num_views * 4)

    @pytest.mark.gpu
    def test_multiview_synchronized_frames(self, cfg_multiview, video_list):
        """Shared reader seed keeps per-view frames synchronized under random shuffle."""
        num_views = 3
        filenames = [video_list] * num_views

        vid_pred_class = PrepareDALI(
            train_stage='train',
            model_type='base',
            filenames=filenames,
            dali_config=cfg_multiview.dali,
            resize_dims=[256, 256],
            imgaug='default',
        )
        loader = vid_pred_class()
        for _ in range(4):
            batch = loader.__next__()
            frames = batch['frames'].cpu().numpy()
            for view in range(1, num_views):
                assert np.allclose(frames[:, 0], frames[:, view])
            # sanity: shuffled sequences still contain distinct frames, not a static clip
            assert not np.allclose(frames[:, 0, 0], frames[:, 0, -1])

    def test_multiview_raises_on_unequal_session_count(self, cfg_multiview, video_list):
        """ValueError is raised when views have different numbers of sessions."""
        vid = video_list[0]
        with pytest.raises(ValueError, match='same number of sessions'):
            PrepareDALI(
                train_stage='train',
                model_type='base',
                filenames=[[vid, vid], [vid]],
                dali_config=cfg_multiview.dali,
                resize_dims=[256, 256],
            )

    def test_multiview_raises_on_unequal_frame_counts(
        self, cfg_multiview, video_list, tmp_path, monkeypatch,
    ):
        """ValueError is raised when frame counts differ across views for the same session."""
        vid = video_list[0]
        vid_copy = str(tmp_path / 'view1_session0.mp4')
        shutil.copy(vid, vid_copy)
        monkeypatch.setattr(
            dali_module, 'count_frames', lambda p: {vid: 100, vid_copy: 90}[p],
        )
        with pytest.raises(ValueError, match='frame counts across views'):
            PrepareDALI(
                train_stage='train',
                model_type='base',
                filenames=[[vid], [vid_copy]],
                dali_config=cfg_multiview.dali,
                resize_dims=[256, 256],
            )

    def test_bbox_df_sets_none_resize_dims_in_predict_pipes(self, cfg, video_list):
        """When bbox_df is provided, resize_dims is None in both predict pipe arg dicts."""
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [100], 'w': [100]})
        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=[128, 128],
            bbox_df=bbox_df,
        )
        assert vid_pred_class._pipe_dict['predict']['base']['resize_dims'] is None
        assert vid_pred_class._pipe_dict['predict']['context']['resize_dims'] is None

    def test_bbox_df_requires_resize_dims(self, cfg, video_list):
        """bbox cropping needs a post-crop resize target, so resize_dims=None raises."""
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [100], 'w': [100]})
        with pytest.raises(ValueError, match='resize_dims is required when bbox_df'):
            PrepareDALI(
                train_stage='predict',
                model_type='base',
                filenames=video_list,
                dali_config=cfg.dali,
                resize_dims=None,
                bbox_df=bbox_df,
            )

    def test_no_resize_dims_without_bbox_df_is_full_resolution(self, cfg, video_list):
        """resize_dims=None with no bbox_df leaves the predict pipes at full resolution."""
        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=None,
        )
        assert vid_pred_class._pipe_dict['predict']['base']['resize_dims'] is None
        assert vid_pred_class._pipe_dict['predict']['context']['resize_dims'] is None

    def test_bbox_df_count_must_match_num_views(self, cfg, video_list):
        """Passing more bbox DataFrames than views raises a ValueError."""
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [100], 'w': [100]})
        with pytest.raises(ValueError, match='one bbox DataFrame per view'):
            PrepareDALI(
                train_stage='predict',
                model_type='base',
                filenames=video_list,
                dali_config=cfg.dali,
                resize_dims=[128, 128],
                bbox_df=[bbox_df, bbox_df],
            )

    def test_bbox_df_preserves_train_pipe_resize_dims(self, cfg, video_list):
        """When bbox_df is provided, train pipe resize_dims is unchanged."""
        resize_dims = [128, 128]
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [100], 'w': [100]})
        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=resize_dims,
            bbox_df=bbox_df,
        )
        assert vid_pred_class._pipe_dict['train']['base']['resize_dims'] == resize_dims
        assert vid_pred_class._pipe_dict['train']['context']['resize_dims'] == resize_dims

    def test_bbox_df_forwarded_to_lit_dali_wrapper(self, cfg, video_list):
        """__call__ passes bbox_df and the original resize_dims to LitDaliWrapper."""
        resize_dims = [128, 128]
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [100], 'w': [100]})
        vid_pred_class = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=resize_dims,
            bbox_df=bbox_df,
        )
        with (
            patch.object(vid_pred_class, '_get_dali_pipe', return_value=MagicMock()),
            patch('lightning_pose.data.video.dali.LitDaliWrapper') as MockWrapper,
        ):
            vid_pred_class()
        call_kwargs = MockWrapper.call_args.kwargs
        assert call_kwargs['bbox_df'] is bbox_df
        assert call_kwargs['resize_dims'] == resize_dims


class TestLitDaliWrapper:
    """Test the LitDaliWrapper class."""

    @pytest.fixture
    def bbox_df(self):
        """Sample bbox DataFrame with 10 rows."""
        return pd.DataFrame({
            'x': [10] * 10,
            'y': [20] * 10,
            'h': [50] * 10,
            'w': [60] * 10,
        })

    def _make_wrapper(
        self,
        bbox_df: pd.DataFrame | list[pd.DataFrame] | None,
        resize_dims: list[int] | None,
        do_context: bool = False,
        frame_idx: int = 0,
    ) -> LitDaliWrapper:
        """Create a LitDaliWrapper without a real DALI pipeline."""
        wrapper = object.__new__(LitDaliWrapper)
        wrapper.do_context = do_context
        wrapper.bbox_df = bbox_df
        wrapper.resize_dims = resize_dims
        wrapper._frame_idx = frame_idx
        return wrapper

    @staticmethod
    def _frame_size(seq_len: int, h: int, w: int) -> torch.Tensor:
        """DALI ``frame_size`` output (order seq_len,C,H,W) for one view."""
        return torch.tensor([[seq_len, 3, h, w]])

    def _make_batch(
        self, seq_len: int, h: int = 100, w: int = 120, transforms: torch.Tensor | None = None,
    ) -> list[dict]:
        """Create a fake raw single-view DALI output with random frames."""
        if transforms is None:
            transforms = torch.zeros(seq_len, 1)
        return [{
            'frames': torch.rand(1, seq_len, 3, h, w),
            'transforms': transforms.unsqueeze(0),
            'frame_size': self._frame_size(seq_len, h, w),
        }]

    def _make_batch_multiview(self, seq_len: int, sizes: list[tuple[int, int]]) -> list[dict]:
        """Create a fake raw multiview DALI output; ``sizes`` is (h, w) of each view."""
        batch = {}
        for idx, (h, w) in enumerate(sizes):
            batch[f'frames_{idx}'] = torch.rand(1, seq_len, 3, h, w)
        for idx in range(len(sizes)):
            batch[f'transforms_{idx}'] = torch.zeros(1)
        for idx, (h, w) in enumerate(sizes):
            batch[f'frame_size_{idx}'] = self._frame_size(seq_len, h, w)
        return [batch]

    def test_output_frames_shape(self, bbox_df):
        """Cropped+resized frames have shape (seq_len, 3, *resize_dims)."""
        wrapper = self._make_wrapper(bbox_df, [64, 64])
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=4))
        assert result['frames'].shape == (4, 3, 64, 64)
        assert not result['is_multiview']

    def test_bbox_tensor_values(self, bbox_df):
        """Output bbox tensor contains [x, y, h, w] values from bbox_df."""
        wrapper = self._make_wrapper(bbox_df, [64, 64])
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=3))
        expected = torch.tensor([10, 20, 50, 60], dtype=torch.float32)
        for i in range(3):
            assert torch.allclose(result['bbox'][i], expected)

    def test_no_bbox_df_uses_full_frame_bbox(self):
        """Without bbox_df the bbox is the full decoded frame [0, 0, H, W]."""
        wrapper = self._make_wrapper(None, None)
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=2, h=100, w=120))
        assert result['frames'].shape == (2, 3, 100, 120)
        expected = torch.tensor([0, 0, 100, 120], dtype=torch.float32).repeat(2, 1)
        assert torch.allclose(result['bbox'], expected)
        assert wrapper._frame_idx == 0

    def test_frame_size_hwc_order(self):
        """frame_size with channels last (seq_len,H,W,C) is parsed as (height, width)."""
        frame_size = torch.tensor([[2, 100, 120, 3]])
        assert LitDaliWrapper._frame_size_hw(frame_size) == (100, 120)

    def test_advances_frame_idx_base(self, bbox_df):
        """_frame_idx advances by seq_len for a base (non-context) model."""
        wrapper = self._make_wrapper(bbox_df, [64, 64], do_context=False)
        wrapper._dali_output_to_tensors(self._make_batch(seq_len=4))
        assert wrapper._frame_idx == 4

    def test_advances_frame_idx_context(self, bbox_df):
        """_frame_idx advances by seq_len - 4 for a context model."""
        wrapper = self._make_wrapper(bbox_df, [64, 64], do_context=True)
        wrapper._dali_output_to_tensors(self._make_batch(seq_len=5))
        assert wrapper._frame_idx == 1  # step = seq_len - 4 = 1

    def test_pads_last_partial_batch(self):
        """Last batch is padded with the final bbox row when fewer rows remain."""
        two_row_df = pd.DataFrame({
            'x': [10, 20],
            'y': [10, 20],
            'h': [50, 60],
            'w': [50, 60],
        })
        wrapper = self._make_wrapper(two_row_df, [64, 64], frame_idx=1)
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=4))
        assert result['bbox'].shape == (4, 4)
        expected_last = torch.tensor([20, 20, 60, 60], dtype=torch.float32)
        for i in range(4):
            assert torch.allclose(result['bbox'][i], expected_last)

    def test_transforms_preserved(self, bbox_df):
        """The transforms field from the raw DALI output is passed through unchanged."""
        wrapper = self._make_wrapper(bbox_df, [64, 64])
        transforms = torch.tensor([[1.0], [2.0], [3.0]])
        batch = self._make_batch(seq_len=3, transforms=transforms)
        result = wrapper._dali_output_to_tensors(batch)
        assert torch.equal(result['transforms'], transforms)

    def test_negative_xy_clamped(self):
        """Negative x/y from edge-frame bbox are clamped to 0; stored bbox reflects actual crop.

        create_bbox computes topleft = centroid - size//2, which goes negative for animals
        near the frame edge. pytorch slice indices are not clamped — a negative start index
        counts from the end of the tensor, producing an empty crop.
        """
        neg_df = pd.DataFrame({'x': [-10], 'y': [-5], 'h': [50], 'w': [60]})
        wrapper = self._make_wrapper(neg_df, [64, 64])
        # frame 100 tall x 120 wide
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=1, h=100, w=120))

        # frame must be cropped+resized without error
        assert result['frames'].shape == (1, 3, 64, 64)
        # clamped: x1=0, y1=0, x2=min(120,-10+60)=50, y2=min(100,-5+50)=45 → [0,0,45,50]
        expected_bbox = torch.tensor([0.0, 0.0, 45.0, 50.0])
        assert torch.allclose(result['bbox'][0], expected_bbox)

    def test_bbox_df_list_for_single_view(self, bbox_df):
        """A one-element list of DataFrames behaves like a bare DataFrame for single-view."""
        wrapper = self._make_wrapper([bbox_df], [64, 64])
        result = wrapper._dali_output_to_tensors(self._make_batch(seq_len=3))
        assert result['frames'].shape == (3, 3, 64, 64)
        assert result['bbox'].shape == (3, 4)

    def test_multiview_no_bbox_df(self):
        """Multiview without bbox_df stacks views and uses each view's full-frame bbox."""
        wrapper = self._make_wrapper(None, None)
        batch = self._make_batch_multiview(seq_len=2, sizes=[(100, 120), (100, 120)])
        result = wrapper._dali_output_to_tensors(batch)
        assert result['is_multiview']
        assert result['frames'].shape == (2, 2, 3, 100, 120)
        assert result['transforms'].shape[0] == 2
        expected = torch.tensor([0, 0, 100, 120, 0, 0, 100, 120], dtype=torch.float32)
        assert torch.allclose(result['bbox'], expected.repeat(2, 1))

    def test_multiview_bbox_differing_view_sizes(self):
        """Views with different native resolutions are cropped independently, then stacked."""
        df_a = pd.DataFrame({'x': [10] * 4, 'y': [20] * 4, 'h': [50] * 4, 'w': [60] * 4})
        df_b = pd.DataFrame({'x': [5] * 4, 'y': [6] * 4, 'h': [30] * 4, 'w': [40] * 4})
        wrapper = self._make_wrapper([df_a, df_b], [64, 64])
        batch = self._make_batch_multiview(seq_len=3, sizes=[(100, 120), (200, 300)])
        result = wrapper._dali_output_to_tensors(batch)

        assert result['is_multiview']
        assert result['frames'].shape == (3, 2, 3, 64, 64)
        assert result['bbox'].shape == (3, 8)
        expected = torch.tensor([10, 20, 50, 60, 5, 6, 30, 40], dtype=torch.float32)
        assert torch.allclose(result['bbox'], expected.repeat(3, 1))

    def test_multiview_bbox_advances_frame_idx_once(self):
        """_frame_idx advances once per batch, not once per view."""
        df = pd.DataFrame({'x': [0] * 10, 'y': [0] * 10, 'h': [50] * 10, 'w': [50] * 10})
        wrapper = self._make_wrapper([df, df], [64, 64])
        batch = self._make_batch_multiview(seq_len=4, sizes=[(100, 120), (100, 120)])
        wrapper._dali_output_to_tensors(batch)
        assert wrapper._frame_idx == 4

    def test_multiview_bbox_df_count_mismatch_raises(self, bbox_df):
        """A bare DataFrame (one view's worth) for a two-view batch raises."""
        wrapper = self._make_wrapper(bbox_df, [64, 64])
        batch = self._make_batch_multiview(seq_len=2, sizes=[(100, 120), (100, 120)])
        with pytest.raises(ValueError, match='expected 2 bbox DataFrame'):
            wrapper._dali_output_to_tensors(batch)
