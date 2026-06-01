"""Test dali dataloading functionality."""

import os
import shutil

import numpy as np
import pytest

from lightning_pose.data import dali as dali_module
from lightning_pose.data.dali import PrepareDALI, video_pipe


class TestVideoPipe:
    """Test the video_pipe function."""

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
