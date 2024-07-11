"""Test dali dataloading functionality."""

import os

import numpy as np
import pytest


def test_video_pipe_single_view(video_list):

    from lightning_pose.data.dali import video_pipe

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

    # remove data from gpu; then cache can be cleared
    del pipe


def test_video_pipe_multiview(video_list):

    from lightning_pose.data.dali import video_pipe

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
        # test sizes of frames from view 0
        sequences_out_0 = pipe_out[0].as_cpu().as_array()
        assert sequences_out_0.shape == (batch_size, seq_len, 3, im_height, im_width)
        # test sizes of frames from view 1
        sequences_out_1 = pipe_out[1].as_cpu().as_array()
        assert sequences_out_1.shape == (batch_size, seq_len, 3, im_height, im_width)
        # make sure frames match (the different "views" correspond to the same video here)
        assert np.allclose(sequences_out_0[0, 0, 0], sequences_out_1[0, 0, 0])
        # make sure frames from different indices don't match
        assert not np.allclose(sequences_out_0[0, 0, 0], sequences_out_0[0, seq_len - 1, 0])

    # remove data from gpu; then cache can be cleared
    del pipe


def test_prepare_dali_single_view(cfg, video_list):

    from lightning_pose.data.dali import PrepareDALI

    im_height = 256
    im_width = 256

    filenames = video_list
    assert os.path.isfile(filenames[0])

    # -----------------------
    # base model
    # -----------------------
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="base",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[im_height, im_width],
    )
    loader = vid_pred_class()
    num_iters = vid_pred_class.num_iters

    # always sequence length should be fixed.
    for i, batch in enumerate(loader):
        assert batch["frames"].shape == (
            cfg.dali.base.predict.sequence_length,
            3,  # channels
            im_height,
            im_width,
        )
    assert i == num_iters - 1  # we have the right number of batches drawn

    # -----------------------
    # context model
    # -----------------------
    # different looking batch and shapes
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="context",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[im_height, im_width],
    )
    loader = vid_pred_class()
    num_iters = vid_pred_class.num_iters

    # this one assumes we have only two images in the last batch
    # NOTE: this is a specific property of this video and the context!
    for i, batch in enumerate(loader):
        assert batch["frames"].shape == (
            cfg.dali.context.predict.sequence_length,
            3,  # channels
            im_height,
            im_width,
        )
    assert i == num_iters - 1

    # error is thrown if one of the video files does not exist
    with pytest.raises(FileNotFoundError):
        PrepareDALI(
            train_stage="predict",
            model_type="base",
            filenames=[filenames[0] + '_bad-id.mp4'],
            dali_config=cfg.dali,
            resize_dims=[im_height, im_width],
        )

    # error is thrown if one of the video files is not a file
    with pytest.raises(FileNotFoundError):
        PrepareDALI(
            train_stage="predict",
            model_type="base",
            filenames=[os.path.dirname(filenames[0])],
            dali_config=cfg.dali,
            resize_dims=[im_height, im_width],
        )


def test_prepare_dali_multiview(cfg_multiview, video_list):

    from lightning_pose.data.dali import PrepareDALI
    from lightning_pose.data.utils import MultiviewUnlabeledBatchDict

    im_height = 256
    im_width = 256

    num_views = 2
    filenames = [video_list] * num_views  # really just copies of the same video

    # -----------------------
    # base model
    # -----------------------
    for train_stage in ["train", "predict"]:

        vid_pred_class = PrepareDALI(
            train_stage=train_stage,
            model_type="base",
            filenames=filenames,
            dali_config=cfg_multiview.dali,
            resize_dims=[im_height, im_width],
        )
        loader = vid_pred_class()

        # sequence length should be fixed for all batches
        if train_stage == "train":
            batch_size = cfg_multiview.dali.base.train.sequence_length
        else:
            batch_size = cfg_multiview.dali.base.predict.sequence_length
        frame_shape = (
            batch_size,
            num_views,
            3,  # channels
            im_height,
            im_width,
        )
        # just check a single batch
        batch = loader.__next__()
        assert batch["frames"].shape == frame_shape
        assert batch["transforms"].shape == (num_views, 1, 1)
        assert batch["bbox"].shape == (batch_size, num_views * 4)  # num_views * xyhw

    # -----------------------
    # context model
    # -----------------------
    for train_stage in ["train", "predict"]:

        vid_pred_class = PrepareDALI(
            train_stage=train_stage,
            model_type="context",
            filenames=filenames,
            dali_config=cfg_multiview.dali,
            resize_dims=[im_height, im_width],
        )
        loader = vid_pred_class()

        # sequence length should be fixed for all batches
        if train_stage == "train":
            batch_size = cfg_multiview.dali.context.train.batch_size
        else:
            batch_size = cfg_multiview.dali.context.predict.sequence_length
        frame_shape = (
            batch_size,
            num_views,
            3,  # channels
            im_height,
            im_width,
        )
        # just check a single batch
        batch = loader.__next__()
        assert batch["frames"].shape == frame_shape
        assert batch["transforms"].shape == (num_views, 1, 1)
        assert batch["bbox"].shape == (batch_size, num_views * 4)  # num_views * xyhw
