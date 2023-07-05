"""Test dali dataloading functionality."""

import torch

from lightning_pose.data.dali import video_pipe

_DALI_DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


def test_video_pipe(video_list):

    batch_size = 2
    seq_len = 7
    im_height = 256
    im_width = 256
    n_iter = 3

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
        sequences_out = pipe_out[0].as_cpu().as_array()
        assert sequences_out.shape == (batch_size, seq_len, 3, im_height, im_width)

    # remove data from gpu; then cache can be cleared
    del pipe


def test_PrepareDALI(cfg, video_list):
    import os

    from lightning_pose.data.dali import PrepareDALI

    filenames = video_list
    assert os.path.isfile(filenames[0])
    # base model: check we can build and run pipe and get a decent looking batch
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="base",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[256, 256],
    )
    pipe = vid_pred_class._get_dali_pipe()
    # can we build pipe?
    pipe.build()
    # can we run pipe?
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    # note: the 1 is there when we run pipe, but not when we obtain it through our lightning
    # wrapper
    assert sequences_out.shape == (
        1,
        cfg.dali.base.predict.sequence_length,
        3,
        256,
        256,
    )

    # starting it over since pipe_run grabs batches
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="base",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[256, 256],
    )
    loader = vid_pred_class()
    num_iters = vid_pred_class.num_iters

    # always sequence length should be fixed.
    for i, batch in enumerate(loader):
        assert batch["frames"].shape == (
            cfg.dali.base.predict.sequence_length,
            3,
            256,
            256,
        )
    assert i == num_iters - 1  # we have the right number of batches drawn

    # context model, different looking batch and shapes
    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="context",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[256, 256],
    )
    pipe = vid_pred_class._get_dali_pipe()
    # can we build pipe?
    pipe.build()
    # can we run pipe?
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    assert sequences_out.shape == (
        1,
        cfg.dali.context.predict.sequence_length,
        3,
        256,
        256,
    )

    vid_pred_class = PrepareDALI(
        train_stage="predict",
        model_type="context",
        filenames=filenames,
        dali_config=cfg.dali,
        resize_dims=[256, 256],
    )
    loader = vid_pred_class()

    num_iters = vid_pred_class.num_iters

    # this one assumes we're gonna have only two images in the last batch. this is a specific
    # property of this video and the context.
    for i, batch in enumerate(loader):
        assert batch["frames"].shape == (
            cfg.dali.context.predict.sequence_length,
            3,
            256,
            256,
        )
    assert i == num_iters - 1

    # OLD TESTS: for when we loaded 5-frame batches
    # now on the final batch, check that we're actually grabing 5-frame sequences
    # assert that frame 1 in batch 0 is frame 0 in batch 1
    # assert torch.allclose(batch["frames"][0][1], batch["frames"][1][0])

    # last sequence of 5 frames, with a step=1, means that only the 0th image is an actual image.
    # the rest is padding. so image 1 and -1 are identical.
    # assert torch.allclose(batch["frames"][1][1], batch["frames"][1][-1])
