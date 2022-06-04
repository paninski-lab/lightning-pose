"""Test dali dataloading functionality."""

from nvidia.dali.plugin.pytorch import LastBatchPolicy
import pytest
import torch

from lightning_pose.data.dali import video_pipe, LightningWrapper

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


def test_dali_wrapper(cfg, video_list):

    from lightning_pose.data.utils import count_frames

    batch_size = 1
    seq_len = 8
    im_height = cfg.data.image_resize_dims.height
    im_width = cfg.data.image_resize_dims.width

    data_pipe = video_pipe(
        filenames=video_list,
        resize_dims=[im_height, im_width],
        sequence_length=seq_len,
        batch_size=batch_size,
        random_shuffle=True,
        seed=0,
        device_id=0,
        num_threads=2,
    )

    # compute number of batches
    total_frames = count_frames(video_list)  # sum across vids
    num_batches = int(total_frames // seq_len)  # assuming batch_size==1

    data_loader = LightningWrapper(
        data_pipe,
        output_map=["x"],
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True,
        num_batches=num_batches,
    )

    for batch in data_loader:
        assert batch.shape == (seq_len, 3, im_height, im_width)
        # just check a single batch
        break

def test_video_prediction_class(video_list):
    from lightning_pose.data.datamodules import VideoPredictionMixin
    import os
    filenames = video_list
    assert os.path.isfile(filenames[0])
    # base model: check we can build and run pipe and get a decent looking batch
    vid_pred_class = VideoPredictionMixin(train_stage="predict", model_type="base", filenames=filenames)
    pipe = vid_pred_class._get_dali_pipe()
    # can we build pipe?
    pipe.build()
    # can we run pipe?
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    # note: the 1 is there when we run pipe, but not when we obtain it through our lightning wrapper
    assert sequences_out.shape == (1, 16, 3, 256, 256)

    # context model, different looking batch and shapes 
    vid_pred_class = VideoPredictionMixin(train_stage="predict", model_type="context", filenames=filenames)
    pipe = vid_pred_class._get_dali_pipe()
    # can we build pipe?
    pipe.build()
    # can we run pipe?
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    assert sequences_out.shape == (4,5, 3, 256, 256)

    vid_pred_class = VideoPredictionMixin(train_stage="predict", model_type="context", filenames=filenames)
    loader = vid_pred_class()
    # how many batches?

    from lightning_pose.data.utils import count_frames
    import numpy as np
    batch_size = 4 # TODO: cuurently hard-coded at the datamodules
    frame_count = count_frames(filenames[0])
    # assuming step=1
    # "how many times should we enumerate the data loader?""
    num_iters = int(np.ceil(frame_count / batch_size))

    for i, batch in enumerate(loader):
        if i < num_iters:
            assert batch.shape == (4, 5, 3, 256, 256)
        elif i == num_iters:
            assert batch.shape == (2, 5, 3, 256, 256)
    assert(i == num_iters)
