import imgaug.augmenters as iaa
import numpy as np
import torch
from pose_est_nets.utils.IO import set_or_open_folder, get_latest_version
from pose_est_nets.models.heatmap_tracker import (
    HeatmapTracker,
    SemiSupervisedHeatmapTracker,
)
from pose_est_nets.models.regression_tracker import (
    RegressionTracker,
    SemiSupervisedRegressionTracker,
)
import matplotlib.pyplot as plt
import os
from typing import Callable, Optional, Tuple, List
from typeguard import typechecked

def get_model_class(map_type: str, semi_supervised: bool):
    if not(semi_supervised):
        if map_type == "regression":
            return RegressionTracker
        elif map_type == "heatmap":
            return HeatmapTracker
    else:
        if map_type == "regression":
            return SemiSupervisedRegressionTracker
        elif map_type == "heatmap":
            return SemiSupervisedHeatmapTracker

def saveNumericalPredictions(model, datamod, threshold):
    i = 0
    # hardcoded for mouse data
    rev_augmenter = []
    rev_augmenter.append(
        iaa.Resize({"height": 406, "width": 396})
    )  # get rid of this for the fish
    rev_augmenter = iaa.Sequential(rev_augmenter)
    model.eval()
    full_dl = datamod.full_dataloader()
    test_dl = datamod.test_dataloader()
    final_gt_keypoints = np.empty(shape=(len(test_dl), model.num_keypoints, 2))
    final_imgs = np.empty(shape=(len(test_dl), 406, 396, 1))
    final_preds = np.empty(shape=(len(test_dl), model.num_keypoints, 2))

    # dpk_final_preds = np.empty(shape = (len(test_dl), model.num_keypoints, 2))

    for idx, batch in enumerate(test_dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if torch.cuda.is_available():
            heatmap_pred = heatmap_pred.cuda()
            y = y.cuda()
        pred_keypoints, y_keypoints = model.computeSubPixMax(heatmap_pred, y, threshold)
        # dpk_final_preds[i] = pred_keypoints
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        x = x[:, 0, :, :]  # only taking one image dimension
        x = np.expand_dims(x, axis=3)
        final_imgs[i], final_gt_keypoints[i] = rev_augmenter(
            images=x, keypoints=np.expand_dims(y_keypoints, axis=0)
        )
        final_imgs[i], final_preds[i] = rev_augmenter(
            images=x, keypoints=np.expand_dims(pred_keypoints, axis=0)
        )
        # final_gt_keypoints[i] = y_keypoints
        # final_preds[i] = pred_keypoints
        i += 1

    final_gt_keypoints = np.reshape(
        final_gt_keypoints, newshape=(len(test_dl), model.num_targets)
    )
    final_preds = np.reshape(final_preds, newshape=(len(test_dl), model.num_targets))
    # dpk_final_preds = np.reshape(dpk_final_preds, newshape = (len(test_dl), model.num_targets))

    # np.savetxt('../../preds/mouse_gt.csv', final_gt_keypoints, delimiter = ',', newline = '\n')
    folder_name = get_latest_version("lightning_logs")
    csv_folder = set_or_open_folder(os.path.join("preds", folder_name))

    np.savetxt(
        os.path.join(csv_folder, "preds.csv"), final_preds, delimiter=",", newline="\n"
    )
    # np.savetxt('../preds/dpk_fish_predictions.csv', dpk_final_preds, delimiter = ',', newline = '\n')
    return


def plotPredictions(model, datamod, save_heatmaps, threshold, mode):
    folder_name = get_latest_version("lightning_logs")
    img_folder = set_or_open_folder(os.path.join("preds", folder_name, "images"))

    if save_heatmaps:
        heatmap_folder = set_or_open_folder(
            os.path.join("preds", folder_name, "heatmaps")
        )

    model.eval()
    if mode == "train":
        dl = datamod.train_dataloader()
    else:
        dl = datamod.test_dataloader()
    i = 0
    for idx, batch in enumerate(dl):
        x, y = batch
        heatmap_pred = model.forward(x)
        if save_heatmaps:
            plt.imshow(heatmap_pred[0, 4].detach().cpu().numpy())
            plt.savefig(os.path.join(heatmap_folder, "pred_map_%i" % i + ".png"))
            plt.clf()
            plt.imshow(y[0, 4].detach().cpu().numpy())
            plt.savefig(os.path.join(heatmap_folder, "gt_map_%i" % i + ".png"))
            plt.clf()
        # if torch.cuda.is_available():
        #     heatmap_pred = heatmap_pred.cuda()
        #     y = y.cuda()
        # TODO: that works, but remove cuda calls! threshold is on model which is on cuda, heatmap_pred and y are on CPU after saving heatmaps
        pred_keypoints, y_keypoints = model.computeSubPixMax(
            heatmap_pred.cuda(), y.cuda(), threshold
        )
        plt.imshow(x[0][0])
        pred_keypoints = pred_keypoints.cpu()
        y_keypoints = y_keypoints.cpu()
        plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c="blue")
        plt.scatter(y_keypoints[:, 0], y_keypoints[:, 1], c="orange")
        plt.savefig(os.path.join(img_folder, "pred_%i" % i + ".png"))
        plt.clf()
        i += 1


def predict_videos(
        video_path, model_file, config_file, save_file=None, sequence_length=16, device="gpu",
        video_pipe_kwargs={}):
    """Loop over a list of videos and process with tracker using DALI for fast inference.

    Args:
        video_path (str): process all videos located in this directory
        model_file (str): .ckpt file for model
        config_file (str): yaml file saved by hydra; must contain
            - config_file.losses
            - config_file.data.image_orig_dims
            - config_file.data.image_resize_dims
            - config_file.model.losses_to_use
            - config_file.model.data_type
            - config_file.model.semi_supervised
        save_file (str): full filename of tracked points; currently supports hdf5 and csv; if
            NoneType, the output will be saved in the video path
        sequence_length (int)
        device (str): "gpu" | "cpu"
        video_pipe_kwargs (dict): extra keyword-value argument pairs for
            `pose_est_nets.datasets.DALI.video_pipe` function

    TODO: support different video formats

    """

    import csv
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
    import nvidia.dali.types as types
    from omegaconf import OmegaConf
    import pandas as pd
    import time

    from pose_est_nets.datasets.DALI import video_pipe, LightningWrapper, count_frames
    from pose_est_nets.datasets.datasets import BaseTrackingDataset, HeatmapDataset
    from pose_est_nets.models.regression_tracker import RegressionTracker, \
        SemiSupervisedRegressionTracker
    from pose_est_nets.models.heatmap_tracker import HeatmapTracker, SemiSupervisedHeatmapTracker
    from pose_est_nets.utils.IO import set_or_open_folder, get_latest_version

    # check input
    if save_file is not None:
        if not (save_file.endswith(".csv") or save_file.endswith(".hdf5")
                or save_file.endswith(".hdf") or save_file.endswith(".h5")
                or save_file.endswith(".h")):
            raise NotImplementedError("Currently only .csv and .h5 files are supported")

    if device == "gpu" or device == "cuda":
        device_pt = "cuda"
        device_dali = "gpu"
    elif device == "cpu":
        device_pt = "cpu"
        device_dali = "cpu"
    else:
        raise NotImplementedError("must choose 'gpu' or 'cpu' for `device` argument")

    # gather videos to process
    assert os.path.exists(video_path)
    all_files = [video_path + "/" + f for f in os.listdir(video_path)]
    video_files = []
    for f in all_files:
        if f.endswith(".mp4"):
            video_files.append(f)
    if len(video_files) == 0:
        raise IOError("Did not find any video files (.mp4) in %s" % video_path)

    # load configuration file
    with open(config_file, 'r') as f:
        cfg = OmegaConf.load(f)

    # load model weights
    if not cfg.model.semi_supervised:
        if cfg.model.data_type == "regression":
            model = RegressionTracker.load_from_checkpoint(model_file)
        elif cfg.model.data_type == "heatmap":
            model = HeatmapTracker.load_from_checkpoint(model_file)
        else:
            raise NotImplementedError
    else:
        loss_param_dict = OmegaConf.to_object(cfg.losses)
        losses_to_use = OmegaConf.to_object(cfg.model.losses_to_use)
        if cfg.model.data_type == "regression":
            model = SemiSupervisedRegressionTracker.load_from_checkpoint(
                model_file, semi_super_losses_to_use=losses_to_use, loss_params=loss_param_dict)
        elif cfg.model.data_type == "heatmap":
            model = SemiSupervisedHeatmapTracker.load_from_checkpoint(
                model_file, semi_super_losses_to_use=losses_to_use, loss_params=loss_param_dict)
        else:
            raise NotImplementedError
    model.to(device_pt)
    model.eval()

    # set some defaults
    batch_size = 1  # don't change this, change sequence length (exposed to user) instead
    video_pipe_kwargs_defaults = {"num_threads": 2, "device_id": 0}
    for key, val in video_pipe_kwargs_defaults.items():
        if key not in video_pipe_kwargs.keys():
            video_pipe_kwargs[key] = val

    # loop over videos
    for video_file in video_files:

        print("Processing video at %s" % video_file)

        # build video loader/pipeline
        pipe = video_pipe(
            resize_dims=(cfg.data.image_resize_dims.height, cfg.data.image_resize_dims.width),
            batch_size=batch_size, sequence_length=sequence_length, filenames=[video_file],
            random_shuffle=False, device=device_dali, name="reader", pad_sequences=True,
            **video_pipe_kwargs)

        predict_loader = LightningWrapper(
            pipe, output_map=["x"], last_batch_policy=LastBatchPolicy.FILL,
            last_batch_padded=False, auto_reset=False, reader_name="reader")

        # iterate through video
        n_frames_ = count_frames(video_file)  # total frames in video
        n_frames = 0  # total frames processed
        keypoints_np = np.zeros((n_frames_, model.num_targets))
        t_beg = time.time()
        n = -1
        with torch.no_grad():
            for n, batch in enumerate(predict_loader):
                outputs = model.forward(batch)
                if cfg.model.data_type == "heatmap":
                    pred_keypoints = model.run_subpixelmaxima(outputs).detach().cpu().numpy()
                else:
                    pred_keypoints = outputs.detach().cpu().numpy()
                n_frames_curr = pred_keypoints.shape[0]
                if n_frames + n_frames_curr > n_frames_:
                    # final sequence
                    final_batch_size = n_frames_ - n_frames
                    keypoints_np[n_frames:] = pred_keypoints[:final_batch_size]
                    n_frames_curr = final_batch_size
                else:
                    keypoints_np[n_frames:n_frames + n_frames_curr] = pred_keypoints
                n_frames += n_frames_curr
            t_end = time.time()
            if n == -1:
                print("WARNING: issue processing %s" % video_file)  # TODO: what can go wrong here?
                continue
            else:
                print("inference speed: %1.2f fr/sec" % ((n * sequence_length) / (t_end - t_beg)))

        # save csv file of predictions in DeepLabCut format
        if save_file is None:
            # create filename based on video name and model type
            video_file_name = os.path.basename(video_file).replace(".mp4", "")
            loss_str = "_".join([""] + cfg.model.losses_to_use) \
                if len(cfg.model.losses_to_use) > 0 else ""
            save_file = os.path.join(
                video_path, "%s_%s%s.csv" % (video_file_name, cfg.model.data_type, loss_str))

        num_joints = int(model.num_targets // 2)
        predictions = np.zeros((keypoints_np.shape[0], num_joints * 3))
        predictions[:, 0] = np.arange(keypoints_np.shape[0])
        # put x vals back in original pixel space
        x_resize = cfg.data.image_resize_dims.width
        x_og = cfg.data.image_orig_dims.width
        predictions[:, 0::3] = keypoints_np[:, 0::2] / x_resize * x_og
        # put y vals back in original pixel space
        y_resize = cfg.data.image_resize_dims.height
        y_og = cfg.data.image_orig_dims.height
        predictions[:, 1::3] = keypoints_np[:, 1::2] / y_resize * y_og

        xyl_labels = ["x", "y", "likelihood"]
        joint_labels = ['bp_%i' % n for n in range(model.num_targets // 2)]
        pdindex = pd.MultiIndex.from_product(
            [["%s_tracker" % cfg.model.data_type], joint_labels, xyl_labels],
            names=["scorer", "bodyparts", "coords"],
        )
        df = pd.DataFrame(predictions, columns=pdindex)
        if save_file.endswith(".csv"):
            df.to_csv(save_file)
        elif save_file.find(".h") > -1:
            df.to_hdf(save_file)
        else:
            raise NotImplementedError("Currently only .csv and .h5 files are supported")
