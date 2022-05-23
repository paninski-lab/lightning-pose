"""Example model training script."""

import hydra
from omegaconf import DictConfig, ListConfig
import os
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd

from lightning_pose.callbacks.callbacks import AnnealWeight
from lightning_pose.utils.io import return_absolute_data_paths
from lightning_pose.utils.predictions import predict_dataset, make_pred_arr_undo_resize, get_csv_file, get_keypoint_names
from lightning_pose.utils.scripts import (
    get_data_module,
    get_dataset,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)
from lightning_pose.data.utils import count_frames
# attempting to predict a video here
from lightning_pose.data.dali import video_pipe, LightningWrapper, ContextLightningWrapper
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from typing import List, Tuple
from torchtyping import TensorType, patch_typeguard
from lightning_pose.utils.predictions_new import PredictionHandler


_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    """Main fitting function, accessed from command line."""

    print("Our Hydra config file:")
    pretty_print(cfg)

    # path handling for toy data
    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg.data)

    # ----------------------------------------------------------------------------------
    # Set up data/model objects
    # ----------------------------------------------------------------------------------

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg)

    # dataset
    dataset = get_dataset(cfg=cfg, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg, dataset=dataset, video_dir=video_dir)

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg, data_module=data_module)

    # model
    model = get_model(cfg=cfg, data_module=data_module, loss_factories=loss_factories)
    # ----------------------------------------------------------------------------------
    # Set up and run training
    # ----------------------------------------------------------------------------------

    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg.model.model_name)
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_supervised_loss",
        patience=cfg.training.early_stop_patience,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor="val_supervised_loss"
    )
    transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
        unfreeze_backbone_at_epoch=cfg.training.unfreezing_epoch,
        lambda_func=lambda epoch: 1.5,
        backbone_initial_ratio_lr=0.1,
        should_align=True,
        train_bn=True,
    )
    anneal_weight_callback = AnnealWeight(**cfg.callbacks.anneal_weight)
    # TODO: add wandb?
    # determine gpu setup
    if _TORCH_DEVICE == "cpu":
        gpus = 0
    elif isinstance(cfg.training.gpu_id, list):
        gpus = cfg.training.gpu_id
    elif isinstance(cfg.training.gpu_id, ListConfig):
        gpus = list(cfg.training.gpu_id)
    elif isinstance(cfg.training.gpu_id, int):
        gpus = [cfg.training.gpu_id]
    else:
        raise NotImplementedError(
            "training.gpu_id must be list or int, not {}".format(
                type(cfg.training.gpu_id)
            )
        )
    trainer = pl.Trainer(  # TODO: be careful with devices when scaling to multiple gpus
        gpus=gpus,
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=[
            early_stopping,
            lr_monitor,
            ckpt_callback,
            transfer_unfreeze_callback,
            anneal_weight_callback,
        ],
        logger=logger,
        limit_train_batches=cfg.training.limit_train_batches,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        multiple_trainloader_mode=cfg.training.multiple_trainloader_mode,
        profiler=cfg.training.profiler,
    )
    trainer.fit(model=model, datamodule=data_module)

    # ----------------------------------------------------------------------------------
    # Post-training cleanup
    # ----------------------------------------------------------------------------------

    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))
    model_ckpt = trainer.checkpoint_callback.best_model_path
    model_ckpt_abs = os.path.abspath(model_ckpt)
    print("Best model path: {}".format(model_ckpt_abs))
    if not os.path.isfile(model_ckpt_abs):
        raise FileNotFoundError(
            "Cannot find model checkpoint. Have you trained for too few epochs?"
        )
    # export predictions on train/val/test data to a csv saved in model directory
    if cfg.training.get("save_heatmaps", True):
        heatmap_file = os.path.join(hydra_output_directory, "heatmaps.h5")
    else:
        heatmap_file = None
    predict_dataset(
        cfg=cfg,
        data_module=data_module,
        ckpt_file=model_ckpt,
        preds_file=os.path.join(hydra_output_directory, "predictions.csv"),
        heatmap_file=heatmap_file,
    )

    # TODO: generate a video
    # TODO: evaluate the network on everything in the video_dir, and make videos.
    # build a video loader and predict dataset
    # build video loader/pipeline. this is identical for context/non-context, given the right batch sizes and sequence lengths.
    # batch_size = 1
    # sequence_length = cfg.eval.dali_parameters.sequence_length
    # step = 1
    # video_file = os.path.join(video_dir, "test.mp4") # just for now .

    # pipe = video_pipe(
    #     filenames=[video_file],
    #     resize_dims=(
    #         cfg.data.image_resize_dims.height,
    #         cfg.data.image_resize_dims.width,
    #     ),
    #     batch_size=batch_size,
    #     sequence_length=sequence_length,
    #     step=step,
    #     random_shuffle=False,
    #     device= "gpu", #device_dict["device_dali"],
    #     name="reader",
    #     pad_sequences=True, # TODO: be aware of that
    #     num_threads=2,
    #     device_id=0,
    #     #**video_pipe_kwargs
    # )

    # # build dataloader
    # # each data loader returns
    # do_context = False # TODO: make this a parameter from eval
    # if do_context:
    #     predict_loader = ContextLightningWrapper(
    #         pipe,
    #         output_map=["x"],
    #         last_batch_policy=LastBatchPolicy.PARTIAL,
    #         auto_reset=False,  # TODO: I removed the auto_reset, but I don't know if it's needed. Think we'll loop over the dataset without resetting.
    #         # num_batches=num_batches, # TODO: works also if num_batches = int
    #     ) # TODO: there are other args in predict_loader that we don't have here. check if it's fine.
    # else:
    #     predict_loader = LightningWrapper(
    #         pipe,
    #         output_map=["x"],
    #         last_batch_policy=LastBatchPolicy.FILL,
    #         last_batch_padded=False,
    #         auto_reset=False,
    #         reader_name="reader",
    #     )
    
    if cfg.model.do_context == False:
    
        # TODO: all of this is just for testing. should ideally go to setup of predict_dataloader.
        # should be called with video arguments, like video_dir. 
        # this allows flexibility to load datamodule and add new vids to it. 
        # need to make sure this all depends on context condition. 
        filenames = ["/home/jovyan/dali-seq-testing/test_vid_with_fr.mp4"]
        resize_dims = [256, 256]
        sequence_length = 16
        batch_size = 1
        step = 16 # to proceed to frame 16 after reading frames 0-15
        seed = 123456
        num_threads = 4
        device_id = 0

        from lightning_pose.data.utils import count_frames
        frame_count = count_frames(filenames[0])
        num_batches_simple = int(np.ceil(frame_count / sequence_length)) # what matt had before. so we can add the num_batches and predict properly. 

        pipe = video_pipe(
                resize_dims=resize_dims,
                batch_size=batch_size,
                sequence_length=sequence_length,
                step=step,
                filenames=filenames,
                random_shuffle=False,
                device="gpu",
                name="reader",
                pad_sequences=True,
                num_threads=num_threads,
                device_id=device_id,
            )

        predict_loader = LightningWrapper(
            pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.FILL,
            last_batch_padded=False,
            auto_reset=False,
            reader_name="reader",
            num_batches = num_batches_simple, # added for testing, so we can have it in predict loader.
        )
        
        # TODO: consider adding more vids and covering that case. or loop over one vid at a time. 
        # now do the prediction. this treats a single vid for now.
        preds = trainer.predict(model=model, ckpt_path=model_ckpt_abs, dataloaders=predict_loader, return_predictions=True)
        
        # initialize prediction handler class, can process multiple vids with a shared cfg and data_module
        pred_handler = PredictionHandler(cfg=cfg, data_module=data_module)
        
        # call this instance on a single vid's preds
        # TODO: potentially loop over files in a directory here
        preds_df = pred_handler(video_file=filenames[0], preds=preds)
        
        # save the predictions to a csv
        # e.g.,: '/home/jovyan/dali-seq-testing/test_vid_with_fr.mp4' -> 'test_vid_with_fr'
        base_vid_name_for_save = os.path.basename(filenames[0]).split('.')[0]
        preds_df.to_csv(os.path.join(hydra_output_directory, "preds_{}.csv".format(base_vid_name_for_save)))
    
    else: # we do context 
        filenames = ["/home/jovyan/dali-seq-testing/test_vid_with_fr.mp4"]
        resize_dims = [256, 256]
        sequence_length = 5 # hard coded for context
        batch_size = 4
        step = 1 
        seed = 123456
        num_threads = 4
        device_id = 0

        from lightning_pose.data.utils import count_frames
        frame_count = count_frames(filenames[0])
        # assuming step=1
        num_batches = int(np.ceil(frame_count / batch_size))

        pipe = video_pipe(
                resize_dims=resize_dims,
                batch_size=batch_size,
                sequence_length=sequence_length,
                step=step,
                filenames=filenames,
                random_shuffle=False,
                device="gpu",
                name="reader",
                pad_sequences=True,
                num_threads=num_threads,
                device_id=device_id,
                pad_last_batch=True,
            )

        predict_loader = ContextLightningWrapper(
            pipe,
            output_map=["x"],
            last_batch_policy=LastBatchPolicy.PARTIAL, # was fill
            last_batch_padded=False, # could work without it too.
            auto_reset=False,
            reader_name="reader",
            num_batches = num_batches, # this is necessary to make the dataloader work with this policy and configs. this number should be right.
        )

        preds = trainer.predict(model=model, ckpt_path=model_ckpt_abs, dataloaders=predict_loader, return_predictions=True)
        
        # TODO: needs to be handled.

        
        num_frames = [pred[0].shape[0] for pred in preds]
        print("num_frames: {}".format(np.array(num_frames).sum()))
        total_num_frames = np.array(num_frames).sum()
        assert(total_num_frames == frame_count)

        # initialize prediction handler class, can process multiple vids with a shared cfg and data_module
        pred_handler = PredictionHandler(cfg=cfg, data_module=data_module)
        
        # call this instance on a single vid's preds
        preds_df = pred_handler(video_file=filenames[0], preds=preds)
        
        # save the predictions to a csv
        # e.g.,: '/home/jovyan/dali-seq-testing/test_vid_with_fr.mp4' -> 'test_vid_with_fr'
        base_vid_name_for_save = os.path.basename(filenames[0]).split('.')[0]
        preds_df.to_csv(os.path.join(hydra_output_directory, "preds_{}.csv".format(base_vid_name_for_save)))
        a= 6


def pretty_print(cfg):

    for key, val in cfg.items():
        if key == "eval":
            continue
        print("--------------------")
        print("%s parameters" % key)
        print("--------------------")
        for k, v in val.items():
            print("{}: {}".format(k, v))
        print()
    print("\n\n")


if __name__ == "__main__":
    train()
