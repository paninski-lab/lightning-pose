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


_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PredictionHandler:
    def __init__(self, cfg: DictConfig, data_module: pl.LightningDataModule) -> None:
        self.cfg = cfg
        self.data_module = data_module
    
    @property
    def keypoint_names(self):
        return self.data_module.dataset.keypoint_names
    
    @property
    def do_context(self):
        return self.data_module.dataset.do_context
    
    def discard_context_rows(self, df):
        # TODO: replace first and last two rows by preds 2 and -2.
        if self.do_context == False:
            pass

            

    
    @staticmethod
    def unpack_preds(preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"], TensorType["batch", "num_keypoints"]]], frame_count: int) -> Tuple[TensorType["num_frames", "two_times_num_keypoints"], TensorType["num_frames", "num_keypoints"]]:
        """ unpack list of preds coming out from pl.trainer.predict, confs tuples into tensors.
        It still returns unnecessary final rows, which should be discarded at the dataframe stage.
        This works for the output of predict_loader, suitable for batch_size=1, sequence_length=16, step=16"""
        # stack the predictions into rows.
        # loop over the batches, and stack 
        stacked_preds = torch.vstack([pred[0] for pred in preds])
        stacked_confs = torch.vstack([pred[1] for pred in preds])
        # eliminate last rows
        # this is true just for the case of e.g., batch_size=1, sequence_length=16, step=sequence_length
        # the context dataloader just doesn't include those extra frames.
        num_rows_to_discard = stacked_preds.shape[0] - frame_count
        if num_rows_to_discard > 0:
            stacked_preds = stacked_preds[:-num_rows_to_discard]
            stacked_confs = stacked_confs[:-num_rows_to_discard]
            
        return stacked_preds, stacked_confs
    
    def make_pred_arr_undo_resize(
        self,
        keypoints_np: np.array,
        confidence_np: np.array,
    ) -> np.array:
        """Resize keypoints and add confidences into one numpy array.

        Args:
            keypoints_np: shape (n_frames, n_keypoints * 2)
            confidence_np: shape (n_frames, n_keypoints)

        Returns:
            np.ndarray: cols are (bp0_x, bp0_y, bp0_likelihood, bp1_x, bp1_y, ...)

        """
        assert keypoints_np.shape[0] == confidence_np.shape[0]  # num frames in the dataset
        assert keypoints_np.shape[1] == (
            confidence_np.shape[1] * 2
        )  # we have two (x,y) coordinates and a single likelihood value

        num_joints = confidence_np.shape[-1]  # model.num_keypoints
        predictions = np.zeros((keypoints_np.shape[0], num_joints * 3))
        predictions[:, 0] = np.arange(keypoints_np.shape[0])
        # put x vals back in original pixel space
        x_resize = self.cfg.data.image_resize_dims.width
        x_og = self.cfg.data.image_orig_dims.width
        predictions[:, 0::3] = keypoints_np[:, 0::2] / x_resize * x_og
        # put y vals back in original pixel space
        y_resize = self.cfg.data.image_resize_dims.height
        y_og = self.cfg.data.image_orig_dims.height
        predictions[:, 1::3] = keypoints_np[:, 1::2] / y_resize * y_og
        predictions[:, 2::3] = confidence_np

        return predictions
    
    def make_dlc_pandas_index(self) -> pd.MultiIndex:
        xyl_labels = ["x", "y", "likelihood"]
        pdindex = pd.MultiIndex.from_product(
            [["%s_tracker" % self.cfg.model.model_type], self.keypoint_names, xyl_labels],
            names=["scorer", "bodyparts", "coords"],
        )
        return pdindex
    
    def __call__(self, video_file: str, preds: List[Tuple[TensorType["batch", "two_times_num_keypoints"], TensorType["batch", "num_keypoints"]]])-> pd.DataFrame:
        """
        Call this function to get a pandas dataframe of the predictions for a single video.
        Assuming you've already run trainer.predict(), and have a list of Tuple predictions.
        Args:
            preds: list of tuples of (predictions, confidences)
            video_file: path to video file
        Returns:
            pd.DataFrame: index is (frame, bodypart, x, y, likelihood)
        """
        frame_count = count_frames(video_file)
        stacked_preds, stacked_confs = self.unpack_preds(preds=preds, frame_count=frame_count)
        pred_arr = self.make_pred_arr_undo_resize(stacked_preds.numpy(), stacked_confs.numpy())
        pdindex = self.make_dlc_pandas_index()
        df = pd.DataFrame(pred_arr, columns=pdindex)
        return df

    

        


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
