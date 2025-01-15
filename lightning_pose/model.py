from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from lightning_pose.model_config import ModelConfig
from lightning_pose.models import ALLOWED_MODELS
from lightning_pose.utils.io import ckpt_path_from_base_path
from lightning_pose.utils.predictions import (
    export_predictions_and_labeled_video,
    load_model_from_checkpoint,
    predict_dataset,
)

# Import as different name to avoid naming conflict with the kwarg `compute_metrics`.
from lightning_pose.utils.scripts import compute_metrics as compute_metrics_fn
from lightning_pose.utils.scripts import (
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)

__all__ = ["Model"]


class Model:
    model_dir: Path
    config: ModelConfig
    model: Optional[ALLOWED_MODELS] = None

    @staticmethod
    def from_dir(model_dir: str | Path):
        model_dir = Path(model_dir)
        config = ModelConfig.from_yaml_file(model_dir / "config.yaml")
        return Model(model_dir, config)

    def __init__(self, model_dir: str | Path, config: ModelConfig):
        self.model_dir = Path(model_dir)
        self.config = config

    @property
    def cfg(self):
        return self.config.cfg

    def _load(self):
        if self.model is None:
            ckpt_file = ckpt_path_from_base_path(
                base_path=str(self.model_dir), model_name=self.cfg.model.model_name
            )
            self.model = load_model_from_checkpoint(
                cfg=self.cfg,
                ckpt_file=ckpt_file,
                eval=True,
                skip_data_module=True,
            )

    def image_preds_dir(self):
        return self.model_dir / "image_preds"

    def video_preds_dir(self):
        return self.model_dir / "video_preds"

    def labeled_videos_dir(self):
        return self.model_dir / "video_preds" / "labeled_videos"

    UNSPECIFIED = "unspecified"

    class PredictionResult(TypedDict):
        predictions: pd.DataFrame
        metrics: pd.DataFrame

    def predict_on_label_csv(
        self,
        csv_file: str | Path,
        data_dir: Optional[str | Path] = None,
        compute_metrics: bool = True,
        generate_labeled_images: bool = False,
        output_dir: Optional[str | Path] = UNSPECIFIED,
    ) -> PredictionResult:
        """Predicts on a labeled dataset and computes error/loss metrics if applicable.

        Args:
            csv_file (str | Path): Path to the CSV file of images, keypoint locations.
            data_dir (str | Path, optional): Root path for relative paths in the CSV file. Defaults to the
                parent directory of the CSV file.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_images (bool, optional): Whether to save labeled images. Defaults to False.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`. If set to None, outputs are not saved.
        Returns:
            PredictionResult: A PredictionResult object containing the predictions
                and metrics.
        """
        return self.predict_on_label_csv_internal(
            csv_file=csv_file,
            data_dir=data_dir,
            compute_metrics=compute_metrics,
            generate_labeled_images=generate_labeled_images,
            output_dir=output_dir,
            output_filename_stem="predictions",
            add_train_val_test_set=False,
        )

    def predict_on_label_csv_internal(
        self,
        csv_file: str | Path,
        data_dir: Optional[str | Path] = None,
        compute_metrics: bool = True,
        generate_labeled_images: bool = False,
        output_dir: Optional[str | Path] = UNSPECIFIED,
        output_filename_stem: str = "predictions",
        add_train_val_test_set: bool = False,
    ) -> PredictionResult:
        """
        See predict_on_label_csv for the rest of the arguments. The following are the
        arguments specific to the internal function.
        Args:
            output_filename_stem (str): The stem of the output filename. Defaults to 'predictions'.
                Used to generate predictions_new for OOD, and predictions_{view_name} for multi-view, in the
                model_dir.
            add_train_val_test_set (bool): When predicting on training dataset, set to true to add the `set`
                column to the prediction output.
        """

        self._load()
        csv_file = Path(csv_file)
        if data_dir is None:
            data_dir = csv_file.parent

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.image_preds_dir() / csv_file.name

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir.mkdir(parents=True, exist_ok=True)

        if generate_labeled_images:
            raise NotImplementedError()

        # Point predict_dataset to the csv_file and data_dir.
        cfg_overrides = {
            "data": {
                "data_dir": str(data_dir),
                "csv_file": str(csv_file),
            }
        }

        # Avoid annotating set=train/val/test for CSV file other than the training CSV file.
        if not add_train_val_test_set:
            cfg_overrides = {"train_prob": 1, "val_prob": 0, "train_frames": 1}

        cfg_pred = OmegaConf.merge(self.cfg, cfg_overrides)
        # HACK: For true multi-view model, trick predict_dataset and compute_metrics
        # into thinking this is a single-view model.
        if self.config.is_multi_view():
            del cfg_pred.data.view_names
            # HACK: If we don't delete mirrored_column_matches, downstream
            # interprets this as a mirrored multiview model, and compute_metrics fails.
            del cfg_pred.data.mirrored_column_matches

        data_module_pred = _build_datamodule_pred(cfg_pred)

        preds_file_path = output_dir / (output_filename_stem + ".csv")
        preds_file = str(preds_file_path)

        df = predict_dataset(
            cfg_pred, data_module_pred, model=self.model, preds_file=preds_file
        )

        if compute_metrics:
            # HACK: True multi-view model treated as single-view model, so preds_file is
            # a string, not a list per-view. This means we can't yet compute pca_multiview.
            compute_metrics_fn(
                cfg=cfg_pred,
                preds_file=preds_file,
                data_module=data_module_pred,
            )

        # TODO: Generate detector outputs.

        return self.PredictionResult(predictions=df)

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: Optional[str | Path] = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
    ) -> PredictionResult:
        self._load()
        video_file = Path(video_file)

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.video_preds_dir()

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_csv_file = output_dir / f"{video_file.stem}.csv"

        labeled_mp4_file = None
        if generate_labeled_video:
            labeled_mp4_file = str(
                self.labeled_videos_dir() / f"{video_file.stem}_labeled.mp4"
            )

        if self.config.cfg.eval.get("predict_vids_after_training_save_heatmaps", False):
            raise NotImplementedError(
                "Implement this after cleaning up _predict_frames: "
                "Set a flag on the model to return heatmaps. "
                "Use trainer.predict instead of side-stepping it."
            )
        df = export_predictions_and_labeled_video(
            video_file=str(video_file),
            cfg=self.config.cfg,
            prediction_csv_file=str(prediction_csv_file),
            labeled_mp4_file=labeled_mp4_file,
            model=self.model,
        )

        # FIXME: This is only used for computing PCA metrics.
        data_module = _build_datamodule_pred(self.cfg)
        if compute_metrics:
            compute_metrics_fn(self.cfg, str(prediction_csv_file), data_module)

        return self.PredictionResult(predictions=df)


def _build_datamodule_pred(cfg: DictConfig):
    cfg_pred = copy.deepcopy(cfg)
    cfg_pred.training.imgaug = "default"
    imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
    dataset_pred = get_dataset(
        cfg=cfg_pred,
        data_dir=cfg_pred.data.data_dir,
        imgaug_transform=imgaug_transform_pred,
    )
    data_module_pred = get_data_module(
        cfg=cfg_pred, dataset=dataset_pred, video_dir=cfg_pred.data.video_dir
    )
    data_module_pred.setup()

    return data_module_pred
