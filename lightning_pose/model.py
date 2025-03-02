from __future__ import annotations

import copy
from pathlib import Path
from typing import TypedDict

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
from lightning_pose.utils.scripts import (
    compute_metrics_single,
    get_data_module,
    get_dataset,
    get_imgaug_transform,
)

__all__ = ["Model"]


class Model:
    model_dir: Path
    """Directory the model is stored in."""

    config: ModelConfig
    """The model configuration stored as a `ModelConfig` object.
    `ModelConfig` wraps the `omegaconf.DictConfig` and provides util functions
    over it.
    """

    model: ALLOWED_MODELS | None = None

    # Just a constant we can use as a default value for kwargs,
    # to differentiate between user omitting a kwarg, vs explicitly passing None.
    UNSPECIFIED = "unspecified"

    @staticmethod
    def from_dir(model_dir: str | Path):
        """Create a `Model` instance for a model stored at `model_dir`."""
        return Model.from_dir2(model_dir)

    @staticmethod
    def from_dir2(model_dir: str | Path, hydra_overrides: list[str] = None):
        """Internal version of from_dir that supports hydra_overrides. Not sure whether to
        promote this to public API yet."""

        model_dir = Path(model_dir).absolute()

        if hydra_overrides is not None:
            import hydra

            with hydra.initialize_config_dir(
                version_base="1.1", config_dir=str(model_dir)
            ):
                cfg = hydra.compose(config_name="config", overrides=hydra_overrides)
                config = ModelConfig(cfg)
        else:
            config = ModelConfig.from_yaml_file(model_dir / "config.yaml")

        return Model(model_dir, config)

    def __init__(self, model_dir: str | Path, config: ModelConfig):
        self.model_dir = Path(model_dir).absolute()
        self.config = config

    @property
    def cfg(self) -> DictConfig:
        """The model configuration as an `omegaconf.DictConfig`."""
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

    def image_preds_dir(self) -> Path:
        return self.model_dir / "image_preds"

    def video_preds_dir(self) -> Path:
        return self.model_dir / "video_preds"

    def labeled_videos_dir(self) -> Path:
        return self.model_dir / "video_preds" / "labeled_videos"

    def cropped_data_dir(self):
        return self.model_dir / "cropped_images"

    def cropped_videos_dir(self):
        return self.model_dir / "cropped_videos"

    def cropped_csv_file_path(self, csv_file_path: str | Path):
        csv_file_path = Path(csv_file_path)
        return (
            self.model_dir
            / "image_preds"
            / csv_file_path.name
            / ("cropped_" + csv_file_path.name)
        )

    class PredictionResult(TypedDict):
        predictions: pd.DataFrame
        metrics: pd.DataFrame

    def predict_on_label_csv(
        self,
        csv_file: str | Path,
        data_dir: str | Path | None = None,
        compute_metrics: bool = True,
        generate_labeled_images: bool = False,
        output_dir: str | Path | None = UNSPECIFIED,
        add_train_val_test_set: bool = False,
    ) -> PredictionResult:
        """Predicts on a labeled dataset and computes error/loss metrics if applicable.

        Args:
            csv_file (str | Path): Path to the CSV file of images, keypoint locations.
            data_dir (str | Path, optional): Root path for relative paths in the CSV file. Defaults to the
                data_dir originally used when training.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_images (bool, optional): Whether to save labeled images. Defaults to False.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`. If set to None, outputs are not saved.
            add_train_val_test_set (bool): When predicting on training dataset, set to true to add the `set`
                column to the prediction output.
        Returns:
            PredictionResult: A PredictionResult object containing the predictions and metrics.
        """
        self._load()
        # Convert this to absolute, because if relative, downstream will
        # assume incorrectly assume its relative to the data_dir.
        csv_file = Path(csv_file).absolute()
        if data_dir is None:
            data_dir = self.config.cfg.data.data_dir

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.image_preds_dir() / csv_file.name

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if generate_labeled_images:
            raise NotImplementedError()

        # Point predict_dataset to the csv_file and data_dir.
        # HACK: For true multi-view model, trick predict_dataset and compute_metrics
        # into thinking this is a single-view model.
        cfg_overrides = {
            "data": {
                "data_dir": str(data_dir),
                "csv_file": str(csv_file),
            }
        }

        # Avoid annotating set=train/val/test for CSV file other than the training CSV file.
        if not add_train_val_test_set:
            cfg_overrides.update({"train_prob": 1, "val_prob": 0, "train_frames": 1})

        cfg_pred = OmegaConf.merge(self.cfg, cfg_overrides)

        # HACK: For true multi-view model, trick predict_dataset and compute_metrics
        # into thinking this is a single-view model.
        if self.config.is_multi_view():
            del cfg_pred.data.view_names
            # HACK: If we don't delete mirrored_column_matches, downstream
            # interprets this as a mirrored multiview model, and compute_metrics fails.
            del cfg_pred.data.mirrored_column_matches

        data_module_pred = _build_datamodule_pred(cfg_pred)

        preds_file_path = output_dir / "predictions.csv"
        preds_file = str(preds_file_path)

        df = predict_dataset(
            cfg_pred, data_module_pred, model=self.model, preds_file=preds_file
        )

        if compute_metrics:
            compute_metrics_single(
                cfg=cfg_pred,
                labels_file=str(csv_file),
                preds_file=preds_file,
                data_module=data_module_pred,
            )

        return self.PredictionResult(predictions=df)

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path | None = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
    ) -> PredictionResult:
        """Predicts on a video file and computes unsupervised loss metrics if applicable.

        Args:
            video_file (str | Path): Path to the video file.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_video (bool, optional): Whether to save a labeled video. Defaults to False.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`. If set to None, outputs are not saved.
        Returns:
            PredictionResult: A PredictionResult object containing the predictions and metrics.
        """
        self._load()
        video_file = Path(video_file)

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.video_preds_dir()

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir = Path(output_dir)
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

        if compute_metrics:
            # FIXME: Data module is only used for computing PCA metrics.
            data_module = _build_datamodule_pred(self.cfg)
            compute_metrics_single(
                cfg=self.cfg,
                labels_file=None,
                preds_file=str(prediction_csv_file),
                data_module=data_module,
            )

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

    return data_module_pred
