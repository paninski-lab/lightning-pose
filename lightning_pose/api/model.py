from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from lightning_pose.api.model_config import ModelConfig
from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.datatypes import MultiviewPredictionResult, PredictionResult
from lightning_pose.data.utils import convert_bbox_coords
from lightning_pose.models import ALLOWED_MODELS
from lightning_pose.utils import io as io_utils
from lightning_pose.utils.predictions import generate_labeled_video as generate_labeled_video_fn
from lightning_pose.utils.predictions import (
    load_model_from_checkpoint,
    predict_dataset,
    predict_video,
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
            ckpt_file = io_utils.ckpt_path_from_base_path(
                base_path=str(self.model_dir), model_name=self.cfg.model.model_name
            )
            if ckpt_file is None:
                raise FileNotFoundError(
                    "Checkpoint file not found, have you trained for enough epochs?"
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

    def predict_frame(
        self,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Single-frame inference. No file I/O, no DALI.

        Preprocessing uses cv2 (not DALI). Results will differ numerically
        from ``predict_on_video_file`` due to interpolation and normalization
        differences. Do not mix results from the two paths in quantitative
        analysis.

        For MHCRNN (context) models, pass a ``(T, H, W, 3)`` array where T
        is the temporal context length (typically 5). Passing a single frame
        to a context model raises ``ValueError`` — use
        ``predict_on_video_file`` for proper temporal inference.

        The first call triggers model loading and CUDA initialization, which
        may take several seconds. Subsequent calls are fast (~5-50ms depending
        on backbone). For latency-sensitive loops, call once on a dummy frame
        before entering the loop.

        Args:
            frame_rgb: ``(H, W, 3)`` uint8 RGB array for standard models, or
                ``(T, H, W, 3)`` uint8 RGB array for context (MHCRNN) models.
            bbox: Optional ``(x, y, w, h)`` crop region. Note: this is
                ``(x, y, width, height)``, NOT ``(x1, y1, x2, y2)``.
                If provided, crops first, then remaps keypoints back to
                original coordinates.

        Returns:
            {"keypoints": (num_kp, 2) float32 array (x, y) in original frame coords,
             "confidence": (num_kp,) float32 in [0, 1] -- likelihood/confidence
              per keypoint. For regression models, confidence is always 1.0.}

        Raises:
            ValueError: If frame_rgb has wrong shape/dtype, bbox has non-positive
                dimensions, bbox produces an empty crop, or a context model
                receives single-frame input.

        """
        self._load()

        # --- Input validation ---
        if frame_rgb.dtype != np.uint8:
            raise ValueError(
                f"frame_rgb must be uint8, got {frame_rgb.dtype}. "
                "Convert with frame.astype(np.uint8) if values are in [0, 255]."
            )

        is_context_input = frame_rgb.ndim == 4
        if is_context_input:
            if frame_rgb.shape[3] != 3:
                raise ValueError(
                    f"frame_rgb must be (T, H, W, 3), got shape {frame_rgb.shape}"
                )
        elif frame_rgb.ndim == 3:
            if frame_rgb.shape[2] != 3:
                raise ValueError(
                    f"frame_rgb must be (H, W, 3), got shape {frame_rgb.shape}"
                )
        else:
            raise ValueError(
                f"frame_rgb must be (H, W, 3) or (T, H, W, 3), "
                f"got {frame_rgb.ndim}D array with shape {frame_rgb.shape}"
            )

        if frame_rgb.size == 0:
            raise ValueError("frame_rgb is empty")

        is_context_model = self.model.do_context
        if is_context_model and not is_context_input:
            raise ValueError(
                "Context model requires frame_rgb of shape (T, H, W, 3) "
                "where T is the temporal context length (typically 5). "
                "Use predict_on_video_file for single-frame input."
            )

        # --- Crop ---
        if bbox is not None:
            bx, by, bw, bh = bbox
            if bx < 0 or by < 0:
                raise ValueError(
                    f"bbox origin must be non-negative, got x={bx}, y={by}"
                )
            if bw <= 0 or bh <= 0:
                raise ValueError(
                    f"bbox width and height must be positive, got w={bw}, h={bh}"
                )
            if is_context_input:
                crop = frame_rgb[:, by:by + bh, bx:bx + bw]
            else:
                crop = frame_rgb[by:by + bh, bx:bx + bw]
            if crop.size == 0:
                raise ValueError(
                    f"bbox (x={bx}, y={by}, w={bw}, h={bh}) produces an empty "
                    f"crop on frame of shape {frame_rgb.shape}"
                )
            # Use actual crop dims for remap -- numpy clips silently when
            # bbox extends beyond frame boundaries.
            if is_context_input:
                actual_h, actual_w = crop.shape[1], crop.shape[2]
            else:
                actual_h, actual_w = crop.shape[0], crop.shape[1]
        else:
            crop = frame_rgb

        # --- Preprocess ---
        resize_h = self.cfg.data.image_resize_dims.height
        resize_w = self.cfg.data.image_resize_dims.width
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
        std = np.array(_IMAGENET_STD, dtype=np.float32)

        def _preprocess_single(img):
            resized = cv2.resize(
                img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR,
            )
            t = resized.astype(np.float32) / 255.0
            t = (t - mean) / std
            return np.transpose(t, (2, 0, 1))  # (3, H, W)

        if is_context_input:
            frames = [_preprocess_single(crop[i]) for i in range(crop.shape[0])]
            tensor = np.stack(frames)  # (T, 3, H, W)
            tensor_t = torch.from_numpy(tensor).unsqueeze(0)  # (1, T, 3, H, W)
        else:
            tensor = _preprocess_single(crop)
            tensor_t = torch.from_numpy(tensor).unsqueeze(0)  # (1, 3, H, W)

        device = self.model.device
        tensor_t = tensor_t.to(device)

        # --- Build batch dict ---
        # Bbox in LP format: [x, y, height, width]
        if bbox is not None:
            bbox_lp = torch.tensor(
                [[bx, by, actual_h, actual_w]], dtype=torch.float32, device=device,
            )
        else:
            if is_context_input:
                fh, fw = frame_rgb.shape[1], frame_rgb.shape[2]
            else:
                fh, fw = frame_rgb.shape[0], frame_rgb.shape[1]
            bbox_lp = torch.tensor(
                [[0, 0, fh, fw]], dtype=torch.float32, device=device,
            )

        num_kp = self.model.num_keypoints
        batch_dict = {
            "images": tensor_t,
            "keypoints": torch.zeros(1, num_kp * 2, dtype=torch.float32, device=device),
            "bbox": bbox_lp,
            "idxs": torch.zeros(1, dtype=torch.long, device=device),
            "heatmaps": torch.zeros(1, num_kp, 1, 1, dtype=torch.float32, device=device),
        }

        # --- Inference via get_loss_inputs_labeled ---
        self.model.eval()
        with torch.inference_mode():
            result = self.model.get_loss_inputs_labeled(batch_dict)

        # --- Extract predictions ---
        kp_pred = result["keypoints_pred"]
        has_confidence = "confidences" in result

        if is_context_model:
            # Context model's get_loss_inputs_labeled concatenates [sf; mf] along batch dim
            n = kp_pred.shape[0] // 2
            kp_sf = kp_pred[:n].reshape(n, -1, 2)
            kp_mf = kp_pred[n:].reshape(n, -1, 2)
            conf_sf = result["confidences"][:n]
            conf_mf = result["confidences"][n:]
            # Merge: pick higher-confidence prediction per keypoint
            mf_better = conf_mf > conf_sf
            kp_sf[mf_better] = kp_mf[mf_better]
            conf_merged = conf_sf.clone()
            conf_merged[mf_better] = conf_mf[mf_better]
            kp = kp_sf[0].cpu().numpy().astype(np.float32)
            conf = conf_merged[0].cpu().numpy().astype(np.float32)
        elif has_confidence:
            # Heatmap model — keypoints already in original frame coords
            # (get_loss_inputs_labeled calls convert_bbox_coords internally)
            kp = kp_pred[0].cpu().numpy().reshape(-1, 2).astype(np.float32)
            conf = result["confidences"][0].cpu().numpy().astype(np.float32)
        else:
            # Regression model — get_loss_inputs_labeled does not call
            # convert_bbox_coords, so we apply the remap ourselves.
            kp_pred = convert_bbox_coords(batch_dict, kp_pred, in_place=False)
            kp = kp_pred[0].cpu().numpy().reshape(-1, 2).astype(np.float32)
            conf = np.ones(num_kp, dtype=np.float32)

        return {"keypoints": kp, "confidence": conf}

    def predict_on_label_csv(
        self,
        csv_file: str | Path,
        data_dir: str | Path | None = None,
        compute_metrics: bool = True,
        add_train_val_test_set: bool = False,
    ) -> PredictionResult:
        """Predicts on a labeled dataset and computes error/loss metrics if applicable.

        Args:
            csv_file (str | Path): Path to the CSV file of images, keypoint locations.
            data_dir (str | Path, optional): Root path for relative paths in the CSV file.
                Defaults to the data_dir originally used when training.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_images (bool, optional): Whether to save labeled images.
                Defaults to False.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`.
                If set to None, outputs are not saved.
            add_train_val_test_set (bool): When predicting on training dataset, set to true to add
                the `set` column to the prediction output.
        Returns:
            PredictionResult: A PredictionResult object containing the predictions and metrics.
        """
        self._load()
        # Convert this to absolute, because if relative, downstream will
        # assume its relative to the data_dir.
        csv_file = Path(csv_file).absolute()
        if data_dir is None:
            data_dir = self.config.cfg.data.data_dir

        output_dir = self.image_preds_dir() / csv_file.name
        output_dir.mkdir(parents=True, exist_ok=True)

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
            metrics = compute_metrics_single(
                cfg=cfg_pred,
                labels_file=str(csv_file),
                preds_file=preds_file,
                data_module=data_module_pred,
            )
        else:
            metrics = None

        return PredictionResult(predictions=df, metrics=metrics)

    def predict_on_label_csv_multiview(
        self,
        csv_file_per_view: list[str] | list[Path],
        bbox_file_per_view: list[str] | list[Path] | None = None,
        camera_params_file: str | Path | None = None,
        data_dir: str | Path | None = None,
        compute_metrics: bool = True,
        add_train_val_test_set: bool = False,
    ) -> MultiviewPredictionResult:
        """Version of `predict_on_label_csv` that gives models access to all views of each frame.

        Arguments:
            csv_file_per_view (list[str] | list[Path]): A list of csv files each from a different
            view of the same session. Order must match the `view_names` in the config file.

        See `predict_on_label_csv` docstring for other arguments."""
        assert self.config.is_multi_view()
        self._load()

        view_names = self.config.cfg.data.view_names
        assert len(csv_file_per_view) == len(
            view_names
        ), f"{len(csv_file_per_view)} != {len(view_names)}"

        # Convert this to absolute, because if relative, downstream will
        # assume its relative to the data_dir.
        csv_file_per_view: list[Path] = [Path(f).absolute() for f in csv_file_per_view]

        if data_dir is None:
            data_dir = self.config.cfg.data.data_dir

        # Point predict_dataset to the csv_file and data_dir.
        cfg_overrides = {
            "data": {
                "data_dir": str(data_dir),
                "csv_file": [str(p) for p in csv_file_per_view],
            }
        }
        if camera_params_file:
            cfg_overrides["data"]["camera_params_file"] = camera_params_file
        if bbox_file_per_view:
            cfg_overrides["data"]["bbox_file"] = [str(p) for p in bbox_file_per_view]
        else:
            cfg_overrides["data"]["bbox_file"] = None

        # Avoid annotating set=train/val/test for CSV file other than the training CSV file.
        if not add_train_val_test_set:
            cfg_overrides.update({"train_prob": 1, "val_prob": 0, "train_frames": 1})

        cfg_pred = OmegaConf.merge(self.cfg, cfg_overrides)

        data_module_pred = _build_datamodule_pred(cfg_pred)

        preds_files = []
        for i, view_name in enumerate(view_names):
            output_dir = self.image_preds_dir() / csv_file_per_view[i].name
            output_dir.mkdir(parents=True, exist_ok=True)
            preds_files.append(str(output_dir / "predictions.csv"))

        # Outputs dict[str, pd.DataFrame] because inputs indicate multiview.
        view_to_df_dict = predict_dataset(
            cfg_pred, data_module_pred, model=self.model, preds_file=preds_files
        )

        if compute_metrics:
            metrics = {}
            for view_name, labels_file, _preds_file in zip(
                view_names, csv_file_per_view, preds_files
            ):
                metrics[view_name] = compute_metrics_single(
                    cfg=self.cfg,
                    labels_file=str(labels_file),
                    preds_file=_preds_file,
                    data_module=data_module_pred,
                )
        else:
            metrics = None

        return MultiviewPredictionResult(predictions=view_to_df_dict, metrics=metrics)

    def predict_on_video_file(
        self,
        video_file: str | Path,
        output_dir: str | Path | None = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
        progress_file: Path | None = None,
    ) -> PredictionResult:
        """Predicts on a video file and computes unsupervised loss metrics if applicable.

        Args:
            video_file (str | Path): Path to the video file.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`.
                If set to None, outputs are not saved.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_video (bool, optional): Whether to save a labeled video.
                Defaults to False.
            progress_file (Path, optional): Path to a file to save progress information for the App.
                Defaults to None.

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

        df: pd.DataFrame = predict_video(
            video_file=str(video_file),
            model=self,
            output_pred_file=str(prediction_csv_file),
            progress_file=progress_file,
        )
        if generate_labeled_video:
            labeled_mp4_file = str(self.labeled_videos_dir() / f"{video_file.stem}_labeled.mp4")
            generate_labeled_video_fn(
                video_file=str(video_file),
                preds_df=df,
                output_mp4_file=labeled_mp4_file,
                confidence_thresh_for_vid=self.cfg.eval.confidence_thresh_for_vid,
                colormap=self.cfg.eval.get("colormap", "cool"),
            )

        if compute_metrics:
            # FIXME: Data module is only used for computing PCA metrics.
            data_module = _build_datamodule_pred(self.cfg)
            metrics = compute_metrics_single(
                cfg=self.cfg,
                labels_file=None,
                preds_file=str(prediction_csv_file),
                data_module=data_module,
            )
        else:
            metrics = None

        return PredictionResult(predictions=df, metrics=metrics)

    def predict_on_video_file_multiview(
        self,
        video_file_per_view: list[str] | list[Path],
        output_dir: str | Path | None = UNSPECIFIED,
        compute_metrics: bool = True,
        generate_labeled_video: bool = False,
        progress_file: Path | None = None,
    ) -> MultiviewPredictionResult:
        """Version of `predict_on_video_file` that accesses to multiple camera views of each frame.

        Arguments:
            video_file_per_view (list[str] | list[Path]): A list of video files each from a
                different view of the same session.
                Number of video files must match the `view_names` in the config file.
                Order of the list does not matter: video files are intelligently matched to views
                by their filename using `utils.io.collect_video_files_by_view`.
            output_dir (str | Path, optional): The directory to save outputs to.
                Defaults to `{model_dir}/image_preds/{csv_file_name}`.
                If set to None, outputs are not saved.
            compute_metrics (bool, optional): Whether to compute pixel error and loss metrics on
                predictions.
            generate_labeled_video (bool, optional): Whether to save a labeled video.
                Defaults to False.
            progress_file (Path, optional): Path to a file to save progress information for the App.

        Returns:
            MultiviewPredictionResult: object containing the predictions and metrics for each view.

        """
        assert self.config.is_multi_view()
        self._load()

        view_names = self.config.cfg.data.view_names
        assert len(video_file_per_view) == len(
            view_names
        ), f"{len(video_file_per_view)} != {len(view_names)}"

        video_file_per_view: list[Path] = [Path(f) for f in video_file_per_view]

        if output_dir == self.__class__.UNSPECIFIED:
            output_dir = self.video_preds_dir()

        elif output_dir is None:
            raise NotImplementedError("Currently we must save predictions")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Arranges video_file_per_view to be in the same order as cfg.data.view_names.
        _view_to_video_file: dict[str, Path] = io_utils.collect_video_files_by_view(
            video_file_per_view, view_names
        )
        video_file_per_view: list[Path] = [
            _view_to_video_file[view_name] for view_name in view_names
        ]

        prediction_csv_file_list = [
            str(output_dir / f"{video_file.stem}.csv")
            for video_file in video_file_per_view
        ]

        df_list: list[pd.DataFrame] = predict_video(
            video_file=list(map(str, video_file_per_view)),
            model=self,
            output_pred_file=prediction_csv_file_list,
            progress_file=progress_file,
        )
        if generate_labeled_video:
            for video_file, preds_df in zip(video_file_per_view, df_list):
                labeled_mp4_file = str(
                    self.labeled_videos_dir() / f"{video_file.stem}_labeled.mp4"
                )
                generate_labeled_video_fn(
                    video_file=str(video_file),
                    preds_df=preds_df,
                    output_mp4_file=labeled_mp4_file,
                    confidence_thresh_for_vid=self.cfg.eval.confidence_thresh_for_vid,
                    colormap=self.cfg.eval.get("colormap", "cool"),
                )

        data_module = _build_datamodule_pred(self.cfg)
        if compute_metrics:
            metrics = {}
            for view_name, preds_file in zip(view_names, prediction_csv_file_list):
                metrics[view_name] = compute_metrics_single(
                    cfg=self.cfg,
                    labels_file=None,
                    preds_file=preds_file,
                    data_module=data_module,
                )
        else:
            metrics = None

        df_dict = {view_name: df for view_name, df in zip(view_names, df_list)}

        return MultiviewPredictionResult(predictions=df_dict, metrics=metrics)


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
