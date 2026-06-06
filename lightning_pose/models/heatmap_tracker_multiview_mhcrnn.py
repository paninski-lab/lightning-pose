"""Multiview transformer with temporal context (Factorized Space-View-Time Transformer).

Extends :class:`HeatmapTrackerMultiviewTransformer` with a temporal stage so the model reasons
both across views and across time:

  Stage 1 (space + view): the parent's ``forward_vit`` cross-view attention, applied per frame.
  Stage 2 (time): a lightweight temporal transformer over the context frames at each
                  (view, spatial-location) token, contributing a *gated residual* to the center
                  frame's features (zero-initialized gate => identity at the start of training).

Like the single-view MHCRNN it produces a single-frame (SF) and a multi-frame (MF) heatmap for
the center frame, read out by a shared head; both are supervised in training and the
higher-confidence one is used at inference. Because the temporal contribution is a zero-gated
residual, MF == SF at initialization and the temporal branch can only help, never silently
degrade predictions before it has learned (important for out-of-distribution robustness).
"""

from typing import Any, Literal

import torch
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig
from torch import nn

from lightning_pose.data.cameras import project_3d_to_2d, project_camera_pairs_to_3d
from lightning_pose.data.datatypes import MultiviewHeatmapLabeledBatchDict, UnlabeledBatchDict
from lightning_pose.data.utils import convert_bbox_coords, convert_original_to_model_coords
from lightning_pose.losses.factory import LossFactory
from lightning_pose.models.backbones import ALLOWED_TRANSFORMER_BACKBONES
from lightning_pose.models.heatmap_tracker_multiview import HeatmapTrackerMultiviewTransformer

# to ignore imports for sphix-autoapidoc
__all__ = []


class HeatmapTrackerMultiviewTransformerMHCRNN(HeatmapTrackerMultiviewTransformer):
    """Multiview transformer with a temporal-context (MHCRNN-style) dual head."""

    def __init__(
        self,
        num_keypoints: int,
        num_views: int,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_TRANSFORMER_BACKBONES = "vits_dino",
        pretrained: bool = True,
        head: str = "heatmap_mhcrnn_multiview",
        downsample_factor: Literal[1, 2, 3] = 2,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | ListConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | ListConfig | dict | None = None,
        image_size: int = 256,
        context_length: int = 5,
        temporal_n_layers: int = 1,
        temporal_n_heads: int = 6,
        **kwargs: Any,
    ) -> None:
        """Initialize the multiview-transformer temporal-context model.

        Args:
            num_keypoints: number of body parts (per view).
            num_views: number of camera views.
            loss_factory: object to orchestrate supervised loss computation.
            backbone: transformer backbone (reused from the parent for Stage 1).
            pretrained: load pretrained backbone weights.
            head: head identifier (kept for config symmetry; the dual head is built here).
            downsample_factor: heatmap downsampling factor.
            torch_seed: weight-init seed.
            context_length: number of context frames per window.
            temporal_n_layers: depth of the Stage-2 temporal transformer.
            temporal_n_heads: number of attention heads in the temporal transformer.

        """
        # Stage 1 reuses the parent's heatmap_cnn cross-view path; build it as such.
        super().__init__(
            num_keypoints=num_keypoints,
            num_views=num_views,
            loss_factory=loss_factory,
            backbone=backbone,
            pretrained=pretrained,
            head="heatmap_cnn",
            downsample_factor=downsample_factor,
            torch_seed=torch_seed,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            image_size=image_size,
            **kwargs,
        )

        self.context_length = context_length
        dim = self.num_fc_input_features

        # Stage 2: temporal transformer over the context frames at each (view, location) token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=temporal_n_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=temporal_n_layers)
        self.temporal_embedding = nn.Parameter(torch.zeros(context_length, dim))
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

        # the multi-frame (temporally-fused) branch is a gated residual refinement of the
        # single-frame center features, read out by the *same* head (self.head). the per-channel
        # LayerScale gate starts at zero, so at initialization MF == SF exactly and the temporal
        # branch can only deviate where it earns it. this prevents the freshly-initialized temporal
        # stack from corrupting (over-confident) predictions on OOD data before it has learned
        # anything useful -- the previous independent head_mf had no such "do no harm" floor.
        self.temporal_gate = nn.Parameter(torch.zeros(dim))

        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def _run_temporal(
        self,
        tokens: Float[torch.Tensor, "rows context_length dim"],
    ) -> Float[torch.Tensor, "rows context_length dim"]:
        """Apply the temporal transformer, chunking along the independent batch axis.

        Attention only spans the temporal (context) axis, so rows along dim 0 are independent and
        can be processed in chunks with identical results. This is required because at inference
        the batch axis (B * num_views * h * w) can exceed the CUDA attention-kernel grid limit
        (~65535 rows), which otherwise raises "CUDA error: invalid configuration argument".
        """
        max_rows = 32768
        if tokens.shape[0] <= max_rows:
            return self.temporal_transformer(tokens)
        return torch.cat(
            [self.temporal_transformer(chunk) for chunk in tokens.split(max_rows, dim=0)],
            dim=0,
        )

    def forward(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
    ) -> tuple[
        Float[torch.Tensor, "batch num_keypoints_x_views heatmap_height heatmap_width"],
        Float[torch.Tensor, "batch num_keypoints_x_views heatmap_height heatmap_width"],
    ]:
        """Forward pass returning single-frame and multi-frame heatmaps for the center frame.

        Expects a labeled multiview context batch of shape ``(B, V, T, C, H, W)``.
        """
        images = batch_dict["images"] if "images" in batch_dict.keys() else batch_dict["frames"]
        batch, views, frames, channels, h_img, w_img = images.shape

        # ---- Stage 1: per-frame cross-view attention (reuse parent forward_vit) ----
        # order (batch, frame, view) so forward_vit sees batch'=B*T, views=V
        imgs = images.permute(0, 2, 1, 3, 4, 5).reshape(-1, channels, h_img, w_img)
        feats = self.forward_vit(imgs)  # (B*T*V, dim, h, w)
        dim, h, w = feats.shape[1], feats.shape[2], feats.shape[3]
        feats = feats.reshape(batch, frames, views, dim, h, w)

        center = frames // 2

        # single-frame branch: center frame view-fused features
        sf_feats = feats[:, center].reshape(batch * views, dim, h, w)
        heatmaps_sf = self.head(sf_feats)

        # ---- Stage 2: temporal attention per (view, spatial location) ----
        # (B, T, V, dim, h, w) -> (B*V*h*w, T, dim)
        tokens_in = feats.permute(0, 2, 4, 5, 1, 3).reshape(batch * views * h * w, frames, dim)
        tokens_out = self._run_temporal(tokens_in + self.temporal_embedding[:frames])
        # temporal residual the transformer added to the center frame -> (B*V, dim, h, w)
        delta = (
            (tokens_out[:, center] - tokens_in[:, center])
            .reshape(batch, views, h, w, dim)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch * views, dim, h, w)
        )
        # gated residual: MF == SF at init (gate=0), reuse the single-frame head as a shared readout
        mf_feats = sf_feats + self.temporal_gate.view(1, -1, 1, 1) * delta
        heatmaps_mf = self.head(mf_feats)

        # group views per example: (B, V*K, Hm, Wm)
        heatmaps_sf = heatmaps_sf.reshape(batch, -1, heatmaps_sf.shape[-2], heatmaps_sf.shape[-1])
        heatmaps_mf = heatmaps_mf.reshape(batch, -1, heatmaps_mf.shape[-2], heatmaps_mf.shape[-1])
        return heatmaps_sf, heatmaps_mf

    def _triangulate_and_reproject(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
        pred_keypoints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Triangulate predicted 2D keypoints to 3D and reproject back to 2D model coords.

        Mirrors the parent multiview transformer so the supervised 3D / reprojection losses work
        with the temporal-context model. ``pred_keypoints`` are in original (un-cropped) image
        coordinates; the reprojection is returned in model-input coordinates (or ``None`` when no
        reprojection loss is active).
        """
        num_views = batch_dict["images"].shape[1]
        num_keypoints = pred_keypoints.shape[1] // 2 // num_views
        keypoints_pred_3d = project_camera_pairs_to_3d(
            points=pred_keypoints.reshape((-1, num_views, num_keypoints, 2)),
            intrinsics=batch_dict["intrinsic_matrix"].float(),
            extrinsics=batch_dict["extrinsic_matrix"].float(),
            dist=batch_dict["distortions"].float(),
        )
        keypoints_pred_2d_reprojected = None
        if (
            self.loss_factory is not None
            and "supervised_reprojection_heatmap_mse" in self.loss_factory.loss_instance_dict
        ):
            reprojected_original = project_3d_to_2d(
                points_3d=torch.mean(keypoints_pred_3d, dim=1),
                intrinsics=batch_dict["intrinsic_matrix"].float(),
                extrinsics=batch_dict["extrinsic_matrix"].float(),
                dist=batch_dict["distortions"].float(),
            )
            keypoints_pred_2d_reprojected = convert_original_to_model_coords(
                batch_dict=batch_dict,
                original_keypoints=reprojected_original,
            ).reshape(-1, num_views * num_keypoints, 2)
        return keypoints_pred_3d, keypoints_pred_2d_reprojected

    def get_loss_inputs_labeled(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict,
    ) -> dict:
        """Return predicted SF+MF heatmaps and keypoints (both supervised), as MHCRNN does.

        When camera calibration is available, the SF and MF predictions are also triangulated to
        3D and reprojected, so the supervised 3D / reprojection losses apply to both heads.
        """
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict)
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)
        target_keypoints = convert_bbox_coords(batch_dict, batch_dict["keypoints"])
        pred_keypoints_sf = convert_bbox_coords(batch_dict, pred_keypoints_sf)
        pred_keypoints_mf = convert_bbox_coords(batch_dict, pred_keypoints_mf)

        loss_inputs = {
            "heatmaps_targ": torch.cat([batch_dict["heatmaps"], batch_dict["heatmaps"]], dim=0),
            "heatmaps_pred": torch.cat([pred_heatmaps_sf, pred_heatmaps_mf], dim=0),
            "keypoints_targ": torch.cat([target_keypoints, target_keypoints], dim=0),
            "keypoints_pred": torch.cat([pred_keypoints_sf, pred_keypoints_mf], dim=0),
            "confidences": torch.cat([confidence_sf, confidence_mf], dim=0),
        }

        # triangulate + reproject both heads when calibration is provided (3D / reprojection loss)
        if "keypoints_3d" in batch_dict and batch_dict["keypoints_3d"].shape[-1] == 3:
            try:
                kp3d_sf, reproj_sf = self._triangulate_and_reproject(batch_dict, pred_keypoints_sf)
                kp3d_mf, reproj_mf = self._triangulate_and_reproject(batch_dict, pred_keypoints_mf)
                loss_inputs["keypoints_targ_3d"] = torch.cat(
                    [batch_dict["keypoints_3d"], batch_dict["keypoints_3d"]], dim=0,
                )
                loss_inputs["keypoints_pred_3d"] = torch.cat([kp3d_sf, kp3d_mf], dim=0)
                loss_inputs["keypoints_pred_2d_reprojected"] = (
                    torch.cat([reproj_sf, reproj_mf], dim=0) if reproj_sf is not None else None
                )
            except Exception as e:
                print(f"Error in 3D projection: {e}")
                loss_inputs["keypoints_targ_3d"] = None
                loss_inputs["keypoints_pred_3d"] = None
                loss_inputs["keypoints_pred_2d_reprojected"] = None

        return loss_inputs

    def predict_step(
        self,
        batch_dict: MultiviewHeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict keypoints, choosing the higher-confidence of SF/MF per keypoint (MHCRNN-style)."""
        pred_heatmaps_sf, pred_heatmaps_mf = self.forward(batch_dict)
        pred_keypoints_sf, confidence_sf = self.head.run_subpixelmaxima(pred_heatmaps_sf)
        pred_keypoints_mf, confidence_mf = self.head.run_subpixelmaxima(pred_heatmaps_mf)

        pred_keypoints_sf = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1, 2)
        pred_keypoints_mf = pred_keypoints_mf.reshape(pred_keypoints_mf.shape[0], -1, 2)
        mf_conf_gt = torch.gt(confidence_mf, confidence_sf)
        pred_keypoints_sf[mf_conf_gt] = pred_keypoints_mf[mf_conf_gt]
        pred_keypoints = pred_keypoints_sf.reshape(pred_keypoints_sf.shape[0], -1)
        confidence_sf[mf_conf_gt] = confidence_mf[mf_conf_gt]

        pred_keypoints = convert_bbox_coords(batch_dict, pred_keypoints)
        if return_heatmaps:
            pred_heatmaps_sf[mf_conf_gt] = pred_heatmaps_mf[mf_conf_gt]
            return pred_keypoints, confidence_sf, pred_heatmaps_sf
        return pred_keypoints, confidence_sf

    def get_parameters(self) -> list[dict]:
        """Optimizer param groups: backbone (frozen lr), shared head, view + temporal modules."""
        return [
            {"params": self.backbone.parameters(), "name": "backbone", "lr": 0.0},
            {"params": self.head.parameters(), "name": "head"},
            {"params": self.temporal_transformer.parameters(), "name": "temporal_transformer"},
            {
                "params": [self.view_embeddings, self.temporal_embedding, self.temporal_gate],
                "name": "embeddings",
            },
        ]
