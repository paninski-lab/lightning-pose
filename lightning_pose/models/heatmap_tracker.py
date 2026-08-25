"""Models that produce heatmaps of keypoints from images."""

from typing import Any, Literal

import torch
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig

from lightning_pose.data.bboxes import model_to_frame_batch
from lightning_pose.data.datatypes import (
    HeatmapLabeledBatchDict,
    MultiviewHeatmapLabeledBatchDict,
    MultiviewUnlabeledBatchDict,
    UnlabeledBatchDict,
)
from lightning_pose.data.utils import undo_affine_transform_batch
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import RegressionRMSELoss
from lightning_pose.models.backbones import ALLOWED_BACKBONES
from lightning_pose.models.base import (
    BaseSupervisedTracker,
    SemiSupervisedTrackerMixin,
)
from lightning_pose.models.heads import HeatmapHead

# to ignore imports for sphinx-autoapidoc
__all__: list[str] = []


class HeatmapTracker(BaseSupervisedTracker):
    """Base model that produces heatmaps of keypoints from images."""

    def __init__(
        self,
        num_keypoints: int,
        num_targets: int | None = None,
        loss_factory: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | ListConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | ListConfig | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a heatmap-based pose estimation model with conv or transformer backbone.

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate loss computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to save memory; subpixel
                operations are performed for increased precision
            pretrained: True to load pretrained imagenet weights
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """

        # for reproducible weight initialization
        self.torch_seed = torch_seed
        torch.manual_seed(torch_seed)

        super().__init__(
            backbone=backbone,
            pretrained=pretrained,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )

        self.num_keypoints = num_keypoints
        if num_targets is None:
            self.num_targets = num_keypoints * 2
        else:
            self.num_targets = num_targets
        self.downsample_factor = downsample_factor

        self.head = HeatmapHead(
            backbone_arch=backbone,
            in_channels=self.num_fc_input_features,
            out_channels=self.num_keypoints,
            downsample_factor=self.downsample_factor,
        )

        self.loss_factory = loss_factory

        # use this to log auxiliary information: pixel_error on labeled data
        self.rmse_loss = RegressionRMSELoss()

        # necessary so we don't have to pass in model arguments when loading
        # also, "loss_factory" and "loss_factory_unsupervised" cannot be pickled
        # (loss_factory_unsupervised might come from SemiSupervisedHeatmapTracker.__super__().
        # otherwise it's ignored, important so that it doesn't try to pickle the dali loaders)
        self.save_hyperparameters(ignore=["loss_factory", "loss_factory_unsupervised"])

    def forward(
        self,
        images: (
            Float[torch.Tensor, "batch channels image_height image_width"]
            | Float[torch.Tensor, "batch views channels image_height image_width"]
        ),
    ) -> Float[torch.Tensor, "num_valid_outputs num_keypoints heatmap_height heatmap_width"]:
        """Forward pass through the network."""
        # we get one representation for each desired output.
        shape = images.shape

        # if len(shape) > 4 we assume we have multiple views and need to combine images across
        # batch/views before passing to network, then we reshape
        if len(shape) > 4:
            images = images.reshape(-1, shape[-3], shape[-2], shape[-1])
            # images = [views * batch, channels, image_height, image_width]
            representations = self.get_representations(images)
            # representations = [views * batch, num_features, rep_height, rep_width]
            heatmaps = self.head(representations)
            # heatmaps = [views * batch, num_keypoints, heatmap_height, heatmap_width]
            heatmaps = heatmaps.reshape(shape[0], -1, heatmaps.shape[-2], heatmaps.shape[-1])
            # heatmaps = [batch, num_keypoints * views, heatmap_height, heatmap_width]
        else:
            representations = self.get_representations(images)
            heatmaps = self.head(representations)

        return heatmaps

    def get_loss_inputs_labeled(
        self,
        batch_dict: HeatmapLabeledBatchDict | MultiviewHeatmapLabeledBatchDict
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        predicted_heatmaps = self.forward(batch_dict["images"])
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        target_keypoints = model_to_frame_batch(batch_dict, batch_dict["keypoints"])
        return {
            "heatmaps_targ": batch_dict["heatmaps"],
            "heatmaps_pred": predicted_heatmaps,
            "keypoints_targ": target_keypoints,
            "keypoints_pred": predicted_keypoints,
            "confidences": confidence,
        }

    def predict_step(
        self,
        batch_dict: (
            HeatmapLabeledBatchDict
            | MultiviewHeatmapLabeledBatchDict
            | UnlabeledBatchDict
            | MultiviewUnlabeledBatchDict
        ),
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Predict heatmaps and keypoints for a batch of video frames.

        Assuming a DALI video loader is passed in
        > trainer = Trainer(devices=8, accelerator="gpu")
        > predictions = trainer.predict(model, data_loader)

        """
        if "images" in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            # labeled image dataloaders
            images = batch_dict["images"]  # type: ignore[typeddict-item]
        else:
            # unlabeled dali video dataloaders
            images = batch_dict["frames"]  # type: ignore[typeddict-item]
        # images -> heatmaps
        predicted_heatmaps = self.forward(images)
        # heatmaps -> keypoints
        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        # bounding box coords -> original image coords
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        if return_heatmaps:
            return predicted_keypoints, confidence, predicted_heatmaps
        else:
            return predicted_keypoints, confidence

    def get_parameters(self) -> list[dict]:
        """Return per-parameter-group optimizer configuration for backbone and head.

        Returns:
            List of dicts with ``"params"``, ``"name"``, and optionally ``"lr"`` keys; the
            backbone starts with learning rate 0 (frozen until unfreezing).
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0, "name": "backbone"},
            {"params": self.head.parameters(), "name": "head"},
        ]
        return params


class MultiHeadHeatmapTracker(HeatmapTracker):
    """Heatmap tracker with one complete head per source dataset, routed by dataset id.

    The backbone is the only shared component: every dataset in ``dataset_names`` owns a
    full :class:`~lightning_pose.models.heads.HeatmapHead` (all ``num_keypoints`` output
    channels, identical shape across heads). A mixed batch is routed by grouping rows on
    their ``dataset_id``, running each group through its own head, and scattering the
    results back into one ``(batch, num_keypoints, H, W)`` tensor in original batch
    order — so loss, metric, and prediction-export shapes are unchanged from the shared
    model. Channels a dataset never labels receive no gradient (their all-zero targets
    are dropped by the loss) and remain at initialization; they must be masked out of
    evaluation rather than read as predictions.

    Supervised training and labeled-frame prediction only. Prediction on unlabeled
    video (no per-frame dataset id) and dataset-blind head combination are not
    implemented here.
    """

    def __init__(
        self,
        dataset_names: list[str],
        **kwargs: Any,
    ) -> None:
        """Initialize a multi-head heatmap tracker.

        Args:
            dataset_names: ordered source-dataset registry; head ``i`` belongs to
                ``dataset_names[i]``, matching the ids parsed from image paths.
            **kwargs: passed through to :class:`HeatmapTracker`.
        """
        super().__init__(**kwargs)
        if not dataset_names:
            raise ValueError('dataset_names must be a non-empty ordered registry')
        self.dataset_names = list(dataset_names)

        # supporting-set mask for blind combination: mask[d, k] is True when head d
        # received training signal for keypoint k (direct visible=2 labels, or the hflip
        # partner of one when hflip augmentation is active). Non-persistent: the factory
        # recomputes it from training data at every construction, so checkpoints saved
        # before this buffer existed load cleanly.
        self.register_buffer(
            'head_keypoint_mask',
            torch.ones(len(self.dataset_names), self.num_keypoints, dtype=torch.bool),
            persistent=False,
        )

        # inference behavior, set post-construction by evaluation code:
        # 'oracle' routes each labeled row through its dataset's head (default);
        # 'blind' combines all supporting heads by confidence-weighted coordinates.
        # predict_dataset names the head for data with no per-row dataset id (videos).
        self.predict_mode: str = 'oracle'
        self.predict_dataset: str | None = None
        # blind-combination knobs; record alongside any reported blind numbers
        self.blind_gamma: float = 2.0
        self.blind_conf_floor: float = 0.0

        # replace the single shared head with one complete head per dataset; heads are
        # built after the parent's seeded init, so initialization stays deterministic
        # given torch_seed (but differs across heads, as it would across seeds)
        del self.head
        self.heads = torch.nn.ModuleList([
            HeatmapHead(
                backbone_arch=kwargs.get('backbone', 'resnet50'),
                in_channels=self.num_fc_input_features,
                out_channels=self.num_keypoints,
                downsample_factor=self.downsample_factor,
            )
            for _ in self.dataset_names
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Raise: a routed forward needs per-frame dataset ids.

        Raises:
            NotImplementedError: always; use :meth:`forward_routed`. Dataset-blind
                combination across heads is a separate inference mode, not a silent
                fallback.
        """
        raise NotImplementedError(
            'MultiHeadHeatmapTracker has no dataset-agnostic forward; use '
            'forward_routed(images, dataset_ids). Blind head combination is a separate '
            'inference path.'
        )

    def forward_routed(
        self,
        images: Float[torch.Tensor, 'batch channels image_height image_width'],
        dataset_ids: Float[torch.Tensor, 'batch'],
    ) -> Float[torch.Tensor, 'batch num_keypoints heatmap_height heatmap_width']:
        """Compute backbone features once, then route each row through its dataset's head.

        Args:
            images: batch of frames.
            dataset_ids: per-row source-dataset id (index into ``dataset_names``).

        Returns:
            heatmaps scattered back into original batch order.
        """
        representations = self.get_representations(images)
        heatmaps = None
        for dataset_id in torch.unique(dataset_ids):
            mask = dataset_ids == dataset_id
            heatmaps_group = self.heads[int(dataset_id)](representations[mask])
            if heatmaps is None:
                heatmaps = heatmaps_group.new_zeros(
                    representations.shape[0], *heatmaps_group.shape[1:],
                )
            heatmaps[mask] = heatmaps_group
        return heatmaps

    def set_head_keypoint_mask(
        self,
        visibility: torch.Tensor,
        dataset_ids: torch.Tensor,
        keypoint_names: list[str],
        hflip: bool,
    ) -> None:
        """Fill the supporting-set mask from training-data visibility.

        Args:
            visibility: (num_frames, num_keypoints) int tensor of {0, 1, 2} flags.
            dataset_ids: (num_frames,) source-dataset id per frame.
            keypoint_names: canonical keypoint names, for hflip partner resolution.
            hflip: True when horizontal-flip augmentation is active, in which case the
                _left/_right partner of a directly-labeled keypoint also receives
                training signal and joins the supporting set.
        """
        direct = torch.zeros(len(self.dataset_names), self.num_keypoints, dtype=torch.bool)
        for dataset_id in range(len(self.dataset_names)):
            rows = dataset_ids == dataset_id
            if rows.any():
                direct[dataset_id] = (visibility[rows] == 2).any(dim=0)
        mask = direct.clone()
        if hflip:
            idx_by_name = {name: i for i, name in enumerate(keypoint_names)}
            for i, name in enumerate(keypoint_names):
                if name.endswith('_left'):
                    partner = f'{name[:-5]}_right'
                elif name.endswith('_right'):
                    partner = f'{name[:-6]}_left'
                else:
                    continue
                if partner in idx_by_name:
                    mask[:, idx_by_name[partner]] |= direct[:, i]
        self.head_keypoint_mask.copy_(mask)
        logger_counts = mask.sum(dim=1).tolist()
        import logging
        logging.getLogger(__name__).info(
            f'head_keypoint_mask set: trainable keypoints per head = '
            f'{dict(zip(self.dataset_names, logger_counts))}'
        )

    def forward_blind(
        self,
        images: Float[torch.Tensor, 'batch channels image_height image_width'],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine all supporting heads without dataset identity (blind / zero-shot).

        Every head runs on the shared backbone features; per keypoint, only heads whose
        supporting-set mask covers that keypoint vote, with weights
        ``w_h ∝ conf_h ** blind_gamma`` and heads below ``blind_conf_floor`` dropped.
        Combination happens in coordinate space, not heatmap space: averaging softmaxed
        heatmaps from disagreeing heads yields a bimodal map whose soft-argmax lands
        between the modes — a location no head predicted. If every supporting head falls
        below the floor, the single most confident supporting head is used as-is, so the
        output carries its genuinely low confidence rather than a fabricated one.

        Returns:
            tuple of (keypoints (batch, 2 * num_keypoints), confidences
            (batch, num_keypoints), inter-head spread (batch, num_keypoints) — the
            weighted RMS distance of head predictions from the combined coordinate,
            a per-model uncertainty signal).
        """
        representations = self.get_representations(images)
        coords_heads = []
        conf_heads = []
        for head in self.heads:
            heatmaps = head(representations)
            keypoints, confidence = head.run_subpixelmaxima(heatmaps)
            coords_heads.append(keypoints.reshape(keypoints.shape[0], -1, 2))
            conf_heads.append(confidence)
        coords = torch.stack(coords_heads)  # (num_heads, batch, K, 2)
        conf = torch.stack(conf_heads)      # (num_heads, batch, K)

        mask = self.head_keypoint_mask[:, None, :]                      # (num_heads, 1, K)
        weights = conf.clamp_min(0) ** self.blind_gamma
        weights = weights * mask
        weights = torch.where(conf >= self.blind_conf_floor, weights, torch.zeros_like(weights))
        weights_sum = weights.sum(dim=0)                                # (batch, K)

        norm = weights / weights_sum.clamp_min(1e-12)
        combined_coords = (norm[..., None] * coords).sum(dim=0)         # (batch, K, 2)
        combined_conf = (norm * conf).sum(dim=0)                        # (batch, K)
        spread = torch.sqrt(
            (norm * ((coords - combined_coords) ** 2).sum(dim=-1)).sum(dim=0)
        )

        # abstention fallback: keypoints where no supporting head cleared the floor take
        # the most confident supporting head verbatim (confidence included)
        masked_conf = torch.where(mask.expand_as(conf), conf, conf.new_full((), -torch.inf))
        idx_best = masked_conf.argmax(dim=0)                            # (batch, K)
        fallback_coords = coords.gather(
            0, idx_best[None, ..., None].expand(1, *idx_best.shape, 2),
        )[0]
        fallback_conf = conf.gather(0, idx_best[None])[0]
        no_vote = weights_sum <= 0
        combined_coords = torch.where(no_vote[..., None], fallback_coords, combined_coords)
        combined_conf = torch.where(no_vote, fallback_conf, combined_conf)
        spread = torch.where(no_vote, torch.zeros_like(spread), spread)

        return combined_coords.reshape(coords.shape[1], -1), combined_conf, spread

    def get_loss_inputs_labeled(self, batch_dict: HeatmapLabeledBatchDict) -> dict:
        """Return predicted heatmaps and keypoints, routing rows by dataset id."""
        if 'dataset_id' not in batch_dict:
            raise ValueError(
                'per-dataset heads require dataset_id in each labeled batch; set '
                'data.dataset_names so the dataset parses ids from image paths'
            )
        predicted_heatmaps = self.forward_routed(
            batch_dict['images'], batch_dict['dataset_id'],
        )
        # all heads share downsample factor and softmax temperature, so head 0's
        # subpixel refinement applies to the scattered tensor as a whole
        predicted_keypoints, confidence = self.heads[0].run_subpixelmaxima(predicted_heatmaps)
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        target_keypoints = model_to_frame_batch(batch_dict, batch_dict['keypoints'])
        return {
            'heatmaps_targ': batch_dict['heatmaps'],
            'heatmaps_pred': predicted_heatmaps,
            'keypoints_targ': target_keypoints,
            'keypoints_pred': predicted_keypoints,
            'confidences': confidence,
        }

    def predict_step(
        self,
        batch_dict: HeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Predict keypoints using the configured inference mode.

        Modes (``self.predict_mode``, set post-construction by evaluation code):

        - ``'oracle'`` (default): each row runs through its own dataset's head. The
          dataset id comes from the batch (labeled CSVs whose paths encode the dataset)
          or, when absent — videos, external frames — from ``self.predict_dataset``.
        - ``'blind'``: no dataset identity is used; all supporting heads vote via
          :meth:`forward_blind`. This is also the zero-shot path for unseen datasets.
          Heatmaps cannot be returned in this mode (combination is in coordinate space).
        """
        if 'images' in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            images = batch_dict['images']  # type: ignore[typeddict-item]
        else:
            images = batch_dict['frames']  # type: ignore[typeddict-item]

        if self.predict_mode == 'blind':
            predicted_keypoints, confidence, _spread = self.forward_blind(images)
            predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
            if return_heatmaps:
                raise NotImplementedError(
                    'blind mode combines coordinates across heads; there is no single '
                    'heatmap tensor to return'
                )
            return predicted_keypoints, confidence

        if self.predict_mode != 'oracle':
            raise ValueError(f"predict_mode must be 'oracle' or 'blind', got '{self.predict_mode}'")

        if 'dataset_id' in batch_dict.keys():
            dataset_ids = batch_dict['dataset_id']  # type: ignore[typeddict-item]
        elif self.predict_dataset is not None:
            if self.predict_dataset not in self.dataset_names:
                raise ValueError(
                    f"predict_dataset '{self.predict_dataset}' is not in the registry "
                    f'{self.dataset_names}'
                )
            dataset_ids = torch.full(
                (images.shape[0],), self.dataset_names.index(self.predict_dataset),
                dtype=torch.long, device=images.device,
            )
        else:
            raise ValueError(
                'oracle prediction needs a dataset identity: the batch carries none and '
                'predict_dataset is unset. For unseen data use predict_mode="blind".'
            )

        predicted_heatmaps = self.forward_routed(images, dataset_ids)
        predicted_keypoints, confidence = self.heads[0].run_subpixelmaxima(predicted_heatmaps)
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        if return_heatmaps:
            return predicted_keypoints, confidence, predicted_heatmaps
        else:
            return predicted_keypoints, confidence

    def get_parameters(self) -> list[dict]:
        """Keep the two-group layout (0=backbone, 1=head) with every head in group 1.

        Callbacks treat parameter group 0 as backbone and group 1 as head for
        unfreezing, so all per-dataset heads share the single 'head' group.
        """
        params = [
            {'params': self.backbone.parameters(), 'lr': 0, 'name': 'backbone'},
            {'params': self.heads.parameters(), 'name': 'head'},
        ]
        return params


class TokenConditionedHeatmapTracker(HeatmapTracker):
    """Shared-head heatmap tracker conditioned on a learned per-dataset token.

    The third head mode: every weight is shared (unlike per-dataset heads, so small
    datasets never starve their own parameters), but a learned embedding per registry
    dataset is injected into the backbone so the whole network can specialize on
    dataset identity (unlike the plain shared head, so conflicting label conventions
    need not fight over one output channel). The token is added to every patch
    embedding before the transformer (see ``VisionEncoderDino.forward``); tokens are
    zero-initialized, so at initialization the model is exactly the shared model and
    conditioning is learned as a delta.

    Requires a DINO ViT backbone (the injection point lives in its wrapper) and
    supervised training on single-view, non-context batches carrying ``dataset_id``.

    Inference modes (``self.predict_mode``, set post-construction):

    - ``'oracle'`` (default): each row uses its own dataset's token, from the batch's
      ``dataset_id`` or, for videos/external frames, from ``self.predict_dataset``.
    - ``'mean_token'``: every row uses the mean of all learned tokens — the zero-shot
      entry point for data from an unseen dataset.

    Few-shot adaptation to a new dataset = append one token row and train only it
    (handled by fine-tuning scripts, not here).
    """

    def __init__(
        self,
        dataset_names: list[str],
        token_lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        """Initialize a token-conditioned heatmap tracker.

        Args:
            dataset_names: ordered source-dataset registry; token ``i`` belongs to
                ``dataset_names[i]``, matching the ids parsed from image paths.
            token_lr: learning rate for the token embedding's own optimizer group.
                Tokens sit behind the full (initially frozen) transformer, so their
                gradients arrive orders of magnitude weaker than the head's — the
                standard prompt-tuning regime, cured with a much larger lr.
            **kwargs: passed through to :class:`HeatmapTracker`.
        """
        super().__init__(**kwargs)
        if not dataset_names:
            raise ValueError('dataset_names must be a non-empty ordered registry')
        self.dataset_names = list(dataset_names)
        self.token_lr = float(token_lr)

        # zero-init: conditioning starts as a no-op and is learned as a per-dataset
        # delta; also makes the mean token exactly neutral at initialization
        self.dataset_tokens = torch.nn.Embedding(
            len(self.dataset_names), self.num_fc_input_features,
        )
        torch.nn.init.zeros_(self.dataset_tokens.weight)

        self.predict_mode: str = 'oracle'
        self.predict_dataset: str | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Raise: a conditioned forward needs per-frame dataset ids.

        Raises:
            NotImplementedError: always; use :meth:`forward_conditioned`, or
                ``predict_mode='mean_token'`` for data with no dataset identity.
        """
        raise NotImplementedError(
            'TokenConditionedHeatmapTracker has no unconditioned forward; use '
            "forward_conditioned(images, dataset_ids), or predict_mode='mean_token' "
            'for unseen data'
        )

    def _conditioned_representations(
        self,
        images: Float[torch.Tensor, 'batch channels image_height image_width'],
        tokens: Float[torch.Tensor, 'batch features'],
    ) -> Float[torch.Tensor, 'batch features rep_height rep_width']:
        """Backbone forward with per-row conditioning tokens (non-context, single view)."""
        if len(images.shape) != 4:
            raise NotImplementedError(
                'token conditioning supports single-view, non-context batches only'
            )
        if not hasattr(self.backbone, 'vision_encoder'):
            raise NotImplementedError(
                'token conditioning requires a DINO ViT backbone '
                '(VisionEncoderDino); got ' + type(self.backbone).__name__
            )
        return self.backbone(images, dataset_tokens=tokens)

    def forward_conditioned(
        self,
        images: Float[torch.Tensor, 'batch channels image_height image_width'],
        dataset_ids: Float[torch.Tensor, 'batch'],
    ) -> Float[torch.Tensor, 'batch num_keypoints heatmap_height heatmap_width']:
        """Forward pass conditioning each row on its dataset's learned token.

        Args:
            images: batch of images.
            dataset_ids: per-row source-dataset id (index into ``dataset_names``).

        Returns:
            heatmaps in the canonical 36-keypoint space, original batch order.
        """
        tokens = self.dataset_tokens(dataset_ids.long())
        representations = self._conditioned_representations(images, tokens)
        return self.head(representations)

    def forward_mean_token(
        self,
        images: Float[torch.Tensor, 'batch channels image_height image_width'],
    ) -> Float[torch.Tensor, 'batch num_keypoints heatmap_height heatmap_width']:
        """Forward pass conditioning every row on the mean of all learned tokens.

        The zero-shot entry point for data whose source dataset is not in the registry.
        """
        mean_token = self.dataset_tokens.weight.mean(dim=0, keepdim=True)
        tokens = mean_token.expand(images.shape[0], -1)
        representations = self._conditioned_representations(images, tokens)
        return self.head(representations)

    def get_loss_inputs_labeled(self, batch_dict: HeatmapLabeledBatchDict) -> dict:
        """Return predicted heatmaps and keypoints, conditioning rows by dataset id."""
        if 'dataset_id' not in batch_dict:
            raise ValueError(
                'token conditioning requires dataset_id in each labeled batch; set '
                'data.dataset_names so the dataset parses ids from image paths'
            )
        predicted_heatmaps = self.forward_conditioned(
            batch_dict['images'], batch_dict['dataset_id'],
        )
        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        target_keypoints = model_to_frame_batch(batch_dict, batch_dict['keypoints'])
        return {
            'heatmaps_targ': batch_dict['heatmaps'],
            'heatmaps_pred': predicted_heatmaps,
            'keypoints_targ': target_keypoints,
            'keypoints_pred': predicted_keypoints,
            'confidences': confidence,
        }

    def predict_step(
        self,
        batch_dict: HeatmapLabeledBatchDict | UnlabeledBatchDict,
        batch_idx: int,
        return_heatmaps: bool | None = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Predict keypoints using the configured conditioning mode.

        See the class docstring for the ``'oracle'`` / ``'mean_token'`` modes.
        """
        if 'images' in batch_dict.keys():  # can't do isinstance(o, c) on TypedDicts
            images = batch_dict['images']  # type: ignore[typeddict-item]
        else:
            images = batch_dict['frames']  # type: ignore[typeddict-item]

        if self.predict_mode == 'mean_token':
            predicted_heatmaps = self.forward_mean_token(images)
        elif self.predict_mode == 'oracle':
            if 'dataset_id' in batch_dict.keys():
                dataset_ids = batch_dict['dataset_id']  # type: ignore[typeddict-item]
            elif self.predict_dataset is not None:
                if self.predict_dataset not in self.dataset_names:
                    raise ValueError(
                        f"predict_dataset '{self.predict_dataset}' is not in the "
                        f'registry {self.dataset_names}'
                    )
                dataset_ids = torch.full(
                    (images.shape[0],), self.dataset_names.index(self.predict_dataset),
                    dtype=torch.long, device=images.device,
                )
            else:
                raise ValueError(
                    'oracle prediction needs a dataset identity: the batch carries '
                    'none and predict_dataset is unset. For unseen data use '
                    "predict_mode='mean_token'."
                )
            predicted_heatmaps = self.forward_conditioned(images, dataset_ids)
        else:
            raise ValueError(
                f"predict_mode must be 'oracle' or 'mean_token', got '{self.predict_mode}'"
            )

        predicted_keypoints, confidence = self.head.run_subpixelmaxima(predicted_heatmaps)
        predicted_keypoints = model_to_frame_batch(batch_dict, predicted_keypoints)
        if return_heatmaps:
            return predicted_keypoints, confidence, predicted_heatmaps
        else:
            return predicted_keypoints, confidence

    def get_parameters(self) -> list[dict]:
        """Backbone and head groups as usual, plus a token group with its own lr.

        The unfreezing callback reads only groups 0 (backbone) and 1 (head), so the
        extra group is safe. Tokens need their own, much larger lr: their gradient
        arrives through the whole (initially frozen) transformer and is orders of
        magnitude weaker than the head's — measured ~1e-4x on this backbone.
        """
        params = [
            {'params': self.backbone.parameters(), 'lr': 0, 'name': 'backbone'},
            {'params': self.head.parameters(), 'name': 'head'},
            {
                'params': self.dataset_tokens.parameters(),
                'lr': self.token_lr,
                'name': 'dataset_tokens',
            },
        ]
        return params


class SemiSupervisedHeatmapTracker(SemiSupervisedTrackerMixin, HeatmapTracker):
    """Model produces heatmaps of keypoints from labeled/unlabeled images."""

    def __init__(
        self,
        num_keypoints: int,
        loss_factory: LossFactory | None = None,
        loss_factory_unsupervised: LossFactory | None = None,
        backbone: ALLOWED_BACKBONES = "resnet50",
        downsample_factor: Literal[1, 2, 3] = 2,
        pretrained: bool = True,
        torch_seed: int = 123,
        optimizer: str = "Adam",
        optimizer_params: DictConfig | ListConfig | dict | None = None,
        lr_scheduler: str = "multisteplr",
        lr_scheduler_params: DictConfig | ListConfig | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """

        Args:
            num_keypoints: number of body parts
            loss_factory: object to orchestrate supervised loss computation
            loss_factory_unsupervised: object to orchestrate unsupervised loss
                computation
            backbone: ResNet or EfficientNet variant to be used
            downsample_factor: make heatmap smaller than original frames to
                save memory; subpixel operations are performed for increased
                precision
            pretrained: True to load pretrained imagenet weights
            torch_seed: make weight initialization reproducible
            lr_scheduler: how to schedule learning rate
                multisteplr
            lr_scheduler_params: params for specific learning rate schedulers
                multisteplr: milestones, gamma

        """
        super().__init__(
            num_keypoints=num_keypoints,
            loss_factory=loss_factory,
            backbone=backbone,
            downsample_factor=downsample_factor,
            pretrained=pretrained,
            torch_seed=torch_seed,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            **kwargs,
        )
        self.loss_factory_unsup = loss_factory_unsupervised

        # this attribute will be modified by AnnealWeight callback during training
        # self.register_buffer("total_unsupervised_importance", torch.tensor(1.0))
        self.total_unsupervised_importance = torch.tensor(1.0)

    def get_loss_inputs_unlabeled(
        self,
        batch_dict: UnlabeledBatchDict | MultiviewUnlabeledBatchDict,
    ) -> dict:
        """Return predicted heatmaps and their softmaxes (estimated keypoints)."""
        # images -> heatmaps
        pred_heatmaps = self.forward(batch_dict["frames"])
        # heatmaps -> keypoints
        pred_keypoints_augmented, confidence = self.head.run_subpixelmaxima(pred_heatmaps)
        # undo augmentation if needed
        pred_keypoints = undo_affine_transform_batch(
            keypoints_augmented=pred_keypoints_augmented,
            transforms=batch_dict["transforms"],
            is_multiview=batch_dict["is_multiview"],
        )
        # keypoints -> original image coords keypoints
        pred_keypoints = model_to_frame_batch(batch_dict, pred_keypoints)
        return {
            "heatmaps_pred": pred_heatmaps,  # if augmented, augmented heatmaps
            "keypoints_pred": pred_keypoints,  # if augmented, original keypoints
            "keypoints_pred_augmented": pred_keypoints_augmented,  # match pred_heatmaps
            "confidences": confidence,
        }
