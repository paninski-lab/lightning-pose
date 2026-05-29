"""Wrapper around an OmegaConf config with convenience methods for common config queries."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

__all__ = ["ModelConfig"]

from lightning_pose.models import ALLOWED_MODEL_TYPES
from lightning_pose.utils.io import (
    check_video_paths,
    find_video_files_for_views,
    return_absolute_path,
)


class ModelConfig:
    """Wraps an OmegaConf config and exposes convenience methods.

    Prefer constructing via `from_yaml_file` when loading a saved model, or by passing
    a `DictConfig` directly when composing configs programmatically with Hydra.

    Attributes:
        cfg: The underlying OmegaConf config object. Access raw config values here.

    Examples:
        >>> config = ModelConfig.from_yaml_file("outputs/2024-01-01/12-00-00/config.yaml")
        >>> config.is_multi_view()
        False
        >>> config.cfg.model.backbone
        'resnet50_animal_ap10k'
    """

    @staticmethod
    def from_yaml_file(filepath: str | Path) -> ModelConfig:
        """Load a config from a YAML file on disk.

        Args:
            filepath: path to a ``config.yaml`` file, typically found in a model output directory.

        Returns:
            ModelConfig wrapping the loaded config.
        """
        return ModelConfig(OmegaConf.load(filepath))

    def __init__(self, cfg: DictConfig | ListConfig) -> None:
        """Initialize from an existing OmegaConf config.

        Automatically normalizes the legacy ``cfg.data.keypoints`` field to
        ``cfg.data.keypoint_names`` for backwards compatibility with models saved by the LP App.

        Args:
            cfg: OmegaConf config, typically produced by Hydra or `OmegaConf.load`.
        """
        # Patch keypoint_names with keypoints in case the user
        # is predicting on an LP App's model.
        # https://github.com/paninski-lab/lightning-pose/issues/268
        if "keypoints" in cfg.data and "keypoint_names" not in cfg.data:
            with open_dict(cfg.data):
                cfg.data.keypoint_names = cfg.data.keypoints
                del cfg.data.keypoints
        self.cfg = cfg

    def is_single_view(self) -> bool:
        """Return True if the model was trained on a single camera view.

        Returns:
            True when ``cfg.data.view_names`` is absent or unset.
        """
        return not self.is_multi_view()

    def is_multi_view(self) -> bool:
        """Return True if the model was trained on multiple camera views.

        Returns:
            True when ``cfg.data.view_names`` is set and contains more than one entry.

        Raises:
            ValueError: if ``view_names`` is set but contains only one entry, which is
                invalid — single-view models should omit ``view_names`` entirely.
        """
        if self.cfg.data.get("view_names") is None:
            return False
        if len(self.cfg.data.view_names) == 1:
            raise ValueError("view_names should not be specified if there is only one view.")
        return True

    def test_video_files_singleview(self) -> list[Path]:
        """Return the list of test video files for a single-view model.

        Resolves paths relative to ``cfg.eval.test_videos_directory``.

        Returns:
            list of absolute paths to video files.

        Raises:
            AssertionError: if called on a multi-view model.
        """
        assert self.is_single_view(), "Use test_video_files_multiview for multi-view"
        files = check_video_paths(return_absolute_path(self.cfg.eval.test_videos_directory))
        return [Path(f) for f in files]

    def test_video_files_multiview(self) -> list[list[Path]]:
        """Return the list of test video files grouped by session for a multi-view model.

        Each inner list contains one video file per view, all from the same recording session,
        in the same order as ``cfg.data.view_names``.

        Returns:
            list of per-session video file groups, where each group is a list of paths
            ordered by view.

        Raises:
            AssertionError: if called on a single-view model.
        """
        assert self.is_multi_view()
        return find_video_files_for_views(
            video_dir=self.cfg.eval.test_videos_directory,
            view_names=self.cfg.data.view_names,
        )

    def validate(self) -> None:
        """Run all config validation checks.

        Raises:
            AssertionError: if any validation check fails.
        """
        self._validate_data()
        self._validate_training()
        self._validate_model()
        self._validate_losses()

    def _validate_data(self) -> None:
        """Validate the ``data`` config section.

        Checks:
        - ``num_keypoints`` is set and positive.
        - ``keypoint_names``, if set, has length equal to ``num_keypoints``.
        - For multi-view models, ``view_names`` and ``csv_file`` have the same length.
        - ``image_resize_dims`` height and width, if set, are multiples of 128.

        Raises:
            AssertionError: if any check fails.
        """
        assert self.cfg.data.num_keypoints is not None, \
            'data.num_keypoints must be set'
        assert self.cfg.data.num_keypoints > 0, \
            f'data.num_keypoints must be positive, got {self.cfg.data.num_keypoints}'

        if self.cfg.data.keypoint_names is not None:
            n_names = len(self.cfg.data.keypoint_names)
            n_kp = self.cfg.data.num_keypoints
            assert n_names == n_kp, (
                f'len(data.keypoint_names) ({n_names}) must equal '
                f'data.num_keypoints ({n_kp})'
            )

        if self.is_multi_view():
            n_views = len(self.cfg.data.view_names)
            n_csv = len(self.cfg.data.csv_file)
            assert n_views == n_csv, (
                f'len(data.view_names) ({n_views}) must equal '
                f'len(data.csv_file) ({n_csv})'
            )

        for dim in ('height', 'width'):
            val = self.cfg.data.image_resize_dims.get(dim)
            if val is not None:
                assert val % 128 == 0, (
                    f'data.image_resize_dims.{dim} ({val}) must be a multiple of 128'
                )

    def _validate_training(self) -> None:
        """Validate the ``training`` config section.

        Checks:
        - ``train_prob + val_prob`` does not exceed 1.0.
        - ``ckpt_every_n_epochs``, if set, is divisible by ``check_val_every_n_epoch``.
        - LR scheduler milestones do not exceed ``max_epochs`` (or ``max_steps``).
        - Step- and epoch-based fields are not mixed (delegates to
          :meth:`_validate_steps_vs_epochs`).

        Raises:
            AssertionError: if any check fails.
        """
        train_prob = self.cfg.training.train_prob
        val_prob = self.cfg.training.val_prob
        assert train_prob + val_prob <= 1.0, (
            f'training.train_prob ({train_prob}) + training.val_prob ({val_prob}) '
            f'must be <= 1.0'
        )

        ckpt = self.cfg.training.get('ckpt_every_n_epochs')
        if ckpt is not None:
            check_val = self.cfg.training.check_val_every_n_epoch
            assert ckpt % check_val == 0, (
                f'training.ckpt_every_n_epochs ({ckpt}) must be divisible by '
                f'training.check_val_every_n_epoch ({check_val})'
            )

        multisteplr = self.cfg.training.lr_scheduler_params.get('multisteplr')
        if multisteplr is not None:
            if 'milestones' in multisteplr:
                max_val = self.cfg.training.max_epochs
                assert all(m <= max_val for m in multisteplr.milestones), (
                    f'all training.lr_scheduler_params.multisteplr.milestones must be '
                    f'<= training.max_epochs ({max_val})'
                )
            elif 'milestone_steps' in multisteplr:
                max_val = self.cfg.training.max_steps
                assert all(m <= max_val for m in multisteplr.milestone_steps), (
                    f'all training.lr_scheduler_params.multisteplr.milestone_steps must be '
                    f'<= training.max_steps ({max_val})'
                )

        self._validate_steps_vs_epochs()

    def _validate_model(self) -> None:
        """Validate the ``model`` config section.

        Checks:
        - ``model_type`` is a recognised value.
        - Multi-view models use a heatmap-based model_type.
        - When ``losses.supervised_reprojection_heatmap_mse`` is active, ``training.imgaug``
          must be ``"dlc"`` and ``training.imgaug_3d`` must be ``true``.

        Raises:
            AssertionError: if any check fails.
        """
        allowed = set(get_args(ALLOWED_MODEL_TYPES))
        model_type = self.cfg.model.model_type
        assert model_type in allowed, (
            f"model.model_type '{model_type}' is not one of {sorted(allowed)}"
        )

        if self.is_multi_view():
            assert model_type in ('heatmap', 'heatmap_mhcrnn', 'heatmap_multiview_transformer'), (
                f"multi-view models require a heatmap-based model_type, got '{model_type}'"
            )

            reprojection = self.cfg.losses.get('supervised_reprojection_heatmap_mse')
            if reprojection is not None and reprojection.get('log_weight') is not None:
                assert self.cfg.training.imgaug == 'dlc', (
                    "training.imgaug must be 'dlc' when "
                    "losses.supervised_reprojection_heatmap_mse is active"
                )
                assert self.cfg.training.get('imgaug_3d') is True, (
                    "training.imgaug_3d must be true when "
                    "losses.supervised_reprojection_heatmap_mse is active"
                )

    def _validate_losses(self) -> None:
        """Validate the ``losses`` config section.

        Checks:
        - For each loss listed in ``model.losses_to_use``, if a ``log_weight`` is
          configured it must be a numeric value (int or float); a string or other type
          would silently produce wrong training behaviour.

        Raises:
            AssertionError: if any check fails.
        """
        losses_to_use = self.cfg.model.get('losses_to_use') or []
        for loss_name in losses_to_use:
            if not loss_name:
                continue
            loss_cfg = self.cfg.losses.get(loss_name)
            if loss_cfg is None:
                continue
            log_weight = loss_cfg.get('log_weight')
            if log_weight is not None:
                assert isinstance(log_weight, (int, float)), (
                    f'losses.{loss_name}.log_weight must be numeric (int or float), '
                    f'got {type(log_weight).__name__}: {log_weight!r}'
                )

    def _validate_steps_vs_epochs(self) -> None:
        """Ensure the training schedule uses either steps or epochs, not a mix of both.

        The config must use one of two mutually exclusive modes:
        - epoch-based: ``min_epochs``, ``max_epochs``, ``unfreezing_epoch``, ``milestones``
        - step-based: ``min_steps``, ``max_steps``, ``unfreezing_step``, ``milestone_steps``

        Raises:
            AssertionError: if epoch- and step-based fields are mixed.
        """
        if "min_steps" in self.cfg.training or "max_steps" in self.cfg.training:
            assert "min_steps" in self.cfg.training
            assert "max_steps" in self.cfg.training
            assert "min_epochs" not in self.cfg.training
            assert "max_epochs" not in self.cfg.training
            assert "unfreezing_step" in self.cfg.training
            assert "unfreezing_epoch" not in self.cfg.training
            if "multisteplr" in self.cfg.training.lr_scheduler_params:
                assert "milestone_steps" in self.cfg.training.lr_scheduler_params.multisteplr
                assert "milestones" not in self.cfg.training.lr_scheduler_params.multisteplr

        else:
            assert "min_steps" not in self.cfg.training
            assert "max_steps" not in self.cfg.training
            assert "min_epochs" in self.cfg.training
            assert "max_epochs" in self.cfg.training
            assert "unfreezing_step" not in self.cfg.training
            assert "unfreezing_epoch" in self.cfg.training
            if "multisteplr" in self.cfg.training.lr_scheduler_params:
                assert "milestone_steps" not in self.cfg.training.lr_scheduler_params.multisteplr
                assert "milestones" in self.cfg.training.lr_scheduler_params.multisteplr
