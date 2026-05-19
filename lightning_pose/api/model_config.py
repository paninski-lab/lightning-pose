"""Wrapper around an OmegaConf config with convenience methods for common config queries."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

__all__ = ["ModelConfig"]

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
        self._validate_steps_vs_epochs()

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
