from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

__all__ = ["ModelConfig"]

from lightning_pose.utils.io import check_video_paths, return_absolute_path


class ModelConfig:

    @staticmethod
    def from_yaml_file(filepath):
        return ModelConfig(OmegaConf.load(filepath))

    def __init__(self, cfg: DictConfig):
        # Patch keypoint_names with keypoints in case the user
        # is predicting on an LP App's model.
        # https://github.com/paninski-lab/lightning-pose/issues/268
        if "keypoints" in cfg.data and "keypoint_names" not in cfg.data:
            with open_dict(cfg.data):
                cfg.data.keypoint_names = cfg.data.keypoints
                del cfg.data.keypoints
        self.cfg = cfg

    def is_single_view(self):
        return not self.is_multi_view()

    def is_multi_view(self):
        if self.cfg.data.get("view_names") is None:
            return False
        if len(self.cfg.data.view_names) == 1:
            raise ValueError(
                "view_names should not be specified if there is only one view."
            )
        return True

    def test_video_files(self) -> list[Path]:
        files = check_video_paths(
            return_absolute_path(self.cfg.eval.test_videos_directory)
        )
        return [Path(f) for f in files]

    def validate(self):
        self._validate_steps_vs_epochs()

    def _validate_steps_vs_epochs(self):
        if "min_steps" in self.cfg.training or "max_steps" in self.cfg.training:
            assert "min_steps" in self.cfg.training
            assert "max_steps" in self.cfg.training
            assert "min_epochs" not in self.cfg.training
            assert "max_epochs" not in self.cfg.training
            assert "unfreezing_step" in self.cfg.training
            assert "unfreezing_epoch" not in self.cfg.training
            if "multisteplr" in self.cfg.training.lr_scheduler_params:
                assert (
                    "milestone_steps"
                    in self.cfg.training.lr_scheduler_params.multisteplr
                )
                assert (
                    "milestones"
                    not in self.cfg.training.lr_scheduler_params.multisteplr
                )

        else:
            assert "min_steps" not in self.cfg.training
            assert "max_steps" not in self.cfg.training
            assert "min_epochs" in self.cfg.training
            assert "max_epochs" in self.cfg.training
            assert "unfreezing_step" not in self.cfg.training
            assert "unfreezing_epoch" in self.cfg.training
            if "multisteplr" in self.cfg.training.lr_scheduler_params:
                assert (
                    "milestone_steps"
                    not in self.cfg.training.lr_scheduler_params.multisteplr
                )
                assert "milestones" in self.cfg.training.lr_scheduler_params.multisteplr
