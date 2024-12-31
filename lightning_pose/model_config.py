from pathlib import Path

from omegaconf import DictConfig, OmegaConf

__all__ = ["ModelConfig"]

from lightning_pose.utils.io import check_video_paths, return_absolute_path


class ModelConfig:

    @staticmethod
    def from_yaml_file(filepath):
        return ModelConfig(OmegaConf.load(filepath))

    def __init__(self, cfg: DictConfig):
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

    ## Eval ##

    def test_video_files(self) -> list[Path]:
        files = check_video_paths(
            return_absolute_path(self.cfg.eval.test_videos_directory)
        )
        return [Path(f) for f in files]
