from pathlib import Path

import tbparse
from omegaconf import DictConfig, OmegaConf

from lightning_pose.train import train


def _get_base_cfg(cfg):
    # Smallest possible test: two steps per epoch.
    return OmegaConf.merge(
        cfg,
        {
            "training": {
                "train_frames": 2,
                "train_batch_size": 1,
            },
            "model": {
                "backbone": "resnet18",  # lightweight
                "losses_to_use": [],  # disable PCA, too few observations
            },
        },
    )


def test_epochs(cfg: DictConfig, tmp_path: Path):
    # Setup so that it trains for 2 full epochs.
    # Then we'll assert that it actually trained that much.
    cfg = _get_base_cfg(cfg)
    cfg = OmegaConf.merge(
        cfg,
        {
            "training": {
                "min_epochs": 2,
                "max_epochs": 2,
            },
        },
    )

    train(cfg, model_dir=tmp_path, skip_evaluation=True)

    reader = tbparse.SummaryReader(
        tmp_path / "tb_logs" / "my_base_toy_model", pivot=True
    )
    # this is the epoch at each step. 2 epochs, 4 steps
    assert reader.scalars["epoch"].tolist() == [0, 0, 1, 1]


def test_steps(cfg: DictConfig, tmp_path: Path):
    # Setup so that it trains for 3 steps (2nd epoch is partial).
    # Then we'll assert that it actually trained that much.
    cfg = _get_base_cfg(cfg)
    cfg = OmegaConf.merge(
        cfg,
        {
            "training": {
                "min_steps": 3,
                "max_steps": 3,
                "unfreezing_step": 5,  # satisfy validation
                "lr_scheduler_params": {
                    "multisteplr": {
                        "milestone_steps": [100],  # satisfy validation
                    }
                },
            },
        },
    )

    # satisfy validation
    del cfg.training.min_epochs
    del cfg.training.max_epochs
    del cfg.training.unfreezing_epoch
    del cfg.training.lr_scheduler_params.multisteplr.milestones

    train(cfg, model_dir=tmp_path, skip_evaluation=True)

    reader = tbparse.SummaryReader(
        tmp_path / "tb_logs" / "my_base_toy_model", pivot=True
    )
    # this is the epoch at each step. only 3 steps total.
    assert reader.scalars["epoch"].tolist() == [0, 0, 1]
