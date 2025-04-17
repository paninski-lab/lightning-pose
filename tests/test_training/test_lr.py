import math
from pathlib import Path

import tbparse
from omegaconf import DictConfig, OmegaConf

from lightning_pose.train import train


def _get_base_cfg(cfg):
    # 2 steps per epoch to exercise difference between stepwise vs epochwise.
    return OmegaConf.merge(
        cfg,
        {
            "training": {
                "train_frames": 2,
                "train_batch_size": 1,
                "check_val_every_n_epoch": 100,  # dont check val for performance
            },
            "model": {
                "backbone": "resnet18",  # lightweight
                "losses_to_use": [],  # disable PCA, too few observations
            },
        },
    )


def test_unfreeze_epoch(cfg: DictConfig, tmp_path: Path):
    cfg = _get_base_cfg(cfg)
    cfg = OmegaConf.merge(
        cfg,
        {
            "training": {
                "min_epochs": 12,
                "max_epochs": 12,
                "unfreezing_epoch": 1,
                "lr_scheduler_params": {
                    "multisteplr": {
                        "milestones": [9, 11],
                    }
                },
            },
        },
    )

    train(cfg, model_dir=tmp_path, skip_evaluation=True)

    reader = tbparse.SummaryReader(
        tmp_path / "tb_logs" / "my_base_toy_model", pivot=True
    )

    # learning rate only gets logged per epoch.
    # 2 steps per epoch, [::2] gets every 2nd element, so the value per epoch.
    backbone_lr_at_epoch = reader.scalars["lr-Adam/backbone"].tolist()[::2]
    upsampling_lr_at_epoch = reader.scalars["lr-Adam/head"].tolist()[::2]

    # Pin some values I saw on tensorboard. Not perfect, but a good change detector
    # and illustrative of the difference between epoch and step-wise unfreezing.

    # Unfreezes to upsampling_lr
    assert math.isclose(backbone_lr_at_epoch[2], 1e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[4], 2.25e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[8], 0.001, abs_tol=1e-9)
    assert math.isclose(upsampling_lr_at_epoch[8], 0.001, abs_tol=1e-9)

    # Upsampling lr annealing (`milestones`)
    assert math.isclose(upsampling_lr_at_epoch[9], 5e-4, abs_tol=1e-9)
    assert math.isclose(upsampling_lr_at_epoch[10], 5e-4, abs_tol=1e-9)
    assert math.isclose(upsampling_lr_at_epoch[11], 2.5e-4, abs_tol=1e-9)

    # Backbone also affected by lr annealing.
    assert math.isclose(backbone_lr_at_epoch[9], 5e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[10], 5e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[11], 2.5e-4, abs_tol=1e-9)


def test_steps(cfg: DictConfig, tmp_path: Path):
    # Setup so that it trains for 3 steps (2nd epoch is partial).
    # Then we'll assert that it actually trained that much.
    cfg = _get_base_cfg(cfg)
    cfg = OmegaConf.merge(
        cfg,
        {
            "training": {
                "min_steps": 13,
                "max_steps": 13,
                "unfreezing_step": 1,
                "lr_scheduler_params": {
                    "multisteplr": {
                        "milestone_steps": [9, 11],
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

    # learning rate only gets logged per epoch.
    # 2 steps per epoch, [::2] gets every 2nd element, so the value per epoch.
    backbone_lr_at_epoch = reader.scalars["lr-Adam/backbone"].tolist()[::2]
    upsampling_lr_at_epoch = reader.scalars["lr-Adam/head"].tolist()[::2]

    # Pin some values I saw on tensorboard. Not perfect, but a good change detector
    # and illustrative of the difference between epoch and step-wise unfreezing.

    # Unfreezes to upsampling_lr
    assert math.isclose(backbone_lr_at_epoch[1], 1e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[2], 2.25e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[4], 0.001, abs_tol=1e-9)
    assert math.isclose(upsampling_lr_at_epoch[4], 0.001, abs_tol=1e-9)

    # Upsampling lr annealing (`milestones`)
    assert math.isclose(upsampling_lr_at_epoch[5], 5e-4, abs_tol=1e-9)
    assert math.isclose(upsampling_lr_at_epoch[6], 2.5e-4, abs_tol=1e-9)

    # Backbone also affected by lr annealing.
    assert math.isclose(backbone_lr_at_epoch[5], 5e-4, abs_tol=1e-9)
    assert math.isclose(backbone_lr_at_epoch[6], 2.5e-4, abs_tol=1e-9)
