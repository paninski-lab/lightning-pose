import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

# to ignore imports for sphix-autoapidoc
__all__ = [
    "AnnealWeight",
]


class AnnealWeight(Callback):
    """Callback to change weight value during training."""

    def __init__(
        self,
        attr_name: str,
        init_val: float = 0.0,
        increase_factor: float = 0.01,
        final_val: float = 1.0,
        freeze_until_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.init_val = init_val
        self.increase_factor = increase_factor
        self.final_val = final_val
        self.freeze_until_epoch = freeze_until_epoch
        self.attr_name = attr_name

    def on_train_start(self, trainer, pl_module) -> None:
        # Dan: removed buffer; seems to complicate checkpoint loading
        # pl_module.register_buffer(self.attr_name, torch.tensor(self.init_val))
        setattr(pl_module, self.attr_name, torch.tensor(self.init_val))

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch <= self.freeze_until_epoch:
            pass
        else:
            eff_epoch: int = pl_module.current_epoch - self.freeze_until_epoch
            value: float = min(
                self.init_val + eff_epoch * self.increase_factor, self.final_val
            )
            # Dan: removed buffer; seems to complicate checkpoint loading
            # pl_module.register_buffer(self.attr_name, torch.tensor(value))
            setattr(pl_module, self.attr_name, torch.tensor(value))


class UnfreezeBackbone(Callback):
    """Callback that ramps up the backbone learning rate from 0 to `upsampling_lr` on
    `unfreeze_epoch`.

    Starts LR at `initial_ratio * upsampling_lr`. Grows lr by a factor of `epoch_ratio` per
    epoch. Once LR reaches `upsampling_lr`, keeps it in sync with `upsampling_lr`.

    Use instead of pl.callbacks.BackboneFinetuning in order to use multi-GPU (DDP). See
    lightning-ai/pytorch-lightning#20340 for context.
    """

    def __init__(
        self,
        unfreeze_epoch,
        initial_ratio=0.1,
        epoch_ratio=1.5,
    ):
        self.unfreeze_epoch = unfreeze_epoch
        self.initial_ratio = initial_ratio
        self.epoch_ratio = epoch_ratio
        self._warmed_up = False

    def on_train_epoch_start(self, trainer, pl_module):
        # This callback is only applicable to heatmap models but we
        # might encounter a RegressionModel.
        if not hasattr(pl_module, "upsampling_layers"):
            return
        
        # Once backbone_lr warms up to upsampling_lr, this callback does nothing.
        # Control of backbone lr is then the sole job of the main lr scheduler.
        if self._warmed_up:
            return

        optimizer = pl_module.optimizers()
        # Check our assumptions about param group indices
        assert optimizer.param_groups[0]["name"] == "backbone"
        assert optimizer.param_groups[1]["name"].startswith("upsampling")

        upsampling_lr = optimizer.param_groups[1]["lr"]

        optimizer.param_groups[0]["lr"] = self._get_backbone_lr(
            pl_module.current_epoch, upsampling_lr
        )

    def _get_backbone_lr(self, current_epoch, upsampling_lr):
        assert not self._warmed_up

        # Before unfreeze, learning_rate is 0.
        if current_epoch < self.unfreeze_epoch:
            return 0.0

        # On unfreeze, initialize learning rate.
        # Remember this initial value for warm up.
        if current_epoch == self.unfreeze_epoch:
            self._initial_lr = self.initial_ratio * upsampling_lr
            return self._initial_lr

        # Warm up: compute inital_ratio * epoch_ratio ** epochs_since_thaw.
        # Use stored initial_ratio rather than recomputing it since
        # upsampling_lr is subject to change via the scheduler.
        if current_epoch > self.unfreeze_epoch:
            epochs_since_thaw = current_epoch - self.unfreeze_epoch
            next_lr = min(
                self._initial_lr * self.epoch_ratio**epochs_since_thaw, upsampling_lr
            )
            if next_lr == upsampling_lr:
                self._warmed_up = True
            return next_lr
