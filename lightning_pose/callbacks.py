import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback

# to ignore imports for sphix-autoapidoc
__all__ = [
    "AnnealWeight",
    "UnfreezeBackbone",
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
    `unfreeze_epoch` or `unfreeze_step`.

    Starts LR at `initial_ratio * upsampling_lr`. Grows lr by a factor of `warm_up_ratio` per
    epoch or step. Once LR reaches `upsampling_lr`, keeps it in sync with `upsampling_lr`.

    Use instead of pl.callbacks.BackboneFinetuning in order to use multi-GPU (DDP). See
    lightning-ai/pytorch-lightning#20340 for context.
    """

    def __init__(
        self,
        unfreeze_epoch: int | None = None,
        unfreeze_step: int | None = None,
        initial_ratio=0.1,
        warm_up_ratio=1.5,
    ):
        assert (unfreeze_epoch is None) != (
            unfreeze_step is None
        ), "Exactly one must be provided."
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_step = unfreeze_step
        self.initial_ratio = initial_ratio
        self.warm_up_ratio = warm_up_ratio
        self._warmed_up = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):

        # Once backbone_lr warms up to upsampling_lr, this callback does nothing.
        # Control of backbone lr is then the sole job of the main lr scheduler.
        if self._warmed_up:
            return

        optimizer = pl_module.optimizers()
        # Check our assumptions about param group indices
        assert optimizer.param_groups[0]["name"] == "backbone"

        head_lr = optimizer.param_groups[1]["lr"]

        optimizer.param_groups[0]["lr"] = self._get_backbone_lr(
            pl_module.global_step, pl_module.current_epoch, head_lr
        )

    def _get_backbone_lr(self, current_step, current_epoch, upsampling_lr):
        """Returns what the backbone LR should be at this point in time.

        Args:
            Only one of `current_step` and `current_epoch` will be used.
            If self.unfreeze_epoch is not None, then we'll use `current_epoch`
            Otherwise, unfreeze_step is not None and we'll use `current_step`.
        """
        assert not self._warmed_up

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # In the code below, the variables are named in terms of epoch,
        # but the same logic applies for steps, conveniently enough.
        # So if we're in "step mode", plug in steps into epoch variables.
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        unfreeze_epoch = self.unfreeze_epoch
        if self.unfreeze_step is not None:
            unfreeze_epoch = self.unfreeze_step
            current_epoch = current_step
        # After this point, use `unfreeze_epoch` instead of `self.unfreeze_[epoch|step]`.
        # Main logic begins:

        # Before unfreeze, learning_rate is 0.
        if current_epoch < unfreeze_epoch:
            return 0.0

        # On unfreeze, initialize learning rate.
        # Remember this initial value for warm up.
        if current_epoch == unfreeze_epoch:
            self._initial_lr = self.initial_ratio * upsampling_lr
            return self._initial_lr

        # Warm up: compute inital_ratio * epoch_ratio ** epochs_since_thaw.
        # Use stored initial_ratio rather than recomputing it since
        # upsampling_lr is subject to change via the scheduler.
        if current_epoch > unfreeze_epoch:
            epochs_since_thaw = current_epoch - unfreeze_epoch
            next_lr = min(
                self._initial_lr * self.warm_up_ratio**epochs_since_thaw, upsampling_lr
            )
            if next_lr == upsampling_lr:
                self._warmed_up = True
            return next_lr
