import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback


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

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
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
