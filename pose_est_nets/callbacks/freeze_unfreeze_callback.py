"""Training callbacks for pytorch lightning training."""

from pytorch_lightning.callbacks.finetuning import BaseFinetuning


class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    """Callback to unfreeze backbone feature extractor at particular epoch."""

    def __init__(self, unfreeze_at_epoch: int = 10) -> None:
        """FeatureExtractorFreezeUnfreeze constructor.

        Args:
            unfreeze_at_epoch: epoch at which to unfreeze feature extractor
                weights

        """
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing ``backbone``
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.backbone,
                optimizer=optimizer,
                train_bn=True,
            )
