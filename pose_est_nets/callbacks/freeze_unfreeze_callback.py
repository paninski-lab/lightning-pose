
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):

    def __init__(self, unfreeze_at_epoch=10):
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing ``feature_extractor``
        self.freeze(pl_module.feature_extractor)

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor,
                optimizer=optimizer,
                train_bn=True,
            )