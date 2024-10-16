from lightning_pose.callbacks import UnfreezeBackbone


def test_unfreeze_backbone():
    unfreeze_backbone = UnfreezeBackbone(2, initial_ratio=0.1, epoch_ratio=1.5)

    # Test unfreezing at epoch 2.
    assert unfreeze_backbone._get_backbone_lr(0, 1e-3) == 0.0
    assert unfreeze_backbone._get_backbone_lr(1, 1e-3) == 0.0
    assert (
        unfreeze_backbone._get_backbone_lr(2, 1e-3) == 1e-3 * 0.1
    )  # upsampling_lr * initial_ratio

    # Test warming up.
    # We thawed at upsampling_lr = 1e-3. Henceforth, backbone_lr should be
    # agnostic to changes in upsampling_lr so long as we are not fully
    # "warmed up".
    assert unfreeze_backbone._get_backbone_lr(3, 1e-3) == 1e-3 * 0.1 * 1.5
    assert unfreeze_backbone._get_backbone_lr(3, 1.5e-3) == 1e-3 * 0.1 * 1.5

    assert unfreeze_backbone._get_backbone_lr(4, 1e-3) == 1e-4 * 1.5 * 1.5
    assert unfreeze_backbone._get_backbone_lr(4, 1.5e-3) == 1e-4 * 1.5 * 1.5

    # Once we hit upsampling_lr, set the _warmed_up bit to stop this callback
    # from setting backbone lr in the future, allowing the normal scheduler to take over.
    assert not unfreeze_backbone._warmed_up
    # current_epoch set to some high value to trigger "warmed up" condition
    assert unfreeze_backbone._get_backbone_lr(15, 1e-3) == 1e-3
    assert unfreeze_backbone._warmed_up
