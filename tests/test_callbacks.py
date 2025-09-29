from lightning_pose.callbacks import UnfreezeBackbone, PatchMasking


def test_unfreeze_backbone_epoch():
    unfreeze_backbone = UnfreezeBackbone(unfreeze_epoch=2, initial_ratio=0.1, warm_up_ratio=1.5)

    # Test unfreezing at epoch 2.
    assert unfreeze_backbone._get_backbone_lr(None, 0, 1e-3) == 0.0
    assert unfreeze_backbone._get_backbone_lr(None, 1, 1e-3) == 0.0
    assert (
        unfreeze_backbone._get_backbone_lr(None, 2, 1e-3) == 1e-3 * 0.1
    )  # upsampling_lr * initial_ratio

    # Test warming up.
    # We thawed at upsampling_lr = 1e-3. Henceforth, backbone_lr should be
    # agnostic to changes in upsampling_lr so long as we are not fully
    # "warmed up".
    assert unfreeze_backbone._get_backbone_lr(None, 3, 1e-3) == 1e-3 * 0.1 * 1.5
    assert unfreeze_backbone._get_backbone_lr(None, 3, 1.5e-3) == 1e-3 * 0.1 * 1.5

    assert unfreeze_backbone._get_backbone_lr(None, 4, 1e-3) == 1e-4 * 1.5 * 1.5
    assert unfreeze_backbone._get_backbone_lr(None, 4, 1.5e-3) == 1e-4 * 1.5 * 1.5

    # Once we hit upsampling_lr, set the _warmed_up bit to stop this callback
    # from setting backbone lr in the future, allowing the normal scheduler to take over.
    assert not unfreeze_backbone._warmed_up
    # current_epoch set to some high value to trigger "warmed up" condition
    assert unfreeze_backbone._get_backbone_lr(None, 15, 1e-3) == 1e-3
    assert unfreeze_backbone._warmed_up


def test_patch_masking_callback():
    """Test PatchMasking callback curriculum learning schedule for both enabled and disabled states."""

    # Test 1: Patch masking enabled
    patch_mask_config_enabled = {
        "init_step": 100,
        "final_step": 500,
        "init_ratio": 0.1,
        "final_ratio": 0.5
    }

    patch_masking_enabled = PatchMasking(
        patch_mask_config=patch_mask_config_enabled,
        num_views=2,
        patch_seed=42
    )

    # Test initialization - access through curriculum_masking attribute
    assert patch_masking_enabled.curriculum_masking.use_patch_masking == True
    assert patch_masking_enabled.curriculum_masking.num_views == 2
    assert patch_masking_enabled.curriculum_masking.patch_seed == 42

    # Test curriculum schedule at different steps
    # Before masking starts
    schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(50)
    assert schedule_info['mask_ratio'] == 0.0
    assert schedule_info['curriculum_progress'] == "0.0%"

    # At masking start
    schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(100)
    assert schedule_info['mask_ratio'] == 0.1
    assert schedule_info['curriculum_progress'] == "0.0%"

    # Mid-way through curriculum
    schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(300)
    expected_ratio = 0.1 + (300 - 100) / (500 - 100) * (0.5 - 0.1)  # 0.3
    assert abs(schedule_info['mask_ratio'] - expected_ratio) < 1e-6
    assert schedule_info['curriculum_progress'] == "50.0%"

    # At final step
    schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(500)
    assert schedule_info['mask_ratio'] == 0.5
    assert schedule_info['curriculum_progress'] == "100.0%"

    # After final step
    schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(700)
    assert schedule_info['mask_ratio'] == 0.5
    assert schedule_info['curriculum_progress'] == "100.0%"

    # Test should_start_patch_masking
    assert not patch_masking_enabled.curriculum_masking.should_start_patch_masking(99)
    assert patch_masking_enabled.curriculum_masking.should_start_patch_masking(100)
    assert not patch_masking_enabled.curriculum_masking.should_start_patch_masking(101)

    # Test 2: Patch masking disabled when we set final_ratio to 0.0
    patch_mask_config_disabled = {
        "init_step": 100,
        "final_step": 500,
        "init_ratio": 0.1,
        "final_ratio": 0.0  # Disabled
    }

    patch_masking_disabled = PatchMasking(
        patch_mask_config=patch_mask_config_disabled,
        num_views=2,
        patch_seed=42
    )

    # Test that curriculum masking is properly disabled
    assert patch_masking_disabled.curriculum_masking.use_patch_masking == False

    # Test that schedule info returns default values when disabled
    schedule_info = patch_masking_disabled.curriculum_masking.get_training_schedule_info(300)
    assert schedule_info['mask_ratio'] == 0.0
    assert schedule_info['curriculum_progress'] == "0.0%"
    assert schedule_info['steps_to_max_masking'] == 0
    assert schedule_info['steps_to_patch_masking'] == 0
