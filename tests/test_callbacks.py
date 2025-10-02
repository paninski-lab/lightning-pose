import pytest
import torch

from lightning_pose.callbacks import PatchMasker, PatchMasking, UnfreezeBackbone


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


class TestPatchMasking:
    """Test PatchMasking callback curriculum schedule for both enabled and disabled states."""

    @pytest.fixture
    def patch_masking_enabled(self):

        patch_mask_config_enabled = {
            "init_step": 100,
            "final_step": 500,
            "init_ratio": 0.1,
            "final_ratio": 0.5,
        }

        patch_masking_enabled = PatchMasking(
            patch_mask_config=patch_mask_config_enabled,
            patch_seed=42,
        )

        return patch_masking_enabled

    @pytest.fixture
    def patch_masking_disabled(self):

        patch_mask_config_disabled = {
            "init_step": 100,
            "final_step": 500,
            "init_ratio": 0.1,
            "final_ratio": 0.0  # Disabled
        }

        patch_masking_disabled = PatchMasking(
            patch_mask_config=patch_mask_config_disabled,
            patch_seed=42,
        )

        return patch_masking_disabled

    def test_initialization_enabled(self, patch_masking_enabled):
        """Test initialization - access through curriculum_masking attribute"""
        assert patch_masking_enabled.curriculum_masking.use_patch_masking is True
        assert patch_masking_enabled.curriculum_masking.patch_seed == 42

    def test_schedule_at_different_steps(self, patch_masking_enabled):
        """Test curriculum schedule at different steps"""

        # before masking starts
        schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(50)
        assert schedule_info['mask_ratio'] == 0.0
        assert schedule_info['curriculum_progress'] == "0.0%"

        # at masking start
        schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(100)
        assert schedule_info['mask_ratio'] == 0.1
        assert schedule_info['curriculum_progress'] == "0.0%"

        # mid-way through curriculum
        schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(300)
        expected_ratio = 0.1 + (300 - 100) / (500 - 100) * (0.5 - 0.1)  # 0.3
        assert abs(schedule_info['mask_ratio'] - expected_ratio) < 1e-6
        assert schedule_info['curriculum_progress'] == "50.0%"

        # at final step
        schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(500)
        assert schedule_info['mask_ratio'] == 0.5
        assert schedule_info['curriculum_progress'] == "100.0%"

        # after final step
        schedule_info = patch_masking_enabled.curriculum_masking.get_training_schedule_info(700)
        assert schedule_info['mask_ratio'] == 0.5
        assert schedule_info['curriculum_progress'] == "100.0%"

    def test_should_start_patch_masking(self, patch_masking_enabled):
        assert not patch_masking_enabled.curriculum_masking.should_start_patch_masking(99)
        assert patch_masking_enabled.curriculum_masking.should_start_patch_masking(100)
        assert not patch_masking_enabled.curriculum_masking.should_start_patch_masking(101)

    def test_initialization_disabled(self, patch_masking_disabled):
        """Test that curriculum masking is properly disabled"""
        assert patch_masking_disabled.curriculum_masking.use_patch_masking is False

    def test_schedule_disabled(self, patch_masking_disabled):
        """Test that schedule info returns default values when disabled"""
        schedule_info = patch_masking_disabled.curriculum_masking.get_training_schedule_info(300)
        assert schedule_info['mask_ratio'] == 0.0
        assert schedule_info['curriculum_progress'] == "0.0%"
        assert schedule_info['steps_to_max_masking'] == 0
        assert schedule_info['steps_to_patch_masking'] == 0


class TestPatchMasker:

    @pytest.fixture
    def basic_masker(self):
        """Create a basic PatchMasker instance for testing."""
        return PatchMasker(
            patch_mask_config={
                "init_step": 100,
                "final_step": 1000,
                "init_ratio": 0.1,
                "final_ratio": 0.5,
            },
            patch_seed=42,
        )

    def test_initialization_with_config(self, basic_masker):
        """Test that PatchMasker initializes with correct attributes."""
        assert basic_masker.patch_seed == 42
        assert basic_masker.patch_init_step == 100
        assert basic_masker.patch_final_step == 1000
        assert basic_masker.patch_init_ratio == 0.1
        assert basic_masker.patch_final_ratio == 0.5
        assert basic_masker.use_patch_masking is True

    def test_initialization_without_config(self):
        """Test that PatchMasker uses default values when config is None."""
        masker = PatchMasker()
        assert masker.patch_seed == 0
        assert masker.patch_init_step == 700
        assert masker.patch_final_step == 5000
        assert masker.patch_init_ratio == 0.1
        assert masker.patch_final_ratio == 0.5

    def test_initialization_empty_config(self):
        """Test that PatchMasker uses default values with empty config."""
        masker = PatchMasker(patch_mask_config={}, patch_seed=0)
        assert masker.patch_init_step == 700
        assert masker.patch_final_step == 5000
        assert masker.patch_init_ratio == 0.1
        assert masker.patch_final_ratio == 0.5

    def test_masking_disabled_when_final_ratio_zero(self):
        """Test that masking is disabled when final_ratio is 0."""
        masker = PatchMasker(
            patch_mask_config={"final_ratio": 0.0},
            patch_seed=0,
        )
        assert masker.use_patch_masking is False

    def test_apply_masking_disabled(self):
        """Test apply_masking returns original images when masking disabled."""
        masker = PatchMasker(
            patch_mask_config={"final_ratio": 0.0},
            patch_seed=0,
        )
        batch_size, num_views = 2, 3
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images, mask = masker.apply_masking(images, training_step=500)

        assert torch.equal(masked_images, images)
        assert mask.shape == (batch_size, num_views)
        assert torch.all(mask == 1)

    def test_apply_patch_masking_not_training(self, basic_masker):
        """Test that no masking is applied when not in training mode."""
        batch_size, num_views = 2, 3
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images, patch_mask = basic_masker.apply_patch_masking(
            images, training_step=500, is_training=False,
        )

        assert torch.equal(masked_images, images)
        assert torch.all(patch_mask == 1)

    def test_apply_patch_masking_before_init_step(self, basic_masker):
        """Test that no masking is applied before init_step."""
        batch_size, num_views = 2, 3
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images, patch_mask = basic_masker.apply_patch_masking(
            images, training_step=50, is_training=True,
        )

        assert torch.equal(masked_images, images)
        assert torch.all(patch_mask == 1)

    def test_apply_patch_masking_at_init_step(self, basic_masker):
        """Test that masking starts at init_step."""
        batch_size, num_views = 2, 3
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images, patch_mask = basic_masker.apply_patch_masking(
            images, training_step=100, is_training=True
        )

        assert not torch.equal(masked_images, images)
        assert torch.any(patch_mask == 0)

    def test_apply_patch_masking_at_final_step(self, basic_masker):
        """Test that masking reaches maximum at final_step."""
        batch_size, num_views = 1, 2
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images, patch_mask = basic_masker.apply_patch_masking(
            images, training_step=1000, is_training=True,
        )

        num_patches = (224 // 16) * (224 // 16)
        expected_masked = int(0.5 * num_patches)

        for b in range(batch_size):
            for v in range(num_views):
                num_masked = (patch_mask[b, v] == 0).sum().item()
                assert num_masked == expected_masked

    def test_deterministic_masking(self, basic_masker):
        """Test that masking is deterministic with same seed and step."""
        batch_size, num_views = 2, 2
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        masked_images_1, patch_mask_1 = basic_masker.apply_patch_masking(
            images.clone(), training_step=500, is_training=True,
        )
        masked_images_2, patch_mask_2 = basic_masker.apply_patch_masking(
            images.clone(), training_step=500, is_training=True,
        )

        assert torch.equal(masked_images_1, masked_images_2)
        assert torch.equal(patch_mask_1, patch_mask_2)

    def test_different_masking_per_step(self, basic_masker):
        """Test that different steps produce different masks."""
        batch_size, num_views = 2, 2
        images = torch.randn(batch_size, num_views, 3, 224, 224)

        _, patch_mask_1 = basic_masker.apply_patch_masking(
            images.clone(), training_step=500, is_training=True
        )
        _, patch_mask_2 = basic_masker.apply_patch_masking(
            images.clone(), training_step=501, is_training=True
        )

        assert not torch.equal(patch_mask_1, patch_mask_2)

    def test_get_training_schedule_info_before_init(self, basic_masker):
        """Test training schedule info before init_step."""
        info = basic_masker.get_training_schedule_info(current_step=50)

        assert info["step"] == 50
        assert info["mask_ratio"] == 0.0
        assert info["curriculum_progress"] == "0.0%"
        assert info["steps_to_patch_masking"] == 50
        assert info["steps_to_max_masking"] == 950

    def test_get_training_schedule_info_at_init(self, basic_masker):
        """Test training schedule info at init_step."""
        info = basic_masker.get_training_schedule_info(current_step=100)

        assert info["step"] == 100
        assert info["mask_ratio"] == 0.1
        assert info["curriculum_progress"] == "0.0%"
        assert info["steps_to_patch_masking"] == 0
        assert info["steps_to_max_masking"] == 900

    def test_get_training_schedule_info_at_final(self, basic_masker):
        """Test training schedule info at final_step."""
        info = basic_masker.get_training_schedule_info(current_step=1000)

        assert info["step"] == 1000
        assert info["mask_ratio"] == 0.5
        assert info["curriculum_progress"] == "100.0%"
        assert info["steps_to_patch_masking"] == 0
        assert info["steps_to_max_masking"] == 0

    def test_get_training_schedule_info_midway(self, basic_masker):
        """Test training schedule info midway through curriculum."""
        info = basic_masker.get_training_schedule_info(current_step=550)

        assert info["step"] == 550
        assert 0.1 < info["mask_ratio"] < 0.5
        assert info["steps_to_patch_masking"] == 0
        assert info["steps_to_max_masking"] == 450

    def test_should_start_patch_masking(self, basic_masker):
        """Test should_start_patch_masking returns correct values."""
        assert basic_masker.should_start_patch_masking(99) is False
        assert basic_masker.should_start_patch_masking(100) is True
        assert basic_masker.should_start_patch_masking(101) is False

    def test_should_start_patch_masking_disabled(self):
        """Test should_start_patch_masking when masking is disabled."""
        masker = PatchMasker(
            patch_mask_config={"final_ratio": 0.0},
            patch_seed=0,
        )
        assert masker.should_start_patch_masking(700) is False

    def test_masked_patches_are_zeroed(self, basic_masker):
        """Test that masked patches have zero values."""
        batch_size, num_views = 1, 1
        images = torch.ones(batch_size, num_views, 3, 224, 224)

        masked_images, patch_mask = basic_masker.apply_patch_masking(
            images, training_step=500, is_training=True,
        )

        patch_size = 16
        num_patches_w = 224 // patch_size

        for b in range(batch_size):
            for v in range(num_views):
                masked_patch_indices = (patch_mask[b, v] == 0).nonzero(as_tuple=True)[0]
                for patch_idx in masked_patch_indices:
                    patch_h = (patch_idx // num_patches_w) * patch_size
                    patch_w = (patch_idx % num_patches_w) * patch_size
                    patch_region = masked_images[
                        b, v, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size
                    ]
                    assert torch.all(patch_region == 0)
