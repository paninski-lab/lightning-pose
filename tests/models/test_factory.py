"""Tests for models/factory.py — get_model and get_model_class."""

import copy

import pytest
import torch

from lightning_pose.data import get_data_module
from lightning_pose.losses import get_loss_factories
from lightning_pose.models import get_model
from lightning_pose.models.factory import get_model_class


class TestGetModel:
    """Test the get_model function."""

    def _make_regression_cfg(self, cfg):
        """Return a minimal supervised regression cfg that avoids network downloads."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.model.model_type = 'regression'
        cfg_tmp.model.backbone = 'resnet18'
        cfg_tmp.model.backbone_pretrained = False
        return cfg_tmp

    def _build_model(self, cfg_tmp, base_dataset):
        """Create a regression model from cfg and base_dataset."""
        data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        loss_factories = get_loss_factories(cfg_tmp, data_module=data_module)
        return get_model(cfg_tmp, data_module=data_module, loss_factories=loss_factories)

    def test_get_model_loads_checkpoint_from_ckpt_file(self, cfg, base_dataset, tmp_path):
        """Loads weights from a .ckpt path directly into the model."""
        cfg_tmp = self._make_regression_cfg(cfg)
        model = self._build_model(cfg_tmp, base_dataset)

        ckpt_path = str(tmp_path / 'model.ckpt')
        torch.save({'state_dict': model.state_dict()}, ckpt_path)

        cfg_tmp.model.checkpoint = ckpt_path
        loaded = self._build_model(cfg_tmp, base_dataset)

        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items(), strict=True,
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)

    def test_get_model_loads_checkpoint_from_directory(self, cfg, base_dataset, tmp_path):
        """Globs for .ckpt inside a directory when checkpoint is a directory path."""
        cfg_tmp = self._make_regression_cfg(cfg)
        model = self._build_model(cfg_tmp, base_dataset)

        ckpt_dir = tmp_path / 'checkpoints'
        ckpt_dir.mkdir()
        torch.save({'state_dict': model.state_dict()}, ckpt_dir / 'best.ckpt')

        cfg_tmp.model.checkpoint = str(ckpt_dir)
        loaded = self._build_model(cfg_tmp, base_dataset)

        for (k1, v1), (k2, v2) in zip(
            model.state_dict().items(), loaded.state_dict().items(), strict=True,
        ):
            assert k1 == k2
            assert torch.allclose(v1, v2)

    def test_get_model_checkpoint_fallback_to_weights_only_false(
        self, cfg, base_dataset, tmp_path, mocker,
    ):
        """Falls back to weights_only=False when the initial torch.load call raises."""
        cfg_tmp = self._make_regression_cfg(cfg)
        model = self._build_model(cfg_tmp, base_dataset)

        ckpt_path = str(tmp_path / 'model.ckpt')
        torch.save({'state_dict': model.state_dict()}, ckpt_path)
        cfg_tmp.model.checkpoint = ckpt_path

        calls = []
        original_load = torch.load

        def patched_load(*args, **kwargs):
            calls.append(kwargs.get('weights_only'))
            if len(calls) == 1:
                raise Exception('cannot load')
            return original_load(*args, **kwargs)

        mocker.patch('lightning_pose.models.factory.torch.load', side_effect=patched_load)
        self._build_model(cfg_tmp, base_dataset)
        assert len(calls) == 2

    def test_get_model_checkpoint_loads_backbone_only_on_head_mismatch(
        self, cfg, base_dataset, tmp_path,
    ):
        """Falls back to backbone-only weights when load_state_dict raises RuntimeError."""
        cfg_tmp = self._make_regression_cfg(cfg)
        model = self._build_model(cfg_tmp, base_dataset)

        state_dict = model.state_dict()
        # corrupt one non-backbone key with a wrong shape to trigger RuntimeError
        non_backbone_key = next(k for k in state_dict if 'backbone' not in k)
        state_dict[non_backbone_key] = torch.zeros(1)

        ckpt_path = str(tmp_path / 'model.ckpt')
        torch.save({'state_dict': state_dict}, ckpt_path)
        cfg_tmp.model.checkpoint = ckpt_path

        # should succeed: RuntimeError triggers backbone-only fallback
        loaded = self._build_model(cfg_tmp, base_dataset)

        # backbone weights loaded from checkpoint must match
        for k, v in model.state_dict().items():
            if 'backbone' in k:
                assert torch.allclose(loaded.state_dict()[k], v), f'backbone mismatch at {k}'


class TestGetModelClass:
    """Test the get_model_class function."""

    def test_get_model_class_supervised_regression(self):
        """Returns RegressionTracker for supervised regression."""
        from lightning_pose.models import RegressionTracker
        assert get_model_class('regression', semi_supervised=False) is RegressionTracker

    def test_get_model_class_supervised_heatmap(self):
        """Returns HeatmapTracker for supervised heatmap."""
        from lightning_pose.models import HeatmapTracker
        assert get_model_class('heatmap', semi_supervised=False) is HeatmapTracker

    def test_get_model_class_supervised_heatmap_mhcrnn(self):
        """Returns HeatmapTrackerMHCRNN for supervised heatmap_mhcrnn."""
        from lightning_pose.models import HeatmapTrackerMHCRNN
        assert get_model_class('heatmap_mhcrnn', semi_supervised=False) is HeatmapTrackerMHCRNN

    def test_get_model_class_supervised_heatmap_multiview_transformer(self):
        """Returns HeatmapTrackerMultiviewTransformer for supervised multiview transformer."""
        from lightning_pose.models import HeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=False)
            is HeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised supervised model_type."""
        with pytest.raises(NotImplementedError, match='invalid model_type for a fully supervised'):
            get_model_class('unknown_type', semi_supervised=False)  # type: ignore[arg-type]

    def test_get_model_class_semi_supervised_regression(self):
        """Returns SemiSupervisedRegressionTracker for semi-supervised regression."""
        from lightning_pose.models import SemiSupervisedRegressionTracker
        assert (
            get_model_class('regression', semi_supervised=True) is SemiSupervisedRegressionTracker
        )

    def test_get_model_class_semi_supervised_heatmap(self):
        """Returns SemiSupervisedHeatmapTracker for semi-supervised heatmap."""
        from lightning_pose.models import SemiSupervisedHeatmapTracker
        assert get_model_class('heatmap', semi_supervised=True) is SemiSupervisedHeatmapTracker

    def test_get_model_class_semi_supervised_heatmap_mhcrnn(self):
        """Returns SemiSupervisedHeatmapTrackerMHCRNN for semi-supervised heatmap_mhcrnn."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN
        assert (
            get_model_class('heatmap_mhcrnn', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMHCRNN
        )

    def test_get_model_class_semi_supervised_heatmap_multiview_transformer(self):
        """Returns SemiSupervisedHeatmapTrackerMultiviewTransformer for semi-supervised variant."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_semi_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised semi-supervised model_type."""
        with pytest.raises(
            NotImplementedError, match='invalid model_type for a semi-supervised',
        ):
            get_model_class('unknown_type', semi_supervised=True)  # type: ignore[arg-type]
