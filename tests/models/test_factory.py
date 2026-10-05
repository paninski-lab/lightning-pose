"""Tests for models/factory.py — get_model and get_model_class."""

import copy
import logging

import pytest
import torch
from omegaconf import OmegaConf

from lightning_pose.data import get_data_module
from lightning_pose.losses import get_loss_factories
from lightning_pose.models import get_model
from lightning_pose.models.factory import get_model_class, resolve_head_groups
from lightning_pose.models.heads.heatmap import BlockLayerNorm2d, LayerNorm2d, MaskedConv2d


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


class TestGetModelRemovedOptions:
    """Removed config options fail loudly instead of training something else."""

    def test_get_model_head_freeze_removed(self, cfg, base_dataset):
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.model.backbone = 'resnet18'
        cfg_tmp.model.backbone_pretrained = False
        cfg_tmp.model.head_freeze_keypoints = [cfg_tmp.data.keypoint_names[0]]
        data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        loss_factories = get_loss_factories(cfg_tmp, data_module=data_module)

        with pytest.raises(ValueError, match='head_freeze_keypoints was removed'):
            get_model(cfg_tmp, data_module=data_module, loss_factories=loss_factories)


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


class TestGetModelClassHeadMode:
    """Test the head_mode argument of get_model_class."""

    def test_get_model_class_head_mode_per_dataset(self):
        """per_dataset selects MultiHeadHeatmapTracker for supervised heatmap models."""
        from lightning_pose.models import MultiHeadHeatmapTracker
        assert (
            get_model_class('heatmap', semi_supervised=False, head_mode='per_dataset')
            is MultiHeadHeatmapTracker
        )

    def test_get_model_class_head_mode_shared_default(self):
        """shared (the default) keeps the stock class."""
        from lightning_pose.models import HeatmapTracker
        model_class = get_model_class('heatmap', semi_supervised=False, head_mode='shared')
        assert model_class is HeatmapTracker

    def test_get_model_class_head_mode_per_dataset_semi_supervised_raises(self):
        """Per-dataset heads cannot train on unlabeled video (no dataset id)."""
        with pytest.raises(NotImplementedError, match='carry no dataset id'):
            get_model_class('heatmap', semi_supervised=True, head_mode='per_dataset')

    def test_get_model_class_head_mode_dataset_token_removed(self):
        """The removed dataset-token mode fails loudly and names the tag that keeps it."""
        with pytest.raises(ValueError, match='pre-cleanup-2026-10-02'):
            get_model_class('heatmap', semi_supervised=False, head_mode='dataset_token')

    def test_get_model_class_head_mode_unknown_raises(self):
        """Any other head_mode is rejected."""
        with pytest.raises(ValueError, match="'shared' or 'per_dataset'"):
            get_model_class('heatmap', semi_supervised=False, head_mode='bogus')


# ── keypoint-group heads ──────────────────────────────────────────────────────

# toy-config (mirror-mouse-example) keypoints in three groups, mapping order = block order
TOY_HEAD_GROUPS = {
    'paw': [
        'paw1LH_top', 'paw2LF_top', 'paw3RF_top', 'paw4RH_top',
        'paw1LH_bot', 'paw2LF_bot', 'paw3RF_bot', 'paw4RH_bot',
    ],
    'tail': ['tailBase_top', 'tailMid_top', 'tailBase_bot', 'tailMid_bot'],
    'other': ['nose_top', 'obs_top', 'nose_bot', 'obsHigh_bot', 'obsLow_bot'],
}


# the canonical mouse-pose vocabulary (44 keypoints, mouse-pose configs/model_zoominout.yaml order)
# and the eye / snout / ear / paw groups of the head-groups experiment
MOUSE_HEAD_GROUPS = {
    'eye': [
        'eye_back_left', 'eye_back_right', 'eye_bottom_left', 'eye_bottom_right',
        'eye_front_left', 'eye_front_right', 'eye_top_left', 'eye_top_right',
        'pupil_center_left', 'pupil_center_right',
    ],
    'snout': [
        'upperlip_right', 'upperlip_left', 'tongue_end_right', 'tongue_end_left', 'nose_tip',
        'nose_top', 'nose_bottom', 'pad_center', 'mouth', 'lowerlip', 'tongue_tip',
        'tongue_center', 'pad_top_left', 'pad_top_right', 'pad_side_left', 'pad_side_right',
    ],
    'ear': [
        'ear_top_left', 'ear_top_right', 'ear_tip_left', 'ear_tip_right', 'ear_bottom_left',
        'ear_bottom_right', 'ear_base_left', 'ear_base_right',
    ],
    'paw': [
        'wrist_left', 'wrist_right', 'd1_tip_left', 'd1_tip_right', 'd2_tip_left',
        'd2_tip_right', 'd3_tip_left', 'd3_tip_right', 'd4_tip_left', 'd4_tip_right',
    ],
}
MOUSE_KEYPOINT_NAMES = [name for names in MOUSE_HEAD_GROUPS.values() for name in names]


def _head_groups_cfg(cfg, **model_keys):
    """Supervised resnet18 heatmap cfg (no download) with extra ``model`` keys."""
    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.losses_to_use = []
    cfg_tmp.model.model_type = 'heatmap'
    cfg_tmp.model.backbone = 'resnet18'
    cfg_tmp.model.backbone_pretrained = False
    for key, value in model_keys.items():
        cfg_tmp.model[key] = value
    return cfg_tmp


def _build_without_data(cfg_tmp):
    """Inference-style construction: no data module, no loss factories."""
    return get_model(
        cfg_tmp, data_module=None, loss_factories={'supervised': None, 'unsupervised': None},
    )


class TestResolveHeadGroups:
    """Test the resolve_head_groups function."""

    names = ['a_left', 'a_right', 'b', 'c_left', 'c_right']

    def _resolve(self, head_groups, hidden=8, shared=None, head_mode='shared'):
        return resolve_head_groups(
            head_groups=head_groups, keypoint_names=self.names, hidden_channels=hidden,
            shared_channels=shared, head_mode=head_mode,
        )

    def test_resolve_head_groups_indices_follow_mapping_order(self):
        groups = {'c': ['c_right', 'c_left'], 'a': ['a_left', 'a_right', 'b']}
        assert self._resolve(groups) == [1, 1, 1, 0, 0]

    def test_resolve_head_groups_accepts_omegaconf(self):
        groups = OmegaConf.create({'a': ['a_left', 'a_right', 'b'], 'c': ['c_left', 'c_right']})
        assert self._resolve(groups, hidden=10, shared=2) == [0, 0, 0, 1, 1]

    def test_resolve_head_groups_none(self):
        assert self._resolve(None, hidden=None) is None

    @pytest.mark.parametrize(
        ('head_groups', 'kwargs', 'match'),
        [
            ({'a': ['a_left', 'a_right', 'b', 'nope'], 'c': ['c_left', 'c_right']}, {},
             "unknown keypoint 'nope' in group 'a'"),
            ({'a': ['a_left', 'a_right', 'b'], 'c': ['c_left', 'c_right', 'b']}, {},
             "keypoint 'b' is in group 'a' and in group 'c'"),
            ({'a': ['a_left', 'a_right'], 'c': ['c_left', 'c_right']}, {},
             r"keypoints in no group: \['b'\]"),
            ({'a': ['a_left', 'a_right', 'b'], 'c': []}, {},
             "group 'c' must be a non-empty list"),
            ({'a': 'a_left'}, {}, "group 'a' must be a non-empty list"),
            ({}, {}, 'non-empty mapping'),
            ([['a_left']], {}, 'non-empty mapping'),
            ({'a': ['a_left', 'a_right', 'b', 'c_left', 'c_right']}, {'hidden': None},
             'model.head_groups requires model.head_hidden_channels'),
            ({'a': ['a_left', 'a_right', 'b', 'c_left', 'c_right']}, {'head_mode': 'per_dataset'},
             'model.head_groups is implemented for the shared head only'),
            ({'a': ['a_left', 'a_right', 'b'], 'c': ['c_left', 'c_right']}, {'shared': 8},
             'model.head_shared_channels must be in'),
            ({'a': ['a_left', 'a_right', 'b'], 'c': ['c_left', 'c_right']}, {'shared': -2},
             'model.head_shared_channels must be in'),
            ({'a': ['a_left', 'a_right', 'b'], 'c': ['c_left', 'c_right']}, {'shared': 1},
             'does not split equally into 2 model.head_groups'),
            (None, {'shared': 4}, 'model.head_shared_channels requires model.head_groups'),
            (None, {'shared': 0}, 'model.head_shared_channels requires model.head_groups'),
        ],
    )
    def test_resolve_head_groups_invalid_raises(self, head_groups, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._resolve(head_groups, **kwargs)


class TestGetModelHeadGroups:
    """Test get_model with model.head_groups / model.head_shared_channels."""

    def test_get_model_head_groups_builds_grouped_head(self, cfg, caplog):
        cfg_tmp = _head_groups_cfg(
            cfg, head_hidden_channels=16, head_shared_channels=4,
            head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
        )

        with caplog.at_level(logging.INFO, logger='lightning_pose.models.factory'):
            model = _build_without_data(cfg_tmp)

        names = list(cfg_tmp.data.keypoint_names)
        expected = [
            next(i for i, g in enumerate(TOY_HEAD_GROUPS.values()) if n in g) for n in names
        ]
        assert model.head.groups == expected
        assert model.hparams['head_groups'] == expected
        assert model.hparams['head_shared_channels'] == 4
        assert isinstance(model.head.upsampling_layers[-3], BlockLayerNorm2d)
        assert model.head.upsampling_layers[-3].block_sizes == [4, 4, 4, 4]
        assert isinstance(model.head.upsampling_layers[-1], MaskedConv2d)
        assert 'head groups (3): paw=8, tail=4, other=5 keypoints; 4 hidden channels per ' \
            'group block, 4 shared channels (H = 16)' in caplog.text

    def test_get_model_without_head_group_keys_builds_plain_nonlinear_head(self, cfg):
        model = _build_without_data(_head_groups_cfg(cfg, head_hidden_channels=16))

        assert model.head.groups is None
        assert model.hparams['head_groups'] is None
        assert model.hparams['head_shared_channels'] == 0
        assert type(model.head.upsampling_layers[-3]) is LayerNorm2d
        assert type(model.head.upsampling_layers[-1]) is torch.nn.Conv2d

    def test_get_model_head_groups_same_initial_tensors_as_nonlinear(self, cfg):
        plain = _build_without_data(_head_groups_cfg(cfg, head_hidden_channels=16))
        grouped = _build_without_data(_head_groups_cfg(
            cfg, head_hidden_channels=16, head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
            head_shared_channels=4,
        ))

        state_plain, state_grouped = plain.state_dict(), grouped.state_dict()
        assert list(state_plain) == list(state_grouped)
        mask = grouped.head.upsampling_layers[-1].mask
        key_readout = f'head.upsampling_layers.{len(plain.head.upsampling_layers) - 1}.weight'
        for key in state_plain:
            expected = state_plain[key] * mask if key == key_readout else state_plain[key]
            assert torch.equal(state_grouped[key], expected), key

    def test_get_model_head_group_fan_in_gain(self, cfg):
        model = _build_without_data(_head_groups_cfg(
            cfg, head_hidden_channels=16, head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
            head_shared_channels=4, head_group_fan_in_gain=True,
        ))

        assert model.hparams['head_group_fan_in_gain'] is True
        assert torch.equal(
            model.head.upsampling_layers[-1].gain.flatten(),
            torch.full((model.num_keypoints,), 16 / 8),
        )

    def test_get_model_head_group_fan_in_gain_without_groups_raises(self, cfg):
        cfg_tmp = _head_groups_cfg(cfg, head_hidden_channels=16, head_group_fan_in_gain=True)
        with pytest.raises(ValueError, match='head_group_fan_in_gain requires model.head_groups'):
            _build_without_data(cfg_tmp)

    def test_get_model_head_groups_per_dataset_raises(self, cfg):
        cfg_tmp = _head_groups_cfg(
            cfg, head_hidden_channels=16, head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
            head_mode='per_dataset',
        )
        cfg_tmp.data.dataset_names = ['a', 'b']
        with pytest.raises(ValueError, match='shared head only'):
            _build_without_data(cfg_tmp)

    def test_get_model_head_groups_other_model_type_raises(self, cfg):
        cfg_tmp = _head_groups_cfg(
            cfg, head_hidden_channels=16, head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
        )
        cfg_tmp.model.model_type = 'regression'
        with pytest.raises(ValueError, match='model.model_type=heatmap only'):
            _build_without_data(cfg_tmp)

    def test_get_model_head_groups_without_hidden_channels_raises(self, cfg):
        cfg_tmp = _head_groups_cfg(cfg, head_groups=copy.deepcopy(TOY_HEAD_GROUPS))
        with pytest.raises(ValueError, match='requires model.head_hidden_channels'):
            _build_without_data(cfg_tmp)

    def test_get_model_head_shared_channels_without_groups_raises(self, cfg):
        cfg_tmp = _head_groups_cfg(cfg, head_hidden_channels=16, head_shared_channels=4)
        with pytest.raises(ValueError, match='requires model.head_groups'):
            _build_without_data(cfg_tmp)

    def test_get_model_head_groups_unknown_keypoint_raises(self, cfg):
        head_groups = copy.deepcopy(TOY_HEAD_GROUPS)
        head_groups['tail'].append('ear_top_left')
        cfg_tmp = _head_groups_cfg(cfg, head_hidden_channels=15, head_groups=head_groups)
        with pytest.raises(ValueError, match="unknown keypoint 'ear_top_left'"):
            _build_without_data(cfg_tmp)

    @pytest.mark.parametrize(
        ('shared', 'block_sizes'), [(0, [64, 64, 64, 64]), (128, [128, 32, 32, 32, 32])],
    )
    def test_get_model_head_groups_g1_g2_real_keypoints_train_step(self, cfg, shared, block_sizes):
        cfg_tmp = _head_groups_cfg(
            cfg, head_hidden_channels=256, head_shared_channels=shared,
            head_groups=copy.deepcopy(MOUSE_HEAD_GROUPS),
        )
        cfg_tmp.data.keypoint_names = list(MOUSE_KEYPOINT_NAMES)
        cfg_tmp.data.num_keypoints = len(MOUSE_KEYPOINT_NAMES)
        model = _build_without_data(cfg_tmp)
        readout = model.head.upsampling_layers[-1]
        optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

        heatmaps = model(torch.randn(2, 3, 128, 128))
        (heatmaps * torch.rand_like(heatmaps)).sum().backward()
        optimizer.step()

        assert heatmaps.shape == (2, 44, 32, 32)
        assert model.head.upsampling_layers[-3].block_sizes == block_sizes
        assert torch.all(readout.weight.grad[readout.mask == 0] == 0)
        assert torch.all(readout.weight[readout.mask == 0] == 0)
        # every keypoint reads its group's block plus the shared block
        assert readout.mask.sum(dim=(1, 2, 3)).unique().tolist() == [shared + block_sizes[-1]]

    def test_get_model_head_groups_with_data_module(self, cfg, heatmap_dataset):
        cfg_tmp = _head_groups_cfg(
            cfg, head_hidden_channels=15, head_groups=copy.deepcopy(TOY_HEAD_GROUPS),
        )
        data_module = get_data_module(cfg_tmp, dataset=heatmap_dataset, video_dir=None)
        loss_factories = get_loss_factories(cfg_tmp, data_module=data_module)

        model = get_model(cfg_tmp, data_module=data_module, loss_factories=loss_factories)

        assert model.head.upsampling_layers[-3].block_sizes == [5, 5, 5]
        images = torch.randn(2, 3, 128, 128)
        assert model(images).shape == (2, 17, 32, 32)
