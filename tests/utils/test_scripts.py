"""Test the scripts module.

Note that many of the functions in the scripts module are explicitly used (and therefore implicitly
tested) in conftest.py

"""

import copy
import os
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError
from PIL import Image

from lightning_pose.callbacks import (
    AnnealWeight,
    JSONTrainingProgressTracker,
    PatchMasking,
    UnfreezeBackbone,
)
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import BaseTrackingDataset
from lightning_pose.utils.scripts import (
    calculate_steps_per_epoch,
    get_callbacks,
    get_data_module,
    get_imgaug_transform,
    get_loss_factories,
    get_model,
)


class TestCalculateStepsPerEpoch:
    """Test the calculate_steps_per_epoch function."""

    def test_calculate_steps_per_epoch_supervised(self, cfg, base_dataset):
        """Test the computation of steps per epoch."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []

        # Small number of train frames
        cfg_tmp.training.train_frames = 3
        cfg_tmp.training.train_batch_size = 2
        base_data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        n_batches = calculate_steps_per_epoch(base_data_module)
        assert n_batches == 2

        # Large number of frames
        cfg_tmp.training.limit_train_batches = None
        cfg_tmp.training.train_frames = 49
        cfg_tmp.training.train_batch_size = 2
        base_data_module = get_data_module(cfg_tmp, dataset=base_dataset, video_dir=None)
        n_batches = calculate_steps_per_epoch(base_data_module)
        assert n_batches == 25  # ceil (49 / 2)

    def test_calculate_steps_per_epoch_unsupervised(self, cfg, base_dataset, toy_data_dir):
        """Test the computation of steps per epoch."""
        video_dir = os.path.join(toy_data_dir, 'videos')
        cfg_tmp = copy.deepcopy(cfg)

        # Small number of train frames - return minimum of 10
        cfg_tmp.training.train_frames = 3
        cfg_tmp.training.train_batch_size = 2
        base_data_module_combined = get_data_module(
            cfg_tmp, dataset=base_dataset, video_dir=video_dir,
        )
        n_batches = calculate_steps_per_epoch(base_data_module_combined)
        assert n_batches == 10

        # Large number of frames
        cfg_tmp.training.limit_train_batches = None
        cfg_tmp.training.train_frames = 49
        cfg_tmp.training.train_batch_size = 2
        base_data_module_combined = get_data_module(
            cfg_tmp, dataset=base_dataset, video_dir=video_dir,
        )
        n_batches = calculate_steps_per_epoch(base_data_module_combined)
        assert n_batches == 25  # ceil (49 / 2)


class TestGetImgaugTransform:
    """Test the get_imgaug_transform function."""

    def test_get_imgaug_transform_default(self, cfg, base_dataset):
        cfg_tmp = copy.deepcopy(cfg)

        idx = 0
        img_name = base_dataset.image_names[idx]
        keypoints_on_image = base_dataset.keypoints[idx]
        file_name = os.path.join(base_dataset.root_directory, img_name)
        image = Image.open(file_name).convert('RGB')

        # default pipeline: resize only
        cfg_tmp.training.imgaug = 'default'
        pipe = get_imgaug_transform(cfg_tmp)
        im_0, kps_0 = pipe(  # type: ignore[misc]
            images=np.expand_dims(np.array(image), axis=0),
            keypoints=np.expand_dims(keypoints_on_image, axis=0),
        )
        im_0 = im_0[0]
        kps_0 = kps_0[0].reshape(-1)
        assert im_0.shape[0] == image.size[1]  # PIL.Image.size is (width, height)
        assert im_0.shape[1] == image.size[0]

        # default pipeline: should be repeatable
        im_1, kps_1 = pipe(  # type: ignore[misc]
            images=np.expand_dims(np.array(image), axis=0),
            keypoints=np.expand_dims(keypoints_on_image, axis=0),
        )
        im_1 = im_1[0]
        kps_1 = kps_1[0].reshape(-1)
        assert np.allclose(im_0, im_1)
        assert np.allclose(kps_0, kps_1, equal_nan=True)

    def test_get_imgaug_transform_raises_for_unknown_preset(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.training.imgaug = 'null'
        with pytest.raises(NotImplementedError):
            get_imgaug_transform(cfg_tmp)

    def test_get_imgaug_transform_dlc(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)

        cfg_tmp.training.imgaug = 'dlc'
        pipe = get_imgaug_transform(cfg_tmp)
        pipe_str = pipe.__str__()
        assert pipe_str.find('Resize') == -1
        assert pipe_str.find('Fliplr') == -1
        assert pipe_str.find('Flipud') == -1
        assert pipe_str.find('Rot90') == -1
        assert pipe_str.find('Affine') != -1
        assert pipe_str.find('MotionBlur') != -1
        assert pipe_str.find('CoarseDropout') != -1
        assert pipe_str.find('CoarseSalt') != -1
        assert pipe_str.find('CoarsePepper') != -1
        assert pipe_str.find('ElasticTransformation') != -1
        assert pipe_str.find('AllChannelsHistogramEqualization') != -1
        assert pipe_str.find('AllChannelsCLAHE') != -1
        assert pipe_str.find('Emboss') != -1
        assert pipe_str.find('CropAndPad') != -1

    def test_get_imgaug_transform_dlc_lr(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)

        cfg_tmp.training.imgaug = 'dlc-lr'
        pipe = get_imgaug_transform(cfg_tmp)
        pipe_str = pipe.__str__()
        assert pipe_str.find('Resize') == -1
        assert pipe_str.find('Fliplr') == -1
        assert pipe_str.find('Flipud') == -1
        assert pipe_str.find('Rot90(name=UnnamedRot90') != -1
        assert pipe_str.find('param=Choice(a=[0, 2]') != -1
        assert pipe_str.find('Affine') != -1
        assert pipe_str.find('MotionBlur') != -1
        assert pipe_str.find('CoarseDropout') != -1
        assert pipe_str.find('CoarseSalt') != -1
        assert pipe_str.find('CoarsePepper') != -1
        assert pipe_str.find('ElasticTransformation') != -1
        assert pipe_str.find('AllChannelsHistogramEqualization') != -1
        assert pipe_str.find('AllChannelsCLAHE') != -1
        assert pipe_str.find('Emboss') != -1
        assert pipe_str.find('CropAndPad') != -1

    def test_get_imgaug_transform_dlc_top_down(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)

        cfg_tmp.training.imgaug = 'dlc-top-down'
        pipe = get_imgaug_transform(cfg_tmp)
        pipe_str = pipe.__str__()
        assert pipe_str.find('Resize') == -1
        assert pipe_str.find('Fliplr') == -1
        assert pipe_str.find('Flipud') == -1
        assert pipe_str.find('Rot90(name=UnnamedRot90') != -1
        assert pipe_str.find('param=Choice(a=[0, 1, 2, 3]') != -1
        assert pipe_str.find('Affine') != -1
        assert pipe_str.find('MotionBlur') != -1
        assert pipe_str.find('CoarseDropout') != -1
        assert pipe_str.find('CoarseSalt') != -1
        assert pipe_str.find('CoarsePepper') != -1
        assert pipe_str.find('ElasticTransformation') != -1
        assert pipe_str.find('AllChannelsHistogramEqualization') != -1
        assert pipe_str.find('AllChannelsCLAHE') != -1
        assert pipe_str.find('Emboss') != -1
        assert pipe_str.find('CropAndPad') != -1

    def test_get_imgaug_transform_dlc_multiview(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)

        cfg_tmp.training.imgaug = 'dlc-mv'
        pipe = get_imgaug_transform(cfg_tmp)
        pipe_str = pipe.__str__()
        assert pipe_str.find('Resize') == -1
        assert pipe_str.find('Fliplr') == -1
        assert pipe_str.find('Flipud') == -1
        assert pipe_str.find('Affine') == -1
        assert pipe_str.find('MotionBlur') != -1
        assert pipe_str.find('CoarseDropout') != -1
        assert pipe_str.find('CoarseSalt') != -1
        assert pipe_str.find('CoarsePepper') != -1
        assert pipe_str.find('ElasticTransformation') == -1
        assert pipe_str.find('AllChannelsHistogramEqualization') != -1
        assert pipe_str.find('AllChannelsCLAHE') != -1
        assert pipe_str.find('Emboss') != -1
        assert pipe_str.find('CropAndPad') == -1

    def test_get_imgaug_transform_custom(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)

        # custom pipeline: should contain Jigsaw and MultiplyAndAddToBrightness, no ShearX
        cfg_tmp.training.imgaug = {
            'ShearX': {'p': 0.0, 'kwargs': {'shear': (-30, 30)}},
            'Jigsaw': {'p': 0.5, 'kwargs': {'nb_rows': (3, 10), 'nb_cols': (5, 8)}},
            'MultiplyAndAddToBrightness': {
                'p': 1.0, 'kwargs': {'mul': (0.5, 1.5), 'add': (-5, 5)},
            },
        }
        pipe = get_imgaug_transform(cfg_tmp)
        assert pipe.__str__().find('ShearX') == -1
        assert pipe.__str__().find('Jigsaw') != -1
        assert pipe.__str__().find('MultiplyAndAddToBrightness') != -1

        # make sure lists are turned into tuples
        cfg_tmp.training.imgaug = {'Affine': {'p': 1.0, 'kwargs': {'rotate': [-30, 30]}}}
        pipe = get_imgaug_transform(cfg_tmp)
        assert pipe.__str__().find('Uniform') != -1  # uniform rotation in (-30, 30)
        assert pipe.__str__().find('Choice') == -1  # categorical rotation from [-30, 30]

        # allow args
        cfg_tmp.training.imgaug = {
            'Resize': {'p': 1.0, 'args': ({'height': 256, 'width': 256},), 'kwargs': {}},
        }
        pipe = get_imgaug_transform(cfg_tmp)
        assert pipe.__str__().find('Resize') != -1

        # allow no p, args, kwargs
        cfg_tmp.training.imgaug = {'FastSnowyLandscape': {}}
        pipe = get_imgaug_transform(cfg_tmp)
        assert pipe.__str__().find('FastSnowyLandscape') != -1

    def test_get_imgaug_transform_raises_for_unknown_augmentation(self, cfg):
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.training.imgaug = {
            'ResizeD': {'p': 1.0, 'args': ({'height': 256, 'width': 256},), 'kwargs': {}},
        }
        with pytest.raises(AttributeError):
            get_imgaug_transform(cfg_tmp)


class TestGetDataModule:
    """Test the get_data_module function."""

    def _supervised_multi_gpu_cfg(self, cfg):
        return OmegaConf.merge(
            cfg,
            OmegaConf.create({
                'model': {'losses_to_use': []},
                'training': {
                    'num_gpus': 2,
                    'train_batch_size': 4,
                    'val_batch_size': 16,
                    'test_batch_size': 16,
                },
            }),
        )

    def _unsupervised_multi_gpu_cfg(self, cfg):
        cfg = self._supervised_multi_gpu_cfg(cfg)
        cfg.model.losses_to_use = ['temporal']  # trigger unsupervised datamodule
        return cfg

    def test_get_data_module_num_gpus_0(self, cfg, mocker):
        cfg = self._supervised_multi_gpu_cfg(cfg)
        # when num_gpus is set to 0, i.e. from a deprecated config
        cfg.training.num_gpus = 0
        mock_init = mocker.patch.object(BaseDataModule, '__init__', return_value=None)
        get_data_module(cfg, Mock(spec=BaseTrackingDataset))

        # assert num_gpus gets modified to 1
        assert cfg.training.num_gpus == 1
        # the rest of the behavior follows correctly
        assert mock_init.call_args.kwargs['train_batch_size'] == cfg.training.train_batch_size
        assert mock_init.call_args.kwargs['val_batch_size'] == cfg.training.val_batch_size
        assert mock_init.call_args.kwargs['test_batch_size'] == cfg.training.test_batch_size

    def test_get_data_module_multi_gpu_batch_size_adjustment_supervised(self, cfg, mocker):
        cfg = self._supervised_multi_gpu_cfg(cfg)
        mock_init = mocker.patch.object(BaseDataModule, '__init__', return_value=None)
        get_data_module(cfg, Mock(spec=BaseTrackingDataset))
        # train, val batch sizes should be divided by num_gpus
        assert (mock_init.call_args.kwargs['train_batch_size']
                == cfg.training.train_batch_size / cfg.training.num_gpus)
        assert (mock_init.call_args.kwargs['val_batch_size']
                == cfg.training.val_batch_size / cfg.training.num_gpus)
        assert mock_init.call_args.kwargs['test_batch_size'] == cfg.training.test_batch_size

    def test_get_data_module_multi_gpu_batch_size_adjustment_unsupervised(
        self, cfg, heatmap_dataset, toy_data_dir, mocker,
    ):
        cfg = self._unsupervised_multi_gpu_cfg(cfg)
        mock_init = mocker.patch.object(UnlabeledDataModule, '__init__', return_value=None)
        get_data_module(cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'))
        # train, val batch sizes should be divided by num_gpus
        assert (mock_init.call_args.kwargs['train_batch_size']
                == cfg.training.train_batch_size / cfg.training.num_gpus)
        assert (mock_init.call_args.kwargs['val_batch_size']
                == cfg.training.val_batch_size / cfg.training.num_gpus)
        assert mock_init.call_args.kwargs['test_batch_size'] == cfg.training.test_batch_size
        # sequence length should be divided by num_gpus
        assert (mock_init.call_args.kwargs['dali_config'].base.train.sequence_length
                == cfg.dali.base.train.sequence_length / cfg.training.num_gpus)
        # context batch size is more nuanced, tested separately

    def test_get_data_module_multi_gpu_batch_size_adjustment_ceiling(
        self, cfg, heatmap_dataset, toy_data_dir,
    ):
        cfg = self._unsupervised_multi_gpu_cfg(cfg)
        get_data_module(cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'))

        # when batch_size is indivisible by 2
        cfg.training.train_batch_size += 1
        cfg.training.val_batch_size += 1
        cfg.dali.base.train.sequence_length += 1
        cfg.dali.context.train.batch_size += 1

        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'),
        )
        assert isinstance(data_module, UnlabeledDataModule)

        # batch size should be the ceiling of batch_size divided by num_gpus
        assert data_module.train_batch_size == int(
            np.ceil(cfg.training.train_batch_size / cfg.training.num_gpus)
        )
        assert data_module.val_batch_size == int(
            np.ceil(cfg.training.val_batch_size / cfg.training.num_gpus)
        )
        assert data_module.test_batch_size == cfg.training.test_batch_size
        assert data_module.dali_config.base.train.sequence_length == int(  # type: ignore[union-attr]
            np.ceil(cfg.dali.base.train.sequence_length / cfg.training.num_gpus)
        )
        # context batch size is more nuanced, tested separately

    def test_get_data_module_multi_gpu_context_batch_size_adjustment(
        self, cfg, heatmap_dataset, toy_data_dir,
    ):
        cfg = self._unsupervised_multi_gpu_cfg(cfg)
        cfg.model.model_type = 'heatmap_mhcrnn'
        get_data_module(cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'))

        # batch size of 6 -> effective 2 -> per-gpu effective 1 -> per-gpu 5
        cfg.dali.context.train.batch_size = 6
        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'),
        )
        assert isinstance(data_module, UnlabeledDataModule)
        assert data_module.dali_config.context.train.batch_size == 5  # type: ignore[union-attr]

        # batch size of 5 -> effective 1 -> per-gpu effective 1 -> per-gpu 5
        cfg.dali.context.train.batch_size = 5
        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'),
        )
        assert isinstance(data_module, UnlabeledDataModule)
        assert data_module.dali_config.context.train.batch_size == 5  # type: ignore[union-attr]

        # batch size of 28 -> effective 24 -> per-gpu effective 12 -> per-gpu 16
        cfg.dali.context.train.batch_size = 28
        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'),
        )
        assert isinstance(data_module, UnlabeledDataModule)
        assert data_module.dali_config.context.train.batch_size == 16  # type: ignore[union-attr]

        # batch size of 27 -> effective 23 -> per-gpu effective 12 -> per-gpu 16
        cfg.dali.context.train.batch_size = 27
        data_module = get_data_module(
            cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'),
        )
        assert isinstance(data_module, UnlabeledDataModule)
        assert data_module.dali_config.context.train.batch_size == 16  # type: ignore[union-attr]

        # batch size of 4 -> effective 0 -> should throw an error
        cfg.dali.context.train.batch_size = 4
        with pytest.raises(ValidationError):
            get_data_module(cfg, heatmap_dataset, os.path.join(toy_data_dir, 'videos'))


class TestGetLossFactories:
    """Test the get_loss_factories function."""

    def test_get_loss_factories_returns_supervised_and_unsupervised(self, cfg, base_data_module):
        """Always returns a dict with 'supervised' and 'unsupervised' LossFactory keys."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert set(factories.keys()) == {'supervised', 'unsupervised'}
        from lightning_pose.losses.factory import LossFactory
        assert isinstance(factories['supervised'], LossFactory)
        assert isinstance(factories['unsupervised'], LossFactory)

    def test_get_loss_factories_heatmap_supervised_loss(self, cfg, base_data_module):
        """Heatmap model_type builds a supervised factory keyed by heatmap_{loss_type}."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.model.model_type = 'heatmap'
        cfg_tmp.model.heatmap_loss_type = 'mse'
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert 'heatmap_mse' in factories['supervised'].loss_instance_dict

    def test_get_loss_factories_regression_supervised_loss(self, cfg, base_data_module):
        """Regression model_type builds a supervised factory keyed by 'regression'."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.model.model_type = 'regression'
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert 'regression' in factories['supervised'].loss_instance_dict

    def test_get_loss_factories_empty_unsupervised(self, cfg, base_data_module):
        """losses_to_use=[] produces an unsupervised factory with no losses."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert len(factories['unsupervised'].loss_instance_dict) == 0

    def test_get_loss_factories_temporal_unsupervised(self, cfg, base_data_module):
        """temporal in losses_to_use populates the unsupervised factory with a TemporalLoss."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['temporal']
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert 'temporal' in factories['unsupervised'].loss_instance_dict

    def test_get_loss_factories_pca_singleview(self, cfg, heatmap_data_module):
        """pca_singleview in losses_to_use populates the unsupervised factory."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['pca_singleview']
        factories = get_loss_factories(cfg_tmp, data_module=heatmap_data_module)
        assert 'pca_singleview' in factories['unsupervised'].loss_instance_dict

    def test_get_loss_factories_pca_multiview_mirrored(self, cfg, heatmap_data_module):
        """pca_multiview with mirrored columns populates the unsupervised factory."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['pca_multiview']
        factories = get_loss_factories(cfg_tmp, data_module=heatmap_data_module)
        assert 'pca_multiview' in factories['unsupervised'].loss_instance_dict

    def test_get_loss_factories_unimodal_raises(self, cfg, base_data_module):
        """unimodal losses raise Exception due to deprecated image_resize_dims path."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['unimodal_mse']
        cfg_tmp.losses.unimodal_mse = {'log_weight': 0.0}
        with pytest.raises(Exception, match='deprecated'):
            get_loss_factories(cfg_tmp, data_module=base_data_module)

    def test_get_loss_factories_temporal_heatmap_raises(self, cfg, base_data_module):
        """temporal_heatmap losses raise Exception due to deprecated image_resize_dims path."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['temporal_heatmap_mse']
        cfg_tmp.losses.temporal_heatmap_mse = {'log_weight': 0.0}
        with pytest.raises(Exception, match='deprecated'):
            get_loss_factories(cfg_tmp, data_module=base_data_module)

    def test_get_loss_factories_pca_singleview_raises_on_multiview(
        self, cfg, base_data_module,
    ):
        """pca_singleview raises NotImplementedError when view_names has multiple entries."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['pca_singleview']
        cfg_tmp.data.view_names = ['top', 'bot']
        with pytest.raises(NotImplementedError):
            get_loss_factories(cfg_tmp, data_module=base_data_module)

    def test_get_loss_factories_multiview_with_camera_params_no_optional_losses(
        self, cfg_multiview, multiview_heatmap_data_module, tmp_path,
    ):
        """multiview model with camera_params_file but no optional losses only adds heatmap."""
        cfg_tmp = copy.deepcopy(cfg_multiview)
        cfg_tmp.model.model_type = 'heatmap_multiview_transformer'
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.data.camera_params_file = str(tmp_path / 'camera_params.yaml')
        factories = get_loss_factories(cfg_tmp, data_module=multiview_heatmap_data_module)
        supervised_keys = set(factories['supervised'].loss_instance_dict.keys())
        assert 'supervised_pairwise_projections' not in supervised_keys
        assert 'supervised_reprojection_heatmap_mse' not in supervised_keys

    def test_get_loss_factories_multiview_supervised_pairwise_projections(
        self, cfg_multiview, multiview_heatmap_data_module, tmp_path,
    ):
        """supervised_pairwise_projections is added when log_weight is set and camera_params."""
        cfg_tmp = copy.deepcopy(cfg_multiview)
        cfg_tmp.model.model_type = 'heatmap_multiview_transformer'
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.data.camera_params_file = str(tmp_path / 'camera_params.yaml')
        cfg_tmp.losses.supervised_pairwise_projections = {'log_weight': 0.0}
        factories = get_loss_factories(cfg_tmp, data_module=multiview_heatmap_data_module)
        assert 'supervised_pairwise_projections' in factories['supervised'].loss_instance_dict

    def test_get_loss_factories_multiview_supervised_reprojection_heatmap_mse(
        self, cfg_multiview, multiview_heatmap_data_module, tmp_path,
    ):
        """supervised_reprojection_heatmap_mse added when log_weight is set and camera_params."""
        cfg_tmp = copy.deepcopy(cfg_multiview)
        cfg_tmp.model.model_type = 'heatmap_multiview_transformer'
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.data.camera_params_file = str(tmp_path / 'camera_params.yaml')
        cfg_tmp.losses.supervised_reprojection_heatmap_mse = {'log_weight': 0.0}
        factories = get_loss_factories(cfg_tmp, data_module=multiview_heatmap_data_module)
        assert (
            'supervised_reprojection_heatmap_mse' in factories['supervised'].loss_instance_dict
        )


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
            model.state_dict().items(), loaded.state_dict().items(), strict=True
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

        mocker.patch('lightning_pose.utils.scripts.torch.load', side_effect=patched_load)
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


class TestGetCallbacks:
    """Test the get_callbacks function."""

    def test_get_callbacks_default(self, cfg):
        """Default args produce UnfreezeBackbone, LearningRateMonitor, and ModelCheckpoint."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp)
        types = [type(cb) for cb in callbacks]
        assert UnfreezeBackbone in types
        assert LearningRateMonitor in types
        assert ModelCheckpoint in types

    def test_get_callbacks_with_early_stopping(self, cfg):
        """early_stopping=True adds an EarlyStopping callback."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp, early_stopping=True)
        types = [type(cb) for cb in callbacks]
        assert EarlyStopping in types

    def test_get_callbacks_without_backbone_unfreeze(self, cfg):
        """backbone_unfreeze=False omits UnfreezeBackbone."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp, backbone_unfreeze=False)
        types = [type(cb) for cb in callbacks]
        assert UnfreezeBackbone not in types

    def test_get_callbacks_without_lr_monitor(self, cfg):
        """lr_monitor=False omits LearningRateMonitor."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp, lr_monitor=False)
        types = [type(cb) for cb in callbacks]
        assert LearningRateMonitor not in types

    def test_get_callbacks_without_checkpointing(self, cfg):
        """checkpointing=False omits the best-model ModelCheckpoint."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp, checkpointing=False)
        types = [type(cb) for cb in callbacks]
        assert ModelCheckpoint not in types

    def test_get_callbacks_with_ckpt_every_n_epochs(self, cfg):
        """ckpt_every_n_epochs adds a second ModelCheckpoint that fires periodically."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        callbacks = get_callbacks(cfg_tmp, ckpt_every_n_epochs=5)
        ckpt_callbacks = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(ckpt_callbacks) == 2

    def test_get_callbacks_with_unsupervised_losses(self, cfg):
        """Non-empty losses_to_use adds an AnnealWeight callback."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = ['temporal']
        callbacks = get_callbacks(cfg_tmp)
        types = [type(cb) for cb in callbacks]
        assert AnnealWeight in types

    def test_get_callbacks_with_status_file(self, cfg, tmp_path):
        """Passing status_file adds a JSONTrainingProgressTracker callback."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        status_file = tmp_path / 'status.json'
        callbacks = get_callbacks(cfg_tmp, status_file=status_file)
        types = [type(cb) for cb in callbacks]
        assert JSONTrainingProgressTracker in types

    def test_get_callbacks_patch_masking(self, cfg):
        """Patch masking is added for multiview transformer when final_ratio > 0."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        cfg_tmp.model.model_type = 'heatmap_multiview_transformer'
        cfg_tmp.training.patch_mask = {'final_ratio': 0.5}
        callbacks = get_callbacks(cfg_tmp)
        types = [type(cb) for cb in callbacks]
        assert PatchMasking in types
