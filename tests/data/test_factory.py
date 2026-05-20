"""Test the data factory module."""

import copy
import os
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError
from PIL import Image

from lightning_pose.data import (
    get_data_module,
    get_imgaug_transform,
)
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import BaseTrackingDataset


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
