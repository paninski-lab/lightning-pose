"""Tests for ModelConfig.validate() and its sub-validation methods."""

import pytest
from omegaconf import OmegaConf

from lightning_pose.api.model_config import ModelConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _singleview_cfg_dict() -> dict:
    """Return a minimal valid single-view config dict."""
    return {
        'data': {
            'num_keypoints': 3,
            'keypoint_names': ['kp1', 'kp2', 'kp3'],
            'image_resize_dims': {'height': 256, 'width': 256},
            'csv_file': 'CollectedData.csv',
        },
        'training': {
            'train_prob': 0.8,
            'val_prob': 0.1,
            'check_val_every_n_epoch': 5,
            'ckpt_every_n_epochs': None,
            'unfreezing_epoch': 20,
            'min_epochs': 300,
            'max_epochs': 300,
            'imgaug': 'dlc',
            'lr_scheduler_params': {
                'multisteplr': {
                    'milestones': [150, 200, 250],
                    'gamma': 0.5,
                },
            },
        },
        'model': {'model_type': 'heatmap'},
        'losses': {},
    }


def _multiview_cfg_dict() -> dict:
    """Return a minimal valid multi-view config dict."""
    return {
        'data': {
            'num_keypoints': 3,
            'keypoint_names': ['kp1', 'kp2', 'kp3'],
            'image_resize_dims': {'height': 256, 'width': 256},
            'csv_file': ['CollectedData_cam0.csv', 'CollectedData_cam1.csv'],
            'view_names': ['cam0', 'cam1'],
        },
        'training': {
            'train_prob': 0.8,
            'val_prob': 0.1,
            'check_val_every_n_epoch': 5,
            'ckpt_every_n_epochs': None,
            'unfreezing_epoch': 20,
            'min_epochs': 300,
            'max_epochs': 300,
            'imgaug': 'dlc',
            'imgaug_3d': True,
            'lr_scheduler_params': {
                'multisteplr': {
                    'milestones': [150, 200, 250],
                    'gamma': 0.5,
                },
            },
        },
        'model': {'model_type': 'heatmap_multiview_transformer'},
        'losses': {
            'supervised_reprojection_heatmap_mse': {'log_weight': None},
        },
    }


def _mc(d: dict) -> ModelConfig:
    """Wrap a plain dict in a ModelConfig."""
    return ModelConfig(OmegaConf.create(d))


# ---------------------------------------------------------------------------
# _validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    """Test ModelConfig._validate_data."""

    def test_validate_data_valid_singleview(self):
        # Arrange / Act / Assert
        _mc(_singleview_cfg_dict())._validate_data()

    def test_validate_data_valid_multiview(self):
        _mc(_multiview_cfg_dict())._validate_data()

    def test_validate_data_num_keypoints_none(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['num_keypoints'] = None
        with pytest.raises(AssertionError, match='num_keypoints must be set'):
            _mc(cfg)._validate_data()

    def test_validate_data_num_keypoints_zero(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['num_keypoints'] = 0
        with pytest.raises(AssertionError, match='num_keypoints must be positive'):
            _mc(cfg)._validate_data()

    def test_validate_data_keypoint_names_length_mismatch(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['keypoint_names'] = ['kp1', 'kp2']  # 2, but num_keypoints=3
        with pytest.raises(AssertionError, match='len\\(data.keypoint_names\\)'):
            _mc(cfg)._validate_data()

    def test_validate_data_keypoint_names_none_skipped(self):
        # None keypoint_names should not trigger the length check.
        cfg = _singleview_cfg_dict()
        cfg['data']['keypoint_names'] = None
        _mc(cfg)._validate_data()

    def test_validate_data_multiview_view_names_csv_mismatch(self):
        cfg = _multiview_cfg_dict()
        cfg['data']['csv_file'] = ['CollectedData_cam0.csv']  # 1, but view_names has 2
        with pytest.raises(AssertionError, match='len\\(data.view_names\\)'):
            _mc(cfg)._validate_data()

    def test_validate_data_height_not_multiple_of_128(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['image_resize_dims']['height'] = 200
        with pytest.raises(AssertionError, match='height.*multiple of 128'):
            _mc(cfg)._validate_data()

    def test_validate_data_width_not_multiple_of_128(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['image_resize_dims']['width'] = 100
        with pytest.raises(AssertionError, match='width.*multiple of 128'):
            _mc(cfg)._validate_data()

    def test_validate_data_resize_dims_none_skipped(self):
        # Null resize dims are valid (model uses native image size).
        cfg = _singleview_cfg_dict()
        cfg['data']['image_resize_dims'] = {'height': None, 'width': None}
        _mc(cfg)._validate_data()


# ---------------------------------------------------------------------------
# _validate_training
# ---------------------------------------------------------------------------

class TestValidateTraining:
    """Test ModelConfig._validate_training."""

    def test_validate_training_valid_epoch_based(self):
        _mc(_singleview_cfg_dict())._validate_training()

    def test_validate_training_valid_step_based(self):
        cfg = _singleview_cfg_dict()
        t = cfg['training']
        del t['unfreezing_epoch']
        del t['min_epochs']
        del t['max_epochs']
        t['unfreezing_step'] = 600
        t['min_steps'] = 9000
        t['max_steps'] = 9000
        t['lr_scheduler_params'] = {
            'multisteplr': {'milestone_steps': [4500, 6000, 7500], 'gamma': 0.5},
        }
        _mc(cfg)._validate_training()

    def test_validate_training_train_val_prob_exceed_one(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['train_prob'] = 0.9
        cfg['training']['val_prob'] = 0.2
        with pytest.raises(AssertionError, match='train_prob.*val_prob'):
            _mc(cfg)._validate_training()

    def test_validate_training_ckpt_not_divisible_by_check_val(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['ckpt_every_n_epochs'] = 7   # not divisible by check_val=5
        with pytest.raises(AssertionError, match='ckpt_every_n_epochs.*divisible'):
            _mc(cfg)._validate_training()

    def test_validate_training_ckpt_none_skipped(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['ckpt_every_n_epochs'] = None
        _mc(cfg)._validate_training()

    def test_validate_training_milestone_exceeds_max_epochs(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['lr_scheduler_params']['multisteplr']['milestones'] = [150, 200, 350]
        with pytest.raises(AssertionError, match='milestones.*max_epochs'):
            _mc(cfg)._validate_training()

    def test_validate_training_milestone_steps_exceeds_max_steps(self):
        cfg = _singleview_cfg_dict()
        t = cfg['training']
        del t['unfreezing_epoch']
        del t['min_epochs']
        del t['max_epochs']
        t['unfreezing_step'] = 600
        t['min_steps'] = 9000
        t['max_steps'] = 9000
        t['lr_scheduler_params'] = {
            'multisteplr': {'milestone_steps': [4500, 6000, 10000], 'gamma': 0.5},
        }
        with pytest.raises(AssertionError, match='milestone_steps.*max_steps'):
            _mc(cfg)._validate_training()

    def test_validate_training_mixed_step_epoch_fields(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['min_steps'] = 9000  # mixed with min_epochs
        with pytest.raises(AssertionError):
            _mc(cfg)._validate_training()


# ---------------------------------------------------------------------------
# _validate_model
# ---------------------------------------------------------------------------

class TestValidateModel:
    """Test ModelConfig._validate_model."""

    def test_validate_model_valid_singleview(self):
        _mc(_singleview_cfg_dict())._validate_model()

    def test_validate_model_valid_multiview(self):
        _mc(_multiview_cfg_dict())._validate_model()

    def test_validate_model_invalid_model_type(self):
        cfg = _singleview_cfg_dict()
        cfg['model']['model_type'] = 'invalid_type'
        with pytest.raises(AssertionError, match="model.model_type 'invalid_type' is not one of"):
            _mc(cfg)._validate_model()

    def test_validate_model_multiview_wrong_model_type(self):
        cfg = _multiview_cfg_dict()
        cfg['model']['model_type'] = 'heatmap'
        with pytest.raises(AssertionError, match='heatmap_multiview_transformer'):
            _mc(cfg)._validate_model()

    def test_validate_model_reprojection_loss_wrong_imgaug(self):
        cfg = _multiview_cfg_dict()
        cfg['losses']['supervised_reprojection_heatmap_mse']['log_weight'] = 3.0
        cfg['training']['imgaug'] = 'default'
        with pytest.raises(AssertionError, match="training.imgaug must be 'dlc'"):
            _mc(cfg)._validate_model()

    def test_validate_model_reprojection_loss_imgaug_3d_false(self):
        cfg = _multiview_cfg_dict()
        cfg['losses']['supervised_reprojection_heatmap_mse']['log_weight'] = 3.0
        cfg['training']['imgaug_3d'] = False
        with pytest.raises(AssertionError, match='imgaug_3d must be true'):
            _mc(cfg)._validate_model()

    def test_validate_model_reprojection_loss_null_log_weight_skips_augmentation_check(self):
        # log_weight=None means the loss is inactive; augmentation check should not fire.
        cfg = _multiview_cfg_dict()
        cfg['losses']['supervised_reprojection_heatmap_mse']['log_weight'] = None
        cfg['training']['imgaug'] = 'default'
        cfg['training']['imgaug_3d'] = False
        _mc(cfg)._validate_model()

    def test_validate_model_no_reprojection_loss_section_skips_augmentation_check(self):
        # Missing losses section entirely should not trigger the augmentation check.
        cfg = _multiview_cfg_dict()
        del cfg['losses']['supervised_reprojection_heatmap_mse']
        cfg['training']['imgaug_3d'] = False
        _mc(cfg)._validate_model()


# ---------------------------------------------------------------------------
# validate (top-level)
# ---------------------------------------------------------------------------

class TestValidate:
    """Test ModelConfig.validate end-to-end."""

    def test_validate_valid_singleview(self):
        _mc(_singleview_cfg_dict()).validate()

    def test_validate_valid_multiview(self):
        _mc(_multiview_cfg_dict()).validate()

    def test_validate_propagates_data_error(self):
        cfg = _singleview_cfg_dict()
        cfg['data']['num_keypoints'] = None
        with pytest.raises(AssertionError, match='num_keypoints must be set'):
            _mc(cfg).validate()

    def test_validate_propagates_training_error(self):
        cfg = _singleview_cfg_dict()
        cfg['training']['train_prob'] = 0.9
        cfg['training']['val_prob'] = 0.2
        with pytest.raises(AssertionError, match='train_prob'):
            _mc(cfg).validate()

    def test_validate_propagates_model_error(self):
        cfg = _singleview_cfg_dict()
        cfg['model']['model_type'] = 'bad'
        with pytest.raises(AssertionError, match="model.model_type 'bad'"):
            _mc(cfg).validate()
