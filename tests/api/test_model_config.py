"""Tests for ModelConfig.validate() and its sub-validation methods."""

from pathlib import Path
from unittest.mock import patch

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
# is_multi_view / is_single_view
# ---------------------------------------------------------------------------

class TestIsMultiView:
    """Test ModelConfig.is_multi_view and is_single_view."""

    def test_is_multi_view_no_view_names(self):
        """Absent view_names → single-view."""
        mc = _mc(_singleview_cfg_dict())
        assert not mc.is_multi_view()
        assert mc.is_single_view()

    def test_is_multi_view_two_views(self):
        """Two view_names → multi-view."""
        mc = _mc(_multiview_cfg_dict())
        assert mc.is_multi_view()
        assert not mc.is_single_view()

    def test_is_multi_view_single_entry_raises(self):
        """Exactly one view_name → ValueError."""
        cfg = _singleview_cfg_dict()
        cfg['data']['view_names'] = ['only_one']
        with pytest.raises(ValueError, match='should not be specified if there is only one view'):
            _mc(cfg).is_multi_view()


# ---------------------------------------------------------------------------
# test_video_files_singleview
# ---------------------------------------------------------------------------

class TestTestVideoFilesSingleview:
    """Test ModelConfig.test_video_files_singleview."""

    def test_test_video_files_singleview_on_multiview_raises(self):
        """AssertionError when called on a multi-view model."""
        with pytest.raises(AssertionError):
            _mc(_multiview_cfg_dict()).test_video_files_singleview()

    def test_test_video_files_singleview_returns_paths(self):
        """Returns list of Paths produced by check_video_paths."""
        cfg = _singleview_cfg_dict()
        cfg['eval'] = {'test_videos_directory': '/fake/videos'}
        with (
            patch(
                'lightning_pose.api.model_config.return_absolute_path',
                return_value='/fake/videos',
            ),
            patch(
                'lightning_pose.api.model_config.check_video_paths',
                return_value=['/fake/videos/a.mp4', '/fake/videos/b.mp4'],
            ),
        ):
            result = _mc(cfg).test_video_files_singleview()
        assert result == [Path('/fake/videos/a.mp4'), Path('/fake/videos/b.mp4')]


# ---------------------------------------------------------------------------
# test_video_files_multiview
# ---------------------------------------------------------------------------

class TestTestVideoFilesMultiview:
    """Test ModelConfig.test_video_files_multiview."""

    def test_test_video_files_multiview_on_singleview_raises(self):
        """AssertionError when called on a single-view model."""
        with pytest.raises(AssertionError):
            _mc(_singleview_cfg_dict()).test_video_files_multiview()

    def test_test_video_files_multiview_returns_grouped_paths(self):
        """Returns the list of per-session path groups from find_video_files_for_views."""
        cfg = _multiview_cfg_dict()
        cfg['eval'] = {'test_videos_directory': '/fake/videos'}
        fake_result = [[Path('/fake/videos/cam0.mp4'), Path('/fake/videos/cam1.mp4')]]
        with patch(
            'lightning_pose.api.model_config.find_video_files_for_views',
            return_value=fake_result,
        ):
            result = _mc(cfg).test_video_files_multiview()
        assert result == fake_result


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
        with pytest.warns(UserWarning, match='heatmap_multiview_transformer'):
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
# _validate_losses
# ---------------------------------------------------------------------------

class TestValidateLosses:
    """Test ModelConfig._validate_losses."""

    def test_validate_losses_no_losses_to_use(self):
        """Absent losses_to_use → passes without inspecting cfg.losses."""
        _mc(_singleview_cfg_dict())._validate_losses()

    def test_validate_losses_empty_list(self):
        """Empty losses_to_use list → passes."""
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = []
        _mc(cfg)._validate_losses()

    def test_validate_losses_valid_float_log_weight(self):
        """A loss with a float log_weight passes."""
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = ['temporal_norm']
        cfg['losses']['temporal_norm'] = {'log_weight': 0.5}
        _mc(cfg)._validate_losses()

    def test_validate_losses_null_log_weight_skipped(self):
        """log_weight=null means the loss is inactive; no assertion fires."""
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = ['temporal_norm']
        cfg['losses']['temporal_norm'] = {'log_weight': None}
        _mc(cfg)._validate_losses()

    def test_validate_losses_missing_loss_cfg_skipped(self):
        """A loss in losses_to_use with no entry in cfg.losses is skipped."""
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = ['temporal_norm']
        # 'temporal_norm' is absent from cfg['losses']
        _mc(cfg)._validate_losses()

    def test_validate_losses_string_log_weight_raises(self):
        """A string log_weight raises AssertionError."""
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = ['temporal_norm']
        cfg['losses']['temporal_norm'] = {'log_weight': '0.5'}
        with pytest.raises(AssertionError, match='log_weight must be numeric'):
            _mc(cfg)._validate_losses()


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

    def test_validate_propagates_losses_error(self):
        cfg = _singleview_cfg_dict()
        cfg['model']['losses_to_use'] = ['temporal_norm']
        cfg['losses']['temporal_norm'] = {'log_weight': 'bad'}
        with pytest.raises(AssertionError, match='log_weight must be numeric'):
            _mc(cfg).validate()
