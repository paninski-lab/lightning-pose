"""Tests for losses/factory.py — LossFactory, get_loss_classes, get_loss_factories."""

import copy

import pytest
import torch

from lightning_pose.losses import get_loss_classes, get_loss_factories
from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import HeatmapMSELoss, Loss, RegressionMSELoss, TemporalLoss

stage = 'train'


class TestGetLossClasses:
    """Test the get_loss_classes function."""

    def test_all_values_are_loss_subclasses(self):
        """Every returned class is a subclass of Loss."""
        loss_classes = get_loss_classes()
        for _loss_name, loss_class in loss_classes.items():
            assert issubclass(loss_class, Loss)


class TestGetLossFactories:
    """Test the get_loss_factories function."""

    def test_get_loss_factories_returns_supervised_and_unsupervised(self, cfg, base_data_module):
        """Always returns a dict with 'supervised' and 'unsupervised' LossFactory keys."""
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.losses_to_use = []
        factories = get_loss_factories(cfg_tmp, data_module=base_data_module)
        assert set(factories.keys()) == {'supervised', 'unsupervised'}
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


class TestLossFactory:
    """Test the LossFactory class."""

    @pytest.fixture
    def heatmap_factory(self):
        """Return a LossFactory containing a single HeatmapMSELoss."""
        return LossFactory(
            losses_params_dict={'heatmap_mse': {'log_weight': 0.0}},
            data_module=None,
        )

    @pytest.fixture
    def regression_factory(self):
        """Return a LossFactory containing a single RegressionMSELoss."""
        return LossFactory(
            losses_params_dict={'regression': {'log_weight': 0.0}},
            data_module=None,
        )

    @pytest.fixture
    def temporal_factory(self):
        """Return a LossFactory containing a single TemporalLoss."""
        return LossFactory(
            losses_params_dict={'temporal': {'log_weight': 0.0}},
            data_module=None,
        )

    def test_initializes_heatmap_mse_loss(self, heatmap_factory):
        """Test that HeatmapMSELoss is instantiated with the correct type."""
        assert 'heatmap_mse' in heatmap_factory.loss_instance_dict
        assert isinstance(heatmap_factory.loss_instance_dict['heatmap_mse'], HeatmapMSELoss)

    def test_initializes_regression_loss(self, regression_factory):
        """Test that RegressionMSELoss is instantiated with the correct type."""
        assert 'regression' in regression_factory.loss_instance_dict
        assert isinstance(
            regression_factory.loss_instance_dict['regression'],
            RegressionMSELoss,
        )

    def test_initializes_temporal_loss(self, temporal_factory):
        """Test that TemporalLoss is instantiated with the correct type."""
        assert isinstance(temporal_factory.loss_instance_dict['temporal'], TemporalLoss)

    def test_initializes_multiple_losses(self):
        """Test that multiple losses are all instantiated."""
        factory = LossFactory(
            losses_params_dict={
                'heatmap_mse': {'log_weight': 0.0},
                'regression': {'log_weight': 0.0},
            },
            data_module=None,
        )
        assert len(factory.loss_instance_dict) == 2

    def test_call_zero_loss_when_preds_equal_targets(self, heatmap_factory):
        """Test that identical targets and predictions yield zero total loss."""
        batch_size, num_keypoints, h, w = 3, 4, 16, 16
        heatmaps = torch.rand(batch_size, num_keypoints, h, w)
        tot_loss, log_list = heatmap_factory(
            stage=stage,
            heatmaps_targ=heatmaps,
            heatmaps_pred=heatmaps,
        )
        assert tot_loss == 0.0
        names = [d['name'] for d in log_list]
        assert f'{stage}_heatmap_mse_loss' in names
        assert f'{stage}_heatmap_mse_loss_weighted' in names

    def test_call_nonnegative_loss_with_mismatched_preds(self, regression_factory):
        """Test that differing predictions yield a non-negative total loss."""
        batch_size = 3
        keypoints_targ = torch.rand(batch_size, 8)
        keypoints_pred = torch.rand(batch_size, 8)
        tot_loss, _ = regression_factory(
            stage='val',
            keypoints_targ=keypoints_targ,
            keypoints_pred=keypoints_pred,
        )
        assert tot_loss >= 0.0

    def test_heatmap_loss_unaffected_by_anneal_weight(self, heatmap_factory):
        """Test that anneal_weight does not scale heatmap losses."""
        batch_size, num_keypoints, h, w = 3, 4, 16, 16
        heatmaps_targ = torch.rand(batch_size, num_keypoints, h, w)
        heatmaps_pred = torch.rand(batch_size, num_keypoints, h, w)
        loss_annealed, _ = heatmap_factory(
            stage=stage,
            anneal_weight=0.0,
            heatmaps_targ=heatmaps_targ,
            heatmaps_pred=heatmaps_pred,
        )
        loss_full, _ = heatmap_factory(
            stage=stage,
            anneal_weight=1.0,
            heatmaps_targ=heatmaps_targ,
            heatmaps_pred=heatmaps_pred,
        )
        assert torch.isclose(loss_annealed, loss_full)

    def test_non_heatmap_loss_scaled_by_anneal_weight(self, temporal_factory):
        """Test that anneal_weight linearly scales non-heatmap losses."""
        # alternating predictions ensure a non-zero temporal loss
        keypoints_pred = torch.zeros(4, 8)
        keypoints_pred[1::2] = 1.0
        loss_half, _ = temporal_factory(
            stage=stage,
            anneal_weight=0.5,
            keypoints_pred=keypoints_pred,
        )
        loss_full, _ = temporal_factory(
            stage=stage,
            anneal_weight=1.0,
            keypoints_pred=keypoints_pred,
        )
        assert torch.isclose(loss_half * 2, loss_full)

    def test_anneal_weight_none_treated_as_one(self, temporal_factory):
        """Test that anneal_weight=None behaves identically to anneal_weight=1."""
        keypoints_pred = torch.zeros(4, 8)
        keypoints_pred[1::2] = 1.0
        loss_none, _ = temporal_factory(
            stage=stage,
            anneal_weight=None,
            keypoints_pred=keypoints_pred,
        )
        loss_one, _ = temporal_factory(
            stage=stage,
            anneal_weight=1.0,
            keypoints_pred=keypoints_pred,
        )
        assert torch.isclose(loss_none, loss_one)

    def test_call_with_no_stage(self, heatmap_factory):
        """Test that stage=None does not raise."""
        batch_size, num_keypoints, h, w = 3, 4, 16, 16
        heatmaps = torch.rand(batch_size, num_keypoints, h, w)
        tot_loss, _ = heatmap_factory(
            stage=None,
            heatmaps_targ=heatmaps,
            heatmaps_pred=heatmaps,
        )
        assert tot_loss == 0.0
