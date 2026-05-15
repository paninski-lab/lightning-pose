"""Test LossFactory class."""

import pytest
import torch

from lightning_pose.losses.factory import LossFactory
from lightning_pose.losses.losses import HeatmapMSELoss, RegressionMSELoss, TemporalLoss

stage = 'train'


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
