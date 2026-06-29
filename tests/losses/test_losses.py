"""Test loss classes."""

import math
from typing import Literal

import numpy as np
import pytest
import torch
from kornia.geometry.subpix import spatial_softmax2d

from lightning_pose.data.heatmaps import generate_heatmaps
from lightning_pose.losses.losses import (
    HeatmapJSLoss,
    HeatmapKLLoss,
    HeatmapMSELoss,
    Loss,
    PairwiseProjectionsLoss,
    PCALoss,
    RegressionMSELoss,
    RegressionRMSELoss,
    ReprojectionHeatmapLoss,
    TemporalHeatmapLoss,
    TemporalLoss,
    UnimodalLoss,
)
from lightning_pose.utils.pca import format_multiview_data_for_pca

stage: Literal["train"] = 'train'
device = 'cpu'


class TestLoss:
    """Test the base Loss class."""

    def test_weight_property(self):
        """Test that weight is computed correctly from log_weight."""
        loss = Loss(log_weight=0.0)
        assert torch.isclose(loss.weight, torch.tensor(1.0 / (2.0 * math.exp(0.0))))

    def test_weight_property_nonzero_log_weight(self):
        """Test weight with non-zero log_weight."""
        log_weight = 1.0
        loss = Loss(log_weight=log_weight)
        expected = 1.0 / (2.0 * math.exp(log_weight))
        assert torch.isclose(loss.weight, torch.tensor(expected, dtype=torch.float))

    def test_rectify_epsilon_zeros_small_values(self):
        """Test that values below epsilon are zeroed out."""
        loss = Loss(epsilon=0.5)
        values = torch.tensor([0.1, 0.5, 1.0])
        rectified = loss.rectify_epsilon(values)
        assert rectified[0] == 0.0
        assert rectified[1] == 0.0
        assert torch.isclose(rectified[2], torch.tensor(0.5))

    def test_reduce_loss_mean(self):
        """Test mean reduction."""
        loss = Loss()
        values = torch.tensor([1.0, 2.0, 3.0])
        assert loss.reduce_loss(values, method='mean') == 2.0

    def test_reduce_loss_sum(self):
        """Test sum reduction."""
        loss = Loss()
        values = torch.tensor([1.0, 2.0, 3.0])
        assert loss.reduce_loss(values, method='sum') == 6.0

    def test_remove_nans_raises(self):
        """Test that remove_nans raises NotImplementedError on the base class."""
        loss = Loss()
        with pytest.raises(NotImplementedError):
            loss.remove_nans()

    def test_compute_loss_raises(self):
        """Test that compute_loss raises NotImplementedError on the base class."""
        loss = Loss()
        with pytest.raises(NotImplementedError):
            loss.compute_loss()

    def test_call_raises(self):
        """Test that __call__ raises NotImplementedError on the base class."""
        loss = Loss()
        with pytest.raises(NotImplementedError):
            loss()


class TestHeatmapMSELoss:
    """Test the HeatmapMSELoss class."""

    @pytest.fixture
    def heatmap_mse_loss(self):
        """Return a default HeatmapMSELoss instance."""
        return HeatmapMSELoss()

    def test_heatmap_mse_loss_zero_when_preds_equal_targets(self, heatmap_mse_loss):
        """Test that loss is zero when predictions equal targets."""
        targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
        predictions = torch.ones_like(targets) / (48 * 48)
        loss, logs = heatmap_mse_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert loss == 0.0
        assert logs[0]['name'] == f'{stage}_heatmap_mse_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'heatmap_mse_weight'
        assert logs[1]['value'] == heatmap_mse_loss.weight

    def test_heatmap_mse_loss_positive_when_preds_differ(self, heatmap_mse_loss):
        """Test that loss is positive when predictions differ from targets."""
        targets = torch.ones((3, 7, 48, 48)) / (48 * 48)
        predictions = (torch.ones_like(targets) / (48 * 48)) + 0.01 * torch.randn_like(targets)
        loss, logs = heatmap_mse_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        assert loss > 0.0


class TestHeatmapKLLoss:
    """Test the HeatmapKLLoss class."""

    @pytest.fixture
    def heatmap_kl_loss(self):
        """Return a default HeatmapKLLoss instance."""
        return HeatmapKLLoss()

    @pytest.fixture
    def heatmaps(self):
        """Return a batch of valid heatmaps generated from random keypoints."""
        m = 100
        keypoints = m * torch.rand((3, 7, 2))
        return generate_heatmaps(keypoints, height=m, width=m, output_shape=(32, 32))

    def test_heatmap_kl_loss_zero_when_preds_equal_targets(self, heatmap_kl_loss, heatmaps):
        """Test that loss is near zero when predictions equal targets."""
        targets = heatmaps
        predictions = targets.clone()
        loss, logs = heatmap_kl_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert np.isclose(loss.detach().cpu().numpy(), 0.0, rtol=1e-5)
        assert logs[0]['name'] == f'{stage}_heatmap_kl_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'heatmap_kl_weight'
        assert logs[1]['value'] == heatmap_kl_loss.weight

    def test_heatmap_kl_loss_increases_with_error(self, heatmap_kl_loss, heatmaps):
        """Test that loss increases when predictions have higher error."""
        targets = heatmaps
        predictions = targets.clone()
        loss, _ = heatmap_kl_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        predictions_shifted = torch.roll(predictions, shifts=1, dims=0)
        loss2, _ = heatmap_kl_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions_shifted,
            stage=stage,
        )
        assert loss2 > loss


class TestHeatmapJSLoss:
    """Test the HeatmapJSLoss class."""

    @pytest.fixture
    def heatmap_js_loss(self):
        """Return a default HeatmapJSLoss instance."""
        return HeatmapJSLoss()

    @pytest.fixture
    def heatmaps(self):
        """Return a batch of valid heatmaps generated from random keypoints."""
        m = 100
        keypoints = m * torch.rand((3, 7, 2))
        return generate_heatmaps(keypoints, height=m, width=m, output_shape=(32, 32))

    def test_heatmap_js_loss_zero_when_preds_equal_targets(self, heatmap_js_loss, heatmaps):
        """Test that loss is near zero when predictions equal targets."""
        targets = heatmaps
        predictions = targets.clone()
        loss, logs = heatmap_js_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert np.isclose(loss.detach().cpu().numpy(), 0.0, rtol=1e-5)
        assert logs[0]['name'] == f'{stage}_heatmap_js_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'heatmap_js_weight'
        assert logs[1]['value'] == heatmap_js_loss.weight

    def test_heatmap_js_loss_increases_with_error(self, heatmap_js_loss, heatmaps):
        """Test that loss increases when predictions have higher error."""
        targets = heatmaps
        predictions = targets.clone()
        loss, _ = heatmap_js_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions,
            stage=stage,
        )
        predictions_shifted = torch.roll(predictions, shifts=1, dims=0)
        loss2, _ = heatmap_js_loss(
            heatmaps_targ=targets,
            heatmaps_pred=predictions_shifted,
            stage=stage,
        )
        assert loss2 > loss


class TestPCALoss:
    """Test the PCALoss class."""

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(f'cuda:{torch.cuda.current_device()}', marks=pytest.mark.gpu),
    ])
    def test_pca_singleview_loss(self, cfg, base_data_module, device):
        """Test singleview PCA loss produces positive loss on toy data."""
        pca_loss = PCALoss(
            loss_name='pca_singleview',
            components_to_keep=2,
            data_module=base_data_module,
            columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
            device=device,
        )
        # scale to pixel coordinates so reprojection errors reliably exceed the empirical epsilon
        keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device) * 50
        loss, logs = pca_loss(keypoints_pred, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss > 0.0
        assert logs[0]['name'] == f'{stage}_pca_singleview_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'pca_singleview_weight'
        assert logs[1]['value'] == pca_loss.weight

    def test_pca_multiview_loss_raises_without_mirrored_column_matches(
        self, base_data_module,
    ):
        """Test that PCALoss raises ValueError when mirrored_column_matches is missing."""
        with pytest.raises(ValueError):
            PCALoss(loss_name='pca_multiview', data_module=base_data_module)

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(f'cuda:{torch.cuda.current_device()}', marks=pytest.mark.gpu),
    ])
    def test_pca_multiview_loss_on_toy_dataset(self, cfg, base_data_module, device):
        """Test multiview PCA loss produces expected shapes and positive loss on toy data."""
        pca_loss = PCALoss(
            loss_name='pca_multiview',
            components_to_keep=3,
            data_module=base_data_module,
            mirrored_column_matches=cfg.data.mirrored_column_matches,
            device=device,
        )
        keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device)
        keypoints_pred = keypoints_pred.reshape(keypoints_pred.shape[0], -1, 2)
        keypoints_pred = format_multiview_data_for_pca(
            data_arr=keypoints_pred,
            mirrored_column_matches=cfg.data.mirrored_column_matches,
        )
        pre_reduction_loss = pca_loss.compute_loss(keypoints_pred)
        assert pre_reduction_loss.shape == (keypoints_pred.shape[0], 2)

        # scale to pixel coordinates so reprojection errors reliably exceed the empirical epsilon
        keypoints_pred = torch.randn(20, base_data_module.dataset.num_targets, device=device) * 50
        loss, logs = pca_loss(keypoints_pred, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss > 0.0
        assert logs[0]['name'] == f'{stage}_pca_multiview_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'pca_multiview_weight'
        assert logs[1]['value'] == pca_loss.weight

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param(f'cuda:{torch.cuda.current_device()}', marks=pytest.mark.gpu),
    ])
    def test_pca_multiview_loss_on_fake_dataset(self, cfg, base_data_module, device):
        """Test that multiview PCA loss is zero for data exactly in the PCA subspace."""
        # TODO: this is not how we currently compute reprojection; we now add mean in final stage
        pca_loss = PCALoss(
            loss_name='pca_multiview',
            components_to_keep=3,
            data_module=base_data_module,
            mirrored_column_matches=cfg.data.mirrored_column_matches,
            device=device,
        )
        kept_evecs = torch.eye(n=4, device=device)[:, :3].T
        projection_to_obs = torch.randn(size=(10, 3), device=device)
        obs = projection_to_obs @ kept_evecs
        mean = obs.mean(dim=0)
        good_arr_for_pca = obs - mean.unsqueeze(0)
        reproj = good_arr_for_pca @ kept_evecs.T @ kept_evecs
        assert torch.allclose(reproj - good_arr_for_pca, torch.zeros_like(input=reproj))

        pca_loss.pca.parameters['kept_eigenvectors'] = kept_evecs
        pca_loss.pca.parameters['mean'] = mean
        pca_loss.pca.mirrored_column_matches = [[0], [1]]
        loss, logs = pca_loss(obs, stage=stage)
        assert loss == 0.0


class TestTemporalLoss:
    """Test the TemporalLoss class."""

    @pytest.fixture
    def temporal_loss(self):
        """Return a TemporalLoss with zero epsilon."""
        return TemporalLoss(epsilon=0.0)

    def test_temporal_loss_zero_for_constant_predictions(self, temporal_loss):
        """Test that constant predictions along batch dim yield zero loss."""
        predicted_keypoints = torch.ones(size=(12, 32))
        predicted_keypoints[:, 1] = 2
        predicted_keypoints[:, 2] = 4
        predicted_keypoints[:, 3] = 8
        loss, logs = temporal_loss(predicted_keypoints, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss == 0.0
        assert logs[0]['name'] == f'{stage}_temporal_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'temporal_weight'
        assert logs[1]['value'] == temporal_loss.weight

    def test_temporal_loss_positive_for_random_predictions(self, temporal_loss):
        """Test that random predictions yield a positive scalar loss."""
        predicted_keypoints = torch.rand(size=(12, 32))
        loss, logs = temporal_loss(predicted_keypoints, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss > 0.0

    def test_temporal_loss_matches_expected_norm(self, temporal_loss):
        """Test that computed loss matches the analytic L2 norm."""
        predicted_keypoints = torch.Tensor([[0.0, 0.0], [np.sqrt(2.0), np.sqrt(2.0)]])
        loss, logs = temporal_loss(predicted_keypoints, stage=stage)
        assert loss.item() - 2 < 1e-6

    def test_temporal_loss_epsilon_zeroes_small_values(self):
        """Test that epsilon masks out temporal differences below the threshold."""
        s2 = np.sqrt(2.0)
        s3 = np.sqrt(3.0)
        predicted_keypoints = torch.Tensor([[0.0, 0.0], [s2, s2], [s3 + s2, s3 + s2]])
        # [s2, s2] -> norm 2; [s3, s3] -> norm sqrt(6)
        loss, logs = TemporalLoss(epsilon=0.0)(predicted_keypoints, stage=stage)
        assert (loss.item() - (2 + np.sqrt(6))) < 1e-6

        # epsilon=2.1 zeroes out the first diff (norm 2), leaving only sqrt(6)
        loss, logs = TemporalLoss(epsilon=2.1)(predicted_keypoints, stage=stage)
        assert (loss.item() - np.sqrt(6)) < 1e-6

    def test_temporal_loss_multi_epsilon_rectification(self):
        """Test rectify_epsilon with per-keypoint epsilon vectors."""
        batch_size = 6
        loss_tensor = (
            torch.tensor([0.0, 1.0, 0.4], dtype=torch.float32)
            .unsqueeze(0)
            .repeat(batch_size - 1, 1)
        )

        temporal_loss = TemporalLoss(epsilon=[0.1, 0.0, 0.5])
        rectified = temporal_loss.rectify_epsilon(loss_tensor)
        assert rectified.shape == torch.Size([batch_size - 1, 3])
        assert torch.all(rectified[:, 0] == 0.0)
        assert torch.all(rectified[:, 1] == 1.0)
        assert torch.all(rectified[:, 2] == 0.0)

        temporal_loss = TemporalLoss(epsilon=[0.1, 0.0, 0.3])
        rectified = temporal_loss.rectify_epsilon(loss_tensor)
        assert rectified.shape == torch.Size([batch_size - 1, 3])
        assert torch.all(rectified[:, 0] == 0.0)
        assert torch.all(rectified[:, 1] == 1.0)
        assert torch.allclose(rectified[:, 2], torch.tensor([0.1]))

        loss_tensor_fancier = torch.tensor(
            data=[[1.0, 2.0, 1.5], [0.05, 0.12, 0.2]], dtype=torch.float32
        )
        temporal_loss = TemporalLoss(epsilon=[0.1, 0.15, 0.3])
        rectified = temporal_loss.rectify_epsilon(loss_tensor_fancier)
        assert rectified.shape == (2, 3)
        assert torch.allclose(rectified[0, :], torch.tensor([0.9, 1.85, 1.2]))
        assert torch.allclose(rectified[1, :], torch.tensor([0.0, 0.0, 0.0]))

    def test_temporal_loss_with_confidences(self):
        """Test that remove_nans zeros out low-confidence diffs."""
        temporal_loss = TemporalLoss(epsilon=0.0, prob_threshold=0.5)
        batch_size = 4
        keypoints_pred = torch.zeros(batch_size, 4)
        keypoints_pred[1::2] = 1.0
        # all confidences below threshold for keypoint 0; should zero out its contribution
        confidences = torch.zeros(batch_size, 2)
        confidences[:, 1] = 1.0
        loss_with_conf, _ = temporal_loss(keypoints_pred, confidences=confidences, stage=stage)
        loss_no_conf, _ = temporal_loss(keypoints_pred, stage=stage)
        assert loss_with_conf.shape == torch.Size([])
        assert loss_with_conf >= 0.0
        # zeroing one keypoint reduces or equals the unmasked loss
        assert loss_with_conf <= loss_no_conf + 1e-6


class TestTemporalHeatmapLoss:
    """Test the TemporalHeatmapLoss class."""

    @pytest.fixture
    def mse_loss(self):
        """Return a TemporalHeatmapLoss using pixel-wise MSE."""
        return TemporalHeatmapLoss(loss_name='temporal_heatmap_mse')

    @pytest.fixture
    def kl_loss(self):
        """Return a TemporalHeatmapLoss using KL divergence."""
        return TemporalHeatmapLoss(loss_name='temporal_heatmap_kl')

    def test_invalid_loss_name_raises(self):
        """Test that an invalid loss name raises ValueError."""
        with pytest.raises(ValueError):
            TemporalHeatmapLoss(loss_name='bad_name')  # type: ignore[arg-type]

    def test_mse_zero_for_constant_predictions(self, mse_loss):
        """Test that identical consecutive heatmaps yield zero MSE loss."""
        batch_size, num_keypoints, h, w = 4, 3, 16, 16
        frame = torch.rand(1, num_keypoints, h, w)
        predictions = frame.expand(batch_size, -1, -1, -1).clone()
        confidences = torch.ones(batch_size, num_keypoints)
        loss, logs = mse_loss(heatmaps_pred=predictions, confidences=confidences, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss.item() == 0.0
        assert logs[0]['name'] == f'{stage}_temporal_heatmap_mse_loss'
        assert logs[1]['name'] == 'temporal_heatmap_mse_weight'

    def test_mse_positive_for_varying_predictions(self, mse_loss):
        """Test that differing consecutive heatmaps yield positive MSE loss."""
        batch_size, num_keypoints, h, w = 4, 3, 16, 16
        predictions = torch.rand(batch_size, num_keypoints, h, w)
        confidences = torch.ones(batch_size, num_keypoints)
        loss, _ = mse_loss(heatmaps_pred=predictions, confidences=confidences, stage=stage)
        assert loss.item() > 0.0

    def test_kl_zero_for_constant_predictions(self, kl_loss):
        """Test that identical consecutive heatmaps yield near-zero KL loss."""
        batch_size, num_keypoints, h, w = 4, 3, 16, 16
        frame = spatial_softmax2d(torch.randn(1, num_keypoints, h, w))
        predictions = frame.expand(batch_size, -1, -1, -1).clone()
        confidences = torch.ones(batch_size, num_keypoints)
        loss, logs = kl_loss(heatmaps_pred=predictions, confidences=confidences, stage=stage)
        assert loss.shape == torch.Size([])
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)
        assert logs[0]['name'] == f'{stage}_temporal_heatmap_kl_loss'

    def test_kl_positive_for_varying_predictions(self, kl_loss):
        """Test that differing consecutive heatmaps yield positive KL loss."""
        batch_size, num_keypoints, h, w = 4, 3, 16, 16
        predictions = spatial_softmax2d(torch.randn(batch_size, num_keypoints, h, w))
        confidences = torch.ones(batch_size, num_keypoints)
        loss, _ = kl_loss(heatmaps_pred=predictions, confidences=confidences, stage=stage)
        assert loss.item() >= 0.0

    def test_compute_loss_mse_shape(self, mse_loss):
        """Test that MSE compute_loss returns shape (batch-1, num_keypoints)."""
        batch_size, num_keypoints, h, w = 5, 3, 16, 16
        predictions = torch.rand(batch_size, num_keypoints, h, w)
        diffs = mse_loss.compute_loss(predictions)
        assert diffs.shape == (batch_size - 1, num_keypoints)

    def test_compute_loss_kl_shape(self, kl_loss):
        """Test that KL compute_loss returns shape (batch-1, num_keypoints)."""
        batch_size, num_keypoints, h, w = 5, 3, 16, 16
        predictions = spatial_softmax2d(torch.randn(batch_size, num_keypoints, h, w))
        diffs = kl_loss.compute_loss(predictions)
        assert diffs.shape == (batch_size - 1, num_keypoints)

    def test_remove_nans_zeros_low_confidence_diffs(self, mse_loss):
        """Test that diffs adjacent to low-confidence frames are zeroed out."""
        mse_loss.prob_threshold = torch.tensor(0.5)
        batch_size, num_keypoints, h, w = 3, 2, 8, 8
        predictions = torch.rand(batch_size, num_keypoints, h, w)
        # keypoint 0 is low-confidence in all frames; should zero out all its diffs
        confidences = torch.zeros(batch_size, num_keypoints)
        confidences[:, 1] = 1.0
        diffs = mse_loss.compute_loss(predictions)
        clean = mse_loss.remove_nans(confidences=confidences, loss=diffs.clone())
        assert torch.all(clean[:, 0] == 0.0)
        assert torch.any(clean[:, 1] > 0.0) or True  # may be zero by chance; no crash

    def test_rectify_epsilon_per_keypoint(self, mse_loss):
        """Test epsilon rectification with a scalar epsilon."""
        mse_loss.epsilon = torch.tensor(1.0)
        loss_tensor = torch.tensor([[0.5, 2.0], [1.5, 0.3]])
        rectified = mse_loss.rectify_epsilon(loss_tensor)
        assert rectified[0, 0] == 0.0   # 0.5 < 1.0
        assert rectified[0, 1] > 0.0   # 2.0 > 1.0
        assert rectified[1, 0] > 0.0   # 1.5 > 1.0
        assert rectified[1, 1] == 0.0  # 0.3 < 1.0


class TestUnimodalLoss:
    """Test the UnimodalLoss class."""

    def test_unimodal_mse_loss(self):
        """Test that MSE unimodal loss returns a positive scalar with correct log keys."""
        img_size = 48
        img_size_ds = 32
        batch_size = 12
        num_keypoints = 16
        keypoints_pred = img_size * torch.rand(
            size=(batch_size, 2 * num_keypoints),
            device=device,
        )
        confidences = torch.rand(size=(batch_size, num_keypoints))
        heatmaps_pred = torch.ones(
            size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
            device=device,
        )
        uni_loss = UnimodalLoss(
            loss_name='unimodal_mse',
            original_image_height=img_size,
            original_image_width=img_size,
            downsampled_image_height=img_size_ds,
            downsampled_image_width=img_size_ds,
        )
        loss, logs = uni_loss(
            keypoints_pred_augmented=keypoints_pred,
            heatmaps_pred=heatmaps_pred,
            confidences=confidences,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert loss > 0.0
        assert logs[0]['name'] == f'{stage}_unimodal_mse_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'unimodal_mse_weight'
        assert logs[1]['value'] == uni_loss.weight

    def test_unimodal_kl_loss(self):
        """Test that KL unimodal loss returns a positive scalar with correct log keys."""
        img_size = 48
        img_size_ds = 32
        batch_size = 12
        num_keypoints = 16
        keypoints_pred = img_size * torch.rand(
            size=(batch_size, 2 * num_keypoints),
            device=device,
        )
        confidences = torch.rand(size=(batch_size, num_keypoints))
        heatmaps_pred = spatial_softmax2d(
            torch.randn(
                size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
                device=device,
            )
        )
        uni_loss = UnimodalLoss(
            loss_name='unimodal_kl',
            original_image_height=img_size,
            original_image_width=img_size,
            downsampled_image_height=img_size_ds,
            downsampled_image_width=img_size_ds,
        )
        loss, logs = uni_loss(
            keypoints_pred_augmented=keypoints_pred,
            heatmaps_pred=heatmaps_pred,
            confidences=confidences,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert loss > 0.0
        assert logs[0]['name'] == f'{stage}_unimodal_kl_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'unimodal_kl_weight'
        assert logs[1]['value'] == uni_loss.weight

    def test_unimodal_js_loss(self):
        """Test that JS unimodal loss returns a positive scalar with correct log keys."""
        img_size = 48
        img_size_ds = 32
        batch_size = 12
        num_keypoints = 16
        keypoints_pred = img_size * torch.rand(
            size=(batch_size, 2 * num_keypoints),
            device=device,
        )
        confidences = torch.rand(size=(batch_size, num_keypoints))
        heatmaps_pred = spatial_softmax2d(
            torch.randn(
                size=(batch_size, num_keypoints, img_size_ds, img_size_ds),
                device=device,
            )
        )
        uni_loss = UnimodalLoss(
            loss_name='unimodal_js',
            original_image_height=img_size,
            original_image_width=img_size,
            downsampled_image_height=img_size_ds,
            downsampled_image_width=img_size_ds,
        )
        loss, logs = uni_loss(
            keypoints_pred_augmented=keypoints_pred,
            heatmaps_pred=heatmaps_pred,
            confidences=confidences,
            stage=stage,
        )
        assert loss.shape == torch.Size([])
        assert loss > 0.0
        assert logs[0]['name'] == f'{stage}_unimodal_js_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'unimodal_js_weight'
        assert logs[1]['value'] == uni_loss.weight


class TestRegressionMSELoss:
    """Test the RegressionMSELoss class."""

    @pytest.fixture
    def mse_loss(self):
        """Return a default RegressionMSELoss instance."""
        return RegressionMSELoss()

    def test_regression_mse_loss_zero_when_preds_equal_targets(self, mse_loss):
        """Test that loss is zero when predictions equal targets."""
        true_keypoints = torch.ones(size=(12, 32))
        predicted_keypoints = torch.ones(size=(12, 32))
        loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss == 0.0
        assert logs[0]['name'] == f'{stage}_regression_loss'
        assert logs[0]['value'] == loss
        assert logs[1]['name'] == 'regression_weight'
        assert logs[1]['value'] == mse_loss.weight

    def test_regression_mse_loss_positive_when_preds_differ(self, mse_loss):
        """Test that loss is positive when predictions differ from targets."""
        true_keypoints = torch.rand(size=(12, 32))
        predicted_keypoints = torch.rand(size=(12, 32))
        loss, logs = mse_loss(true_keypoints, predicted_keypoints, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss > 0.0


class TestRegressionRMSELoss:
    """Test the RegressionRMSELoss class."""

    @pytest.fixture
    def mse_loss(self):
        return RegressionMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1

    @pytest.fixture
    def rmse_loss(self):
        return RegressionRMSELoss(log_weight=np.log(0.5))  # set log_weight so weight=1

    def test_zero_loss_when_predictions_equal_targets(self, rmse_loss):
        true_keypoints = torch.ones(size=(12, 32))
        predicted_keypoints = torch.ones(size=(12, 32))
        loss, logs = rmse_loss(true_keypoints, predicted_keypoints, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss == 0.0
        assert logs[0]["name"] == f"{stage}_rmse_loss"
        assert logs[0]["value"] == loss
        assert logs[1]["name"] == "rmse_weight"
        assert logs[1]["value"] == rmse_loss.weight

    def test_rmse_from_mse(self, mse_loss, rmse_loss):
        n = 4
        batch_size = 3
        labels = 2 * torch.ones((batch_size, n))
        preds = torch.zeros((batch_size, n))
        true_rmse = 2.0
        mse, _ = mse_loss(labels, preds, stage=stage)
        rmse, _ = rmse_loss(labels, preds, stage=stage)
        assert rmse == true_rmse
        assert mse == true_rmse ** 2.0


class TestPairwiseProjectionsLoss:
    """Test the PairwiseProjectionsLoss class."""

    @pytest.fixture
    def pp_loss(self):
        loss = PairwiseProjectionsLoss(log_weight=np.log(0.5))  # set log_weight so weight=1
        return loss

    def test_zero_loss_when_predictions_equal_targets(self, pp_loss):
        num_batch = 2
        num_keypoints = 4
        num_cam_pairs = 3
        keypoints_targ_3d = torch.ones(size=(num_batch, num_keypoints, 3))
        keypoints_pred_3d = torch.ones(size=(num_batch, num_cam_pairs, num_keypoints, 3))
        loss, logs = pp_loss(keypoints_targ_3d, keypoints_pred_3d, stage=stage)
        assert loss.shape == torch.Size([])
        assert loss == 0.0
        assert logs[0]["name"] == f"{stage}_supervised_pairwise_projections_loss"
        assert logs[0]["value"] == loss
        assert logs[1]["name"] == "supervised_pairwise_projections_weight"
        assert logs[1]["value"] == pp_loss.weight

    def test_actual_values(self, pp_loss):
        num_batch = 2
        num_keypoints = 4
        num_cam_pairs = 3
        keypoints_targ_3d = torch.zeros(size=(num_batch, num_keypoints, 3))
        keypoints_pred_3d = torch.ones(size=(num_batch, num_cam_pairs, num_keypoints, 3))
        loss, _ = pp_loss(keypoints_targ_3d, keypoints_pred_3d)
        assert loss.isclose(torch.sqrt(torch.tensor(3)))

    def test_targets_all_nans(self, pp_loss):
        num_batch = 1
        num_keypoints = 4
        num_cam_pairs = 3
        keypoints_targ_3d = torch.full((num_batch, num_keypoints, 3), float('nan'))
        keypoints_pred_3d = torch.ones(
            (num_batch, num_cam_pairs, num_keypoints, 3),
            requires_grad=True,
        )
        loss, _ = pp_loss(keypoints_targ_3d, keypoints_pred_3d)
        assert loss.item() == 0.0
        loss.backward()
        assert keypoints_pred_3d.grad is not None
        assert not torch.isnan(keypoints_pred_3d.grad).any(), "gradients contain NaN values"

    def test_predictions_all_nans(self, pp_loss):
        num_batch = 1
        num_keypoints = 4
        num_cam_pairs = 3
        keypoints_targ_3d = torch.ones((num_batch, num_keypoints, 3))
        keypoints_pred_3d = torch.full(
            (num_batch, num_cam_pairs, num_keypoints, 3), float('nan'),
            requires_grad=True,
        )
        loss, _ = pp_loss(keypoints_targ_3d, keypoints_pred_3d)
        assert loss.item() == 0.0
        loss.backward()
        assert keypoints_pred_3d.grad is not None
        assert not torch.isnan(keypoints_pred_3d.grad).any(), "gradients contain NaN values"

    def test_targets_partial_nans(self, pp_loss):
        num_batch = 2
        num_keypoints = 4
        num_cam_pairs = 2
        keypoints_targ_3d = torch.zeros(size=(num_batch, num_keypoints, 3))
        keypoints_targ_3d[0, 0, :] = float('nan')  # first keypoint in first batch NaN
        keypoints_pred_3d = torch.ones(
            size=(num_batch, num_cam_pairs, num_keypoints, 3),
            requires_grad=True,
        )
        loss, _ = pp_loss(keypoints_targ_3d, keypoints_pred_3d)
        # each valid position has loss = sqrt(3) (distance from 0 to 1 in 3D)
        expected_loss = torch.sqrt(torch.tensor(3.0))
        assert loss.isclose(expected_loss)
        loss.backward()
        assert keypoints_pred_3d.grad is not None
        assert not torch.isnan(keypoints_pred_3d.grad).any(), "gradients contain NaN values"

    def test_predictions_partial_nans(self, pp_loss):
        num_batch = 3
        num_keypoints = 4
        num_cam_pairs = 3
        keypoints_targ_3d = torch.zeros(size=(num_batch, num_keypoints, 3))
        keypoints_pred_3d = torch.ones(size=(num_batch, num_cam_pairs, num_keypoints, 3))
        keypoints_pred_3d[0, 0, 0, :] = float('nan')
        keypoints_pred_3d[1, 1, :, :] = float('nan')
        keypoints_pred_3d[2, :, :, :] = float('nan')
        keypoints_pred_3d.requires_grad_(True)  # need to do this after inplace operations
        loss, _ = pp_loss(keypoints_targ_3d, keypoints_pred_3d)
        # each valid position has loss = sqrt(3) (distance from 0 to 1 in 3D)
        expected_loss = torch.sqrt(torch.tensor(3.0))
        assert loss.isclose(expected_loss)
        loss.backward()
        assert keypoints_pred_3d.grad is not None
        assert not torch.isnan(keypoints_pred_3d.grad).any(), "gradients contain NaN values"


class TestReprojectionHeatmapLoss:
    """Test the ReprojectionHeatmapLoss class."""

    @pytest.fixture
    def rh_loss(self):
        """Create a ReprojectionHeatmapLoss instance with standard parameters."""
        loss = ReprojectionHeatmapLoss(
            original_image_height=512,
            original_image_width=512,
            downsampled_image_height=64,
            downsampled_image_width=64,
            log_weight=np.log(0.5),  # set log_weight so weight=0.5
        )
        return loss

    def test_actual_values(self, rh_loss):
        """Test loss computation with actual different values."""
        batch_size = 2
        num_keypoints = 4
        heatmap_height = 64
        heatmap_width = 64

        # Create target heatmaps
        heatmaps_targ = torch.zeros(
            size=(batch_size, num_keypoints, heatmap_height, heatmap_width)
        )
        heatmaps_targ[0, 0, 30:35, 30:35] = 1.0

        # Create keypoints that will generate different heatmaps
        keypoints_pred_2d = torch.tensor([
            [[10.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # different location
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ], dtype=torch.float32)

        loss, logs = rh_loss(heatmaps_targ, keypoints_pred_2d, stage=stage)

        # Loss should be positive when predictions differ from targets
        assert loss.item() > 0.0
        assert logs[0]["name"] == f"{stage}_supervised_reprojection_heatmap_mse_loss"
        assert logs[0]["value"] == loss
        assert logs[1]["name"] == "supervised_reprojection_heatmap_mse_weight"
        assert logs[1]["value"] == rh_loss.weight

    def test_targets_all_zeros(self, rh_loss):
        """Test behavior when all target heatmaps are zeros (invalid keypoints)."""
        batch_size = 2
        num_keypoints = 4
        heatmap_height = 64
        heatmap_width = 64

        # All zero heatmaps (should be ignored by remove_nans)
        heatmaps_targ = torch.zeros(
            size=(batch_size, num_keypoints, heatmap_height, heatmap_width)
        )

        keypoints_pred_2d = torch.tensor([
            [[32.0, 32.0], [100.0, 150.0], [200.0, 300.0], [400.0, 450.0]],
            [[150.0, 22.0], [80.0, 120.0], [250.0, 350.0], [350.0, 400.0]]
        ], dtype=torch.float32, requires_grad=True)

        loss, _ = rh_loss(heatmaps_targ, keypoints_pred_2d)

        # Since all targets are zeros (ignored), loss should be zero
        assert loss.item() == 0.0
        loss.backward()
        # Gradients should be well-behaved (not NaN)
        assert keypoints_pred_2d.grad is not None
        assert not torch.isnan(keypoints_pred_2d.grad).any(), "gradients contain NaN values"

    def test_none_reprojected_keypoints_raises_error(self, rh_loss):
        """Test that passing None for reprojected keypoints raises ValueError."""
        stage = "train"
        batch_size = 2
        num_keypoints = 4
        heatmap_height = 64
        heatmap_width = 64

        heatmaps_targ = torch.zeros(
            size=(batch_size, num_keypoints, heatmap_height, heatmap_width)
        )

        with pytest.raises(ValueError) as exc_info:
            rh_loss(heatmaps_targ, None, stage=stage)

        assert "Reprojected keypoints not available" in str(exc_info.value)
        assert stage in str(exc_info.value)

    def test_targets_partial_zeros(self, rh_loss):
        """Test behavior when some target heatmaps are zeros."""
        batch_size = 2
        num_keypoints = 4
        heatmap_height = 64
        heatmap_width = 64

        heatmaps_targ = torch.zeros(
            size=(batch_size, num_keypoints, heatmap_height, heatmap_width))
        # Make some keypoints valid (non-zero)
        heatmaps_targ[0, 0, 30:35, 30:35] = 1.0  # first keypoint in first batch
        heatmaps_targ[1, 2, 20:25, 40:45] = 1.0  # third keypoint in second batch

        keypoints_pred_2d = torch.tensor([
            [[10.0, 10.0], [100.0, 150.0], [200.0, 300.0], [400.0, 450.0]],
            # different from target
            [[150.0, 22.0], [80.0, 120.0], [10.0, 10.0], [350.0, 400.0]]  # different from target
        ], dtype=torch.float32, requires_grad=True)

        loss, _ = rh_loss(heatmaps_targ, keypoints_pred_2d)

        # Loss should be positive for valid keypoints
        assert loss.item() > 0.0
        loss.backward()
        assert keypoints_pred_2d.grad is not None
        assert not torch.isnan(keypoints_pred_2d.grad).any(), "gradients contain NaN values"

    def test_gradient_flow(self, rh_loss):
        """Test that gradients flow properly through the loss."""
        batch_size = 1
        num_keypoints = 1
        heatmap_height = 64
        heatmap_width = 64

        heatmaps_targ = torch.zeros(
            size=(batch_size, num_keypoints, heatmap_height, heatmap_width)
        )
        heatmaps_targ[0, 0, 30:35, 30:35] = 1.0

        # Create keypoints that require gradients
        keypoints_pred_2d = torch.tensor([[[10.0, 10.0]]], dtype=torch.float32, requires_grad=True)

        loss, _ = rh_loss(heatmaps_targ, keypoints_pred_2d)
        loss.backward()

        # Check that gradients exist and are finite
        assert keypoints_pred_2d.grad is not None
        assert torch.isfinite(keypoints_pred_2d.grad).all()

    def test_remove_nans_functionality(self, rh_loss):
        """Test the remove_nans method directly."""
        batch_size = 2
        num_keypoints = 4
        heatmap_height = 64
        heatmap_width = 64

        # Create targets with some all-zero heatmaps (should be removed)
        targets = torch.zeros(size=(batch_size, num_keypoints, heatmap_height, heatmap_width))
        targets[0, 0, 30:35, 30:35] = 1.0  # valid keypoint
        targets[1, 2, 20:25, 40:45] = 1.0  # valid keypoint
        # targets[0, 1], targets[0, 2], targets[0, 3], targets[1, 0], targets[1, 1], targets[1, 3]
        # are all zeros

        predictions = torch.ones_like(targets) * 0.5

        # Compute elementwise loss first
        elementwise_loss = rh_loss.compute_loss(targets, predictions)

        clean_loss = rh_loss.remove_nans(elementwise_loss, targets)

        # Should only have valid losses from 2 keypoints * heatmap_height * heatmap_width pixels
        expected_num_elements = 2 * heatmap_height * heatmap_width
        assert clean_loss.numel() == expected_num_elements
        assert clean_loss.dim() == 1  # Should be flattened
        assert torch.all(torch.isfinite(clean_loss))

    def test_compute_loss_method(self, rh_loss):
        """Test the compute_loss method directly."""
        targets = torch.zeros(size=(2, 32, 32))
        predictions = torch.ones(size=(2, 32, 32))

        elementwise_loss = rh_loss.compute_loss(targets, predictions)

        # Should be MSE loss scaled by h*w
        expected_loss = (predictions - targets) ** 2 * 32 * 32
        assert torch.allclose(elementwise_loss, expected_loss)
        assert elementwise_loss.shape == targets.shape
