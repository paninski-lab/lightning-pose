"""Tests for heatmap generation and evaluation utilities."""

import copy

import torch
from kornia.geometry.subpix import spatial_expectation2d, spatial_softmax2d

from lightning_pose.data.heatmaps import generate_heatmaps


class TestGenerateHeatmaps:

    def test_basic(self, cfg, heatmap_dataset):

        im_height = cfg.data.image_resize_dims.height
        im_width = cfg.data.image_resize_dims.width

        batch = heatmap_dataset.__getitem__(idx=0)
        heatmap_gt = batch['heatmaps'].unsqueeze(0)
        keypts_gt = batch['keypoints'].unsqueeze(0).reshape(1, -1, 2)
        heatmap_torch = generate_heatmaps(
            keypts_gt,
            height=im_height,
            width=im_width,
            output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
        )

        # find soft argmax and confidence of ground truth heatmap
        softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
        preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
        confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

        # find soft argmax and confidence of generated heatmap
        softmaxes_torch = spatial_softmax2d(heatmap_torch, temperature=torch.tensor(100))
        preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
        confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

        assert (preds_gt == preds_torch).all()
        assert (confidences_gt == confidences_torch).all()

        # cleanup
        del batch
        del heatmap_gt, keypts_gt
        del softmaxes_gt, preds_gt, confidences_gt
        del softmaxes_torch, preds_torch, confidences_torch
        torch.cuda.empty_cache()  # remove tensors from gpu

    def test_uniform_heatmaps(self, cfg, toy_data_dir):
        """uniform_heatmaps=True: dataset heatmaps match generate_heatmaps with synthesized vis."""

        from lightning_pose.data import get_dataset, get_imgaug_transform
        from lightning_pose.data.datasets import HeatmapDataset

        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.model_type = 'heatmap'
        cfg_tmp.training.uniform_heatmaps_for_nan_keypoints = True

        imgaug_transform = get_imgaug_transform(cfg_tmp)
        heatmap_dataset = get_dataset(
            cfg_tmp,
            data_dir=toy_data_dir,
            imgaug_transform=imgaug_transform,
        )

        assert isinstance(heatmap_dataset, HeatmapDataset)
        im_height = cfg.data.image_resize_dims.height
        im_width = cfg.data.image_resize_dims.width

        batch = heatmap_dataset.__getitem__(idx=0)
        heatmap_gt = batch['heatmaps'].unsqueeze(0)  # type: ignore[typeddict-item]
        keypts_gt = batch['keypoints'].unsqueeze(0).reshape(1, -1, 2)
        assert heatmap_dataset.visibility is not None
        vis = heatmap_dataset.visibility[0].unsqueeze(0)  # (1, K) synthesized visibility

        heatmap_uniform_torch = generate_heatmaps(
            keypts_gt,
            height=im_height,
            width=im_width,
            output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
            visibility=vis,
        )

        softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
        preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
        confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

        softmaxes_torch = spatial_softmax2d(heatmap_uniform_torch, temperature=torch.tensor(100))
        preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
        confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

        assert (preds_gt == preds_torch).all()
        assert (confidences_gt == confidences_torch).all()

        torch.cuda.empty_cache()

    def test_weird_shape(self, cfg, toy_data_dir):

        from lightning_pose.data import get_dataset, get_imgaug_transform

        img_shape = (384, 256)

        # update config
        cfg_tmp = copy.deepcopy(cfg)
        cfg_tmp.model.model_type = 'heatmap'
        cfg_tmp.data.image_resize_dims.height = img_shape[0]
        cfg_tmp.data.image_resize_dims.width = img_shape[1]

        # build dataset with these new image dimensions
        imgaug_transform = get_imgaug_transform(cfg_tmp)
        dataset = get_dataset(
            cfg_tmp,
            data_dir=toy_data_dir,
            imgaug_transform=imgaug_transform,
        )

        # now same test as `test_basic`
        batch = dataset.__getitem__(idx=0)
        heatmap_gt = batch['heatmaps'].unsqueeze(0)  # type: ignore[typeddict-item]
        keypts_gt = batch['keypoints'].unsqueeze(0).reshape(1, -1, 2)
        heatmap_torch = generate_heatmaps(
            keypts_gt,
            height=img_shape[0],
            width=img_shape[1],
            output_shape=(heatmap_gt.shape[2], heatmap_gt.shape[3]),
        )

        # find soft argmax and confidence of ground truth heatmap
        softmaxes_gt = spatial_softmax2d(heatmap_gt, temperature=torch.tensor(100))
        preds_gt = spatial_expectation2d(softmaxes_gt, normalized_coordinates=False)
        confidences_gt = torch.amax(softmaxes_gt, dim=(2, 3))

        # find soft argmax and confidence of generated heatmap
        softmaxes_torch = spatial_softmax2d(heatmap_torch, temperature=torch.tensor(100))
        preds_torch = spatial_expectation2d(softmaxes_torch, normalized_coordinates=False)
        confidences_torch = torch.amax(softmaxes_torch, dim=(2, 3))

        assert (preds_gt == preds_torch).all()
        assert (confidences_gt == confidences_torch).all()

        # cleanup
        del batch
        del heatmap_gt, keypts_gt
        del softmaxes_gt, preds_gt, confidences_gt
        del softmaxes_torch, preds_torch, confidences_torch
        torch.cuda.empty_cache()  # remove tensors from gpu

    def test_keep_gradients(self):
        """Test that gradients flow through keypoints when keep_gradients=True."""

        # Create mock data
        im_height = 256
        im_width = 256
        output_height = 64
        output_width = 64

        # Create keypoints that require gradients
        keypts_with_grad = torch.tensor([
            [[32.0, 64.0], [128.0, 96.0], [200.0, 150.0], [100.0, 200.0]],  # batch 1
            [[64.0, 32.0], [160.0, 120.0], [180.0, 180.0], [120.0, 220.0]]  # batch 2
        ], dtype=torch.float32, requires_grad=True)

        # Generate heatmaps with gradients enabled
        heatmap_torch = generate_heatmaps(
            keypts_with_grad,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            keep_gradients=True,
        )

        # Compute a simple loss and backpropagate
        loss = torch.sum(heatmap_torch)
        loss.backward()

        # Check that gradients exist and are finite
        assert keypts_with_grad.grad is not None, 'No gradients computed for keypoints'
        assert torch.isfinite(keypts_with_grad.grad).all(), 'Gradients contain NaN or inf values'
        assert not torch.all(keypts_with_grad.grad == 0), 'All gradients are zero'

        # Test the opposite: gradients should NOT flow when keep_gradients=False
        keypts_no_grad = torch.tensor([
            [[32.0, 64.0], [128.0, 96.0], [200.0, 150.0], [100.0, 200.0]],  # batch 1
            [[64.0, 32.0], [160.0, 120.0], [180.0, 180.0], [120.0, 220.0]]  # batch 2
        ], dtype=torch.float32, requires_grad=True)

        heatmap_no_grad = generate_heatmaps(
            keypts_no_grad,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            keep_gradients=False,
        )

        # Check that the heatmap doesn't require gradients (computation graph is detached)
        assert not heatmap_no_grad.requires_grad, \
            'Heatmap should not require gradients when keep_gradients=False'

        # Since heatmap_no_grad doesn't require gradients, we can't call backward on it
        # Instead, verify that the original keypoints have no gradients after this operation
        # (they shouldn't since the computation was detached)
        assert keypts_no_grad.grad is None, 'Original keypoints should have no gradients yet'

    def test_out_of_bounds_nan_indices(self):
        """OOB/NaN keypoints produce zero, uniform, or Gaussian heatmaps depending on vis."""

        im_height = 256
        im_width = 256
        output_height = 64
        output_width = 64

        # batch=2, 4 keypoints each — mix of valid, OOB, and explicit NaN
        keypoints = torch.tensor([
            [
                [32.0, 32.0],    # valid
                [-10.0, 50.0],   # x OOB (< -1 after scaling)
                [500.0, 32.0],   # x OOB (> width + 1 after scaling)
                [32.0, 500.0],   # y OOB (> height + 1 after scaling)
            ],
            [
                [32.0, -10.0],   # y OOB (< -1 after scaling)
                [64.0, 64.0],    # valid
                [float('nan'), 32.0],  # explicit NaN
                [128.0, 128.0],  # valid
            ],
        ], dtype=torch.float32)

        zeros = torch.zeros(output_height, output_width)
        uniform = torch.ones(output_height, output_width) / (output_height * output_width)

        # visibility=None: OOB and NaN → zeros; valid → Gaussian
        heatmaps = generate_heatmaps(
            keypoints,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
        )
        assert torch.allclose(heatmaps[0, 1], zeros)  # x OOB
        assert torch.allclose(heatmaps[0, 2], zeros)  # x OOB
        assert torch.allclose(heatmaps[0, 3], zeros)  # y OOB
        assert torch.allclose(heatmaps[1, 0], zeros)  # y OOB
        assert torch.allclose(heatmaps[1, 2], zeros)  # explicit NaN
        assert not torch.allclose(heatmaps[0, 0], zeros)  # valid
        assert not torch.allclose(heatmaps[1, 1], zeros)  # valid
        assert not torch.allclose(heatmaps[1, 3], zeros)  # valid

        # vis=1 (occluded): all keypoints → uniform, regardless of OOB or NaN
        vis1 = torch.ones(2, 4, dtype=torch.long)
        heatmaps_v1 = generate_heatmaps(
            keypoints,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            visibility=vis1,
        )
        assert torch.allclose(heatmaps_v1[0, 1], uniform)  # x OOB → uniform
        assert torch.allclose(heatmaps_v1[1, 2], uniform)  # NaN → uniform

        # vis=0 (not labeled): all keypoints → zeros, including valid ones
        vis0 = torch.zeros(2, 4, dtype=torch.long)
        heatmaps_v0 = generate_heatmaps(
            keypoints,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            visibility=vis0,
        )
        assert torch.allclose(heatmaps_v0[0, 0], zeros)  # valid kp → zeros when vis=0
        assert torch.allclose(heatmaps_v0[1, 1], zeros)  # valid kp → zeros when vis=0

        # vis=2 (visible) + OOB/NaN → zeros (defensive); vis=2 + valid → Gaussian
        vis2 = torch.full((2, 4), 2, dtype=torch.long)
        heatmaps_v2 = generate_heatmaps(
            keypoints,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            visibility=vis2,
        )
        assert torch.allclose(heatmaps_v2[0, 1], zeros)    # x OOB → zero despite vis=2
        assert torch.allclose(heatmaps_v2[1, 2], zeros)    # NaN → zero despite vis=2
        assert not torch.allclose(heatmaps_v2[0, 0], zeros)  # valid → Gaussian
        assert not torch.allclose(heatmaps_v2[1, 1], zeros)  # valid → Gaussian

    def test_extreme_keypoint_clamping(self):
        """Extreme OOB keypoints are clamped; visibility flags still control heatmap type."""

        im_height = 256
        im_width = 256
        output_height = 64
        output_width = 64

        # batch=1, four keypoints all with extreme (way OOB) coordinates
        extreme_keypoints = torch.tensor([
            [
                [-100000000.0, 32.0],   # extremely negative x
                [100000000.0, 32.0],    # extremely positive x
                [32.0, -100000000.0],   # extremely negative y
                [32.0, 100000000.0],    # extremely positive y
            ]
        ], dtype=torch.float32, requires_grad=True)

        # --- gradients and finiteness (visibility=None path) ---
        heatmaps = generate_heatmaps(
            extreme_keypoints,
            height=im_height,
            width=im_width,
            output_shape=(output_height, output_width),
            keep_gradients=True,
        )
        assert torch.isfinite(heatmaps).all()
        assert heatmaps.shape == (1, 4, output_height, output_width)
        loss = heatmaps.sum()
        loss.backward()
        assert extreme_keypoints.grad is not None
        assert torch.isfinite(extreme_keypoints.grad).all()

        # visibility=None: OOB → zeros
        zero = torch.zeros(output_height, output_width)
        assert torch.allclose(heatmaps[0, 0], zero)
        assert torch.allclose(heatmaps[0, 1], zero)
        assert torch.allclose(heatmaps[0, 2], zero)
        assert torch.allclose(heatmaps[0, 3], zero)

        # vis=0: not labeled → zeros regardless of OOB
        vis0 = torch.zeros(1, 4, dtype=torch.long)
        heatmaps_v0 = generate_heatmaps(
            extreme_keypoints.detach(),
            height=im_height, width=im_width, output_shape=(output_height, output_width),
            visibility=vis0,
        )
        assert torch.allclose(heatmaps_v0[0, 0], zero)

        # vis=1: occluded → uniform regardless of OOB
        uniform = torch.ones(output_height, output_width) / (output_height * output_width)
        vis1 = torch.ones(1, 4, dtype=torch.long)
        heatmaps_v1 = generate_heatmaps(
            extreme_keypoints.detach(),
            height=im_height, width=im_width, output_shape=(output_height, output_width),
            visibility=vis1,
        )
        assert torch.allclose(heatmaps_v1[0, 0], uniform)

        # vis=2: visible but OOB → zero (defensive fallback)
        vis2 = torch.full((1, 4), 2, dtype=torch.long)
        heatmaps_v2 = generate_heatmaps(
            extreme_keypoints.detach(),
            height=im_height, width=im_width, output_shape=(output_height, output_width),
            visibility=vis2,
        )
        assert torch.allclose(heatmaps_v2[0, 0], zero)

    def test_generate_heatmaps_visibility_none_nan_produces_zeros(self):
        """visibility=None: NaN keypoints produce zero heatmaps."""

        keypoints = torch.tensor([[[64.0, 64.0], [float('nan'), float('nan')]]])

        result = generate_heatmaps(keypoints, height=128, width=128, output_shape=(32, 32))

        zero_heatmap = torch.zeros(32, 32)
        assert not torch.allclose(result[0, 0], zero_heatmap)  # valid keypoint → non-zero
        assert torch.allclose(result[0, 1], zero_heatmap)       # NaN keypoint → zero

    def test_generate_heatmaps_visibility_2_produces_gaussian(self):
        """vis=2 keypoints get a Gaussian heatmap identical to the legacy valid-keypoint path."""

        # Arrange
        keypoints = torch.tensor([[[64.0, 64.0]]])
        vis = torch.tensor([[2]])

        # Act
        expected = generate_heatmaps(keypoints, height=128, width=128, output_shape=(32, 32))
        result = generate_heatmaps(
            keypoints, height=128, width=128, output_shape=(32, 32), visibility=vis,
        )

        # Assert
        assert torch.allclose(result, expected)

    def test_generate_heatmaps_visibility_1_produces_uniform(self):
        """vis=1 keypoints get a uniform heatmap regardless of coordinate values."""

        H, W = 32, 32
        uniform_heatmap = torch.ones(H, W) / (H * W)

        # Case 1: NaN coordinates with vis=1
        keypoints_nan = torch.tensor([[[float('nan'), float('nan')]]])
        vis = torch.tensor([[1]])
        result_nan = generate_heatmaps(
            keypoints_nan, height=128, width=128, output_shape=(H, W), visibility=vis,
        )
        assert torch.allclose(result_nan[0, 0], uniform_heatmap)

        # Case 2: valid coordinates with vis=1 — coords are ignored, still uniform
        keypoints_valid = torch.tensor([[[64.0, 64.0]]])
        result_valid = generate_heatmaps(
            keypoints_valid, height=128, width=128, output_shape=(H, W), visibility=vis,
        )
        assert torch.allclose(result_valid[0, 0], uniform_heatmap)

    def test_generate_heatmaps_visibility_0_produces_zeros(self):
        """vis=0 keypoints get an all-zero heatmap regardless of coordinate values."""

        H, W = 32, 32
        zero_heatmap = torch.zeros(H, W)

        # Case 1: NaN coordinates with vis=0
        keypoints_nan = torch.tensor([[[float('nan'), float('nan')]]])
        vis = torch.tensor([[0]])
        result_nan = generate_heatmaps(
            keypoints_nan, height=128, width=128, output_shape=(H, W), visibility=vis,
        )
        assert torch.allclose(result_nan[0, 0], zero_heatmap)

        # Case 2: valid coordinates with vis=0 — coords are ignored, still zeros
        keypoints_valid = torch.tensor([[[64.0, 64.0]]])
        result_valid = generate_heatmaps(
            keypoints_valid, height=128, width=128, output_shape=(H, W), visibility=vis,
        )
        assert torch.allclose(result_valid[0, 0], zero_heatmap)

    def test_generate_heatmaps_visibility_mixed(self):
        """Mixed visibility flags produce per-keypoint correct heatmap types."""

        H, W = 32, 32
        # batch=1, K=3: vis=2 (visible), vis=1 (occluded), vis=0 (not labeled)
        keypoints = torch.tensor([[[64.0, 64.0], [float('nan'), float('nan')], [10.0, 10.0]]])
        vis = torch.tensor([[2, 1, 0]])

        result = generate_heatmaps(
            keypoints, height=128, width=128, output_shape=(H, W), visibility=vis,
        )

        zero_heatmap = torch.zeros(H, W)
        uniform_heatmap = torch.ones(H, W) / (H * W)

        # vis=2: Gaussian (non-zero, sums to 1)
        assert not torch.allclose(result[0, 0], zero_heatmap)
        assert torch.isclose(result[0, 0].sum(), torch.tensor(1.0))
        # vis=1: uniform (every pixel equal, sums to 1)
        assert torch.allclose(result[0, 1], uniform_heatmap)
        # vis=0: zeros
        assert torch.allclose(result[0, 2], zero_heatmap)

    def test_generate_heatmaps_visibility_2_nan_coords_falls_back_to_zeros(self):
        """vis=2 with NaN coordinates (invalid label) falls back to a zero heatmap."""

        H, W = 32, 32
        keypoints = torch.tensor([[[float('nan'), float('nan')]]])
        vis = torch.tensor([[2]])

        result = generate_heatmaps(
            keypoints, height=128, width=128, output_shape=(H, W), visibility=vis,
        )
        assert torch.allclose(result[0, 0], torch.zeros(H, W))


def test_evaluate_heatmaps_at_location():

    from lightning_pose.data.heatmaps import evaluate_heatmaps_at_location

    height = 24
    width = 12

    # make sure this works when we have a single frame and/or keypoint
    for n_batch in [1, 5]:
        for n_keypoints in [1, 6]:

            heatmaps = torch.zeros((n_batch, n_keypoints, height, width))

            h_locs = torch.randint(0, height, (n_batch, n_keypoints))
            w_locs = torch.randint(0, width, (n_batch, n_keypoints))
            locs = torch.stack([w_locs, h_locs], dim=2)  # x then y
            # set heatmaps values to .2 at 5 locations near the central pixel.
            for i, l1 in enumerate(locs):
                for j, l2 in enumerate(l1):
                    l2_1_offset, l2_0_offset = l2[1] + 1, l2[0] + 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] - 1, l2[0] - 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1], l2[0]
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] + 1, l2[0] - 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2

                    l2_1_offset, l2_0_offset = l2[1] - 1, l2[0] + 1
                    l2_1_offset = torch.clamp(l2_1_offset, min=0, max=height - 1)
                    l2_0_offset = torch.clamp(l2_0_offset, min=0, max=width - 1)
                    heatmaps[i, j, l2_1_offset, l2_0_offset] += 0.2
            # heatmap values should sum to 1 even when values are spread across the heatmap
            vals = evaluate_heatmaps_at_location(heatmaps=heatmaps, locs=locs)
            assert torch.all(vals == 1.0)

    # more tests

    batch = 1
    num_keypoints = 1
    heat_height = 32
    heat_width = 32

    # ----------------------------------
    # make delta heatmap
    # ----------------------------------
    idx0 = 5
    heatmaps = torch.zeros((batch, num_keypoints, heat_height, heat_width))
    heatmaps[0, 0, idx0, idx0] = 1

    # if we choose the correct location, do we get 1?
    locs0 = torch.zeros((batch, num_keypoints, 2))
    locs0[0, 0, 0] = idx0
    locs0[0, 0, 1] = idx0
    confs0 = evaluate_heatmaps_at_location(heatmaps, locs0)
    assert confs0.shape == (batch, num_keypoints)
    assert torch.allclose(confs0[0], torch.tensor(1.0))

    # if we choose almost the correct location, do we get 1?
    idx1 = idx0 + 1
    locs1 = torch.zeros((batch, num_keypoints, 2))
    locs1[0, 0, 0] = idx1
    locs1[0, 0, 1] = idx1
    confs1 = evaluate_heatmaps_at_location(heatmaps, locs1)
    assert torch.allclose(confs1[0], torch.tensor(1.0))

    # if we choose a completely wrong location, do we get 0?
    idx2 = 25
    locs2 = torch.zeros((batch, num_keypoints, 2))
    locs2[0, 0, 0] = idx2
    locs2[0, 0, 1] = idx2
    confs2 = evaluate_heatmaps_at_location(heatmaps, locs2)
    assert torch.allclose(confs2[0], torch.tensor(0.0))

    # ----------------------------------
    # make a gaussain heatmap
    # ----------------------------------
    heatmaps_g = generate_heatmaps(
        locs0,
        height=heat_height,
        width=heat_width,
        output_shape=(heat_height, heat_width),
    )

    # if we choose the correct location, do we get close to 1?
    confs0_g = evaluate_heatmaps_at_location(heatmaps_g, locs0)
    assert confs0_g[0] > 0
    assert confs0_g[0] <= 1.0

    # if we choose almost the correct location, do we get less than the correct location?
    confs1_g = evaluate_heatmaps_at_location(heatmaps_g, locs1)
    assert confs0_g[0] > confs1_g[0]

    # if we choose a completely wrong location, do we get 0?
    confs2_g = evaluate_heatmaps_at_location(heatmaps_g, locs2)
    assert torch.allclose(confs2_g[0], torch.tensor(0.0))
