import io

import numpy as np
import pytest
import torch
import torch.nn as nn

from lightning_pose.models.heads.heatmap import (
    BlockLayerNorm2d,
    HeatmapHead,
    LayerNorm2d,
    MaskedConv2d,
    initialize_upsampling_layers,
    make_group_mask,
    make_upsampling_layers,
    run_subpixelmaxima,
    upsample,
)


class TestMakeUpsamplingLayers:

    @pytest.mark.parametrize("n_layers", [1, 2, 3])
    def test_basic_parameters(self, n_layers):
        """Test with different numbers of layers."""
        model = make_upsampling_layers(
            in_channels=256,
            out_channels=64,
            int_channels=128,
            n_layers=n_layers,
        )

        assert isinstance(model, nn.Sequential)
        # should have 1 PixelShuffle + n_layers ConvTranspose2d
        assert len(model) == n_layers + 1
        assert isinstance(model[0], nn.PixelShuffle)
        for i in range(1, len(model)):
            assert isinstance(model[i], nn.ConvTranspose2d)

        # first ConvTranspose2d should take in_channels // 4
        assert model[1].in_channels == 256 // 4
        if n_layers > 1:
            assert model[1].out_channels == 128

        # last ConvTranspose2d should output out_channels
        assert model[-1].out_channels == 64
        if n_layers > 1:
            assert model[-1].in_channels == 128

    def test_conv_parameters(self):
        """Test that ConvTranspose2d layers have correct parameters."""
        model = make_upsampling_layers(
            in_channels=256,
            out_channels=64,
            int_channels=128,
            n_layers=2,
        )

        for i in range(1, len(model)):
            conv = model[i]
            assert conv.kernel_size == (3, 3)
            assert conv.stride == (2, 2)
            assert conv.padding == (1, 1)
            assert conv.output_padding == (1, 1)

    def test_forward_pass(self):
        """Test that the model can process input tensors correctly."""
        batch_size = 2
        in_channels = 256
        h, w = 8, 8

        model = make_upsampling_layers(
            in_channels=in_channels,
            out_channels=64,
            int_channels=128,
            n_layers=3
        )

        x = torch.randn(batch_size, in_channels, h, w)
        output = model(x)

        # PixelShuffle reduces channels by 4 and doubles spatial dims
        # Then each ConvTranspose2d doubles spatial dims
        # Total upsampling: 2 * 2^3 = 16x
        expected_h = h * 2 * (2 ** 3)  # 128
        expected_w = w * 2 * (2 ** 3)  # 128

        assert output.shape == (batch_size, 64, expected_h, expected_w)


class TestUpsample:

    def test_shapes(self):

        batch = 4
        num_keypoints = 2
        height = 8
        width = 16

        data = torch.rand((batch, num_keypoints, height, width))

        data_upsampled = upsample(data)

        assert data_upsampled.shape == (batch, num_keypoints, 2 * height, 2 * width)

    def test_align_corners(self):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 2, 2] = 1.0  # peak at (2, 2)
        inputs[0, 1, 6, 6] = 1.0  # peak at (6, 6)

        result = upsample(inputs)

        # with align_corners=False and 2x upsampling:
        # input pixel at (i, j) maps to output pixel at (2*i, 2*j)
        # verify peaks are near expected locations (with tolerance for interpolation)
        peak_0_y, peak_0_x = torch.where(result[0, 0] == result[0, 0].max())
        peak_1_y, peak_1_x = torch.where(result[0, 1] == result[0, 1].max())

        assert abs(peak_0_y[0].item() - 4) <= 1  # 2 * 2 = 4
        assert abs(peak_0_x[0].item() - 4) <= 1
        assert abs(peak_1_y[0].item() - 12) <= 1  # 6 * 2 = 12
        assert abs(peak_1_x[0].item() - 12) <= 1


class TestRunSubpixelMaxima:

    @pytest.mark.parametrize("downsample_factor", [1, 2])
    def test_interior_points(self, downsample_factor):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        val_0 = 2
        inputs[0, 0, val_0, val_0] = 1.0  # peak at (2, 2)
        val_1 = 4
        inputs[0, 1, val_1, val_1] = 1.0  # peak at (4, 4)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=downsample_factor, temperature=torch.tensor(1000.0),
        )

        xy = val_0 * (2 ** downsample_factor)
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([float(xy), float(xy)]), dtype=preds.dtype),
        )
        assert confidences[0, 0] == torch.tensor(1.0)

        xy = val_1 * (2 ** downsample_factor)
        assert torch.allclose(
            preds[0, 2:4],
            torch.tensor(np.array([float(xy), float(xy)]), dtype=preds.dtype),
        )
        assert confidences[0, 1] == torch.tensor(1.0)

    @pytest.mark.parametrize("downsample_factor", [1, 2])
    def test_boundary_points(self, downsample_factor):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 2
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 0, 0] = 1.0  # peak at (0, 0) (corner)
        inputs[0, 1, 0, 1] = 1.0  # peak at (0, 1) (edge)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=downsample_factor, temperature=torch.tensor(1000.0),
        )

        assert torch.isclose(preds[0, 0], torch.tensor(0.), atol=0.5)
        assert torch.isclose(preds[0, 1], torch.tensor(0.), atol=0.5)
        assert torch.isclose(confidences[0, 0], torch.tensor(1.0), rtol=1e-3)

        assert torch.isclose(preds[0, 2], torch.tensor(float(2 ** downsample_factor)), rtol=1e-1)
        assert torch.isclose(preds[0, 3], torch.tensor(0.), atol=0.5)
        assert torch.isclose(confidences[0, 1], torch.tensor(1.0), rtol=1e-3)

    def test_temperature(self):

        # test with multiple peaks to verify consistent scaling
        batch_size = 1
        num_keypoints = 1
        height, width = 8, 8

        inputs = torch.zeros(batch_size, num_keypoints, height, width)
        inputs[0, 0, 4, 4] = 1.0  # peak at (0, 0)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(1000.0),
        )
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert confidences[0, 0] == torch.tensor(1.0)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(100.0),
        )
        assert torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert confidences[0, 0] != torch.tensor(1.0)
        assert torch.isclose(confidences[0, 0], torch.tensor(1.0), 1e-3)

        preds, confidences = run_subpixelmaxima(
            inputs, downsample_factor=2, temperature=torch.tensor(10.0),
        )
        assert not torch.allclose(
            preds[0, 0:2],
            torch.tensor(np.array([16.0, 16.0]), dtype=preds.dtype),
        )
        assert not torch.isclose(confidences[0, 0], torch.tensor(1.0), 1e-2)
        assert confidences[0, 0] < torch.tensor(0.5)


class TestHeatmapHead:

    @pytest.fixture
    def basic_head(self):
        """Create a basic HeatmapHead instance for testing."""
        return HeatmapHead(
            backbone_arch="resnet50",
            in_channels=256,
            out_channels=17,
            deconv_out_channels=32,
            downsample_factor=2,
            final_softmax=True,
        )

    def test_initialization(self, basic_head):
        """Test that HeatmapHead initializes with correct attributes."""
        assert basic_head.backbone_arch == "resnet50"
        assert basic_head.in_channels == 256
        assert basic_head.out_channels == 17
        assert basic_head.deconv_out_channels == 32
        assert basic_head.downsample_factor == 2
        assert basic_head.final_softmax is True
        assert isinstance(basic_head.temperature, torch.Tensor)
        assert basic_head.temperature.item() == 1000.0

    def test_forward_shape(self, basic_head):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        features_h, features_w = 16, 16
        features = torch.randn(batch_size, 256, features_h, features_w)

        output = basic_head(features)

        assert output.shape[0] == batch_size
        assert output.shape[1] == 17  # out_channels
        assert output.shape[2] == features_h * (2 ** (basic_head.downsample_factor + 1))
        assert output.shape[3] == features_w * (2 ** (basic_head.downsample_factor + 1))

    def test_forward_with_softmax(self, basic_head):
        """Test that output is normalized when final_softmax=True."""

        features = torch.randn(1, 256, 8, 8)
        output = basic_head(features)

        # check that values are in valid probability range
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

        # check that each heatmap approximately sums to 1 (spatial softmax)
        for i in range(output.shape[1]):
            heatmap_sum = output[0, i].sum()
            assert torch.isclose(heatmap_sum, torch.tensor(1.0), atol=1e-5)

    def test_forward_without_softmax(self):
        """Test that output is not normalized when final_softmax=False."""
        head = HeatmapHead(
            backbone_arch="resnet50",
            in_channels=128,
            out_channels=10,
            downsample_factor=2,
            final_softmax=False,
        )

        features = torch.randn(1, 128, 8, 8)
        output = head(features)

        # check that each heatmap does not approximately sum to 1
        for i in range(output.shape[1]):
            heatmap_sum = output[0, i].sum()
            assert not torch.isclose(heatmap_sum, torch.tensor(1.0), atol=1e-5)

    def test_different_downsample_factors(self):
        """Test initialization with different downsample factors."""
        for downsample_factor in [1, 2, 3]:
            head = HeatmapHead(
                backbone_arch="resnet50",
                in_channels=256,
                out_channels=17,
                downsample_factor=downsample_factor,
                final_softmax=True,
            )
            assert head.downsample_factor == downsample_factor
            assert len(head.upsampling_layers) == 4 - downsample_factor + 1  # +1 for PixelShuffle

    def test_vit_backbone_architecture(self):
        """Test that ViT backbone reduces number of layers by 1."""
        head_vit = HeatmapHead(
            backbone_arch="vitb_imagenet",
            in_channels=128,
            out_channels=17,
            downsample_factor=2,
            final_softmax=True,
        )

        head_resnet = HeatmapHead(
            backbone_arch="resnet50",
            in_channels=128,
            out_channels=17,
            downsample_factor=2,
            final_softmax=True,
        )

        # ViT should have one fewer layer
        assert len(head_vit.upsampling_layers) == len(head_resnet.upsampling_layers) - 1

    # ── keypoint-group heads ──────────────────────────────────────────────────

    @staticmethod
    def _grouped(**kwargs):
        """resnet18 head (two ConvT layers), 5 keypoints in 3 groups, H = 4 shared + 3 x 4."""
        kwargs = {
            'groups': [0, 0, 1, 1, 2], 'shared_channels': 4, 'hidden_channels': 16, **kwargs,
        }
        return HeatmapHead(backbone_arch='resnet18', in_channels=64, out_channels=5, **kwargs)

    def test_heatmap_head_no_groups_builds_like_reference_nonlinear(self):
        torch.manual_seed(0)
        head = HeatmapHead(
            backbone_arch='resnet18', in_channels=64, out_channels=5, hidden_channels=16,
        )
        # the nonlinear head as built before keypoint groups existed
        torch.manual_seed(0)
        reference = nn.Sequential(
            *make_upsampling_layers(in_channels=64, out_channels=16, int_channels=16, n_layers=2),
            LayerNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 5, kernel_size=1),
        )
        initialize_upsampling_layers(reference)

        assert [type(m) for m in head.upsampling_layers] == [type(m) for m in reference]
        assert head.groups is None and head.shared_channels == 0
        state, state_ref = head.upsampling_layers.state_dict(), reference.state_dict()
        assert list(state) == list(state_ref)
        assert all(torch.equal(state[k], state_ref[k]) for k in state)

    def test_heatmap_head_no_groups_builds_like_reference_linear(self):
        torch.manual_seed(0)
        head = HeatmapHead(backbone_arch='resnet18', in_channels=64, out_channels=5)
        torch.manual_seed(0)
        reference = make_upsampling_layers(
            in_channels=64, out_channels=5, int_channels=5, n_layers=2,
        )
        initialize_upsampling_layers(reference)

        assert [type(m) for m in head.upsampling_layers] == [type(m) for m in reference]
        state, state_ref = head.upsampling_layers.state_dict(), reference.state_dict()
        assert list(state) == list(state_ref)
        assert all(torch.equal(state[k], state_ref[k]) for k in state)

    def test_heatmap_head_groups_same_tensors_as_nonlinear(self):
        torch.manual_seed(0)
        plain = HeatmapHead(
            backbone_arch='resnet18', in_channels=64, out_channels=5, hidden_channels=16,
        )
        torch.manual_seed(0)
        grouped = self._grouped()

        state_plain, state_grouped = plain.state_dict(), grouped.state_dict()
        assert list(state_plain) == list(state_grouped)
        assert all(state_plain[k].shape == state_grouped[k].shape for k in state_plain)
        key_readout = f'upsampling_layers.{len(plain.upsampling_layers) - 1}.weight'
        mask = grouped.upsampling_layers[-1].mask
        for key in state_plain:
            expected = state_plain[key] * mask if key == key_readout else state_plain[key]
            assert torch.equal(state_grouped[key], expected), key
        # masked entries hold exact zeros, the rest are the nonlinear head's initial values
        assert torch.all(state_grouped[key_readout][mask == 0] == 0)
        assert torch.all(state_grouped[key_readout][mask == 1] != 0)

    def test_heatmap_head_groups_layout(self):
        head = self._grouped()

        assert isinstance(head.upsampling_layers[-3], BlockLayerNorm2d)
        assert isinstance(head.upsampling_layers[-2], nn.ReLU)
        assert isinstance(head.upsampling_layers[-1], MaskedConv2d)
        assert head.upsampling_layers[-3].block_sizes == [4, 4, 4, 4]
        assert head.groups == [0, 0, 1, 1, 2] and head.shared_channels == 4
        # the mask is rebuilt from the constructor arguments, never stored in checkpoints
        assert not any(key.endswith('mask') for key in head.state_dict())

    def test_heatmap_head_groups_forward_shape(self):
        head = self._grouped()
        heatmaps = head(torch.randn(2, 64, 4, 4))
        assert heatmaps.shape == (2, 5, 32, 32)

    def test_heatmap_head_groups_gradient_isolation(self):
        torch.manual_seed(1)
        head = self._grouped()
        features = torch.randn(2, 64, 4, 4)
        target = torch.rand(2, 32, 32)
        convt, norm, readout = (
            head.upsampling_layers[-4], head.upsampling_layers[-3], head.upsampling_layers[-1],
        )
        shared = slice(0, 4)
        blocks = [slice(4, 8), slice(8, 12), slice(12, 16)]

        for idx_keypoint, idx_group in enumerate(head.groups):
            head.zero_grad()
            # loss on one keypoint's heatmap only
            (head(features)[:, idx_keypoint] * target).sum().backward()

            # masked readout weights never receive gradient
            assert torch.all(readout.weight.grad[readout.mask == 0] == 0)
            for idx_other, block in enumerate(blocks):
                grads = [
                    convt.weight.grad[:, block], convt.bias.grad[block],
                    norm.norm.weight.grad[block], norm.norm.bias.grad[block],
                ]
                if idx_other == idx_group:
                    assert all(g.abs().sum() > 0 for g in grads)
                else:
                    assert all(torch.all(g == 0) for g in grads), (idx_keypoint, idx_other)
            # the shared block is trained by every group
            assert convt.weight.grad[:, shared].abs().sum() > 0
            assert norm.norm.weight.grad[shared].abs().sum() > 0

    def test_heatmap_head_groups_masked_weights_stay_zero_under_adam(self):
        torch.manual_seed(2)
        head = self._grouped()
        readout = head.upsampling_layers[-1]
        weight_init = readout.weight.detach().clone()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)

        for _ in range(3):
            optimizer.zero_grad()
            (head(torch.randn(2, 64, 4, 4)) * torch.rand(2, 5, 32, 32)).sum().backward()
            optimizer.step()

        assert torch.all(readout.weight[readout.mask == 0] == 0)
        assert not torch.equal(readout.weight[readout.mask == 1], weight_init[readout.mask == 1])

    def test_heatmap_head_one_group_equals_nonlinear_head(self):
        torch.manual_seed(3)
        plain = HeatmapHead(
            backbone_arch='resnet18', in_channels=64, out_channels=5, hidden_channels=16,
        )
        with torch.no_grad():  # non-trivial weights everywhere, including the LayerNorm affine
            for param in plain.parameters():
                param.add_(torch.randn_like(param))
        grouped = self._grouped(groups=[0] * 5, shared_channels=0)
        grouped.load_state_dict(plain.state_dict())
        features = torch.randn(2, 64, 4, 4)

        assert torch.equal(grouped(features), plain(features))

    @pytest.mark.parametrize('shared_channels', [0, 4])
    def test_heatmap_head_groups_perturbing_a_block(self, shared_channels):
        torch.manual_seed(4)
        groups = [0, 0, 1, 1, 2]
        head = self._grouped(
            groups=groups, shared_channels=shared_channels,
            **({'hidden_channels': 15} if shared_channels == 0 else {}),
        ).eval()
        with torch.no_grad():
            for param in head.parameters():
                param.add_(torch.randn_like(param))
            head.upsampling_layers[-1].zero_masked_weights()
        block = (head.upsampling_layers[-4].out_channels - shared_channels) // 3
        features = torch.randn(2, 64, 4, 4)
        noise = {}

        def perturb(module, inputs, output):  # noise on the chosen channels of the last ConvT
            return output + noise['mask'] * torch.randn_like(output)

        head.upsampling_layers[-4].register_forward_hook(perturb)
        noise['mask'] = torch.zeros(1, head.upsampling_layers[-4].out_channels, 1, 1)
        with torch.no_grad():
            clean = head(features)
        blocks = {
            ('group', g): slice(shared_channels + g * block, shared_channels + (g + 1) * block)
            for g in range(3)
        }
        if shared_channels:
            blocks['shared'] = slice(0, shared_channels)

        for name, channels in blocks.items():
            noise['mask'] = torch.zeros_like(noise['mask'])
            noise['mask'][:, channels] = 1.0
            with torch.no_grad():
                perturbed = head(features)
            for idx_keypoint, idx_group in enumerate(groups):
                changed = not torch.equal(perturbed[:, idx_keypoint], clean[:, idx_keypoint])
                # a keypoint depends on the shared block and its own group's block only
                assert changed == (name == 'shared' or name == ('group', idx_group)), (
                    name, idx_keypoint,
                )

    def test_heatmap_head_groups_state_dict_round_trip(self):
        torch.manual_seed(5)
        head = self._grouped().eval()
        with torch.no_grad():
            for param in head.parameters():
                param.add_(torch.randn_like(param))
            head.upsampling_layers[-1].zero_masked_weights()
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)

        reloaded = self._grouped().eval()
        reloaded.load_state_dict(torch.load(buffer), strict=True)

        features = torch.randn(2, 64, 4, 4)
        assert torch.equal(reloaded(features), head(features))
        assert torch.equal(reloaded.upsampling_layers[-1].mask, head.upsampling_layers[-1].mask)

    def test_heatmap_head_groups_without_hidden_channels_raises(self):
        with pytest.raises(ValueError, match='groups require hidden_channels'):
            HeatmapHead(
                backbone_arch='resnet18', in_channels=64, out_channels=2, groups=[0, 1],
            )

    def test_heatmap_head_shared_channels_without_groups_raises(self):
        with pytest.raises(ValueError, match='shared_channels requires groups'):
            HeatmapHead(
                backbone_arch='resnet18', in_channels=64, out_channels=2, hidden_channels=8,
                shared_channels=2,
            )

    def test_heatmap_head_groups_wrong_length_raises(self):
        with pytest.raises(ValueError, match='one per keypoint'):
            self._grouped(groups=[0, 1, 2])

    def test_heatmap_head_groups_indivisible_raises(self):
        with pytest.raises(ValueError, match='does not split equally'):
            self._grouped(shared_channels=5)


    def test_heatmap_head_group_fan_in_gain_values(self):
        head = self._grouped(group_fan_in_gain=True)   # rows read 4 shared + 4 own of 16

        gain = head.upsampling_layers[-1].gain.flatten()

        assert torch.equal(gain, torch.full((5,), 2.0))

    def test_heatmap_head_group_fan_in_gain_keeps_initial_output(self):
        torch.manual_seed(5)
        plain = self._grouped()
        torch.manual_seed(5)
        gained = self._grouped(group_fan_in_gain=True)
        x = torch.randn(2, 64, 4, 4)

        assert torch.allclose(plain(x), gained(x), atol=1e-6)

    def test_heatmap_head_group_fan_in_gain_speeds_up_readout(self):
        x, target = torch.randn(2, 64, 4, 4), torch.rand(2, 5, 32, 32)
        deltas = {}
        for gain in (False, True):
            torch.manual_seed(6)
            head = self._grouped(group_fan_in_gain=gain)
            readout = head.upsampling_layers[-1]
            before = (readout.weight * readout.mask * readout.gain).detach().clone()
            optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
            optimizer.zero_grad()
            (head(x) * target).sum().backward()
            optimizer.step()
            after = readout.weight * readout.mask * readout.gain
            deltas[gain] = (after - before).detach()[readout.mask.expand_as(after) == 1].abs()

        # Adam's first step moves each raw weight by ~lr: the effective step scales with the gain
        assert torch.allclose(deltas[True], 2.0 * deltas[False], rtol=1e-3)

    def test_heatmap_head_group_fan_in_gain_off_by_default(self):
        head = self._grouped()

        gain = head.upsampling_layers[-1].gain

        assert torch.equal(gain, torch.ones_like(gain))

    def test_heatmap_head_group_fan_in_gain_state_dict_round_trip(self):
        torch.manual_seed(7)
        head = self._grouped(group_fan_in_gain=True)
        buffer = io.BytesIO()
        torch.save(head.state_dict(), buffer)
        buffer.seek(0)
        torch.manual_seed(8)
        loaded = self._grouped(group_fan_in_gain=True)
        loaded.load_state_dict(torch.load(buffer))
        x = torch.randn(2, 64, 4, 4)

        assert not any(key.endswith('.gain') for key in head.state_dict())   # non-persistent
        assert torch.equal(head(x), loaded(x))


class TestBlockLayerNorm2d:
    """Test the BlockLayerNorm2d module."""

    @staticmethod
    def _random_affine(module):
        with torch.no_grad():
            module.norm.weight.copy_(torch.randn_like(module.norm.weight))
            module.norm.bias.copy_(torch.randn_like(module.norm.bias))

    def test_block_layer_norm_single_block_equals_layer_norm(self):
        torch.manual_seed(0)
        block_norm = BlockLayerNorm2d(12, block_sizes=[12])
        self._random_affine(block_norm)
        norm = LayerNorm2d(12)
        norm.load_state_dict(block_norm.state_dict())
        x = 3.0 * torch.randn(2, 12, 5, 7) + 1.0

        assert torch.equal(block_norm(x), norm(x))

    def test_block_layer_norm_equals_layer_norm_per_block(self):
        torch.manual_seed(0)
        block_norm = BlockLayerNorm2d(12, block_sizes=[4, 2, 6])
        self._random_affine(block_norm)
        x = torch.randn(2, 12, 5, 7)

        out = block_norm(x)

        start = 0
        for size in (4, 2, 6):
            norm = LayerNorm2d(size)
            with torch.no_grad():
                norm.norm.weight.copy_(block_norm.norm.weight[start:start + size])
                norm.norm.bias.copy_(block_norm.norm.bias[start:start + size])
            expected = norm(x[:, start:start + size])
            assert torch.allclose(out[:, start:start + size], expected, atol=1e-6, rtol=1e-5)
            start += size

    def test_block_layer_norm_blocks_are_independent(self):
        block_norm = BlockLayerNorm2d(8, block_sizes=[4, 4])
        x = torch.randn(1, 8, 3, 3)
        x_changed = x.clone()
        x_changed[:, 4:] = 10.0 * torch.randn(1, 4, 3, 3)

        assert torch.equal(block_norm(x)[:, :4], block_norm(x_changed)[:, :4])

    def test_block_layer_norm_state_dict_matches_layer_norm(self):
        state = BlockLayerNorm2d(8, block_sizes=[4, 4]).state_dict()
        state_ref = LayerNorm2d(8).state_dict()
        assert list(state) == list(state_ref)
        assert all(state[k].shape == state_ref[k].shape for k in state)

    @pytest.mark.parametrize('block_sizes', [[4, 3], [], [8, 0]])
    def test_block_layer_norm_invalid_blocks_raise(self, block_sizes):
        with pytest.raises(ValueError, match='block_sizes'):
            BlockLayerNorm2d(8, block_sizes=block_sizes)


class TestMaskedConv2d:
    """Test the MaskedConv2d module."""

    def test_masked_conv_forward_uses_masked_weight(self):
        torch.manual_seed(0)
        mask = (torch.rand(3, 6) > 0.5).float()
        conv = MaskedConv2d(6, 3, mask=mask)
        x = torch.randn(2, 6, 4, 4)

        expected = nn.functional.conv2d(x, conv.weight * mask[:, :, None, None], conv.bias)
        assert torch.equal(conv(x), expected)

    def test_masked_conv_state_dict_matches_conv(self):
        conv = MaskedConv2d(6, 3, mask=torch.ones(3, 6))
        state, state_ref = conv.state_dict(), nn.Conv2d(6, 3, kernel_size=1).state_dict()
        assert list(state) == list(state_ref)
        assert all(state[k].shape == state_ref[k].shape for k in state)

    def test_masked_conv_zero_masked_weights(self):
        mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        conv = MaskedConv2d(2, 2, mask=mask)
        weight = conv.weight.detach().clone()

        conv.zero_masked_weights()

        assert torch.equal(conv.weight[:, :, 0, 0], weight[:, :, 0, 0] * mask)

    def test_masked_conv_wrong_mask_shape_raises(self):
        with pytest.raises(ValueError, match='mask shape'):
            MaskedConv2d(6, 3, mask=torch.ones(6, 3))


class TestMakeGroupMask:
    """Test the make_group_mask function."""

    def test_make_group_mask_layout(self):
        mask, block_sizes = make_group_mask([1, 0, 1], hidden_channels=10, shared_channels=2)

        assert block_sizes == [2, 4, 4]
        expected = torch.zeros(3, 10)
        expected[:, :2] = 1.0
        expected[1, 2:6] = 1.0           # group 0 block
        expected[[0, 2], 6:10] = 1.0     # group 1 block
        assert torch.equal(mask, expected)

    def test_make_group_mask_without_shared_channels(self):
        mask, block_sizes = make_group_mask([0, 1], hidden_channels=8)

        assert block_sizes == [4, 4]
        assert mask.sum(dim=1).tolist() == [4.0, 4.0]
        assert torch.equal(mask[0, :4], torch.ones(4)) and torch.equal(mask[1, 4:], torch.ones(4))

    @pytest.mark.parametrize(
        ('groups', 'hidden', 'shared', 'match'),
        [
            ([0, 2], 8, 0, 'every index 0..G-1'),
            ([], 8, 0, 'every index 0..G-1'),
            ([0, 1], 8, 8, 'shared_channels must be in'),
            ([0, 1], 8, -1, 'shared_channels must be in'),
            ([0, 1], 8, 1, 'does not split equally'),
        ],
    )
    def test_make_group_mask_invalid_raises(self, groups, hidden, shared, match):
        with pytest.raises(ValueError, match=match):
            make_group_mask(groups, hidden_channels=hidden, shared_channels=shared)
