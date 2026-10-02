"""Test the initialization and training of heatmap models."""

import copy

import pytest
import torch


@pytest.mark.gpu
def test_supervised_heatmap(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vitb_sam(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vitb_sam"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vits_sam2(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vits_sam2"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vitb_imagenet(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vitb_imagenet"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vits_dino(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vits_dino"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vits_dinov2(
    cfg,
    heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vits_dinov2"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_supervised_heatmap_vits_dinov3(
        cfg,
        heatmap_data_module,
        video_dataloader,
        trainer,
        run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.backbone = "vits_dinov3"
    cfg_tmp.model.losses_to_use = []

    # Check if we have HuggingFace auth
    has_hf_auth = False
    try:
        from huggingface_hub import get_token
        has_hf_auth = get_token() is not None
    except ImportError:
        # huggingface_hub not installed (e.g., in CI)
        has_hf_auth = False

    if has_hf_auth:
        # with auth - should run normally
        run_model_test(
            cfg=cfg_tmp,
            data_module=heatmap_data_module,
            video_dataloader=video_dataloader,
            trainer=trainer,
        )
    else:
        # CI or no auth - should raise proper error
        with pytest.raises(RuntimeError, match="Cannot access gated model"):
            run_model_test(
                cfg=cfg_tmp,
                data_module=heatmap_data_module,
                video_dataloader=video_dataloader,
                trainer=trainer,
            )


@pytest.mark.gpu
def test_supervised_multiview_heatmap(
    cfg_multiview,
    multiview_heatmap_data_module,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_semisupervised_heatmap_temporal_pcasingleview(
    cfg,
    heatmap_data_module_combined,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = ["temporal", "pca_singleview"]

    run_model_test(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


@pytest.mark.gpu
def test_semisupervised_multiview_heatmap_multiview(
    cfg_multiview,
    multiview_heatmap_data_module_combined,
    video_dataloader,
    trainer,
    run_model_test,
):
    """Test the initialization and training of a semi-supervised multiview heatmap model."""

    cfg_tmp = copy.deepcopy(cfg_multiview)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = ["pca_multiview"]

    run_model_test(
        cfg=cfg_tmp,
        data_module=multiview_heatmap_data_module_combined,
        video_dataloader=video_dataloader,
        trainer=trainer,
    )


# ── per-dataset heads (CPU) ───────────────────────────────────────────────────


def _multihead(**kwargs):
    from lightning_pose.models import MultiHeadHeatmapTracker
    return MultiHeadHeatmapTracker(
        dataset_names=['a', 'b'], num_keypoints=3, backbone='resnet18', pretrained=False,
        image_size=64, **kwargs,
    )


class TestMultiHeadHeatmapTracker:
    """Test the per-dataset-head tracker (routing, prediction modes, optimizer groups)."""

    def test_multihead_routed_matches_each_head(self):
        model = _multihead().eval()
        images = torch.randn(4, 3, 64, 64)
        ids = torch.tensor([0, 1, 1, 0])

        with torch.no_grad():
            routed = model.forward_routed(images, ids)
            reps = model.get_representations(images)
            per_head = [model.heads[i](reps) for i in (0, 1)]

        for row, i in enumerate(ids.tolist()):
            assert torch.allclose(routed[row], per_head[i][row])

    def test_multihead_forward_raises(self):
        with pytest.raises(NotImplementedError, match='forward_routed'):
            _multihead().forward(torch.randn(1, 3, 64, 64))

    def test_multihead_loss_inputs_need_dataset_id(self):
        batch = {'images': torch.randn(2, 3, 64, 64)}
        with pytest.raises(ValueError, match='dataset_id'):
            _multihead()._heatmaps_labeled(batch)

    def test_multihead_predict_oracle_uses_predict_dataset(self):
        model = _multihead().eval()
        images = torch.randn(2, 3, 64, 64)
        model.predict_dataset = 'b'

        with torch.no_grad():
            heatmaps = model._heatmaps_predict({'frames': images}, images)
            expected = model.heads[1](model.get_representations(images))

        assert torch.allclose(heatmaps, expected)

    def test_multihead_predict_oracle_without_identity_raises(self):
        model = _multihead().eval()
        images = torch.randn(2, 3, 64, 64)
        with pytest.raises(ValueError, match='predict_dataset is unset'):
            model._heatmaps_predict({'frames': images}, images)

    def test_multihead_predict_mode_invalid_raises(self):
        model = _multihead().eval()
        model.predict_mode = 'nope'
        with pytest.raises(ValueError, match="'oracle' or 'blind'"):
            model.predict_step({'frames': torch.randn(1, 3, 64, 64)}, 0)

    def test_multihead_blind_masked_head_has_no_vote(self):
        model = _multihead().eval()
        model.blind_gamma = 1.0
        # head 1 supports no keypoint: blind output must equal head 0 alone
        model.head_keypoint_mask[1] = False
        images = torch.randn(2, 3, 64, 64)

        with torch.no_grad():
            coords, conf, spread = model.forward_blind(images)
            hm0 = model.heads[0](model.get_representations(images))
            coords0, conf0 = model.heads[0].run_subpixelmaxima(hm0)

        assert torch.allclose(coords, coords0, atol=1e-5)
        assert torch.allclose(conf, conf0, atol=1e-6)
        assert torch.all(spread.abs() < 1e-4)

    def test_multihead_set_mask_from_dataset_with_hflip(self):
        model = _multihead()

        class _Dataset:
            visibility = torch.tensor([[2, 0, 0], [0, 0, 2]])
            dataset_ids = torch.tensor([0, 1])
            keypoint_names = ['nose', 'ear_left', 'ear_right']

        assert model.set_head_keypoint_mask_from_dataset(_Dataset(), hflip=True)
        assert model.head_keypoint_mask.tolist() == [[True, False, False], [False, True, True]]

    def test_multihead_set_mask_from_dataset_without_ids(self):
        model = _multihead()

        class _Dataset:
            visibility = torch.tensor([[2, 0, 0]])
            dataset_ids = None
            keypoint_names = ['nose', 'ear_left', 'ear_right']

        assert not model.set_head_keypoint_mask_from_dataset(_Dataset(), hflip=False)
        assert bool(model.head_keypoint_mask.all())

    def test_multihead_parameter_groups(self):
        model = _multihead()

        groups = {g['name']: g for g in model.get_parameters()}

        assert list(groups) == ['backbone', 'head']
        assert groups['backbone']['lr'] == 0
        head_ids = {id(p) for p in groups['head']['params']}
        assert head_ids == {id(p) for p in model.heads.parameters()}
