"""Test functionality of base model classes."""

import gc
from pathlib import Path

from lightning.pytorch import LightningModule, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import pytest
import torch
import torchvision

from lightning_pose.models.backbones import ALLOWED_BACKBONES
from lightning_pose.models.base import (
    BaseFeatureExtractor,
    LrNotImplementedError,
    OptimizerNotImplementedError,
)


class TestLrNotImplementedError:
    """Test the LrNotImplementedError exception."""

    def test_is_not_implemented_error(self):
        """Subclasses NotImplementedError."""
        assert issubclass(LrNotImplementedError, NotImplementedError)

    def test_message_contains_scheduler_name(self):
        """Error message includes the invalid scheduler name."""
        exc = LrNotImplementedError('cosine')
        assert 'cosine' in str(exc)

    def test_stores_lr_scheduler(self):
        """lr_scheduler attribute holds the passed value."""
        exc = LrNotImplementedError('cosine')
        assert exc.lr_scheduler == 'cosine'

    def test_can_be_raised_and_caught(self):
        """Can be raised and caught as NotImplementedError."""
        with pytest.raises(NotImplementedError):
            raise LrNotImplementedError('cosine')


class TestOptimizerNotImplementedError:
    """Test the OptimizerNotImplementedError exception."""

    def test_is_not_implemented_error(self):
        """Subclasses NotImplementedError."""
        assert issubclass(OptimizerNotImplementedError, NotImplementedError)

    def test_message_contains_optimizer_name(self):
        """Error message includes the invalid optimizer name."""
        exc = OptimizerNotImplementedError('SGD')
        assert 'SGD' in str(exc)

    def test_stores_optimizer(self):
        """optimizer attribute holds the passed value."""
        exc = OptimizerNotImplementedError('SGD')
        assert exc.optimizer == 'SGD'

    def test_can_be_raised_and_caught(self):
        """Can be raised and caught as NotImplementedError."""
        with pytest.raises(NotImplementedError):
            raise OptimizerNotImplementedError('SGD')


_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
HEIGHTS = [128, 256, 384]  # standard numbers, not going to bigger images due to memory
WIDTHS = [120, 246, 380]  # similar but not square
RESNET_BACKBONES: list[ALLOWED_BACKBONES] = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
]
EFFICIENTNET_BACKBONES: list[ALLOWED_BACKBONES] = [
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
]
VIT_BACKBONES: list[ALLOWED_BACKBONES] = [
    "vits_dino",
    "vitb_dino",
    "vits_dinov2",
    "vitb_dinov2",
    # "vits_dinov3",
    # "vitb_dinov3",
    "vitb_imagenet",
    "vitb_sam",
    "vitt_sam2",
    "vits_sam2",
    "vitb_sam2",
]

RESNET_SHAPES: dict[str, dict[int, torch.Size]] = {
    'resnet18': {
        128: torch.Size([BATCH_SIZE, 512, 4, 4]),
        256: torch.Size([BATCH_SIZE, 512, 8, 8]),
        384: torch.Size([BATCH_SIZE, 512, 12, 12]),
    },
    'resnet34': {
        128: torch.Size([BATCH_SIZE, 512, 4, 4]),
        256: torch.Size([BATCH_SIZE, 512, 8, 8]),
        384: torch.Size([BATCH_SIZE, 512, 12, 12]),
    },
    'resnet50': {
        128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
        256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
        384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
    },
    'resnet101': {
        128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
        256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
        384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
    },
    'resnet152': {
        128: torch.Size([BATCH_SIZE, 2048, 4, 4]),
        256: torch.Size([BATCH_SIZE, 2048, 8, 8]),
        384: torch.Size([BATCH_SIZE, 2048, 12, 12]),
    },
}

EFFICIENTNET_SHAPES: dict[str, dict[int, torch.Size]] = {
    'efficientnet_b0': {
        128: torch.Size([BATCH_SIZE, 1280, 4, 4]),
        256: torch.Size([BATCH_SIZE, 1280, 8, 8]),
        384: torch.Size([BATCH_SIZE, 1280, 12, 12]),
    },
    'efficientnet_b1': {
        128: torch.Size([BATCH_SIZE, 1280, 4, 4]),
        256: torch.Size([BATCH_SIZE, 1280, 8, 8]),
        384: torch.Size([BATCH_SIZE, 1280, 12, 12]),
    },
    'efficientnet_b2': {
        128: torch.Size([BATCH_SIZE, 1408, 4, 4]),
        256: torch.Size([BATCH_SIZE, 1408, 8, 8]),
        384: torch.Size([BATCH_SIZE, 1408, 12, 12]),
    },
}

VIT_SHAPES: dict[str, dict[int, torch.Size]] = {
    "vits_dino": {
        128: torch.Size([BATCH_SIZE, 384, 8, 8]),
        256: torch.Size([BATCH_SIZE, 384, 16, 16]),
        384: torch.Size([BATCH_SIZE, 384, 24, 24]),
    },
    "vitb_dino": {
        128: torch.Size([BATCH_SIZE, 768, 8, 8]),
        256: torch.Size([BATCH_SIZE, 768, 16, 16]),
        384: torch.Size([BATCH_SIZE, 768, 24, 24]),
    },
    "vits_dinov2": {
        128: torch.Size([BATCH_SIZE, 384, 8, 8]),
        256: torch.Size([BATCH_SIZE, 384, 16, 16]),
        384: torch.Size([BATCH_SIZE, 384, 24, 24]),
    },
    "vitb_dinov2": {
        128: torch.Size([BATCH_SIZE, 768, 8, 8]),
        256: torch.Size([BATCH_SIZE, 768, 16, 16]),
        384: torch.Size([BATCH_SIZE, 768, 24, 24]),
    },
    # "vits_dinov3": {
    #     128: torch.Size([BATCH_SIZE, 384, 8, 8]),
    #     256: torch.Size([BATCH_SIZE, 384, 16, 16]),
    #     384: torch.Size([BATCH_SIZE, 384, 24, 24]),
    # },
    # "vitb_dinov3": {
    #     128: torch.Size([BATCH_SIZE, 768, 8, 8]),
    #     256: torch.Size([BATCH_SIZE, 768, 16, 16]),
    #     384: torch.Size([BATCH_SIZE, 768, 24, 24]),
    # },
    "vitb_imagenet": {
        128: torch.Size([BATCH_SIZE, 768, 8, 8]),
        256: torch.Size([BATCH_SIZE, 768, 16, 16]),
        384: torch.Size([BATCH_SIZE, 768, 24, 24]),
    },
    "vitb_sam": {
        128: torch.Size([BATCH_SIZE, 768, 8, 8]),
        256: torch.Size([BATCH_SIZE, 768, 16, 16]),
        384: torch.Size([BATCH_SIZE, 768, 24, 24]),
    },
    "vitt_sam2": {
        128: torch.Size([BATCH_SIZE, 768, 4, 4]),
        256: torch.Size([BATCH_SIZE, 768, 8, 8]),
        384: torch.Size([BATCH_SIZE, 768, 12, 12]),
    },
    "vits_sam2": {
        128: torch.Size([BATCH_SIZE, 768, 4, 4]),
        256: torch.Size([BATCH_SIZE, 768, 8, 8]),
        384: torch.Size([BATCH_SIZE, 768, 12, 12]),
    },
    "vitb_sam2": {
        128: torch.Size([BATCH_SIZE, 896, 4, 4]),
        256: torch.Size([BATCH_SIZE, 896, 8, 8]),
        384: torch.Size([BATCH_SIZE, 896, 12, 12]),
    },
}


@pytest.mark.parametrize("backbone", RESNET_BACKBONES)
def test_backbones_resnet(backbone):
    model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
    resnet_v = int(backbone.replace("resnet", ""))
    if resnet_v <= 34:  # last block is BasicBlock
        assert isinstance(
            list(model.backbone.children())[-3][-1],  # type: ignore[index]
            torchvision.models.resnet.BasicBlock,
        )
    else:  # different arch; BottleneckBlock
        assert isinstance(
            list(model.backbone.children())[-3][-1],  # type: ignore[index]
            torchvision.models.resnet.Bottleneck,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("backbone", EFFICIENTNET_BACKBONES)
def test_backbones_efficientnet(backbone):
    model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
    assert isinstance(
        list(model.backbone.children())[-1][-2][0],  # type: ignore[index]
        torchvision.models.efficientnet.MBConv,  # type: ignore[attr-defined]
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("backbone", VIT_BACKBONES)
def test_backbones_vit(backbone):
    model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
    if backbone == "vitb_sam":
        from transformers.models.sam.modeling_sam import SamPatchEmbeddings
        assert isinstance(model.backbone.vision_encoder.patch_embed, SamPatchEmbeddings)  # type: ignore[attr-defined]
    elif backbone in ("vitb_sam2", "vits_sam2", "vitt_sam2"):
        from transformers.models.sam2.modeling_sam2 import Sam2PatchEmbeddings
        assert isinstance(model.backbone.vision_encoder.patch_embed, Sam2PatchEmbeddings)  # type: ignore[attr-defined]
    elif backbone in ("vits_dino", "vitb_dino", "vitb_imagenet"):
        from transformers.models.vit.modeling_vit import ViTEmbeddings
        assert isinstance(model.backbone.vision_encoder.embeddings, ViTEmbeddings)  # type: ignore[attr-defined]
    elif backbone in ("vits_dinov2", "vitb_dinov2"):
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Embeddings
        assert isinstance(model.backbone.vision_encoder.embeddings, Dinov2Embeddings)  # type: ignore[attr-defined]
    elif backbone in ("vits_dinov3", "vitb_dinov3"):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import (
            Dinov3ViTEmbeddings,  # type: ignore[attr-defined]
        )
        assert isinstance(model.backbone.vision_encoder.embeddings, Dinov3ViTEmbeddings)  # type: ignore[attr-defined]
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("backbone", RESNET_BACKBONES)
def test_representation_shapes_resnet(backbone):
    if _TORCH_DEVICE == "cuda":
        torch.cuda.empty_cache()
    model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
    for height, width in zip(HEIGHTS, WIDTHS, strict=True):
        fake_image_batch = torch.rand(
            size=(BATCH_SIZE, 3, height, width),
            device=_TORCH_DEVICE,
        )
        representations = model(fake_image_batch)
        assert representations.shape == RESNET_SHAPES[backbone][height]
        del fake_image_batch
        del representations
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("backbone", EFFICIENTNET_BACKBONES)
def test_representation_shapes_efficientnet(backbone):
    if _TORCH_DEVICE == "cuda":
        torch.cuda.empty_cache()
    model = BaseFeatureExtractor(backbone=backbone).to(_TORCH_DEVICE)
    for height, width in zip(HEIGHTS, WIDTHS, strict=True):
        fake_image_batch = torch.rand(
            size=(BATCH_SIZE, 3, height, width),
            device=_TORCH_DEVICE,
        )
        representations = model(fake_image_batch)
        assert representations.shape == EFFICIENTNET_SHAPES[backbone][height]
        del fake_image_batch
        del representations
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("backbone", VIT_BACKBONES)
def test_representation_shapes_vit(backbone):
    for height in HEIGHTS:
        if _TORCH_DEVICE == "cuda":
            torch.cuda.empty_cache()
        model = BaseFeatureExtractor(backbone=backbone, image_size=height).to(_TORCH_DEVICE)
        fake_image_batch = torch.rand(
            size=(BATCH_SIZE, 3, height, height),
            device=_TORCH_DEVICE,
        )
        representations = model(fake_image_batch)
        assert representations.shape == VIT_SHAPES[backbone][height]
        del fake_image_batch
        del representations
        del model
    gc.collect()
    torch.cuda.empty_cache()


class _ScheduleProbe(BaseFeatureExtractor):
    """Exercise the production optimizer setup without loading a pretrained backbone."""

    def __init__(self, step_based: bool = True, milestones: list[int] | None = None) -> None:
        LightningModule.__init__(self)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(3)])
        self.optimizer = 'Adam'
        self.optimizer_params = OmegaConf.create({'learning_rate': 0.01})
        self.lr_scheduler = 'multisteplr'
        key = 'milestone_steps' if step_based else 'milestones'
        self.lr_scheduler_params = OmegaConf.create({
            key: [2, 4] if milestones is None else milestones, 'gamma': 0.5,
        })
        self.trace = []

    def get_parameters(self) -> list[dict]:
        """Keep distinct backbone, head and LoRA rates to detect group drift."""
        return [dict(params=[p], name=name, lr=lr) for p, name, lr in zip(
            self.weights, ['backbone', 'head', 'lora'], [0.001, 0.01, 0.0001],
        )]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Record the rates used for each update, including accumulated microbatches."""
        self.trace.append((self.global_step, [g['lr'] for g in self.optimizers().param_groups]))
        return sum(p.square().sum() for p in self.weights)


def _fit_schedule_probe(
    model: _ScheduleProbe,
    root: Path,
    steps: int = 6,
    accumulate: int = 1,
    batches: int = 3,
    checkpoint: str | None = None,
    accelerator: str = 'cpu',
) -> Trainer:
    """Run actual Lightning automatic optimization across short epoch boundaries."""
    trainer = Trainer(
        default_root_dir=root, accelerator=accelerator, devices=1, max_steps=steps,
        accumulate_grad_batches=accumulate, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(model, DataLoader(torch.ones(batches, 1), batch_size=1), ckpt_path=checkpoint)
    return trainer


class TestOptimizerStepSchedule:
    """Verify schedule semantics with the installed Lightning training loop."""

    @pytest.mark.parametrize('accumulate,batches', [(1, 3), (2, 3), (3, 7)])
    def test_exact_updates(self, tmp_path: Path, accumulate: int, batches: int) -> None:
        """Decay after updates two and four regardless of epoch and microbatch counts."""
        model = _ScheduleProbe()
        trainer = _fit_schedule_probe(model, tmp_path, accumulate=accumulate, batches=batches)
        assert trainer.global_step == 6
        for step, rates in model.trace:
            factor = 0.5 ** sum(step >= milestone for milestone in [2, 4])
            assert rates == pytest.approx([lr * factor for lr in [0.001, 0.01, 0.0001]])

    def test_resume(self, tmp_path: Path) -> None:
        """Restored optimizer and scheduler match uninterrupted training after a decay."""
        full = _ScheduleProbe()
        _fit_schedule_probe(full, tmp_path / 'full')
        partial = _ScheduleProbe()
        trainer = _fit_schedule_probe(partial, tmp_path / 'partial', steps=3)
        checkpoint = str(tmp_path / 'resume.ckpt')
        trainer.save_checkpoint(checkpoint)
        resumed = _ScheduleProbe()
        _fit_schedule_probe(resumed, tmp_path / 'resumed', checkpoint=checkpoint)
        assert resumed.trace == full.trace[3:]
        for actual, expected in zip(resumed.weights, full.weights):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_epochs(self, tmp_path: Path) -> None:
        """Explicit epoch milestones retain epoch-based behavior."""
        model = _ScheduleProbe(step_based=False, milestones=[1])
        _fit_schedule_probe(model, tmp_path, batches=3)
        for step, rates in model.trace:
            factor = 1 if step < 3 else 0.5
            assert rates == pytest.approx([lr * factor for lr in [0.001, 0.01, 0.0001]])

    def test_empty_steps_override_historical_derived_epochs(self, tmp_path: Path) -> None:
        """An empty explicit step schedule stays constant even with a saved epoch field."""
        model = _ScheduleProbe(milestones=[])
        model.lr_scheduler_params.milestones = [1]
        _fit_schedule_probe(model, tmp_path)
        for _, rates in model.trace:
            assert rates == pytest.approx([0.001, 0.01, 0.0001])

    @pytest.mark.gpu
    def test_cuda_accumulation(self, tmp_path: Path) -> None:
        """Verify exact update rates in the actual CUDA Lightning loop."""
        model = _ScheduleProbe()
        _fit_schedule_probe(model, tmp_path, accumulate=2, accelerator='gpu')
        for step, rates in model.trace:
            factor = 0.5 ** sum(step >= milestone for milestone in [2, 4])
            assert rates == pytest.approx([lr * factor for lr in [0.001, 0.01, 0.0001]])
