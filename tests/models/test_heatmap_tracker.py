"""Test the initialization and training of heatmap models."""

import copy
import pytest
import torch

from lightning_pose.utils.scripts import get_loss_factories, get_model


def test_supervised_heatmap(cfg, heatmap_data_module, trainer, remove_logs):
    """Test the initialization and training of a supervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = []
    
    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_tmp, data_module=heatmap_data_module)

    # build model
    model = get_model(
        cfg=cfg_tmp, data_module=heatmap_data_module, loss_factories=loss_factories
    )

    # train model for a couple epochs
    trainer.fit(model=model, datamodule=heatmap_data_module)

    # remove tensors from gpu
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()


def test_unsupervised_heatmap_temporal(
    cfg, heatmap_data_module_combined, trainer, remove_logs,
):
    """Test the initialization and training of an unsupervised heatmap model."""

    cfg_tmp = copy.deepcopy(cfg)
    cfg_tmp.model.model_type = "heatmap"
    cfg_tmp.model.losses_to_use = ["temporal"]

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(
        cfg=cfg_tmp, data_module=heatmap_data_module_combined
    )

    # build model
    model = get_model(
        cfg=cfg_tmp,
        data_module=heatmap_data_module_combined,
        loss_factories=loss_factories,
    )

    # train model for a couple epochs
    trainer.fit(model=model, datamodule=heatmap_data_module_combined)

    # remove tensors from gpu
    torch.cuda.empty_cache()

    # clean up logging
    remove_logs()


# def test_create_double_upsampling_layer():
#     fake_image_batch = torch.rand(
#         size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
#     )
#     heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
#     upsampling_layer = heatmap_model.create_double_upsampling_layer(
#         in_channels=512, out_channels=heatmap_model.num_keypoints
#     ).to(_TORCH_DEVICE)
#     representations = heatmap_model.get_representations(fake_image_batch)
#     upsampled = upsampling_layer(representations)
#     assert (
#         torch.tensor(upsampled.shape[-2:])
#         == torch.tensor(representations.shape[-2:]) * 2
#     ).all()
#     # now another upsampling layer with a different number of in channels
#     upsampling_layer_two = heatmap_model.create_double_upsampling_layer(
#         in_channels=heatmap_model.num_keypoints,
#         out_channels=heatmap_model.num_keypoints,
#     ).to(_TORCH_DEVICE)
#     twice_upsampled = upsampling_layer_two(upsampled)
#     assert (
#         torch.tensor(twice_upsampled.shape[-2:])
#         == torch.tensor(upsampled.shape[-2:]) * 2
#     ).all()
#
#     # remove model/data from gpu; then cache can be cleared
#     del fake_image_batch
#     del heatmap_model
#     del upsampling_layer
#     del representations
#     del upsampled
#     del upsampling_layer_two
#     del twice_upsampled
#     torch.cuda.empty_cache()  # remove tensors from gpu
#
#     # TODO: revisit this test
#     # # test the output
#     # pix_shuff = torch.nn.PixelShuffle(2)
#     # pix_shuffled = pix_shuff(representations)
#     # print(pix_shuffled.shape)
#     # print(representations.shape)
#     # assert (
#     #     torch.tensor(pix_shuffled.shape[-2:])
#     #     == torch.tensor(representations.shape[-2:] * 2)
#     # ).all()
#
#
# def test_heatmaps_from_representations():
#     fake_image_batch = torch.rand(
#         size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
#     )
#     heatmap_model = HeatmapTracker(num_targets=34).to(_TORCH_DEVICE)
#     representations = heatmap_model.get_representations(fake_image_batch)
#     heatmaps = heatmap_model.heatmaps_from_representations(representations)
#     assert (
#         torch.tensor(heatmaps.shape[-2:])
#         == torch.tensor(fake_image_batch.shape[-2:])
#         // (2 ** heatmap_model.downsample_factor)
#     ).all()
#
#     # remove model/data from gpu; then cache can be cleared
#     del fake_image_batch
#     del heatmap_model
#     del representations
#     del heatmaps
#     torch.cuda.empty_cache()  # remove tensors from gpu
#
#
# def test_softmax():
#     fake_image_batch = torch.rand(
#         size=(_BATCH_SIZE, 3, _HEIGHT, _WIDTH), device=_TORCH_DEVICE
#     )
#     heatmap_model = HeatmapTracker(
#         num_targets=34, output_shape=(_HEIGHT//4, _WIDTH//4)).to(_TORCH_DEVICE)
#     representations = heatmap_model.get_representations(fake_image_batch)
#     heatmaps = heatmap_model.heatmaps_from_representations(representations)
#     print(torch.sum(heatmaps[0][0]))
#     print(torch.sum(heatmaps[1][1]))  # before heatmaps sums would vary widely
#     valid_probability_heatmaps = heatmap_model.forward(fake_image_batch)
#     sums = torch.sum(valid_probability_heatmaps, dim=(2,3))
#     print(sums)
#     ones = torch.ones(
#         size=(_BATCH_SIZE, heatmap_model.num_keypoints),
#         dtype=torch.int32
#     ).to(_TORCH_DEVICE)
#     sums = torch.round(sums)  # rounding error was causing assert to fail
#     assert(sums.eq(ones)).all()
