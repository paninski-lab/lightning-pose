from pose_est_nets.models.heatmap_tracker import HeatmapTracker
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.utils.heatmaps import generate_heatmaps
from pose_est_nets.datasets.datamodules import BaseDataModule
import torch
import pytest
import pytorch_lightning as pl
import imgaug.augmenters as iaa
from geomloss import SamplesLoss

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HEIGHT = 256
_WIDTH = 256
_REACH = 100

data_transform = []
data_transform.append(
    iaa.Resize({"height": _HEIGHT, "width": _WIDTH})
)  # dlc dimensions need to be repeatably divisable by 2
imgaug_transform = iaa.Sequential(data_transform)
dataset = HeatmapDataset(
    root_directory="toy_datasets/toymouseRunningData",
    csv_path="CollectedData_.csv",
    header_rows=[1, 2],
    imgaug_transform=imgaug_transform,
)
datamodule = BaseDataModule(dataset=dataset)
datamodule.setup()
dataloader = iter(datamodule.train_dataloader())

model = HeatmapTracker(
    num_targets=34,
    supervised_loss="wasserstein",
    reach=100.0,
).to(_TORCH_DEVICE)


def test_wass():
    batch = next(dataloader)
    print(len(batch))
    gt_heatmaps = batch["heatmaps"]
    wass_loss_reach = SamplesLoss(loss="sinkhorn", reach=_REACH)
    wass_loss_nr = SamplesLoss(loss="sinkhorn")
    print(gt_heatmaps.shape)
    y2 = y1 = gt_heatmaps[0][0]
    loss_one_heatmap_with_self = wass_loss_reach(y1, y2)
    loss_one_heatmap_with_self_nr = wass_loss_nr(y1, y2)
    print("heatmap with self")
    print(loss_one_heatmap_with_self, loss_one_heatmap_with_self_nr)
    y2 = gt_heatmaps[-1][-1]
    loss_two_different_heatmaps = wass_loss_reach(y1, y2)
    loss_two_different_heatmaps_nr = wass_loss_nr(y1, y2)
    print("two different heatmaps:")
    print(loss_two_different_heatmaps, loss_two_different_heatmaps_nr)

    constant = torch.ones(size=y1.shape) * 10
    y1 = y1 + constant
    y2 = y2 + constant

    loss_two_different_heatmaps_plus_constant = wass_loss_reach(y1, y2)
    loss_two_different_heatmaps_plus_constant_nr = wass_loss_nr(y1, y2)
    print("two different heatmaps plus constant:")
    print(
        loss_two_different_heatmaps_plus_constant,
        loss_two_different_heatmaps_plus_constant_nr,
    )

    ############################################################################
    middle = (
        torch.tensor([y1.shape[0] // 2, y1.shape[1] // 2], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    test_heatmap = generate_heatmaps(
        keypoints=middle,
        height=y1.shape[0],
        width=y1.shape[1],
        output_shape=dataset.output_shape,
    )[0][0]
    vals, row_idx = test_heatmap.max(0)
    col_idx = vals.argmax(0)
    row_idx = row_idx[col_idx]
    print("row, col, max:")
    print(row_idx, col_idx, test_heatmap[row_idx][col_idx])
    test_heatmap_shifted_scaled = (
        generate_heatmaps(
            keypoints=middle + 10,
            height=y1.shape[0],
            width=y1.shape[1],
            output_shape=dataset.output_shape,
        )[0][0]
        / 1.5
    )
    vals, row_idx = test_heatmap_shifted_scaled.max(0)
    col_idx = vals.argmax(0)
    row_idx = row_idx[col_idx]
    print("row, col, max for shifted, scaled:")
    print(row_idx, col_idx, test_heatmap_shifted_scaled[row_idx][col_idx])
    shifted_scaled_heatmap_loss = wass_loss_reach(
        test_heatmap, test_heatmap_shifted_scaled
    )
    bimodal_heatmap = test_heatmap + test_heatmap_shifted_scaled
    bimodal_heatmap_loss = wass_loss_reach(test_heatmap, bimodal_heatmap)
    print("shifted heatmap loss:")
    print(shifted_scaled_heatmap_loss)
    print("bimodal heatmap loss:")
    print(bimodal_heatmap_loss)

    ##########################################

    e1 = e2 = gt_heatmaps[0]
    loss_one_example_with_self = wass_loss_reach(e1, e2)  # set of 17 keypoints
    loss_one_example_with_self_nr = wass_loss_nr(e1, e2)
    print("example (17 keypoints) with self:")
    print(loss_one_example_with_self, loss_one_example_with_self_nr)
    e2 = gt_heatmaps[1]
    loss_two_examples = wass_loss_reach(e1, e2)
    loss_two_examples_nr = wass_loss_nr(e1, e2)
    print("two exampless:")
    print(loss_two_examples, loss_two_examples_nr)

    # Now I will do tests with masking on a full batch like we do when actually computing the supervised loss
    images = batch["images"]
    preds = model.forward(
        images.to(_TORCH_DEVICE)
    )  # gibberish because the model hasn't been trained
    gt_heatmaps = gt_heatmaps.to(_TORCH_DEVICE)
    batch_size, num_keypoints, height, width = gt_heatmaps.shape
    max_vals = torch.amax(gt_heatmaps, dim=(2, 3))
    zeros = torch.zeros(
        size=(gt_heatmaps.shape[0], gt_heatmaps.shape[1]), device=gt_heatmaps.device
    )
    non_zeros = ~torch.eq(max_vals, zeros)
    mask = torch.reshape(non_zeros, [non_zeros.shape[0], non_zeros.shape[1], 1, 1])
    preds = torch.masked_select(preds, mask).unsqueeze(0)
    gt_heatmaps = torch.masked_select(gt_heatmaps, mask).unsqueeze(0)
    batch_loss = wass_loss_reach(preds, gt_heatmaps)
    batch_loss_nr = wass_loss_nr(preds, gt_heatmaps)
    print("one batch vs preds with masking:")
    print(batch_loss, batch_loss_nr)
    print("one batch vs preds with masking scaled:")
    print(
        batch_loss / (batch_size * num_keypoints),
        batch_loss_nr / (batch_size * num_keypoints),
    )
    min_val = torch.min(preds)
    print(min_val)
    constant = torch.ones(size=preds.shape, device=preds.device) * (min_val * -1)
    preds = preds + constant
    gt_heatmaps = gt_heatmaps + constant
    batch_loss_c = wass_loss_reach(preds, gt_heatmaps)
    batch_loss_c_nr = wass_loss_nr(preds, gt_heatmaps)
    print("batch vs preds with masking + constant scaled")
    print(
        batch_loss_c / (batch_size * num_keypoints),
        batch_loss_c_nr / (batch_size * num_keypoints),
    )


# model_no_reach = SemiSupervisedHeatmapTracker(
#     num_targets=34,
#     resnet_version=50,
#     downsample_factor=2,
#     pretrained=True,
#     learn_weights=False,
#     reach=None,
# )
# model_reach = SemiSupervisedHeatmapTracker(
#     num_targets=34,
#     resnet_version=50,
#     downsample_factor=2,
#     pretrained=True,
#     learn_weights=False,
#     reach=100.0,
# )
