from pose_est_nets.utils.heatmap_tracker_utils import SubPixelMaxima
from pose_est_nets.datasets.datasets import HeatmapDataset
import imgaug.augmenters as iaa
import torch
from torch.utils.data import DataLoader

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_subpixmaxima():
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 384, "width": 384})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)
    dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )
    SubPixMax = SubPixelMaxima(
        output_shape = (96, 96), #384 // 2 * 2
        output_sigma = torch.tensor(1.25, device = _TORCH_DEVICE),
        upsample_factor = torch.tensor(100, device = _TORCH_DEVICE),
        coordinate_scale = torch.tensor(4, device = _TORCH_DEVICE), # 2 ** 2
        confidence_scale = torch.tensor(1, device = _TORCH_DEVICE), #was originally 255.0
        threshold = None,
        device = _TORCH_DEVICE
    )
    test_img, gt_heatmap, gt_keypts = dataset.__getitem__(idx = 0)
    maxima, confidence = SubPixMax.run(gt_heatmap.unsqueeze(0).to(_TORCH_DEVICE))
    maxima = maxima.squeeze(0)
    assert(maxima.shape == gt_keypts.shape)
    assert(maxima.shape[0]//2 == confidence.shape[1])
    dl = DataLoader(dataset, batch_size = 5)
    img_batch, gt_heatmap_batch, gt_keypts_batch = next(iter(dl))
    
    (maxima1, confidence1), (maxima2, confidence2) = SubPixMax.run(gt_heatmap_batch.to(_TORCH_DEVICE), gt_heatmap_batch.to(_TORCH_DEVICE))
    print(maxima1.shape, confidence1.shape, maxima2.shape, confidence2.shape)

