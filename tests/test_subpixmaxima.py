from pose_est_nets.utils.heatmap_tracker_utils import SubPixelMaxima
from pose_est_nets.datasets.datasets import HeatmapDataset
import imgaug.augmenters as iaa
import torch
from torch.utils.data import DataLoader

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_draw_keypoints():
    


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
        output_shape=(96, 96),
        output_sigma=torch.tensor(1.25, device=_TORCH_DEVICE),
        upsample_factor=torch.tensor(2, device=_TORCH_DEVICE),
        coordinate_scale=torch.tensor(4, device=_TORCH_DEVICE),  # 2 ** 2
        confidence_scale=torch.tensor(1, device=_TORCH_DEVICE),  # was originally 255.0
        threshold=None,
        device=_TORCH_DEVICE
    )
    test_img, gt_heatmap, gt_keypts = dataset.__getitem__(idx=0)
    maxima, confidence = SubPixMax.run(gt_heatmap.unsqueeze(0).to(_TORCH_DEVICE))
    maxima = maxima.squeeze(0)
    assert(maxima.shape == gt_keypts.shape)
    assert(maxima.shape[0]//2 == confidence.shape[1])

    # remove model/data from gpu; then cache can be cleared
    del gt_heatmap
    del test_img, gt_keypts
    del maxima, confidence
    torch.cuda.empty_cache()  # remove tensors from gpu

    dl = DataLoader(dataset, batch_size=2)
    img_batch, gt_heatmap_batch, gt_keypts_batch = next(iter(dl))

    del dataset
    del dl
    del img_batch, gt_keypts_batch
    torch.cuda.empty_cache()  # remove tensors from gpu
    
    (maxima1, confidence1), (maxima2, confidence2) = SubPixMax.run(
        gt_heatmap_batch.to(_TORCH_DEVICE),
        gt_heatmap_batch.to(_TORCH_DEVICE)
    )
    print(maxima1.shape, confidence1.shape, maxima2.shape, confidence2.shape)

    # remove model/data from gpu; then cache can be cleared
    del gt_heatmap_batch
    del maxima1, confidence1, maxima2, confidence2
    torch.cuda.empty_cache()  # remove tensors from gpu
