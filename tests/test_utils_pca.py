from numpy.lib.npyio import load
import torch
import numpy as np
import pytest
import yaml
import sklearn
import os
import imgaug.augmenters as iaa
from lightning_pose.data.datasets import BaseTrackingDataset, HeatmapDataset
from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from typing import List
from lightning_pose.utils.pca import KeypointPCA
import unittest


def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)


@pytest.fixture
def heatmap_dataset() -> HeatmapDataset:
    """creates a basic heatmap dataset from toy_datasets/toymouseRunningData

    Returns:
        HeatmapDataset: [description]
    """
    data_transform = []
    data_transform.append(
        iaa.Resize({"height": 256, "width": 256})
    )  # dlc dimensions need to be repeatably divisable by 2
    imgaug_transform = iaa.Sequential(data_transform)

    heatmap_dataset = HeatmapDataset(
        root_directory="toy_datasets/toymouseRunningData",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2],
        imgaug_transform=imgaug_transform,
    )

    return heatmap_dataset


@pytest.fixture
def loss_param_dict() -> dict:
    # TODO: check if needed
    # grab example loss config file from repo
    base_dir = os.path.dirname(os.path.dirname(os.path.join(__file__)))
    loss_cfg = os.path.join(
        base_dir, "scripts", "configs", "losses", "losses_params.yaml"
    )
    with open(loss_cfg) as f:
        loss_param_dict = yaml.load(f, Loader=yaml.FullLoader)
    # hard code multivew pca info for now
    loss_param_dict["pca_multiview"]["mirrored_column_matches"] = [
        [0, 1, 2, 3, 4, 5, 6],
        [8, 9, 10, 11, 12, 13, 14],
    ]
    return loss_param_dict


@pytest.fixture
def data_module(heatmap_dataset, video_list, loss_param_dict) -> UnlabeledDataModule:
    unlabeled_module_heatmap = UnlabeledDataModule(
        heatmap_dataset, video_paths_list=video_list, loss_param_dict=loss_param_dict
    )
    return unlabeled_module_heatmap


def test_train_loader_iter(data_module):
    # data_module.setup()
    # access both loaders
    # TODO this is just messing around with dataloaders. good educationally, not great as a test. keep somehow.
    dataset_length = len(data_module.train_dataset)
    from pytorch_lightning.trainer.supporters import CombinedLoader

    loaders = data_module.train_dataloader()
    combined_loader = CombinedLoader(loaders)
    image_counter = 0
    for i, batch in enumerate(combined_loader):
        image_counter += len(batch["labeled"]["keypoints"])
        assert type(batch) is dict
        assert type(batch["labeled"]) is dict
        assert type(batch["unlabeled"]) is torch.Tensor
        assert checkEqual(
            list(batch["labeled"].keys()), ["images", "keypoints", "idxs", "heatmaps"]
        )
        # print(type(batch["labeled"]))
        # print(batch["labeled"])
        # print(batch)
    assert image_counter == dataset_length

    # access only the labeled loader
    labeled_loader = data_module.train_dataloader()["labeled"]
    print(type(labeled_loader))
    for i, b in enumerate(labeled_loader):
        print(i, b.keys(), len(b["keypoints"]))


def test_data_extractor(data_module):
    # TODO: move to data once the fixtures are set there. more rigorous testing.
    from lightning_pose.data.utils import DataExtractor

    data_tensor = DataExtractor(data_module=data_module, cond="train")()

    assert data_tensor.shape == (31, 34)


def test_pca_keypoint_class(data_module):
    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_multiview",
        data_module=data_module,
        components_to_keep=0.9,
        empirical_epsilon_percentile=0.3,
    )
    kp_pca._get_data()
    assert kp_pca.data_arr.shape == (31, 17, 2)  # 31 is 0.8*39 images

    # from lightning_pose.utils.pca import get_train_data_for_pca

    # old_data = get_train_data_for_pca(data_module=data_module)
    # print(kp_pca.data_arr)
    # print(old_data)
    # flat_old = old_data.flatten()
    # flat_new = kp_pca.data_arr.flatten()

    # assert torch.allclose(
    #     flat_old[~torch.isnan(flat_old)], flat_new[~torch.isnan(flat_old)]
    # )
    # we know that there are nan keypoints in this toy dataset, assert that
    nan_count_pre_cleanup = torch.sum(torch.isnan(kp_pca.data_arr))
    assert nan_count_pre_cleanup > 0

    kp_pca._format_data()
    assert kp_pca.data_arr.shape == (
        7 * 31,
        4,
    )  # 7 keypoints seen from both views (specified in loss param dict), 31 images, 4 coords per keypoint

    # again, it should still contain nans
    nan_count_pre_cleanup = torch.sum(torch.isnan(kp_pca.data_arr))
    assert nan_count_pre_cleanup > 0

    # now clean nans
    kp_pca._clean_any_nans()
    assert kp_pca.data_arr.shape[0] < (31 * 7)  # we've eliminated some rows

    # no nans allowed at this stage
    nan_count = torch.sum(torch.isnan(kp_pca.data_arr))
    assert nan_count == 0

    # check that we have enough ovservations
    kp_pca._check_data()  # raises ValueErrors if fails

    # fit the pca model
    kp_pca._fit_pca()

    # we specified 0.9 components to keep but we'll take 3
    kp_pca._choose_n_components()

    kp_pca.pca_prints()

    kp_pca._set_parameter_dict()

    checkEqual(
        list(kp_pca.parameters.keys()),
        ["mean", "kept_eigenvectors", "discarded_eigenvectors", "epsilon"],
    )

    # assert that the results of running the .__call__() method are the same as separately running each of the subparts
    kp_pca_2 = KeypointPCA(
        loss_type="pca_multiview",
        data_module=data_module,
        components_to_keep=0.9,
        empirical_epsilon_percentile=0.3,
    )
    kp_pca_2.__call__()

    assert (kp_pca_2.data_arr == kp_pca.data_arr).all()


def test_format_multiview_data_for_pca():

    from lightning_pose.utils.pca import format_multiview_data_for_pca

    n_batches = 12
    n_keypoints = 20
    keypoints = torch.rand(
        size=(n_batches, n_keypoints, 2),
        device="cpu",
    )

    # basic two-view functionality
    column_matches = [[0, 1, 2, 3], [4, 5, 6, 7]]
    arr = format_multiview_data_for_pca(keypoints, column_matches)
    assert arr.shape == torch.Size(
        [2 * len(column_matches), n_batches * len(column_matches[0])]
    )

    # basic error checking
    column_matches = [[0, 1, 2, 3], [4, 5, 6]]
    with pytest.raises(AssertionError):
        arr = format_multiview_data_for_pca(keypoints, column_matches)

    # basic three-view functionality
    column_matches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    arr = format_multiview_data_for_pca(keypoints, column_matches)
    assert arr.shape == torch.Size(
        [2 * len(column_matches), n_batches * len(column_matches[0])]
    )


def test_component_chooser():
    # create fake data for PCA
    from sklearn.datasets import load_diabetes
    from sklearn.decomposition import PCA
    import numpy as np

    diabetes = load_diabetes()
    data_for_pca = diabetes.data
    assert np.sum(np.isnan(data_for_pca)) == 0  # no nan-handling needed here
    assert data_for_pca.shape == (
        442,
        10,
    )  # just to illustrate the dimensions of the data

    # now fit pca
    pca = PCA(svd_solver="full")
    pca.fit(data_for_pca)

    from lightning_pose.utils.pca import ComponentChooser

    # regular integer behavior
    comp_chooser_int = ComponentChooser(pca, 4)
    assert comp_chooser_int() == 4

    # can't keep more than 10 componets for diabetes data (obs dim = 10)
    with pytest.raises(ValueError):
        comp_chooser_int = ComponentChooser(pca, 11)

    # we return ints, so basically checking that 2 < 3
    assert ComponentChooser(pca, 2)() < ComponentChooser(pca, 3)()

    # can't explain more than 1.0 of the variance
    with pytest.raises(ValueError):
        comp_chooser_float = ComponentChooser(pca, 1.04)

    # no negative proportions
    with pytest.raises(ValueError):
        comp_chooser_float = ComponentChooser(pca, -0.2)

    # typical behavior
    n_comps = ComponentChooser(pca, 0.95)()
    assert (n_comps > 0) and (n_comps <= 10)

    # for explaining exactly 1.0 of the variance, you should keep all 10 components
    assert ComponentChooser(pca, 1.0)() == 10

    # less explained variance -> less components kept
    assert ComponentChooser(pca, 0.20)() < ComponentChooser(pca, 0.90)()
