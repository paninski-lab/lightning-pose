"""Test the PCA module used in the PCA losses."""

import numpy as np
import pytest
import torch
from lightning.pytorch.utilities import CombinedLoader

from lightning_pose.utils.pca import KeypointPCA


def check_lists_equal(list_1, list_2):
    return len(list_1) == len(list_2) and sorted(list_1) == sorted(list_2)


def test_train_loader_iter(base_data_module_combined):

    # TODO: this is just messing around with dataloaders
    # good educationally, not great as a test. keep somehow.
    dataset_length = len(base_data_module_combined.train_dataset)

    combined_loader = base_data_module_combined.train_dataloader()
    # the default mode of CombinedLoader changes in Lightning 2.0
    # we manually take the iterbles inside the combined_loader, and make a new class that cycles
    # only over the labeled dataloader
    combined_loader = CombinedLoader(combined_loader.iterables, mode="min_size")
    image_counter = 0
    for i, batch in enumerate(combined_loader):
        print(i)

        image_counter += len(batch["labeled"]["keypoints"])
        assert type(batch) is dict
        assert type(batch["labeled"]) is dict
        assert type(batch["unlabeled"]) is dict
        assert type(batch["unlabeled"]["frames"]) is torch.Tensor
        assert check_lists_equal(
            list(batch["labeled"].keys()), ["images", "keypoints", "idxs"]
        )
    assert image_counter == dataset_length


def test_pca_keypoint_class(cfg, base_data_module_combined):

    num_train_ims = (
        len(base_data_module_combined.dataset)
        * base_data_module_combined.train_probability
    )
    num_keypoints = base_data_module_combined.dataset.num_keypoints
    num_keypoints_both_views = 7

    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_multiview",
        data_module=base_data_module_combined,
        components_to_keep=3,
        empirical_epsilon_percentile=0.3,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
    )

    kp_pca._get_data()
    assert kp_pca.data_arr.shape == (num_train_ims, 2 * num_keypoints)

    # from lightning_pose.utils.pca import get_train_data_for_pca

    # old_data = get_train_data_for_pca(data_module=base_data_module_combined)
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

    kp_pca.data_arr = kp_pca._format_data(data_arr=kp_pca.data_arr)
    assert kp_pca.data_arr.shape == (
        num_keypoints_both_views * num_train_ims,
        4,  # 4 coords per keypoint (2 views)
    )

    # again, it should still contain nans
    nan_count_pre_cleanup = torch.sum(torch.isnan(kp_pca.data_arr))
    assert nan_count_pre_cleanup > 0

    # now clean nans
    kp_pca._clean_any_nans()
    # we've eliminated some rows
    assert kp_pca.data_arr.shape[0] < (num_keypoints_both_views * num_train_ims)

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

    check_lists_equal(
        list(kp_pca.parameters.keys()),
        ["mean", "kept_eigenvectors", "discarded_eigenvectors", "epsilon"],
    )

    # assert that the results of running the .__call__() method are the same as
    # separately running each of the subparts
    # MW: for some reason initializing this second object gave me a seg fault; I reduced
    # the number of workers in the datamodule from 8 to 4 and it seems to be working now
    kp_pca_2 = KeypointPCA(
        loss_type="pca_multiview",
        data_module=base_data_module_combined,
        components_to_keep=3,
        empirical_epsilon_percentile=0.3,
        mirrored_column_matches=cfg.data.mirrored_column_matches,
    )
    kp_pca_2()
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
        [n_batches * len(column_matches[0]), 2 * len(column_matches)]
    )

    # basic error checking
    column_matches = [[0, 1, 2, 3], [4, 5, 6]]
    with pytest.raises(AssertionError):
        format_multiview_data_for_pca(keypoints, column_matches)

    # basic three-view functionality
    column_matches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    arr = format_multiview_data_for_pca(keypoints, column_matches)
    assert arr.shape == torch.Size(
        [n_batches * len(column_matches[0]), 2 * len(column_matches)]
    )


def test_singleview_format_and_loss(cfg, base_data_module_combined):
    # generate fake data
    n_batches = 12
    n_keypoints = 17
    pred_keypoints = torch.rand(
        size=(n_batches, n_keypoints * 2),
        device="cpu",
    )
    # initialize an instance
    singleview_pca = KeypointPCA(
        loss_type="pca_singleview",
        data_module=base_data_module_combined,
        components_to_keep=6,
        empirical_epsilon_percentile=1.0,
        columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
    )
    singleview_pca()  # fit it to have all the parameters

    # push pred_keypoints through the reformatter
    data_arr = singleview_pca._format_data(data_arr=pred_keypoints)
    # num selected keypoints in the cfg is 14, so we should have 14 * 2 columns
    assert data_arr.shape == (n_batches, 14 * 2)

    err = singleview_pca.compute_reprojection_error(data_arr=data_arr)
    assert err.shape == (n_batches, 14)


def test_component_chooser():

    # create fake data for PCA
    from sklearn.datasets import load_diabetes
    from sklearn.decomposition import PCA

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
        ComponentChooser(pca, 11)

    # we return ints, so basically checking that 2 < 3
    assert ComponentChooser(pca, 2)() < ComponentChooser(pca, 3)()

    # can't explain more than 1.0 of the variance
    with pytest.raises(ValueError):
        ComponentChooser(pca, 1.04)

    # no negative proportions
    with pytest.raises(ValueError):
        ComponentChooser(pca, -0.2)

    # typical behavior
    n_comps = ComponentChooser(pca, 0.95)()
    assert (n_comps > 0) and (n_comps <= 10)

    # for explaining exactly 1.0 of the variance, you should keep all 10 components
    assert ComponentChooser(pca, 1.0)() == 10

    # less explained variance -> less components kept
    assert ComponentChooser(pca, 0.20)() < ComponentChooser(pca, 0.90)()
