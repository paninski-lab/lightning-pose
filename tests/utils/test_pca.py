"""Test the PCA module used in the PCA losses."""

import numpy as np
import pytest
import torch

from lightning_pose.utils.fiftyone import check_lists_equal
from lightning_pose.utils.pca import KeypointPCA


def test_pca_keypoint_class_singleview(cfg, base_data_module_combined):
    num_train_ims = int(
        len(base_data_module_combined.dataset)
        * base_data_module_combined.train_probability
    )
    num_keypoints = base_data_module_combined.dataset.num_keypoints
    num_keypoints_for_pca = len(cfg.data.columns_for_singleview_pca)

    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_singleview",
        data_module=base_data_module_combined,
        components_to_keep=0.99,
        empirical_epsilon_percentile=1.0,
        columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
    )

    kp_pca._get_data()
    assert kp_pca.data_arr.shape == (num_train_ims, 2 * num_keypoints)

    kp_pca.data_arr = kp_pca._format_data(data_arr=kp_pca.data_arr)
    assert kp_pca.data_arr.shape == (
        num_train_ims,
        2 * num_keypoints_for_pca,  # 2 coords per keypoint
    )

    # check that we have enough observations
    kp_pca._check_data()  # raises ValueErrors if fails

    # fit the pca model
    kp_pca._fit_pca()
    kp_pca._choose_n_components()
    kp_pca.pca_prints()
    kp_pca._set_parameter_dict()

    check_lists_equal(
        list(kp_pca.parameters.keys()),
        ["mean", "kept_eigenvectors", "discarded_eigenvectors", "epsilon"],
    )

    # --------------------------------------
    # test reprojection error computation
    # --------------------------------------

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


def test_pca_keypoint_class_median_centering(cfg, base_data_module_combined):
    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_singleview",
        data_module=base_data_module_combined,
        components_to_keep=0.99,
        empirical_epsilon_percentile=1.0,
        columns_for_singleview_pca=[0, 1, 2],
        centering_method="median",
    )

    fmt_data_arr = kp_pca._format_data(
        data_arr=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],  # Median of 1,2
                [2, -100, 3, 4, 4, 100],  # Median of 3,4
                [12, -90, 13, 14, 14, 110],  # 2nd vec translated by +(10,10)
            ],
            dtype=torch.float32,
        )
    )

    assert torch.equal(
        fmt_data_arr,
        torch.tensor(
            [
                [-1, -1, 0, 0, 1, 1],
                [-1, -104, 0, 0, 1, 96],
                [-1, -104, 0, 0, 1, 96],
            ],
            dtype=torch.float32,
        ),
    )


def test_pca_keypoint_class_mean_centering(cfg, base_data_module_combined):
    # initialize an instance
    kp_pca = KeypointPCA(
        loss_type="pca_singleview",
        data_module=base_data_module_combined,
        components_to_keep=0.99,
        empirical_epsilon_percentile=1.0,
        columns_for_singleview_pca=[0, 1, 2],
        centering_method="mean",
    )

    fmt_data_arr = kp_pca._format_data(
        data_arr=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],  # Mean of 1,2
                [2, -100, 3, 6, 4, 100],  # Mean of 3, 2
                [12, -90, 13, 16, 14, 110],  # 2nd vec translated by +(10,10)
            ],
            dtype=torch.float32,
        )
    )

    assert torch.equal(
        fmt_data_arr,
        torch.tensor(
            [
                [-1, -1, 0, 0, 1, 1],
                [-1, -102, 0, 4, 1, 98],
                [-1, -102, 0, 4, 1, 98],
            ],
            dtype=torch.float32,
        ),
    )


def test_pca_keypoint_class_multiview(
    cfg,
    base_data_module_combined,
    cfg_multiview,
    multiview_heatmap_data_module_combined,
):

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

    kp_pca.data_arr = kp_pca._format_data(data_arr=kp_pca.data_arr)
    assert kp_pca.data_arr.shape == (
        num_keypoints_both_views * num_train_ims,
        4,  # 4 coords per keypoint (2 views)
    )

    # check that we have enough observations
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

    # make sure we don't get errors when using a true multiview dataset
    kp_pca_2 = KeypointPCA(
        loss_type="pca_multiview",
        data_module=multiview_heatmap_data_module_combined,
        components_to_keep=3,
        empirical_epsilon_percentile=0.3,
        mirrored_column_matches=cfg_multiview.data.mirrored_column_matches,
    )
    kp_pca_2()


def test_nan_pca():

    from sklearn.datasets import load_diabetes
    from sklearn.decomposition import PCA

    from lightning_pose.utils.pca import NaNPCA

    # load non-nan example data
    diabetes = load_diabetes()
    data_for_pca = diabetes.data

    # no nan-handling needed here
    assert np.sum(np.isnan(data_for_pca)) == 0
    # just to illustrate the dimensions of the data
    assert data_for_pca.shape == (442, 10)

    # ------------------------------------------------------
    # TEST 1: standard PCA vs our custom PCA with no NaNs
    # ------------------------------------------------------

    # fit standard pca
    pca_skl = PCA(svd_solver="full")
    pca_skl.fit(data_for_pca)

    # fit our custom NaN-handling PCA
    pca_cust = NaNPCA()
    pca_cust.fit(data_for_pca)

    # check outputs are the same
    assert pca_skl.noise_variance_ == pca_cust.noise_variance_
    assert pca_skl.n_samples_ == pca_cust.n_samples_
    assert pca_skl.n_components_ == pca_cust.n_components_
    assert np.allclose(pca_skl.components_, pca_cust.components_, rtol=1e-10)
    assert np.allclose(pca_skl.explained_variance_, pca_cust.explained_variance_, rtol=1e-10)
    assert np.allclose(
        pca_skl.explained_variance_ratio_, pca_cust.explained_variance_ratio_, rtol=1e-10)
    assert np.allclose(pca_skl.singular_values_, pca_cust.singular_values_, rtol=1e-10)

    # ------------------------------------------------------
    # TEST 2: standard PCA vs custom PCA with a single NaN
    # ------------------------------------------------------

    # set a single value to NaN
    data_for_pca_nans1 = np.copy(data_for_pca)
    data_for_pca_nans1[0, 0] = np.nan

    # fit our custom NaN-handling PCA
    pca_cust_nan1 = NaNPCA()
    pca_cust_nan1.fit(data_for_pca_nans1)

    # check deterministic attributes are the same
    assert pca_cust.noise_variance_ == pca_cust_nan1.noise_variance_
    assert pca_cust.n_samples_ == pca_cust_nan1.n_samples_
    assert pca_cust.n_components_ == pca_cust_nan1.n_components_

    # check outputs are close, but not the same
    # for components, we don't consider values close to zero, as these can change sign easily
    mask = np.abs(pca_cust.components_) > 0.05
    assert not np.allclose(
        pca_cust.components_[mask], pca_cust_nan1.components_[mask], rtol=1e-10)
    assert np.allclose(
        pca_cust.components_[mask], pca_cust_nan1.components_[mask], rtol=1e-1)
    assert not np.allclose(
        pca_cust.explained_variance_, pca_cust_nan1.explained_variance_, rtol=1e-10)
    assert np.allclose(
        pca_cust.explained_variance_, pca_cust_nan1.explained_variance_, rtol=1e-2)
    assert not np.allclose(
        pca_cust.explained_variance_ratio_, pca_cust_nan1.explained_variance_ratio_, rtol=1e-10)
    assert np.allclose(
        pca_cust.explained_variance_ratio_, pca_cust_nan1.explained_variance_ratio_, rtol=1e-2)
    assert not np.allclose(
        pca_cust.singular_values_, pca_cust_nan1.singular_values_, rtol=1e-10)
    assert np.allclose(
        pca_cust.singular_values_, pca_cust_nan1.singular_values_, rtol=1e-2)

    # ------------------------------------------------------
    # TEST 3: custom PCA with single vs many NaNs
    # ------------------------------------------------------
    # set many values to NaN
    data_for_pca_nans2 = np.copy(data_for_pca)
    rows = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for r, c in zip(rows, cols):
        data_for_pca_nans2[c, r] = np.nan

    # fit our custom NaN-handling PCA
    pca_cust_nan2 = NaNPCA()
    pca_cust_nan2.fit(data_for_pca_nans2)

    # check outputs are close, but not the same
    # for components, we don't consider values close to zero, as these can change sign easily
    mask = np.abs(pca_cust.components_) > 0.05
    assert not np.allclose(
        pca_cust_nan1.components_[mask], pca_cust_nan2.components_[mask], rtol=1e-10)
    assert np.allclose(
        pca_cust_nan1.components_[mask], pca_cust_nan2.components_[mask], rtol=1e0)
    # just look at the 'd' dims with highest variance
    d = 7
    assert not np.allclose(
        pca_cust_nan1.explained_variance_[:d], pca_cust_nan2.explained_variance_[:d], rtol=1e-10)
    assert np.allclose(
        pca_cust_nan1.explained_variance_[:d], pca_cust_nan2.explained_variance_[:d], rtol=1e-2)
    assert not np.allclose(
        pca_cust_nan1.explained_variance_ratio_[:d], pca_cust_nan2.explained_variance_ratio_[:d],
        rtol=1e-10,
    )
    assert np.allclose(
        pca_cust_nan1.explained_variance_ratio_[:d], pca_cust_nan2.explained_variance_ratio_[:d],
        rtol=1e-2,
    )
    assert not np.allclose(
        pca_cust_nan1.singular_values_[:d], pca_cust_nan2.singular_values_[:d], rtol=1e-10)
    assert np.allclose(
        pca_cust_nan1.singular_values_[:d], pca_cust_nan2.singular_values_[:d], rtol=1e-2)


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
