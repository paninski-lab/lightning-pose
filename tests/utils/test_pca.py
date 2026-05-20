"""Test the PCA module used in the PCA losses."""

import numpy as np
import pytest
import torch

from lightning_pose.utils.pca import (
    ComponentChooser,
    EmpiricalEpsilon,
    KeypointPCA,
    NaNPCA,
    format_multiview_data_for_pca,
)


def check_lists_equal(list_0: list, list_1: list) -> bool:
    return (len(list_0) == len(list_1)) and sorted(list_0) == sorted(list_1)


class TestKeypointPCA:
    """Test the class KeypointPCA."""

    def test_keypoint_pca_singleview(self, cfg, base_data_module_combined):
        num_train_ims = int(
            len(base_data_module_combined.dataset)
            * base_data_module_combined.train_probability
        )
        num_keypoints = base_data_module_combined.dataset.num_keypoints
        num_keypoints_for_pca = len(cfg.data.columns_for_singleview_pca)

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

        kp_pca._check_data()  # raises ValueErrors if fails

        kp_pca._fit_pca()
        kp_pca._choose_n_components()
        kp_pca.pca_prints()
        kp_pca._set_parameter_dict()

        check_lists_equal(
            list(kp_pca.parameters.keys()),
            ["mean", "kept_eigenvectors", "discarded_eigenvectors", "epsilon"],
        )

        # test reprojection error computation
        n_batches = 12
        n_keypoints = 17
        pred_keypoints = torch.rand(size=(n_batches, n_keypoints * 2), device="cpu")
        singleview_pca = KeypointPCA(
            loss_type="pca_singleview",
            data_module=base_data_module_combined,
            components_to_keep=6,
            empirical_epsilon_percentile=1.0,
            columns_for_singleview_pca=cfg.data.columns_for_singleview_pca,
        )
        singleview_pca()

        data_arr = singleview_pca._format_data(data_arr=pred_keypoints)
        # num selected keypoints in the cfg is 14, so we should have 14 * 2 columns
        assert data_arr.shape == (n_batches, 14 * 2)

        err = singleview_pca.compute_reprojection_error(data_arr=data_arr)
        assert err.shape == (n_batches, 14)

    def test_keypoint_pca_median_centering(self, cfg, base_data_module_combined):
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

    def test_keypoint_pca_mean_centering(self, cfg, base_data_module_combined):
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

    def test_keypoint_pca_multiview(
        self,
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

        kp_pca._check_data()  # raises ValueErrors if fails

        kp_pca._fit_pca()
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


class TestNaNPCA:
    """Test the class NaNPCA."""

    @pytest.fixture
    def base_pca(self):
        from sklearn.datasets import load_diabetes
        from sklearn.decomposition import PCA

        diabetes = load_diabetes()
        data = diabetes.data  # type: ignore[union-attr]
        pca_skl = PCA(svd_solver="full")
        pca_skl.fit(data)
        pca_cust = NaNPCA()
        pca_cust.fit(data)
        xhat_cust = pca_cust.transform(data)
        return data, pca_skl, pca_cust, xhat_cust

    def test_nan_pca_no_nans(self, base_pca):

        data, pca_skl, pca_cust, xhat_cust = base_pca
        assert np.sum(np.isnan(data)) == 0
        assert data.shape == (442, 10)
        xhat_skl = pca_skl.transform(data)
        assert pca_skl.noise_variance_ == pca_cust.noise_variance_
        assert pca_skl.n_samples_ == pca_cust.n_samples_
        assert pca_skl.n_components_ == pca_cust.n_components_
        assert np.allclose(  # type: ignore[arg-type]
            pca_skl.components_, pca_cust.components_, rtol=1e-10)
        assert np.allclose(pca_skl.explained_variance_, pca_cust.explained_variance_, rtol=1e-10)
        assert np.allclose(
            pca_skl.explained_variance_ratio_, pca_cust.explained_variance_ratio_, rtol=1e-10)
        assert np.allclose(pca_skl.singular_values_, pca_cust.singular_values_, rtol=1e-10)
        assert np.allclose(xhat_skl, xhat_cust, rtol=1e-10)

    def test_nan_pca_single_nan(self, base_pca):

        data, _, pca_cust, xhat_cust = base_pca
        data_nans1 = np.copy(data)
        data_nans1[0, 0] = np.nan
        pca_cust_nan1 = NaNPCA()
        pca_cust_nan1.fit(data_nans1)

        assert pca_cust.noise_variance_ == pca_cust_nan1.noise_variance_
        assert pca_cust.n_samples_ == pca_cust_nan1.n_samples_
        assert pca_cust.n_components_ == pca_cust_nan1.n_components_

        # for components, we don't consider values close to zero, as these can change sign easily
        mask = np.abs(pca_cust.components_) > 0.05
        assert not np.allclose(
            pca_cust.components_[mask], pca_cust_nan1.components_[mask], rtol=1e-10)
        assert np.allclose(pca_cust.components_[mask], pca_cust_nan1.components_[mask], rtol=1e-1)
        assert not np.allclose(
            pca_cust.explained_variance_, pca_cust_nan1.explained_variance_, rtol=1e-10)
        assert np.allclose(
            pca_cust.explained_variance_, pca_cust_nan1.explained_variance_, rtol=1e-2)
        assert not np.allclose(
            pca_cust.explained_variance_ratio_,
            pca_cust_nan1.explained_variance_ratio_,
            rtol=1e-10,
        )
        assert np.allclose(
            pca_cust.explained_variance_ratio_,
            pca_cust_nan1.explained_variance_ratio_,
            rtol=1e-2,
        )
        assert not np.allclose(
            pca_cust.singular_values_, pca_cust_nan1.singular_values_, rtol=1e-10)
        assert np.allclose(pca_cust.singular_values_, pca_cust_nan1.singular_values_, rtol=1e-2)

        xhat_cust_nan1 = pca_cust_nan1.transform(data_nans1)
        assert np.allclose(xhat_cust[1:], xhat_cust_nan1[1:], atol=1e-2)

    def test_nan_pca_many_nans(self, base_pca):

        data, _, pca_cust, xhat_cust = base_pca

        data_nans1 = np.copy(data)
        data_nans1[0, 0] = np.nan
        pca_cust_nan1 = NaNPCA()
        pca_cust_nan1.fit(data_nans1)

        rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6]
        data_nans2 = np.copy(data)
        for r, c in zip(rows, cols, strict=True):
            data_nans2[r, c] = np.nan
        pca_cust_nan2 = NaNPCA()
        pca_cust_nan2.fit(data_nans2)

        # for components, we don't consider values close to zero, as these can change sign easily
        mask = np.abs(pca_cust.components_) > 0.05
        assert not np.allclose(
            pca_cust_nan1.components_[mask], pca_cust_nan2.components_[mask], rtol=1e-10)
        assert np.allclose(
            pca_cust_nan1.components_[mask], pca_cust_nan2.components_[mask], rtol=1e0)
        # just look at the 'd' dims with highest variance
        d = 7
        assert not np.allclose(
            pca_cust_nan1.explained_variance_[:d],
            pca_cust_nan2.explained_variance_[:d],
            rtol=1e-10,
        )
        assert np.allclose(
            pca_cust_nan1.explained_variance_[:d],
            pca_cust_nan2.explained_variance_[:d],
            rtol=1e-2,
        )
        assert not np.allclose(
            pca_cust_nan1.explained_variance_ratio_[:d],
            pca_cust_nan2.explained_variance_ratio_[:d],
            rtol=1e-10,
        )
        assert np.allclose(
            pca_cust_nan1.explained_variance_ratio_[:d],
            pca_cust_nan2.explained_variance_ratio_[:d],
            rtol=1e-2,
        )
        assert not np.allclose(
            pca_cust_nan1.singular_values_[:d], pca_cust_nan2.singular_values_[:d], rtol=1e-10)
        assert np.allclose(
            pca_cust_nan1.singular_values_[:d], pca_cust_nan2.singular_values_[:d], rtol=1e-2)

        xhat_cust_nan2 = pca_cust_nan2.transform(data_nans2)
        assert np.allclose(xhat_cust[rows[-1] + 1:], xhat_cust_nan2[rows[-1] + 1:], atol=1e-2)

    def test_nan_pca_whole_row_nan(self, base_pca):

        data, _, _, xhat_cust = base_pca
        data_nans3 = np.copy(data)
        data_nans3[0, :] = np.nan
        pca_cust_nan3 = NaNPCA()
        pca_cust_nan3.fit(data_nans3)

        xhat_cust_nan3 = pca_cust_nan3.transform(data_nans3)
        assert np.allclose(xhat_cust[1:], xhat_cust_nan3[1:], atol=1e-2)
        assert np.all(xhat_cust_nan3[0, :] == 0)


class TestEmpiricalEpsilon:
    """Test the class EmpiricalEpsilon."""

    def test_empirical_epsilon_numpy(self):
        ee = EmpiricalEpsilon(percentile=90)
        loss = np.arange(101, dtype='float')
        assert ee(loss) == 90

    def test_empirical_epsilon_tensor(self):
        ee = EmpiricalEpsilon(percentile=90)
        loss = np.arange(101, dtype='float')
        assert ee(torch.tensor(loss)) == 90

    def test_empirical_epsilon_nans(self):
        ee = EmpiricalEpsilon(percentile=90)
        loss = np.arange(101, dtype='float')
        loss[1::2] = np.nan
        assert ee(loss) == 90


class TestComponentChooser:
    """Test the class ComponentChooser."""

    @pytest.fixture
    def fitted_pca(self):
        from sklearn.datasets import load_diabetes
        from sklearn.decomposition import PCA

        diabetes = load_diabetes()
        data = diabetes.data  # type: ignore[union-attr]
        pca = PCA(svd_solver="full")
        pca.fit(data)
        return pca

    def test_component_chooser_integer(self, fitted_pca):
        assert ComponentChooser(fitted_pca, 4)() == 4
        assert ComponentChooser(fitted_pca, 2)() < ComponentChooser(fitted_pca, 3)()

    def test_component_chooser_integer_invalid(self, fitted_pca):
        # can't keep more than 10 components for diabetes data (obs dim = 10)
        with pytest.raises(ValueError):
            ComponentChooser(fitted_pca, 11)

    def test_component_chooser_proportion(self, fitted_pca):
        n_comps = ComponentChooser(fitted_pca, 0.95)()
        assert (n_comps > 0) and (n_comps <= 10)
        # explaining exactly 1.0 of the variance requires keeping all 10 components
        assert ComponentChooser(fitted_pca, 1.0)() == 10
        # less explained variance -> fewer components kept
        assert ComponentChooser(fitted_pca, 0.20)() < ComponentChooser(fitted_pca, 0.90)()

    def test_component_chooser_proportion_invalid(self, fitted_pca):
        with pytest.raises(ValueError):
            ComponentChooser(fitted_pca, 1.04)
        with pytest.raises(ValueError):
            ComponentChooser(fitted_pca, -0.2)


class TestFormatMultiviewDataForPca:
    """Test the function format_multiview_data_for_pca."""

    @pytest.fixture
    def keypoints(self):
        return torch.rand(size=(12, 20, 2), device="cpu")

    def test_format_multiview_two_views(self, keypoints):

        column_matches = [[0, 1, 2, 3], [4, 5, 6, 7]]
        arr = format_multiview_data_for_pca(keypoints, column_matches)
        assert arr.shape == torch.Size(
            [keypoints.shape[0] * len(column_matches[0]), 2 * len(column_matches)]
        )

    def test_format_multiview_mismatched_lengths(self, keypoints):

        column_matches = [[0, 1, 2, 3], [4, 5, 6]]
        with pytest.raises(AssertionError):
            format_multiview_data_for_pca(keypoints, column_matches)

    def test_format_multiview_three_views(self, keypoints):

        column_matches = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
        arr = format_multiview_data_for_pca(keypoints, column_matches)
        assert arr.shape == torch.Size(
            [keypoints.shape[0] * len(column_matches[0]), 2 * len(column_matches)]
        )


def test_convert_dict_values_to_tensors():

    from lightning_pose.utils.pca import convert_dict_values_to_tensors

    test_dict = {
        'param_a': 4.0,
        'param_b': 10.1,
        'param_c': 4,
    }
    test_dict_tensor = convert_dict_values_to_tensors(test_dict, device='cpu')
    for _, val in test_dict_tensor.items():
        assert isinstance(val, torch.Tensor)
        assert val.dtype == torch.float32
