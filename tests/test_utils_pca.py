from numpy.lib.npyio import load
import torch
import numpy as np
import pytest
import yaml
import sklearn


def test_format_multiview_data_for_pca():

    from pose_est_nets.utils.pca import format_multiview_data_for_pca

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

    from pose_est_nets.utils.pca import ComponentChooser

    # regular integer behavior
    comp_chooser_int = ComponentChooser(pca, 4)
    assert comp_chooser_int() == 4

    # can't keep more than 10 componets for diabetes data (obs dim = 10)
    with pytest.raises(AssertionError):
        comp_chooser_int = ComponentChooser(pca, 11)

    # we return ints, so basically checking that 2 < 3
    assert ComponentChooser(pca, 2)() < ComponentChooser(pca, 3)()

    # can't explain more than 1.0 of the variance
    with pytest.raises(AssertionError):
        comp_chooser_float = ComponentChooser(pca, 1.04)

    # no negative proportions
    with pytest.raises(AssertionError):
        comp_chooser_float = ComponentChooser(pca, -0.2)

    # typical behavior
    n_comps = ComponentChooser(pca, 0.95)()
    assert (n_comps > 0) and (n_comps <= 10)

    # for explaining exactly 1.0 of the variance, you should keep all 10 components
    assert ComponentChooser(pca, 1.0)() == 10

    # less explained variance -> less components kept
    assert ComponentChooser(pca, 0.20)() < ComponentChooser(pca, 0.90)()
