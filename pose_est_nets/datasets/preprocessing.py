"""Preprocessing for specialized losses."""

from omegaconf import DictConfig, ListConfig
import numpy as np
from sklearn.decomposition import PCA
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Union

from pose_est_nets.datasets.datamodules import UnlabeledDataModule
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.utils import clean_any_nans

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked


@typechecked
def compute_multiview_pca_params(
        data_module: UnlabeledDataModule,
        components_to_keep: int = 3,
        empirical_epsilon_percentile: float = 90.0
) -> None:
    """Compute eigenvalues and eigenvectors of labeled data for multiview pca loss.

    Note: this function updates attributes of `data_module`

    Args:
        data_module: initialized unlabeled data module, which contains all the relevant
            information
        components_to_keep: projections of predicted keypoints onto remaining components
            will be penalized; enforces a low-dimensional prediction from the network
        empirical_epsilon_percentile: ?

    """
    print("Computing PCA on multiview keypoints...")

    # collect data on which to run pca from data module
    # Subset inherits from dataset, it doesn't have access to dataset.keypoints
    if type(data_module.train_dataset) == torch.utils.data.dataset.Subset:

        # copy data module to manipulate it without interfering with original
        if type(data_module.dataset) == HeatmapDataset:
            pca_data = super(type(data_module.dataset), data_module.dataset)
        else:
            pca_data = data_module.dataset

        indxs = torch.tensor(data_module.train_dataset.indices)
        data_arr = torch.index_select(
            data_module.dataset.keypoints.detach().clone(), 0, indxs
        )  # data_arr is shape (train_batches, keypoints, 2)

        # apply augmentation which *downsamples* the frames
        if data_module.dataset.imgaug_transform:
            i = 0
            for idx in indxs:
                batch = pca_data.__getitem__(idx)
                data_arr[i] = batch["keypoints"].reshape(-1, 2)
                i += 1
    else:
        data_arr = (
            data_module.train_dataset.keypoints.detach().clone()
        )  # won't work for random splitting

        # apply augmentation which *downsamples* the frames/keypoints
        if data_module.train_dataset.imgaug_transform:
            for i in range(len(data_arr)):
                data_arr[i] = super(
                    type(data_module.train_dataset), data_module.train_dataset
                ).__getitem__(i)["keypoints"]

    # format data and run pca
    #shape will be (2 * num_views, num_batches * num_keypoints)
    arr_for_pca = format_multiview_data_for_pca(
        data_arr,
        data_module.loss_param_dict["pca_multiview"]["mirrored_column_matches"],
    )
    print("Initial array for pca shape: {}".format(arr_for_pca.shape))
    good_arr_for_pca = clean_any_nans(arr_for_pca, dim=0)
    pca = PCA(n_components=good_arr_for_pca.shape[0], svd_solver="full")
    pca.fit(good_arr_for_pca.T)
    print("Done!")
    print(
        "good_arr_for_pca shape: {}".format(good_arr_for_pca.shape)
    )  # TODO: have prints as tests
    pca_prints(pca, components_to_keep)  # print important params
    data_module.loss_param_dict["pca_multiview"]["kept_eigenvectors"] = torch.tensor(
        pca.components_[:components_to_keep],
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )
    data_module.loss_param_dict["pca_multiview"]["discarded_eigenvectors"] = torch.tensor(
        pca.components_[components_to_keep:],
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )

    # compute the keypoints' projections on the discarded components, to
    # estimate the e.g., 90th percentile and determine epsilon.
    # absolute value is important -- projections can be negative.
    discarded_eigs = data_module.loss_param_dict["pca_multiview"]["discarded_eigenvectors"]
    proj_discarded = torch.abs(
        torch.matmul(
            arr_for_pca.T,
            discarded_eigs.clone().detach().cpu().T,
        )
    )
    # setting axis = 0 generalizes to multiple discarded components
    epsilon = np.nanpercentile(
        proj_discarded.numpy(), empirical_epsilon_percentile, axis=0
    )
    print(epsilon)
    data_module.loss_param_dict["pca_multiview"]["epsilon"] = torch.tensor(
        epsilon,
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )

@typechecked
def compute_singleview_pca_params(
        data_module: UnlabeledDataModule,
        empirical_epsilon_percentile: float = 90.0
) -> None:
    """Compute eigenvalues and eigenvectors of labeled data for singleview pca loss.

    Note: this function updates attributes of `data_module`

    Args:
        data_module: initialized unlabeled data module, which contains all the relevant
            information
        empirical_epsilon_percentile: ?

    """
    print("Computing PCA on singleview keypoints...")

    # collect data on which to run pca from data module
    # Subset inherits from dataset, it doesn't have access to dataset.keypoints
    if type(data_module.train_dataset) == torch.utils.data.dataset.Subset:

        # copy data module to manipulate it without interfering with original
        if type(data_module.dataset) == HeatmapDataset:
            pca_data = super(type(data_module.dataset), data_module.dataset)
        else:
            pca_data = data_module.dataset

        indxs = torch.tensor(data_module.train_dataset.indices)
        data_arr = torch.index_select(
            data_module.dataset.keypoints.detach().clone(), 0, indxs
        )  # data_arr is shape (train_batches, keypoints, 2)

        # apply augmentation which *downsamples* the frames/keypoints
        if data_module.dataset.imgaug_transform:
            i = 0
            for idx in indxs:
                batch = pca_data.__getitem__(idx)
                data_arr[i] = batch["keypoints"].reshape(-1, 2)
                i += 1
    else:
        data_arr = (
            data_module.train_dataset.keypoints.detach().clone()
        )  # won't work for random splitting

        # apply augmentation which *downsamples* the frames
        if data_module.train_dataset.imgaug_transform:
            for i in range(len(data_arr)):
                data_arr[i] = super(
                    type(data_module.train_dataset), data_module.train_dataset
                ).__getitem__(i)["keypoints"]

    # format data and run pca
    #shape is (num_batches, num_keypoints * 2)
    arr_for_pca = data_arr.reshape(data_arr.shape[0], -1)
    print("Initial array for pca shape: {}".format(arr_for_pca.shape))

    good_arr_for_pca = clean_any_nans(arr_for_pca, dim=1)
    print(
        "good_arr_for_pca shape: {}".format(good_arr_for_pca.shape)
    )  # TODO: have prints as tests
    #want to make sure we have more rows than columns after doing nan filtering
    assert(good_arr_for_pca.shape[0] >= good_arr_for_pca.shape[1]), "filtered out too many nan frames"
    pca = PCA(n_components=good_arr_for_pca.shape[1], svd_solver="full")
    pca.fit(good_arr_for_pca)
    print("Done!")
    tot_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    components_to_keep = int(np.where(tot_explained_variance >= data_module.loss_param_dict["pca_singleview"]["min_variance_explained"])[0][0])
    components_to_keep += 1 #cumsum is a d - 1 dimensional vector where the 0th element is the sum of the 0th and 1st element of the d dimensional vector it is summing over
    pca_prints(pca, components_to_keep)  # print important params
    data_module.loss_param_dict["pca_singleview"]["kept_eigenvectors"] = torch.tensor(
        pca.components_[:components_to_keep],
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )
    data_module.loss_param_dict["pca_singleview"]["discarded_eigenvectors"] = torch.tensor(
        pca.components_[components_to_keep:],
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )

    # compute the keypoints' projections on the discarded components, to
    # estimate the e.g., 90th percentile and determine epsilon.
    # absolute value is important -- projections can be negative.
    #shape is (num_discarded_components, num_keypoints * 2)
    discarded_eigs = data_module.loss_param_dict["pca_singleview"]["discarded_eigenvectors"]
    #array for pca shape is (num_batches, num_keypoints * 2)
    proj_discarded = torch.abs(
        torch.matmul(
            arr_for_pca,
            discarded_eigs.clone().detach().cpu().T,
        )
    )
    # setting axis = 0 generalizes to multiple discarded components
    #shape (num_discarded_components, 1)
    epsilon = np.nanpercentile(
        proj_discarded.numpy(), empirical_epsilon_percentile, axis=0
    )
    print(epsilon)
    data_module.loss_param_dict["pca_singleview"]["epsilon"] = torch.tensor(
        epsilon,
        dtype=torch.float32,
        device=_TORCH_DEVICE,  # TODO: be careful for multinode
    )

@typechecked
def pca_prints(pca: PCA, components_to_keep: int) -> None:
    print("Results of running PCA on keypoints:")
    evr = np.round(pca.explained_variance_ratio_, 3)
    print("components kept: {}".format(components_to_keep))
    print("explained_variance_ratio_: {}".format(evr))
    tev = np.round(np.sum(pca.explained_variance_ratio_[:components_to_keep]), 3)
    print("total_explained_var: {}".format(tev))


@typechecked
def format_multiview_data_for_pca(
        data_arr: TensorType["batch", "num_keypoints", "2"],
        mirrored_column_matches: Union[ListConfig, List],
) -> TensorType["two_time_num_views", "batch_times_num_keypoints"]:
    """

    Args:
        data_arr: keypoints from training data
        mirrored_column_matches: one element for each camera view; each element is
            itself a list that contains indices into the overall ordering of the
            keypoints

    Returns:
        formatted data to run pca

    """
    n_views = len(mirrored_column_matches)
    n_keypoints = len(mirrored_column_matches[0])
    data_arr_views = []
    # separate views and reformat
    for view in range(n_views):
        assert len(mirrored_column_matches[view]) == n_keypoints
        data_arr_tmp = data_arr[:, np.array(mirrored_column_matches[view]), :]
        data_arr_tmp = data_arr_tmp.permute(2, 0, 1).reshape(2, -1)
        data_arr_views.append(data_arr_tmp)
    # concatenate views
    data_arr = torch.cat(data_arr_views, dim=0)
    return data_arr
