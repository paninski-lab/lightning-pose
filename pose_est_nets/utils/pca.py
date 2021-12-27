"""Preprocessing for specialized losses."""

from omegaconf import DictConfig, ListConfig
import numpy as np
from pytorch_lightning.core import datamodule
from sklearn.decomposition import PCA
import sklearn
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Union, Literal

from pose_est_nets.datasets.datamodules import UnlabeledDataModule
from pose_est_nets.datasets.datasets import HeatmapDataset
from pose_est_nets.datasets.utils import clean_any_nans

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked

# TODO: think about exporting out the data-getting procedure to its own class so that we can support general arrays, like arrays with predictions.


@typechecked
class KeypointPCA:
    def __init__(
        self,
        loss_type: Literal["pca_singleview", "pca_multiview"],
        data_module: UnlabeledDataModule,
        components_to_keep: Optional[Union[int, float]],
        empirical_epsilon_percentile: float = 90.0,
    ):
        self.loss_type = loss_type
        self.data_module = data_module
        self.components_to_keep = components_to_keep
        self.empirical_epsilon_percentile = empirical_epsilon_percentile

    def _get_training_data(self) -> None:
        # TODO: try to simplify.
        # iterate over train loader and organize data as a tensor
        # collect data on which to run pca from data module
        # Subset inherits from dataset, it doesn't have access to dataset.keypoints
        if type(self.data_module.train_dataset) == torch.utils.data.dataset.Subset:

            # copy data module to manipulate it without interfering with original
            if type(self.data_module.dataset) == HeatmapDataset:
                pca_data = super(
                    type(self.data_module.dataset), self.data_module.dataset
                )
            else:
                pca_data = self.data_module.dataset

            indxs = torch.tensor(self.data_module.train_dataset.indices)
            data_arr = torch.index_select(
                self.data_module.dataset.keypoints.detach().clone(), 0, indxs
            )  # data_arr is shape (train_batches, keypoints, 2)

            # apply augmentation which *downsamples* the frames/keypoints
            if self.data_module.dataset.imgaug_transform:
                i = 0
                for idx in indxs:
                    batch_dict = pca_data.__getitem__(idx)
                    data_arr[i] = batch_dict["keypoints"].reshape(-1, 2)
                    i += 1
        else:
            data_arr = (
                self.data_module.train_dataset.keypoints.detach().clone()
            )  # won't work for random splitting

            # apply augmentation which *downsamples* the frames
            if self.data_module.train_dataset.imgaug_transform:
                for i in range(len(data_arr)):
                    data_arr[i] = super(
                        type(self.data_module.train_dataset),
                        self.data_module.train_dataset,
                    ).__getitem__(i)["keypoints"]

        self.data_arr = data_arr

    def _format_data(self) -> None:
        # TODO: check that the two format end up having same rows/columns division
        if self.data_arr is not None:
            if self.loss_type == "pca_multiview":
                self.data_arr = format_multiview_data_for_pca(
                    data_arr=self.data_arr,
                    mirrored_column_matches=self.data_module.loss_param_dict[
                        "pca_multiview"
                    ]["mirrored_column_matches"],
                )
            else:  # no need to format single-view data
                pass

    def _clean_any_nans(self) -> None:
        # we count nans along the first dimension, i.e., columns. We remove those rows whose column nan sum > 0, i.e., more than zero nan.
        self.data_arr = clean_any_nans(self.data_arr, dim=1)

    def _check_data(self) -> None:
        # ensure we have more rows than columns after doing nan filtering
        # TODO: raise a more informative error?
        assert self.data_arr.shape[0] >= self.data_arr.shape[1]

    def _fit_pca(self) -> None:
        # fit PCA with the full number of comps
        pass

    def _choose_n_components(self) -> None:
        # TODO: should we return an integer and not override?
        # call the ComponentChooser class which we just tested.
        pass

    def __call__(self):
        # TODO: think if we always like to override data_arr. should we need a copy of it to undo the nan stuff?
        self._get_training_data()  # save training data in self.data_arr, TODO: consider putting in init
        self._format_data()  # modify self.data_arr in the case of multiview pca, else keep the same
        self._clean_any_nans()  # remove those observations with more than one Nan. TODO: consider infilling somehow
        self._check_data()  # check that we have more observations than observation-dimensions


class ComponentChooser:
    """determines the number of components to keep."""

    def __init__(
        self,
        pca_object: sklearn.decomposition.PCA,
        components_to_keep: Optional[Union[int, float]],
    ):
        self.pca_object = pca_object
        self.components_to_keep = components_to_keep  # can be either a float indicating proportion of explained variance, or an integer specifying the number of components.
        self._check_components_to_keep()

    # TODO: I dislike the confusing names (components to keep versus min_variance_explained)
    @property
    def cumsum_explained_variance(self):
        return np.cumsum(self.pca_object.explained_variance_ratio_)

    def _check_components_to_keep(self) -> None:
        # if int, ensure it's not too big
        if type(self.components_to_keep) is int:
            assert self.components_to_keep <= len(
                self.pca_object.explained_variance_ratio_
            )  # TODO: check if there's a clearer attribute like n_components
        # if float, ensure a proportion between 0.0-1.0
        elif type(self.components_to_keep) is float:
            assert self.components_to_keep >= 0.0 and self.components_to_keep <= 1.0

    def _find_first_threshold_cross(self) -> int:
        # find the index of the first element above a min_variance_explained threshold
        assert type(self.components_to_keep is float)
        components_to_keep = int(
            np.where(self.cumsum_explained_variance >= self.components_to_keep)[0][0]
        )
        # cumsum is a d - 1 dimensional vector where the 0th element is the sum of the 0th and 1st element of the d dimensional vector it is summing over
        return components_to_keep + 1

    def __call__(self) -> int:
        if type(self.components_to_keep) is int:
            return self.components_to_keep  # return integer as is
        elif type(self.components_to_keep) is float:
            return (
                self._find_first_threshold_cross()
            )  # find that integer that crosses the minimum explained variance


@typechecked
def compute_pca_params(
    loss_type: Literal["pca_singleview", "pca_multiview"],
    data_module: UnlabeledDataModule,
    components_to_keep: Optional[Union[int, float]],
    empirical_epsilon_percentile: float = 90.0,
):
    # TODO:
    # unify the two funcs to one. handle prints nicely.
    # handle the components_to_keep nicely. make sure you pass the right arguments in datamodule.
    # make sure you cover the case where the two are passed? or assume it won't happen?
    # call the formatting function accordingly
    return None


@typechecked
def compute_multiview_pca_params(
    data_module: UnlabeledDataModule,
    components_to_keep: int = 3,
    empirical_epsilon_percentile: float = 90.0,
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

    # format data and run pca
    data_arr = get_train_data_for_pca(data_module=data_module)
    # shape will be (2 * num_views, num_batches * num_keypoints)
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

    # send parameters to loss_param_dict["pca_singleview"]
    add_params_to_loss_dict(
        data_module=data_module,
        loss_key="pca_multiview",
        mean=pca.mean_,
        kept_eigenvectors=pca.components_[:components_to_keep],
        discarded_eigenvectors=pca.components_[components_to_keep:],
    )

    epsilon = compute_epsilon_for_PCA(
        good_arr_for_pca=good_arr_for_pca.to(_TORCH_DEVICE),
        kept_eigenvectors=data_module.loss_param_dict["pca_multiview"][
            "kept_eigenvectors"
        ],
        mean=data_module.loss_param_dict["pca_multiview"]["mean"],
        empirical_epsilon_percentile=empirical_epsilon_percentile,
    )

    add_params_to_loss_dict(
        data_module=data_module,
        loss_key="pca_multiview",
        epsilon=epsilon,
    )

    return


# TODO: add TensorType
@typechecked
def compute_PCA_reprojection_error(
    good_arr_for_pca: torch.Tensor,
    kept_eigenvectors: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    # first verify that the pca array has observations divisible by 2 (corresponding to (x,y) coords)
    good_arr_for_pca = good_arr_for_pca.T - mean.unsqueeze(
        0
    )  # transpose and mean-center
    assert good_arr_for_pca.shape[1] % 2 == 0
    reprojection_arr = (
        good_arr_for_pca @ kept_eigenvectors.T @ kept_eigenvectors
    )  # e.g., (214, 4) X (4, 3) X (3, 4) = (214, 4) as we started
    diff = (
        good_arr_for_pca - reprojection_arr
    )  # shape: (num_samples: for multiview (num_samples=num_samples X num_bodyparts), observation_dim: for multiview (2 X num_views), for singleview (2 X num_bodyparts))
    # reshape:
    diff_arr_per_keypoint = diff.reshape(diff.shape[0], diff.shape[1] // 2, 2)
    reprojection_loss = torch.linalg.norm(diff_arr_per_keypoint, dim=2)
    # print("reprojection_loss.shape: {}".format(reprojection_loss.shape))
    # print("diff: {}".format(diff[:5, :]))
    # print("diff_arr_per_keypoint: {}".format(diff_arr_per_keypoint[:5, :, :]))
    return reprojection_loss


@typechecked
def compute_epsilon_for_PCA(
    good_arr_for_pca: torch.Tensor,
    kept_eigenvectors: torch.Tensor,
    mean: torch.Tensor,
    empirical_epsilon_percentile: float,
):

    reprojection_loss = compute_PCA_reprojection_error(
        good_arr_for_pca, kept_eigenvectors, mean
    )

    epsilon = np.nanpercentile(
        reprojection_loss.flatten().clone().detach().cpu().numpy(),
        empirical_epsilon_percentile,
        axis=0,
    )
    return epsilon


@typechecked
def compute_singleview_pca_params(
    data_module: UnlabeledDataModule, empirical_epsilon_percentile: float = 90.0
) -> None:
    """Compute eigenvalues and eigenvectors of labeled data for singleview pca loss.

    Note: this function updates attributes of `data_module`

    Args:
        data_module: initialized unlabeled data module, which contains all the relevant
            information
        empirical_epsilon_percentile (float): a percentile of all errors, below which errors are zeroed out

    """
    print("Computing PCA on singleview keypoints...")

    data_arr = get_train_data_for_pca(data_module=data_module)
    # format data and run pca
    # shape is (num_batches, num_keypoints * 2)
    arr_for_pca = data_arr.reshape(data_arr.shape[0], -1)
    print("Initial array for pca shape: {}".format(arr_for_pca.shape))

    good_arr_for_pca = clean_any_nans(arr_for_pca, dim=1)
    print(
        "good_arr_for_pca shape: {}".format(good_arr_for_pca.shape)
    )  # TODO: have prints as tests
    # want to make sure we have more rows than columns after doing nan filtering
    assert (
        good_arr_for_pca.shape[0] >= good_arr_for_pca.shape[1]
    ), "filtered out too many nan frames"
    pca = PCA(n_components=good_arr_for_pca.shape[1], svd_solver="full")
    pca.fit(good_arr_for_pca)
    print("Done!")
    tot_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    components_to_keep = int(
        np.where(
            tot_explained_variance
            >= data_module.loss_param_dict["pca_singleview"]["min_variance_explained"]
        )[0][0]
    )
    components_to_keep += 1  # cumsum is a d - 1 dimensional vector where the 0th element is the sum of the 0th and 1st element of the d dimensional vector it is summing over
    pca_prints(pca, components_to_keep)  # print important params

    # send parameters to loss_param_dict["pca_singleview"]
    add_params_to_loss_dict(
        data_module=data_module,
        loss_key="pca_singleview",
        mean=pca.mean_,
        kept_eigenvectors=pca.components_[:components_to_keep],
        discarded_eigenvectors=pca.components_[components_to_keep:],
    )

    # now compute epsilon
    epsilon = compute_epsilon_for_PCA(
        good_arr_for_pca=good_arr_for_pca.to(_TORCH_DEVICE).T,
        kept_eigenvectors=data_module.loss_param_dict["pca_singleview"][
            "kept_eigenvectors"
        ],
        mean=data_module.loss_param_dict["pca_singleview"]["mean"],
        empirical_epsilon_percentile=empirical_epsilon_percentile,
    )

    # send it to loss dict
    add_params_to_loss_dict(
        data_module=data_module,
        loss_key="pca_singleview",
        epsilon=epsilon,
    )


@typechecked
def get_train_data_for_pca(data_module: UnlabeledDataModule) -> torch.Tensor:
    """collect training data on which to run pca from data module. extract the training frames, transform them, and spit out a tensor of data.

    Args:
        data_module (UnlabeledDataModule): an instance of UnlabeledDataModule

    Returns:
        torch.Tensor:
    """
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
                batch_dict = pca_data.__getitem__(idx)
                data_arr[i] = batch_dict["keypoints"].reshape(-1, 2)
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

    return data_arr


@typechecked
def add_params_to_loss_dict(
    data_module: UnlabeledDataModule,
    loss_key: Literal["pca_singleview", "pca_multiview"],
    **kwargs
):
    # TODO: be careful of dtype (for half-precision training) and device (for multinode)
    print("original data_module.loss_param_dict")
    print(data_module.loss_param_dict[loss_key])
    for param_name, param_val in kwargs.items():
        # make it a tensor and send to device
        tensor_param = torch.tensor(
            param_val,
            dtype=torch.float32,
            device=_TORCH_DEVICE,
        )
        print(
            "modifying data_module.loss_param_dict[%s][%s] with:"
            % (loss_key, param_name)
        )
        print(tensor_param)
        # save in dict
        data_module.loss_param_dict[loss_key][param_name] = tensor_param


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
) -> TensorType["batch_times_num_keypoints", "two_times_num_views"]:
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
    return data_arr.T  # note the transpose
