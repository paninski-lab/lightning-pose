"""PCA class to assist with computing PCA losses."""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
from omegaconf import ListConfig
from sklearn.decomposition import PCA
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import DataExtractor, clean_any_nans
from lightning_pose.losses.helpers import EmpiricalEpsilon, convert_dict_values_to_tensors

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: think about exporting out the data-getting procedure to its own class so that we
# can support general arrays, like arrays with predictions.


class KeypointPCA(object):
    """Class to collect data from a dataloader and compute PCA params."""

    def __init__(
        self,
        loss_type: Literal["pca_singleview", "pca_multiview"],
        data_module: Union[UnlabeledDataModule, BaseDataModule],
        components_to_keep: Optional[Union[int, float]] = 0.95,
        empirical_epsilon_percentile: float = 90.0,
        mirrored_column_matches: Optional[Union[ListConfig, List]] = None,
        columns_for_singleview_pca: Optional[Union[ListConfig, List]] = None,
        device: Union[Literal["cuda", "cpu"], torch.device] = "cpu",
    ) -> None:
        self.loss_type = loss_type
        self.data_module = data_module
        self.components_to_keep = components_to_keep
        self.empirical_epsilon_percentile = empirical_epsilon_percentile
        self.mirrored_column_matches = mirrored_column_matches
        self.columns_for_singleview_pca = columns_for_singleview_pca
        self.pca_object = None
        self.device = device

    @property
    def _format_factory(self) -> Dict[str, Any]:
        formats = {
            "pca_multiview": self._multiview_format,
            "pca_singleview": self._singleview_format,
        }
        return formats

    def _get_data(self) -> None:
        self.data_arr, _ = DataExtractor(
            data_module=self.data_module, cond="train", extract_images=False,
            remove_augmentations=True,
        )()

    def _multiview_format(
        self, data_arr: TensorType["num_original_samples", "num_original_dims"]
    ) -> TensorType["num_original_samples_times_num_selected_keypoints", "two_times_num_views"]:
        # original shape = (batch, 2 * num_keypoints) where `num_keypoints` includes
        # keypoints views from multiple views.
        data_arr = data_arr.reshape(data_arr.shape[0], data_arr.shape[1] // 2, 2)
        # shape = (batch_size, num_keypoints, 2)
        data_arr = format_multiview_data_for_pca(
            data_arr=data_arr,
            mirrored_column_matches=self.mirrored_column_matches,
        )  # shape = (batch_size * num_keypoints, views * 2)
        return data_arr

    def _singleview_format(
        self, data_arr: TensorType["num_original_samples", "num_original_dims"]
    ) -> Union[
        TensorType["num_original_samples", "num_selected_dims"],
        TensorType["num_original_samples", "num_original_dims"],
    ]:
        # original shape = (batch, 2 * num_keypoints)
        # optionally choose a subset of the keypoints for the singleview pca
        if self.columns_for_singleview_pca is not None:
            # reshape to (batch, num_keypoints, 2) to easily select columns
            data_arr = data_arr.reshape(data_arr.shape[0], data_arr.shape[1] // 2, 2)
            # select columns
            data_arr = data_arr[:, np.array(self.columns_for_singleview_pca), :]
            # reshape back to (batch, num_selected_keypoints * 2)
            data_arr = data_arr.reshape(data_arr.shape[0], data_arr.shape[1] * 2)
        return data_arr

    def _format_data(
        self, data_arr: TensorType["num_original_samples", "num_original_dims"]
    ) -> TensorType:
        # Union[
        #     TensorType["num_original_samples", "num_selected_dims"],  # singleview filtered
        #     TensorType[
        #         "num_original_samples", "num_original_dims"
        #     ],  # singleview unfiltered
        #     TensorType[
        #         "num_original_samples_times_num_selected_keypoints", "two_times_num_views"
        #     ]],  # multiview
        return self._format_factory[self.loss_type](data_arr=data_arr)

    def _clean_any_nans(self) -> None:
        # we count nans along the first dimension, i.e., columns. We remove those rows
        # whose column nan sum > 0, i.e., more than zero nan.
        self.data_arr = clean_any_nans(self.data_arr, dim=1)

    def _ensure_no_nans(self) -> None:
        nan_count = torch.sum(torch.isnan(self.data_arr))
        if nan_count > 0:
            raise ValueError(
                "data array includes {} missing values (nans), cannot fit vanilla PCA.".format(
                    nan_count
                )
            )

    def _ensure_enough_data(self) -> None:
        # ensure we have more rows than columns after doing nan filtering
        if self.data_arr.shape[0] < self.data_arr.shape[1]:
            raise ValueError(
                "cannot fit PCA with {} samples < {} observation dimensions".format(
                    self.data_arr.shape[0], self.data_arr.shape[1]
                )
            )

    def _check_data(self) -> None:
        self._ensure_no_nans()
        self._ensure_enough_data()

    def _fit_pca(self) -> None:
        # fit PCA with the full number of comps on the cleaned-up data array.
        # note self.data_arr is a tensor but sklearn's PCA function is fine with it
        self.pca_object = PCA(svd_solver="full")
        self.pca_object.fit(X=self.data_arr)

    def _choose_n_components(self) -> None:
        # TODO: should we return an integer and not override?
        if self.loss_type == "pca_multiview":
            # all views can be explained by 3 (x,y,z) coords
            # ignore self.components_to_keep_argument
            self._n_components_kept = 3
            if self._n_components_kept != self.components_to_keep:
                warnings.warn(
                    f"for {self.loss_type} loss, you specified {self.components_to_keep} "
                    f"components_to_keep, but we will instead keep {self._n_components_kept} "
                    f"components"
                )
        elif self.loss_type == "pca_singleview":
            if self.pca_object is not None:
                self._n_components_kept = ComponentChooser(
                    fitted_pca_object=self.pca_object,
                    components_to_keep=self.components_to_keep,
                ).__call__()
        assert isinstance(self._n_components_kept, int)

    def pca_prints(self) -> None:
        # call after we've fitted a pca object and selected how many components to keep
        pca_prints(self.pca_object, self.loss_type, self._n_components_kept)

    def _set_parameter_dict(self) -> None:
        self.parameters = {}  # dict with same keys as loss_param_dict
        self.parameters["mean"] = self.pca_object.mean_
        self.parameters["kept_eigenvectors"] = \
            self.pca_object.components_[:self._n_components_kept]
        self.parameters["discarded_eigenvectors"] = \
            self.pca_object.components_[self._n_components_kept:]
        self.parameters = convert_dict_values_to_tensors(self.parameters, self.device)
        self.parameters["epsilon"] = EmpiricalEpsilon(
            percentile=self.empirical_epsilon_percentile)(loss=self.compute_reprojection_error())

    def reproject(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> TensorType["num_samples", "sample_dim"]:
        """reproject a data array using the fixed pca parameters. everything happens in torch.
        applying the transformation as it is in scikit learn.
        https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/decomposition/_base.py#L125
        """
        # TODO: more type assertions
        # assert that datashape is valid
        if data_arr is None:
            data_arr = self.data_arr.to(self.device)
        evecs = self.parameters["kept_eigenvectors"]
        mean = self.parameters["mean"].unsqueeze(0)

        # assert that the observation dimension is equal for all objects
        assert data_arr.shape[1] == evecs.shape[1] and evecs.shape[1] == mean.shape[1]
        # verify that data array has observations divisible by 2 (corresponding to (x,y) coords)
        assert data_arr.shape[1] % 2 == 0

        # transform data into low-d space as in scikit learn's _BasePCA.transform()
        # https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_base.py#L97
        centered_data = data_arr - mean
        low_d_projection = centered_data @ evecs.T

        # project back up to observation space, as in scikit learn's _BasePCA.inverse_transform()
        # https://github.com/scikit-learn/scikit-learn/blob/37ac6788c9504ee409b75e5e24ff7d86c90c2ffb/sklearn/decomposition/_base.py#L125
        reprojection = low_d_projection @ evecs + mean
        return reprojection

    def compute_reprojection_error(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> TensorType["num_samples", "sample_dim_over_two"]:
        """returns error per 2D keypoint"""
        if data_arr is None:
            data_arr = self.data_arr.to(self.device)
        reprojection = self.reproject(data_arr=data_arr)
        diff = data_arr - reprojection
        # reshape to get a per-keypoint differences:
        diff_arr_per_keypoint = diff.reshape(diff.shape[0], diff.shape[1] // 2, 2)
        # compute the prediction error for each 2D bodypart
        reprojection_loss = torch.linalg.norm(diff_arr_per_keypoint, dim=2)

        return reprojection_loss

    def __call__(self) -> None:

        # TODO: think if we always like to override data_arr.
        # should we need a copy of it to undo the nan stuff?

        # save training data in self.data_arr
        # TODO: consider putting in init
        self._get_data()

        # modify self.data_arr in the case of multiview pca, else keep the same
        self.data_arr = self._format_data(data_arr=self.data_arr)

        # remove those observations with more than one Nan.
        # TODO: consider infilling somehow
        self._clean_any_nans()

        # check no nans, and that we have more observations than observation-dimensions
        self._check_data()

        self._fit_pca()
        self._choose_n_components()
        self.pca_prints()

        # save all the meaningful quantities
        self._set_parameter_dict()


@typechecked
class ComponentChooser:
    """Determine the number of PCA components to keep."""

    def __init__(
        self,
        fitted_pca_object: PCA,
        components_to_keep: Optional[Union[int, float]],
    ) -> None:
        self.fitted_pca_object = fitted_pca_object
        # can be either a float indicating proportion of explained variance, or an
        # integer specifying the number of components
        self.components_to_keep = components_to_keep
        self._check_components_to_keep()

    # TODO: I dislike the confusing names (components_to_keep vs min_variance_explained)

    @property
    def cumsum_explained_variance(self):
        return np.cumsum(self.fitted_pca_object.explained_variance_ratio_)

    def _check_components_to_keep(self) -> None:
        # if int, ensure it's not too big
        if type(self.components_to_keep) is int:
            if self.components_to_keep > self.fitted_pca_object.n_components_:
                raise ValueError(
                    f"components_to_keep was set to {self.components_to_keep}, exceeding the "
                    f"maximum value of {self.fitted_pca_object.n_components_} observation dims"
                )
        # if float, ensure a proportion between 0.0-1.0
        elif type(self.components_to_keep) is float:
            if self.components_to_keep < 0.0 or self.components_to_keep > 1.0:
                raise ValueError(
                    f"components_to_keep was set to {self.components_to_keep} while it has to be "
                    f"between 0.0 and 1.0"
                )

    def _find_first_threshold_cross(self) -> int:
        # find the index of the first element above a min_variance_explained threshold
        assert type(
            self.components_to_keep is float
        )  # i.e., threshold crossing doesn't make sense with integer components_to_keep
        if self.components_to_keep != 1.0:
            components_to_keep = int(
                np.where(self.cumsum_explained_variance >= self.components_to_keep)[0][0]
            )
            # cumsum is a d - 1 dimensional vector where the 0th element is the sum of the
            # 0th and 1st element of the d dimensional vector it is summing over
            return components_to_keep + 1
        else:  # if we want to keep all components, we need to return the number of components
            # we do this because there's an issue with == 1.0 in the cumsum_explained_variance
            return len(self.fitted_pca_object.explained_variance_)

    def __call__(self) -> int:
        if type(self.components_to_keep) is int:
            return self.components_to_keep  # return integer as is
        elif type(self.components_to_keep) is float:
            return (
                self._find_first_threshold_cross()
            )  # find that integer that crosses the minimum explained variance


@typechecked
def pca_prints(pca: PCA, condition: str, components_to_keep: int) -> None:
    print("Results of running PCA ({}) on keypoints:".format(condition))
    print(
        "Kept {}/{} components, and found:".format(
            components_to_keep, pca.n_components_
        )
    )
    evr = np.round(pca.explained_variance_ratio_, 3)
    print("Explained variance ratio: {}".format(evr))
    tev = np.round(np.sum(pca.explained_variance_ratio_[:components_to_keep]), 3)
    print("Variance explained by {} components: {}".format(components_to_keep, tev))


# @typechecked
def format_multiview_data_for_pca(
    data_arr: TensorType["batch", "num_keypoints", "2"],
    mirrored_column_matches: Union[ListConfig, list],
) -> TensorType["batch_times_num_selected_keypoints", "two_times_num_views"]:
    """Reformat multiview data so each observation is a single body part across views.

    Args:
        data_arr: keypoints from training data
        mirrored_column_matches: one element for each camera view; each element is
            itself a list that contains indices into the overall ordering of the
            keypoints

    Returns:
        formatted data to run pca

    """
    n_views = len(mirrored_column_matches)
    n_keypoints = len(mirrored_column_matches[0])  # only the ones used for all views
    data_arr_views = []
    # separate views and reformat
    for view in range(n_views):
        assert len(mirrored_column_matches[view]) == n_keypoints
        # all the (x,y) coordinates of all bodyparts from current view
        data_arr_tmp = data_arr[:, np.array(mirrored_column_matches[view]), :]
        # first permute to: 2, batch, num_keypoints, then reshape to: 2, batch * num_keypoints
        data_arr_tmp = data_arr_tmp.permute(2, 0, 1).reshape(
            2, -1
        )  # -> 2 X num_frames*num_bodyparts
        data_arr_views.append(data_arr_tmp)
    # concatenate views
    data_arr = torch.cat(
        data_arr_views, dim=0
    )  # -> 2 * n_views X num_frames*num_bodyparts
    return data_arr.T  # note the transpose
