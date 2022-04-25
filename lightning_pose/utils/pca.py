"""PCA class to assist with computing PCA losses."""

import numpy as np
from omegaconf import DictConfig, ListConfig
from sklearn.decomposition import PCA
import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import List, Optional, Union, Literal, Dict, Any, Tuple
import warnings

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.utils import clean_any_nans, DataExtractor
from lightning_pose.losses.helpers import (
    EmpiricalEpsilon,
    convert_dict_values_to_tensors,
)

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

patch_typeguard()  # use before @typechecked

# TODO: think about exporting out the data-getting procedure to its own class so that we
# can support general arrays, like arrays with predictions.


@typechecked
class KeypointPCA(object):
    """Class to collect data from a dataloader and compute PCA params."""

    def __init__(
        self,
        loss_type: Literal["pca_singleview", "pca_multiview"],
        error_metric: Literal["reprojection_error", "proj_on_discarded_evecs"],
        data_module: Union[UnlabeledDataModule, BaseDataModule],
        components_to_keep: Optional[Union[int, float]] = 0.95,
        empirical_epsilon_percentile: float = 90.0,
        mirrored_column_matches: Optional[Union[ListConfig, List]] = None,
        columns_for_singleview_pca: Optional[Union[ListConfig, List]] = None,
        device: Union[Literal["cuda", "cpu"], torch.device] = "cpu",
    ):
        self.loss_type = loss_type
        self.error_metric = error_metric
        self.data_module = data_module
        self.components_to_keep = components_to_keep
        self.empirical_epsilon_percentile = empirical_epsilon_percentile
        self.mirrored_column_matches = mirrored_column_matches
        self.columns_for_singleview_pca = columns_for_singleview_pca
        self.pca_object = None
        self.device = device

    @property
    def _error_metric_factory(self):
        metrics = {
            "reprojection_error": self.compute_reprojection_error,
            "proj_on_discarded_evecs": self.compute_discarded_evec_error,
        }
        return metrics

    @property
    def _format_factory(self) -> Dict[str, Any]:
        formats = {
            "pca_multiview": self._multiview_format,
            "pca_singleview": self._singleview_format,
        }
        return formats

    def compute_error(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> TensorType["num_samples", -1]:
        return self._error_metric_factory[self.error_metric](data_arr=data_arr)

    def _get_data(self) -> None:
        self.data_arr, _ = DataExtractor(data_module=self.data_module, cond="train", extract_images=False)()

    def _multiview_format(
        self, data_arr: TensorType["num_original_samples", "num_original_dims"]
    ) -> TensorType[
        "num_original_samples_times_num_selected_keypoints", "two_times_num_views"
    ]:
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
                    "for {} loss, you specified {} components_to_keep, but we will instead keep {} components".format(
                        self.loss_type, self.components_to_keep, self._n_components_kept
                    )
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
        # TODO: parameters of pca, need to send over to the loss param dict
        self.parameters = {}  # dict with same keys as loss_param_dict
        self.parameters["mean"] = self.pca_object.mean_
        self.parameters["kept_eigenvectors"] = self.pca_object.components_[
            : self._n_components_kept
        ]
        self.parameters["discarded_eigenvectors"] = self.pca_object.components_[
            self._n_components_kept :
        ]
        self.parameters = convert_dict_values_to_tensors(self.parameters, self.device)

        self.parameters["epsilon"] = EmpiricalEpsilon(
            percentile=self.empirical_epsilon_percentile
        )(
            loss=self.compute_error()
        )  # self.compute_reprojection_error()
        # was loss=self._compute_reprojection_error() relying on the external func

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

    def project_onto_discarded_evecs(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> Optional[TensorType["num_samples", "num_discarded_evecs"]]:
        if data_arr is None:
            data_arr = self.data_arr.to(self.device)
        discarded_evecs = self.parameters["discarded_eigenvectors"]
        mean = self.parameters["mean"].unsqueeze(0)
        # mean subtract
        centered_data = data_arr - mean
        # project onto discarded components
        return centered_data @ discarded_evecs.T

    def compute_discarded_evec_error(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> TensorType["num_samples", 1]:
        """returns a single error for all keypoints in a sample"""
        if data_arr is None:
            data_arr = self.data_arr.to(self.device)

        proj = self.project_onto_discarded_evecs(data_arr=data_arr)
        loss = torch.linalg.norm(proj, dim=1)
        return loss.reshape(data_arr.shape[0], 1)

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

    # def _compute_reproj_error(self) -> torch.Tensor:
    #     return compute_pca_reprojection_error(
    #         self.data_arr.to(self.device),
    #         self.parameters["kept_eigenvectors"],
    #         self.parameters["mean"],
    #     )

    def __call__(self):

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
class LinearGaussian(KeypointPCA):
    """
    A linear Gaussian model for keypoint detection.
    Parametrized as in Bishop PRML Appendix B.32-B.51
    p(z) = N(z; \mu, Lambda^{-1})
    p(x|z) = N(x; Az + b, L^{-1})
    """

    def __init__(
        self,
        parametrization: Literal["Bishop","Paninski"] = "Bishop",
        **kwargs,
    ):
        super().__init__(**kwargs)
        super().__call__() # clean data, fit pca, generate pca_object...

        self.parametrization = parametrization
        # derived quantities needed for both optional parametrizations
        self.evals: TensorType["n_components_kept"]= torch.tensor(self.pca_object.explained_variance_, dtype=self.parameters["mean"].dtype)
        # sigma_2: mean of the discarded eigenvalues
        self.sigma_2: TensorType[(), float] =  torch.mean(self.evals[self._n_components_kept:])
        # D: M \times M diagonal matrix with top M e-vals as entries
        self.D: TensorType["n_components_kept", "n_components_kept"] = torch.diag(self.evals[:self._n_components_kept])

        self.evecs: TensorType["n_components_kept", "observation_dim"] = self.parameters["kept_eigenvectors"]
    
    @property
    def observation_mean(self) -> TensorType["observation_dim", 1]:
        """ mean of the data as computed by the PCA class """
        return self.parameters["mean"]
    
    @property
    def prior_mean(self) -> TensorType["n_components_kept"]:
        """ X \sim N(0, D) for Paninski, X \sim N(0, I) for Bishop """
        return torch.zeros(self._n_components_kept, 1)
    
    @property
    def prior_precision(self) -> TensorType["n_components_kept", "n_components_kept"]:
        if self.parametrization == "Bishop":
            return torch.eye(self._n_components_kept)
        elif self.parametrization == "Paninski":
            return torch.linalg.inv(self.D)
    
    @property
    def observation_precision(self) -> TensorType["observation_dim", "observation_dim"]:
        """ precision of the data """
        if self.parametrization == "Bishop":
            # identity matrix of size obs_dim X obs_dim 
            I = torch.eye(self.parameters["mean"].shape[0])
            return torch.linalg.inv(I * self.sigma_2)
        elif self.parametrization == "Paninski":
            # P: (M \times 2K), our A matrix
            PDP_T = self.evecs.T @ self.D @ self.evecs
            cov = torch.tensor(self.pca_object.get_covariance(), device=self.evecs.device)
            R = cov - PDP_T # R is the covariance of the residuals, low-rank by construction
            R += torch.eye(R.shape[0]) * 1e-5 # add a small diagonal jitter to avoid singularity
            return torch.linalg.inv(R)
    
    @property
    def observation_projection(self) -> TensorType["observation_dim", "n_components_kept"]:
        """ projection matrix from latent space to data space """
        if self.parametrization == "Bishop":
            # Eq. 7 in Tipping & Bishop, assuming R=I and the sqrt of a diagonal matrix is the sqrt of the diag entries
            # W_{ML} = U_q (\Lambda_q - \sigma^2 I)^{1/2}R
            return self.evecs.T @ torch.sqrt(self.D - self.sigma_2*torch.eye(self.D.shape[0]))
        elif self.parametrization == "Paninski":
            return self.evecs.T

@typechecked
def tile_inds(inds: Union[List[int], TensorType["num_valid_inds", int]]) -> Tuple[TensorType["num_valid_inds_squared", int], TensorType["num_valid_inds_squared", int]]:
    # if list, make it into a tensor
    if isinstance(inds, list):
        inds = torch.tensor(inds)
    row_inds = inds.repeat_interleave(len(inds), dim=0)
    col_inds = inds.tile(len(inds))
    return (row_inds, col_inds)

@typechecked
def extract_blocks_from_inds(valid_inds: Union[List[int], TensorType["num_valid_inds", int]], cov_mat: TensorType["num_data_points", "num_data_points"]) -> TensorType[Any, Any]:
    '''recieves a covariance matrix and extracts the relevant blocks'''
    if isinstance(valid_inds, list):
        valid_inds = torch.tensor(valid_inds)
    assert (np.diff(valid_inds)>=0).all() # ensure valid_inds are sorted
    inds_tuple = tile_inds(valid_inds)
    return cov_mat[inds_tuple].reshape(len(valid_inds), len(valid_inds))


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
                    "components_to_keep was set to {}, exceeding the maximum value of {} observation dims".format(
                        self.components_to_keep, self.fitted_pca_object.n_components_
                    )
                )
        # if float, ensure a proportion between 0.0-1.0
        elif type(self.components_to_keep) is float:
            if self.components_to_keep < 0.0 or self.components_to_keep > 1.0:
                raise ValueError(
                    "components_to_keep was set to {} while it has to be between 0.0 and 1.0".format(
                        self.components_to_keep
                    )
                )

    def _find_first_threshold_cross(self) -> int:
        # find the index of the first element above a min_variance_explained threshold
        assert type(
            self.components_to_keep is float
        )  # i.e., threshold crossing doesn't make sense with integer components_to_keep
        components_to_keep = int(
            np.where(self.cumsum_explained_variance >= self.components_to_keep)[0][0]
        )
        # cumsum is a d - 1 dimensional vector where the 0th element is the sum of the
        # 0th and 1st element of the d dimensional vector it is summing over
        return components_to_keep + 1

    def __call__(self) -> int:
        if type(self.components_to_keep) is int:
            return self.components_to_keep  # return integer as is
        elif type(self.components_to_keep) is float:
            return (
                self._find_first_threshold_cross()
            )  # find that integer that crosses the minimum explained variance


# TODO: the function below was in usage until Mar 21, 2022. Integrating it into the pca class
@typechecked
def compute_pca_reprojection_error(
    clean_pca_arr: TensorType["samples", "observation_dim"],
    kept_eigenvectors: TensorType["latent_dim", "observation_dim"],
    mean: TensorType["observation_dim"],
) -> TensorType["samples", "observation_dim_div_by_two"]:

    # first verify that the pca array has observations divisible by 2
    # (corresponding to (x,y) coords)
    assert clean_pca_arr.shape[1] % 2 == 0

    # mean-center
    clean_pca_arr = clean_pca_arr - mean.unsqueeze(0)

    # project down into low-d space, then back up to observation space
    reprojection_arr = (
        clean_pca_arr @ kept_eigenvectors.T @ kept_eigenvectors
    )  # e.g., (214, 4) X (4, 3) X (3, 4) = (214, 4) as we started

    # compute difference between original array and its reprojection
    # multiview:
    # - num_samples: num_samples X num_bodyparts
    # - observation_dim: 2 x num_views
    # singleview:
    # - num_samples: num_samples
    # - observation_dim: 2 X num_bodyparts
    diff = clean_pca_arr - reprojection_arr

    # reshape to get a per sample/keypoint values:
    diff_arr_per_keypoint = diff.reshape(diff.shape[0], diff.shape[1] // 2, 2)
    reprojection_loss = torch.linalg.norm(diff_arr_per_keypoint, dim=2)

    # print("reprojection_loss.shape: {}".format(reprojection_loss.shape))
    # print("diff: {}".format(diff[:5, :]))
    # print("diff_arr_per_keypoint: {}".format(diff_arr_per_keypoint[:5, :, :]))

    return reprojection_loss


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


@typechecked
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
