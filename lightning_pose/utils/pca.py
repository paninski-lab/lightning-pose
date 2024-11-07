"""PCA class to assist with computing PCA losses."""

import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from omegaconf import ListConfig
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _infer_dimension
from sklearn.utils._array_api import _convert_to_numpy, get_namespace
from sklearn.utils.extmath import stable_cumsum, svd_flip
from torchtyping import TensorType
from typeguard import typechecked

from lightning_pose.data.datamodules import BaseDataModule, UnlabeledDataModule
from lightning_pose.data.datasets import MultiviewHeatmapDataset
from lightning_pose.data.utils import DataExtractor
from lightning_pose.losses.helpers import EmpiricalEpsilon, convert_dict_values_to_tensors

# to ignore imports for sphix-autoapidoc
__all__ = [
    "KeypointPCA",
    "ComponentChooser",
    "format_multiview_data_for_pca",
]


class KeypointPCA(object):
    """Class to collect data from a dataloader and compute PCA params."""

    def __init__(
        self,
        loss_type: Literal["pca_singleview", "pca_multiview"],
        data_module: Union[UnlabeledDataModule, BaseDataModule],
        components_to_keep: Optional[Union[int, float]] = 0.99,
        empirical_epsilon_percentile: float = 90.0,
        mirrored_column_matches: Optional[Union[ListConfig, List]] = None,
        columns_for_singleview_pca: Optional[Union[ListConfig, List]] = None,
        device: Union[Literal["cuda", "cpu"], torch.device] = "cpu",
        centering_method: Optional[Literal["mean", "median"]] = None,
    ) -> None:
        self.loss_type = loss_type
        self.data_module = data_module
        self.components_to_keep = components_to_keep
        self.empirical_epsilon_percentile = empirical_epsilon_percentile
        # check if this is a mirrored or true multiview dataset
        # the former will be a list of lists, the latter a list of ints
        if mirrored_column_matches is not None and isinstance(mirrored_column_matches[0], int):
            if not isinstance(data_module.dataset, MultiviewHeatmapDataset):
                raise ValueError(
                    "cfg.data.mirrored_column_matches must contain a list of indices for each "
                    "mirrored view"
                )
            num_views = len(data_module.dataset.view_names)
            num_keypoints = data_module.dataset.num_keypoints / num_views  # keypoints per view
            mirrored_column_matches = [
                (v * num_keypoints + np.array(mirrored_column_matches, dtype=int)).tolist()
                for v in range(num_views)
            ]
        self.mirrored_column_matches = mirrored_column_matches
        self.columns_for_singleview_pca = columns_for_singleview_pca
        self.pca_object = None
        self.device = device
        self.centering_method = centering_method

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
        # reshape to (batch, num_keypoints, 2) to easily select columns
        ## [1,2,3,4]
        ## [[1,2]],[3,4]]
        data_arr = data_arr.reshape(data_arr.shape[0], data_arr.shape[1] // 2, 2)
        # optionally choose a subset of the keypoints for the singleview pca
        if self.columns_for_singleview_pca is not None:
            data_arr = data_arr[:, np.array(self.columns_for_singleview_pca), :]

        if self.centering_method is not None:
            if self.centering_method == "mean":
                center = data_arr.mean(dim=1, keepdim=True)
            elif self.centering_method == "median":
                # When there are an even # of keypoints, Tensor.quantile averages
                # while Tensor.median is non-deterministic.
                center = data_arr.quantile(dim=1, q=0.5, keepdim=True)
            else:
                raise NotImplementedError(f"centering_method: {self.centering_method}")
            data_arr = data_arr - center

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
        if self.loss_type == "pca_multiview":
            return self._multiview_format(data_arr=data_arr)
        elif self.loss_type == "pca_singleview":
            return self._singleview_format(data_arr=data_arr)
        else:
            raise NotImplementedError

    def _check_data(self) -> None:

        # ensure we have more rows than columns after doing nan filtering
        if self.data_arr.shape[0] < self.data_arr.shape[1]:
            raise ValueError(
                f"cannot fit PCA with {self.data_arr.shape[0]} samples < {self.data_arr.shape[1]} "
                "observation dimensions"
            )

    def _fit_pca(self) -> None:
        # fit PCA with the full number of comps on the cleaned-up data array.
        # note self.data_arr is a tensor but sklearn's PCA function is fine with it
        self.pca_object = NaNPCA(svd_solver="covariance_eigh")
        self.pca_object.fit(X=self.data_arr)

    def _choose_n_components(self) -> None:
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

        self.parameters = {  # dict with same keys as loss_param_dict
            "mean": self.pca_object.mean_,
            "kept_eigenvectors": self.pca_object.components_[:self._n_components_kept],
            "discarded_eigenvectors": self.pca_object.components_[self._n_components_kept:],
        }

        self.parameters = convert_dict_values_to_tensors(self.parameters, self.device)

        self.parameters["epsilon"] = EmpiricalEpsilon(
            percentile=self.empirical_epsilon_percentile
        )(loss=self.compute_reprojection_error())

    def reproject(
        self, data_arr: Optional[TensorType["num_samples", "sample_dim"]] = None
    ) -> TensorType["num_samples", "sample_dim"]:
        """Reproject a data array using the fixed pca parameters.

        This transformation is implemented as in scikit-learn
        https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/decomposition/_base.py#L125

        """
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

        # save training data in self.data_arr
        self._get_data()

        # modify self.data_arr in the case of multiview pca, else keep the same
        self.data_arr = self._format_data(data_arr=self.data_arr)

        # check that we have more observations than observation-dimensions
        self._check_data()

        self._fit_pca()
        self._choose_n_components()
        self.pca_prints()

        # save all the meaningful quantities
        self._set_parameter_dict()


class NaNPCA(PCA):

    def __init__(
        self,
        n_components: Optional[int] = None,
        *,
        copy: bool = True,
        whiten: bool = False,
        svd_solver: str = "covariance_eigh",
        tol: float = 0.0,
        iterated_power: str = "auto",
        n_oversamples: int = 10,
        power_iteration_normalizer: str = "auto",
        random_state: Optional[int] = None,
    ) -> None:

        # force solver to be "covariance_eigh"
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver="covariance_eigh",
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )

    def _fit(self, X: np.ndarray) -> tuple:
        """Dispatch to the right submethod depending on the chosen solver.

        This is a modification of the sklearn fit function, with data validation
        removed since we want to include NaNs.

        """
        xp, is_array_api_compliant = get_namespace(X)

        # Validate the data, without ever forcing a copy as any solver that
        # supports sparse input data and the `covariance_eigh` solver are
        # written in a way to avoid the need for any inplace modification of
        # the input data contrary to the other solvers.
        # The copy will happen
        # later, only if needed, once the solver negotiation below is done.
        # X = self._validate_data(
        #     X,
        #     dtype=[xp.float64, xp.float32],
        #     force_writeable=True,
        #     accept_sparse=("csr", "csc"),
        #     ensure_2d=True,
        #     copy=False,
        # )

        self._fit_svd_solver = self.svd_solver
        assert self._fit_svd_solver == "covariance_eigh"

        if self.n_components is None:
            if self._fit_svd_solver != "arpack":
                n_components = min(X.shape)
            else:
                n_components = min(X.shape) - 1
        else:
            n_components = self.n_components

        # Call fit for full SVD
        return self._fit_full(X, n_components, xp, is_array_api_compliant)

    def _fit_full(self, X, n_components, xp, is_array_api_compliant):
        """Fit the model by computing full SVD on X.

        This is a modification of the sklearn function that now allows for NaNs in the inputs.

        """

        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                f"n_components={n_components} must be between 0 and "
                f"min(n_samples, n_features)={min(n_samples, n_features)} with "
                f"svd_solver={self._fit_svd_solver!r}"
            )

        self.mean_ = np.nanmean(X, axis=0)
        # When X is a scipy sparse matrix, self.mean_ is a numpy matrix, so we need
        # to transform it to a 1D array. Note that this is not the case when X
        # is a scipy sparse array.
        self.mean_ = xp.reshape(xp.asarray(self.mean_), (-1,))

        # BEGIN ORIGINAL ELSE BLOCK

        assert self._fit_svd_solver == "covariance_eigh"
        x_is_centered = False

        # -- original computation of covariance matrix
        # C = X.T @ X
        # C -= (
        #     n_samples
        #     * xp.reshape(self.mean_, (-1, 1))
        #     * xp.reshape(self.mean_, (1, -1))
        # )
        # C /= n_samples - 1

        # -- masked computation of covariance matrix
        C = np.ma.cov(np.ma.masked_invalid(X), rowvar=False).data

        # -- picking up original sklearn pca code here
        eigenvals, eigenvecs = xp.linalg.eigh(C)

        # When X is a scipy sparse matrix, the following two datastructures
        # are returned as instances of the soft-deprecated numpy.matrix
        # class. Note that this problem does not occur when X is a scipy
        # sparse array (or another other kind of supported array).
        # TODO: remove the following two lines when scikit-learn only
        # depends on scipy versions that no longer support scipy.sparse
        # matrices.
        eigenvals = xp.reshape(xp.asarray(eigenvals), (-1,))
        eigenvecs = xp.asarray(eigenvecs)

        eigenvals = xp.flip(eigenvals, axis=0)
        eigenvecs = xp.flip(eigenvecs, axis=1)

        # The covariance matrix C is positive semi-definite by
        # construction. However, the eigenvalues returned by xp.linalg.eigh
        # can be slightly negative due to numerical errors. This would be
        # an issue for the subsequent sqrt, hence the manual clipping.
        eigenvals[eigenvals < 0.0] = 0.0
        explained_variance_ = eigenvals

        # Re-construct SVD of centered X indirectly and make it consistent
        # with the other solvers.
        S = xp.sqrt(eigenvals * (n_samples - 1))
        Vt = eigenvecs.T
        U = None

        # END ORIGINAL ELSE BLOCK

        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt, u_based_decision=False)

        components_ = Vt

        # Get variance explained by singular values
        total_var = xp.sum(explained_variance_)
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = xp.asarray(S, copy=True)  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            if is_array_api_compliant:
                # Convert to numpy as xp.cumsum and xp.searchsorted are not
                # part of the Array API standard yet:
                #
                # https://github.com/data-apis/array-api/issues/597
                # https://github.com/data-apis/array-api/issues/688
                #
                # Furthermore, it's not always safe to call them for namespaces
                # that already implement them: for instance as
                # cupy.searchsorted does not accept a float as second argument.
                explained_variance_ratio_np = _convert_to_numpy(
                    explained_variance_ratio_, xp=xp
                )
            else:
                explained_variance_ratio_np = explained_variance_ratio_
            ratio_cumsum = stable_cumsum(explained_variance_ratio_np)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = xp.mean(explained_variance_[n_components:])
        else:
            self.noise_variance_ = 0.0

        self.n_samples_ = n_samples
        self.n_components_ = n_components
        # Assign a copy of the result of the truncation of the components in
        # order to:
        # - release the memory used by the discarded components,
        # - ensure that the kept components are allocated contiguously in
        #   memory to make the transform method faster by leveraging cache
        #   locality.
        self.components_ = xp.asarray(components_[:n_components, :], copy=True)

        # We do the same for the other arrays for the sake of consistency.
        self.explained_variance_ = xp.asarray(
            explained_variance_[:n_components], copy=True
        )
        self.explained_variance_ratio_ = xp.asarray(
            explained_variance_ratio_[:n_components], copy=True
        )
        self.singular_values_ = xp.asarray(singular_values_[:n_components], copy=True)

        return U, S, Vt, X, x_is_centered, xp


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
    def cumsum_explained_variance(self) -> np.ndarray:
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
            # return integer as is
            return self.components_to_keep
        elif type(self.components_to_keep) is float:
            # find that integer that crosses the minimum explained variance
            return self._find_first_threshold_cross()


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
