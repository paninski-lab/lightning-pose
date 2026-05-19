"""Tests for PredictionResult.to_dict() and MultiviewPredictionResult.to_dict()."""

import numpy as np
import pandas as pd

from lightning_pose.data.datatypes import (
    ComputeMetricsSingleResult,
    MultiviewPredictionResult,
    PredictionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KP_NAMES = ['nose', 'left_ear', 'right_ear']
_N_FRAMES = 4
_SCORER = 'heatmap_tracker'


def _make_predictions_df(
    keypoint_names: list[str] = _KP_NAMES,
    n_frames: int = _N_FRAMES,
    scorer: str = _SCORER,
) -> pd.DataFrame:
    """Build a minimal DLC-format predictions DataFrame."""
    index = pd.MultiIndex.from_product(
        [[scorer], keypoint_names, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'],
    )
    rng = np.random.default_rng(0)
    data = rng.random((n_frames, len(keypoint_names) * 3))
    return pd.DataFrame(
        data,
        columns=index,
        index=pd.Index([f'frame_{i}' for i in range(n_frames)]),
    )


def _make_metric_df(
    keypoint_names: list[str] = _KP_NAMES,
    n_frames: int = _N_FRAMES,
    include_set: bool = False,
    seed: int = 1,
) -> pd.DataFrame:
    """Build a minimal per-frame per-keypoint metric DataFrame."""
    rng = np.random.default_rng(seed)
    data = rng.random((n_frames, len(keypoint_names)))
    df = pd.DataFrame(
        data,
        columns=pd.Index(keypoint_names),
        index=pd.Index([f'frame_{i}' for i in range(n_frames)]),
    )
    if include_set:
        df['set'] = 'train'
    return df


# ---------------------------------------------------------------------------
# PredictionResult.to_dict
# ---------------------------------------------------------------------------

class TestPredictionResultToDict:
    """Test PredictionResult.to_dict."""

    def test_to_dict_contains_expected_keys(self):
        result = PredictionResult(predictions=_make_predictions_df()).to_dict()
        expected = {
            'keypoint_names', 'index', 'x', 'y', 'confidence',
            'pixel_error', 'temporal_norm', 'pca_singleview_error', 'pca_multiview_error',
        }
        assert set(result.keys()) == expected

    def test_to_dict_keypoint_names(self):
        result = PredictionResult(predictions=_make_predictions_df()).to_dict()
        assert result['keypoint_names'] == _KP_NAMES

    def test_to_dict_index(self):
        result = PredictionResult(predictions=_make_predictions_df()).to_dict()
        assert result['index'] == [f'frame_{i}' for i in range(_N_FRAMES)]

    def test_to_dict_array_shapes(self):
        result = PredictionResult(predictions=_make_predictions_df()).to_dict()
        expected_shape = (_N_FRAMES, len(_KP_NAMES))
        assert result['x'].shape == expected_shape
        assert result['y'].shape == expected_shape
        assert result['confidence'].shape == expected_shape

    def test_to_dict_xy_values_match_dataframe(self):
        preds = _make_predictions_df()
        result = PredictionResult(predictions=preds).to_dict()
        expected_x = preds.xs('x', level=2, axis=1).to_numpy()
        expected_y = preds.xs('y', level=2, axis=1).to_numpy()
        np.testing.assert_array_equal(result['x'], expected_x)
        np.testing.assert_array_equal(result['y'], expected_y)

    def test_to_dict_confidence_values_match_dataframe(self):
        preds = _make_predictions_df()
        result = PredictionResult(predictions=preds).to_dict()
        expected = preds.xs('likelihood', level=2, axis=1).to_numpy()
        np.testing.assert_array_equal(result['confidence'], expected)

    def test_to_dict_all_metrics_none_when_not_provided(self):
        result = PredictionResult(predictions=_make_predictions_df()).to_dict()
        assert result['pixel_error'] is None
        assert result['temporal_norm'] is None
        assert result['pca_singleview_error'] is None
        assert result['pca_multiview_error'] is None

    def test_to_dict_pixel_error_shape_and_values(self):
        pixel_error_df = _make_metric_df()
        metrics = ComputeMetricsSingleResult(pixel_error_df=pixel_error_df)
        result = PredictionResult(
            predictions=_make_predictions_df(), metrics=metrics,
        ).to_dict()
        assert result['pixel_error'].shape == (_N_FRAMES, len(_KP_NAMES))
        np.testing.assert_array_equal(result['pixel_error'], pixel_error_df.to_numpy())

    def test_to_dict_temporal_norm_shape_and_values(self):
        temporal_norm_df = _make_metric_df(seed=2)
        metrics = ComputeMetricsSingleResult(temporal_norm_df=temporal_norm_df)
        result = PredictionResult(
            predictions=_make_predictions_df(), metrics=metrics,
        ).to_dict()
        assert result['temporal_norm'].shape == (_N_FRAMES, len(_KP_NAMES))
        np.testing.assert_array_equal(result['temporal_norm'], temporal_norm_df.to_numpy())

    def test_to_dict_set_column_excluded_from_metrics(self):
        # "set" column is added when predicting on training data and must not appear
        # in the output array — it would break the (n_frames, n_keypoints) shape.
        pixel_error_df = _make_metric_df(include_set=True)
        metrics = ComputeMetricsSingleResult(pixel_error_df=pixel_error_df)
        result = PredictionResult(
            predictions=_make_predictions_df(), metrics=metrics,
        ).to_dict()
        assert result['pixel_error'].shape == (_N_FRAMES, len(_KP_NAMES))

    def test_to_dict_partial_metrics_none(self):
        # only pixel_error provided; other metrics should be None
        metrics = ComputeMetricsSingleResult(pixel_error_df=_make_metric_df())
        result = PredictionResult(
            predictions=_make_predictions_df(), metrics=metrics,
        ).to_dict()
        assert result['pixel_error'] is not None
        assert result['temporal_norm'] is None
        assert result['pca_singleview_error'] is None
        assert result['pca_multiview_error'] is None

    def test_to_dict_all_metric_arrays_populated(self):
        metrics = ComputeMetricsSingleResult(
            pixel_error_df=_make_metric_df(seed=1),
            temporal_norm_df=_make_metric_df(seed=2),
            pca_sv_df=_make_metric_df(seed=3),
            pca_mv_df=_make_metric_df(seed=4),
        )
        result = PredictionResult(
            predictions=_make_predictions_df(), metrics=metrics,
        ).to_dict()
        for key in ('pixel_error', 'temporal_norm', 'pca_singleview_error', 'pca_multiview_error'):
            assert result[key] is not None
            assert result[key].shape == (_N_FRAMES, len(_KP_NAMES))


# ---------------------------------------------------------------------------
# MultiviewPredictionResult.to_dict
# ---------------------------------------------------------------------------

class TestMultiviewPredictionResultToDict:
    """Test MultiviewPredictionResult.to_dict."""

    def test_to_dict_keys_are_view_names(self):
        view_names = ['cam0', 'cam1']
        preds = {v: _make_predictions_df() for v in view_names}
        result = MultiviewPredictionResult(predictions=preds).to_dict()
        assert set(result.keys()) == set(view_names)

    def test_to_dict_each_view_has_expected_structure(self):
        view_names = ['cam0', 'cam1']
        preds = {v: _make_predictions_df() for v in view_names}
        result = MultiviewPredictionResult(predictions=preds).to_dict()
        expected_keys = {
            'keypoint_names', 'index', 'x', 'y', 'confidence',
            'pixel_error', 'temporal_norm', 'pca_singleview_error', 'pca_multiview_error',
        }
        for view in view_names:
            assert set(result[view].keys()) == expected_keys

    def test_to_dict_no_metrics(self):
        view_names = ['cam0', 'cam1']
        preds = {v: _make_predictions_df() for v in view_names}
        result = MultiviewPredictionResult(predictions=preds).to_dict()
        for view in view_names:
            assert result[view]['pixel_error'] is None
            assert result[view]['temporal_norm'] is None

    def test_to_dict_metrics_routed_per_view(self):
        # each view should receive its own metrics, not the other view's
        view_names = ['cam0', 'cam1']
        preds = {v: _make_predictions_df() for v in view_names}
        cam0_df = _make_metric_df(seed=10)
        cam1_df = _make_metric_df(seed=20)
        metrics = {
            'cam0': ComputeMetricsSingleResult(pixel_error_df=cam0_df),
            'cam1': ComputeMetricsSingleResult(pixel_error_df=cam1_df),
        }
        result = MultiviewPredictionResult(predictions=preds, metrics=metrics).to_dict()
        np.testing.assert_array_equal(result['cam0']['pixel_error'], cam0_df.to_numpy())
        np.testing.assert_array_equal(result['cam1']['pixel_error'], cam1_df.to_numpy())
