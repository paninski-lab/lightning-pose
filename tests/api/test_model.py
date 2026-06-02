import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from lightning_pose.api import Model
from lightning_pose.api.model import get_model_class
from tests.fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(tmp_path, request, multiview=False) -> Model:
    # get the trained model for testing
    dataset_name = (
        "test_model_mirror_mouse"
        if not multiview
        else "test_model_mirror_mouse_multiview"
    )
    fetch_test_data_if_needed(request.path.parent, dataset_name)
    # copy to tmpdir because prediction will create output artifacts in model_dir
    tmp_model_dir = tmp_path / dataset_name
    shutil.copytree(request.path.parent / dataset_name, tmp_model_dir)

    model = Model.from_dir(tmp_model_dir)

    assert model.model_dir == tmp_model_dir
    assert model.image_preds_dir() == tmp_model_dir / "image_preds"
    assert model.video_preds_dir() == tmp_model_dir / "video_preds"
    assert (
        model.labeled_videos_dir() == tmp_model_dir / "video_preds" / "labeled_videos"
    )

    # confirm predictions don't exist yet; if they do, tests pass even if prediction did nothing
    assert not model.image_preds_dir().exists()
    assert not model.video_preds_dir().exists()
    assert not model.labeled_videos_dir().exists()

    return model


class TestPredictOnLabelCsv:
    """Test the predict_on_label_csv method."""

    def test_predict_on_label_csv_singleview(self, tmp_path, request, toy_data_dir):
        """Singleview model writes predictions and per-metric error CSVs."""
        model = _setup_test_model(tmp_path, request)

        model.predict_on_label_csv(Path(toy_data_dir) / "CollectedData.csv")

        assert (model.image_preds_dir() / "CollectedData.csv" / "predictions.csv").is_file()
        assert (
            model.image_preds_dir() / "CollectedData.csv" / "predictions_pixel_error.csv"
        ).is_file()
        assert (
            model.image_preds_dir()
            / "CollectedData.csv"
            / "predictions_pca_singleview_error.csv"
        ).is_file()

    def test_predict_on_label_csv_with_multiview_model(self, tmp_path, request, toy_mdata_dir):
        """Multiview model can predict on a single-view CSV."""
        model = _setup_test_model(tmp_path, request, multiview=True)

        model.predict_on_label_csv(Path(toy_mdata_dir) / "top.csv")

        assert (model.image_preds_dir() / "top.csv" / "predictions.csv").is_file()
        assert (model.image_preds_dir() / "top.csv" / "predictions_pixel_error.csv").is_file()

    def test_predict_on_label_csv_multiview(self, tmp_path, request, toy_mdata_dir):
        """predict_on_label_csv_multiview writes predictions for all views."""
        model = _setup_test_model(tmp_path, request, multiview=True)

        model.predict_on_label_csv_multiview(
            [
                Path(toy_mdata_dir) / "top.csv",
                Path(toy_mdata_dir) / "bot.csv",
            ]
        )

        assert (model.image_preds_dir() / "top.csv" / "predictions.csv").is_file()
        assert (model.image_preds_dir() / "top.csv" / "predictions_pixel_error.csv").is_file()
        assert (model.image_preds_dir() / "bot.csv" / "predictions.csv").is_file()
        assert (model.image_preds_dir() / "bot.csv" / "predictions_pixel_error.csv").is_file()


class TestPredictOnVideoFile:
    """Test the predict_on_video_file method."""

    def test_predict_on_video_file_singleview(self, tmp_path, request, toy_data_dir):
        """Singleview model writes prediction CSVs and optionally a labeled video."""
        model = _setup_test_model(tmp_path, request)

        model.predict_on_video_file(Path(toy_data_dir) / "videos" / "test_vid.mp4")

        assert (model.video_preds_dir() / "test_vid.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_temporal_norm.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_pca_singleview_error.csv").is_file()
        assert not model.labeled_videos_dir().exists()

        model.predict_on_video_file(
            Path(toy_data_dir) / "videos" / "test_vid.mp4",
            generate_labeled_video=True,
        )
        assert (model.labeled_videos_dir() / "test_vid_labeled.mp4").is_file()

    def test_predict_on_video_file_with_multiview_model(self, tmp_path, request, toy_mdata_dir):
        """Multiview model can predict on a single video file."""
        model = _setup_test_model(tmp_path, request, multiview=True)

        model.predict_on_video_file(
            Path(toy_mdata_dir) / "videos" / "test_vid_top.mp4",
            generate_labeled_video=True,
        )

        assert (model.video_preds_dir() / "test_vid_top.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_top_temporal_norm.csv").is_file()
        assert (model.labeled_videos_dir() / "test_vid_top_labeled.mp4").is_file()

    def test_predict_on_video_file_multiview(self, tmp_path, request, toy_mdata_dir):
        """predict_on_video_file_multiview writes predictions and labeled videos for all views."""
        model = _setup_test_model(tmp_path, request, multiview=True)

        model.predict_on_video_file_multiview(
            [
                Path(toy_mdata_dir) / "videos" / "test_vid_top.mp4",
                Path(toy_mdata_dir) / "videos" / "test_vid_bot.mp4",
            ],
            generate_labeled_video=True,
        )

        assert (model.video_preds_dir() / "test_vid_top.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_top.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_top_temporal_norm.csv").is_file()
        assert (model.video_preds_dir() / "test_vid_bot_temporal_norm.csv").is_file()
        assert (model.labeled_videos_dir() / "test_vid_top_labeled.mp4").is_file()
        assert (model.labeled_videos_dir() / "test_vid_bot_labeled.mp4").is_file()


class TestPredictFrame:
    """Test the predict_frame method."""

    def test_predict_frame_basic(self, tmp_path, request):
        """predict_frame returns keypoints and confidences for a synthetic RGB frame."""
        model = _setup_test_model(tmp_path, request)

        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = model.predict_frame(frame)

        assert "keypoints" in result
        assert "confidence" in result

        kp = result["keypoints"]
        conf = result["confidence"]

        assert kp.dtype == np.float32
        assert conf.dtype == np.float32
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        assert conf.shape[0] == kp.shape[0]
        assert kp.shape[0] > 0  # at least one keypoint
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)
        # tolerance for subpixel overshoot at frame boundary
        assert np.all(kp[:, 0] <= 256 + 1)
        assert np.all(kp[:, 1] <= 256 + 1)

    def test_predict_frame_with_bbox(self, tmp_path, request):
        """predict_frame with bbox remaps keypoints to original frame coordinates."""
        model = _setup_test_model(tmp_path, request)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 50, 200, 150)  # (x, y, w, h)
        result = model.predict_frame(frame, bbox=bbox)

        kp = result["keypoints"]
        conf = result["confidence"]

        assert kp.dtype == np.float32
        assert conf.dtype == np.float32
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        assert conf.shape[0] == kp.shape[0]
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)
        assert np.all(kp[:, 0] >= 0)
        assert np.all(kp[:, 1] >= 0)
        assert np.all(kp[:, 0] <= 640 + 1)
        assert np.all(kp[:, 1] <= 480 + 1)

    def test_predict_frame_bbox_clipping(self, tmp_path, request):
        """Bbox extending past the frame edge is clipped silently; keypoints stay valid."""
        model = _setup_test_model(tmp_path, request)

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # extends 60px past right edge: requested width 200, actual crop width 140
        bbox = (500, 100, 200, 150)
        result = model.predict_frame(frame, bbox=bbox)

        kp = result["keypoints"]
        conf = result["confidence"]

        assert kp.dtype == np.float32
        assert conf.dtype == np.float32
        assert kp.ndim == 2
        assert kp.shape[1] == 2
        assert conf.shape[0] == kp.shape[0]
        assert np.all(conf >= 0)
        assert np.all(conf <= 1)
        assert np.all(kp[:, 0] >= 0)
        assert np.all(kp[:, 1] >= 0)
        assert np.all(kp[:, 0] <= 640 + 1)
        assert np.all(kp[:, 1] <= 480 + 1)


class TestModelErrors:
    """Test that Model public methods raise informative errors on bad inputs."""

    @pytest.fixture()
    def singleview_model(self, tmp_path, request):
        """Singleview model, not yet loaded."""
        return _setup_test_model(tmp_path, request, multiview=False)

    @pytest.fixture()
    def multiview_model(self, tmp_path, request):
        """Multiview model, not yet loaded."""
        return _setup_test_model(tmp_path, request, multiview=True)

    def test_predict_frame_errors(self, singleview_model):
        """predict_frame raises on bad inputs and when the model failed to load."""
        model = singleview_model

        # RuntimeError when _load() is a no-op and model.model stays None
        with patch.object(model, '_load'):
            with pytest.raises(RuntimeError, match='model failed to load'):
                model.predict_frame(np.zeros((256, 256, 3), dtype=np.uint8))

        # Wrong dtype (float32 instead of uint8)
        float_frame = np.random.rand(256, 256, 3).astype(np.float32)
        with pytest.raises(ValueError, match='must be uint8'):
            model.predict_frame(float_frame)

        # Wrong shape (grayscale -- missing channel dim)
        gray_frame = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        with pytest.raises(ValueError, match=r'must be \(H, W, 3\)'):
            model.predict_frame(gray_frame)

        # Wrong shape (RGBA -- 4 channels)
        rgba_frame = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match=r'must be \(H, W, 3\)'):
            model.predict_frame(rgba_frame)

        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Negative bbox origin
        with pytest.raises(ValueError, match='non-negative'):
            model.predict_frame(frame, bbox=(-10, 0, 50, 50))

        # Zero-width bbox
        with pytest.raises(ValueError, match='must be positive'):
            model.predict_frame(frame, bbox=(10, 10, 0, 50))

        # Bbox completely off-frame (empty crop)
        with pytest.raises(ValueError, match='empty crop'):
            model.predict_frame(frame, bbox=(1000, 1000, 50, 50))

    def test_predict_on_label_csv_multiview_requires_multiview_model(self, singleview_model):
        """Raises ValueError when called on a single-view model."""
        with pytest.raises(ValueError, match='requires a multi-view model'):
            singleview_model.predict_on_label_csv_multiview(['a.csv', 'b.csv'])

    def test_predict_on_label_csv_multiview_wrong_csv_count(self, multiview_model):
        """Raises ValueError when the number of csv files doesn't match the view count."""
        with patch.object(multiview_model, '_load'):
            with pytest.raises(ValueError, match='expected.*csv files'):
                multiview_model.predict_on_label_csv_multiview(['only_one.csv'])

    def test_predict_on_video_file_multiview_requires_multiview_model(self, singleview_model):
        """Raises ValueError when called on a single-view model."""
        with pytest.raises(ValueError, match='requires a multi-view model'):
            singleview_model.predict_on_video_file_multiview(['a.mp4', 'b.mp4'])

    def test_predict_on_video_file_multiview_wrong_video_count(self, multiview_model):
        """Raises ValueError when the number of video files doesn't match the view count."""
        with patch.object(multiview_model, '_load'):
            with pytest.raises(ValueError, match='expected.*video files'):
                multiview_model.predict_on_video_file_multiview(['only_one.mp4'])


class TestGetModelClass:
    """Test the get_model_class function."""

    def test_get_model_class_supervised_regression(self):
        """Returns RegressionTracker for supervised regression."""
        from lightning_pose.models import RegressionTracker
        assert get_model_class('regression', semi_supervised=False) is RegressionTracker

    def test_get_model_class_supervised_heatmap(self):
        """Returns HeatmapTracker for supervised heatmap."""
        from lightning_pose.models import HeatmapTracker
        assert get_model_class('heatmap', semi_supervised=False) is HeatmapTracker

    def test_get_model_class_supervised_heatmap_mhcrnn(self):
        """Returns HeatmapTrackerMHCRNN for supervised heatmap_mhcrnn."""
        from lightning_pose.models import HeatmapTrackerMHCRNN
        assert get_model_class('heatmap_mhcrnn', semi_supervised=False) is HeatmapTrackerMHCRNN

    def test_get_model_class_supervised_heatmap_multiview_transformer(self):
        """Returns HeatmapTrackerMultiviewTransformer for supervised multiview transformer."""
        from lightning_pose.models import HeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=False)
            is HeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised supervised model_type."""
        with pytest.raises(NotImplementedError, match='invalid model_type for a fully supervised'):
            get_model_class('unknown_type', semi_supervised=False)  # type: ignore[arg-type]

    def test_get_model_class_semi_supervised_regression(self):
        """Returns SemiSupervisedRegressionTracker for semi-supervised regression."""
        from lightning_pose.models import SemiSupervisedRegressionTracker
        assert (
            get_model_class('regression', semi_supervised=True) is SemiSupervisedRegressionTracker
        )

    def test_get_model_class_semi_supervised_heatmap(self):
        """Returns SemiSupervisedHeatmapTracker for semi-supervised heatmap."""
        from lightning_pose.models import SemiSupervisedHeatmapTracker
        assert get_model_class('heatmap', semi_supervised=True) is SemiSupervisedHeatmapTracker

    def test_get_model_class_semi_supervised_heatmap_mhcrnn(self):
        """Returns SemiSupervisedHeatmapTrackerMHCRNN for semi-supervised heatmap_mhcrnn."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMHCRNN
        assert (
            get_model_class('heatmap_mhcrnn', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMHCRNN
        )

    def test_get_model_class_semi_supervised_heatmap_multiview_transformer(self):
        """Returns SemiSupervisedHeatmapTrackerMultiviewTransformer for semi-supervised variant."""
        from lightning_pose.models import SemiSupervisedHeatmapTrackerMultiviewTransformer
        assert (
            get_model_class('heatmap_multiview_transformer', semi_supervised=True)
            is SemiSupervisedHeatmapTrackerMultiviewTransformer
        )

    def test_get_model_class_semi_supervised_raises_for_unknown(self):
        """Raises NotImplementedError for an unrecognised semi-supervised model_type."""
        with pytest.raises(
            NotImplementedError, match='invalid model_type for a semi-supervised',
        ):
            get_model_class('unknown_type', semi_supervised=True)  # type: ignore[arg-type]
