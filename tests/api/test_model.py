import shutil
from pathlib import Path

import numpy as np
import pytest

from lightning_pose.api.model import Model
from tests.fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(tmp_path, request, multiview=False) -> Model:
    # Get the trained model for testing.
    dataset_name = (
        "test_model_mirror_mouse"
        if not multiview
        else "test_model_mirror_mouse_multiview"
    )
    fetch_test_data_if_needed(request.path.parent, dataset_name)
    # Copy to tmpdir because prediction will create output artifacts in model_dir.
    tmp_model_dir = tmp_path / dataset_name
    shutil.copytree(request.path.parent / dataset_name, tmp_model_dir)

    # Set up a model object.
    model = Model.from_dir(tmp_model_dir)

    # We're going to use these in tests, so make sure they are correct
    # before tests just assume they work correctly.
    assert model.model_dir == tmp_model_dir
    assert model.image_preds_dir() == tmp_model_dir / "image_preds"
    assert model.video_preds_dir() == tmp_model_dir / "video_preds"
    assert (
        model.labeled_videos_dir() == tmp_model_dir / "video_preds" / "labeled_videos"
    )

    # Confirm predictions don't exist yet. If they do, our tests will pass even
    # if model prediction did nothing.
    assert not model.image_preds_dir().exists()
    assert not model.video_preds_dir().exists()
    assert not model.labeled_videos_dir().exists()

    return model


def test_predict_on_label_csv_singleview(tmp_path, request, toy_data_dir):
    model = _setup_test_model(tmp_path, request)

    # Test prediction on a CSV file.
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


def test_predict_on_label_csv_method_multiview_model(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on a CSV file.
    model.predict_on_label_csv(Path(toy_mdata_dir) / "top.csv")

    assert (model.image_preds_dir() / "top.csv" / "predictions.csv").is_file()
    assert (
        model.image_preds_dir() / "top.csv" / "predictions_pixel_error.csv"
    ).is_file()


def test_predict_on_label_csv_multiview(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on CSV files.
    model.predict_on_label_csv_multiview(
        [
            Path(toy_mdata_dir) / "top.csv",
            Path(toy_mdata_dir) / "bot.csv",
        ]
    )

    assert (model.image_preds_dir() / "top.csv" / "predictions.csv").is_file()
    assert (
        model.image_preds_dir() / "top.csv" / "predictions_pixel_error.csv"
    ).is_file()

    assert (model.image_preds_dir() / "bot.csv" / "predictions.csv").is_file()
    assert (
        model.image_preds_dir() / "bot.csv" / "predictions_pixel_error.csv"
    ).is_file()


def test_predict_on_video_file_singleview(tmp_path, request, toy_data_dir):
    model = _setup_test_model(tmp_path, request)

    # Test prediction on a test video.
    model.predict_on_video_file(Path(toy_data_dir) / "videos" / "test_vid.mp4")
    assert (model.video_preds_dir() / "test_vid.csv").is_file()
    assert (model.video_preds_dir() / "test_vid_temporal_norm.csv").is_file()
    assert (model.video_preds_dir() / "test_vid_pca_singleview_error.csv").is_file()

    # Labeled video generation should have been off by default.
    assert not model.labeled_videos_dir().exists()

    # Test labeled_video generation.
    model.predict_on_video_file(
        Path(toy_data_dir) / "videos" / "test_vid.mp4",
        generate_labeled_video=True,
    )
    assert (model.labeled_videos_dir() / "test_vid_labeled.mp4").is_file()


def test_predict_on_video_file_method_multiview_model(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on a test video.
    model.predict_on_video_file(
        Path(toy_mdata_dir) / "videos" / "test_vid_top.mp4",
        generate_labeled_video=True,
    )
    assert (model.video_preds_dir() / "test_vid_top.csv").is_file()
    assert (model.video_preds_dir() / "test_vid_top_temporal_norm.csv").is_file()
    assert (model.labeled_videos_dir() / "test_vid_top_labeled.mp4").is_file()


def test_predict_on_video_file_multiview(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on a test video.
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


def test_predict_frame(tmp_path, request):
    model = _setup_test_model(tmp_path, request)

    # Synthetic RGB frame
    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    result = model.predict_frame(frame)

    assert "keypoints" in result
    assert "confidence" in result

    kp = result["keypoints"]
    conf = result["confidence"]

    # Check types and shapes
    assert kp.dtype == np.float32
    assert conf.dtype == np.float32
    assert kp.ndim == 2
    assert kp.shape[1] == 2
    assert conf.shape[0] == kp.shape[0]
    assert kp.shape[0] > 0  # at least one keypoint

    # Confidence must be in [0, 1] (softmax peak intensity)
    assert np.all(conf >= 0)
    assert np.all(conf <= 1)

    # Keypoints should be within frame bounds (with tolerance for subpixel overshoot)
    assert np.all(kp[:, 0] <= 256 + 1)
    assert np.all(kp[:, 1] <= 256 + 1)


def test_predict_frame_with_bbox(tmp_path, request):
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

    # Confidence must be in [0, 1]
    assert np.all(conf >= 0)
    assert np.all(conf <= 1)

    # Keypoints should be remapped to original frame coordinates
    assert np.all(kp[:, 0] >= 0)
    assert np.all(kp[:, 1] >= 0)
    assert np.all(kp[:, 0] <= 640 + 1)
    assert np.all(kp[:, 1] <= 480 + 1)


def test_predict_frame_errors(tmp_path, request):
    """Error-path tests for predict_frame validation guards."""
    model = _setup_test_model(tmp_path, request)

    # Wrong dtype (float32 instead of uint8)
    float_frame = np.random.rand(256, 256, 3).astype(np.float32)
    with pytest.raises(ValueError, match="must be uint8"):
        model.predict_frame(float_frame)

    # Wrong shape (grayscale -- missing channel dim)
    gray_frame = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
        model.predict_frame(gray_frame)

    # Wrong shape (RGBA -- 4 channels)
    rgba_frame = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
        model.predict_frame(rgba_frame)

    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Negative bbox origin
    with pytest.raises(ValueError, match="non-negative"):
        model.predict_frame(frame, bbox=(-10, 0, 50, 50))

    # Zero-width bbox
    with pytest.raises(ValueError, match="must be positive"):
        model.predict_frame(frame, bbox=(10, 10, 0, 50))

    # Bbox completely off-frame (empty crop)
    with pytest.raises(ValueError, match="empty crop"):
        model.predict_frame(frame, bbox=(1000, 1000, 50, 50))


def test_predict_frame_bbox_clipping(tmp_path, request):
    """Bbox extending past frame edge -- numpy clips silently, remap should still be valid."""
    model = _setup_test_model(tmp_path, request)

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # Bbox extends 60px past right edge: requested width 200, actual crop width 140
    bbox = (500, 100, 200, 150)
    result = model.predict_frame(frame, bbox=bbox)

    kp = result["keypoints"]
    conf = result["confidence"]

    assert kp.dtype == np.float32
    assert conf.dtype == np.float32
    assert kp.ndim == 2
    assert kp.shape[1] == 2
    assert conf.shape[0] == kp.shape[0]

    # Confidence in [0, 1]
    assert np.all(conf >= 0)
    assert np.all(conf <= 1)

    # Keypoints should remap into the clipped region, not the requested region
    assert np.all(kp[:, 0] >= 0)
    assert np.all(kp[:, 1] >= 0)
    assert np.all(kp[:, 0] <= 640 + 1)
    assert np.all(kp[:, 1] <= 480 + 1)
