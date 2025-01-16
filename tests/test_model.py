import shutil
from pathlib import Path

from lightning_pose.model import Model

from .fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(tmp_path, request):
    # Get the trained model for testing.
    fetch_test_data_if_needed(request.path.parent, "test_model_mirror_mouse")
    # Copy to tmpdir because prediction will create output artifacts in model_dir.
    tmp_model_dir = tmp_path / "test_model_mirror_mouse"
    shutil.copytree(
        request.path.parent / "test_model_mirror_mouse", tmp_model_dir
    )

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

    return model


def test_predict_on_label_csv(tmp_path, request, toy_data_dir):
    model = _setup_test_model(tmp_path, request)

    # Test prediction on a test video.
    model.predict_on_label_csv(Path(toy_data_dir) / "CollectedData.csv")

    assert (model.image_preds_dir() / "CollectedData.csv" / "predictions.csv").is_file()
    assert (
        model.image_preds_dir() / "CollectedData.csv" / "predictions_pixel_error.csv"
    ).is_file()


def test_predict_on_video_file(tmp_path, request, toy_data_dir):
    model = _setup_test_model(tmp_path, request)

    # Test prediction on a test video.
    model.predict_on_video_file(Path(toy_data_dir) / "videos" / "test_vid.mp4")
    assert (model.video_preds_dir() / "test_vid.csv").is_file()
    assert (model.video_preds_dir() / "test_vid_temporal_norm.csv").is_file()
    # Labeled video generation should have been off by default.
    assert not model.labeled_videos_dir().exists()

    # Test labeled_video generation.
    model.predict_on_video_file(
        Path(toy_data_dir) / "videos" / "test_vid.mp4", generate_labeled_video=True
    )
    assert (model.labeled_videos_dir() / "test_vid_labeled.mp4").is_file()
