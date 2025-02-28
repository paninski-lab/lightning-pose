import shutil
from pathlib import Path

from lightning_pose.model import Model

from .fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(tmp_path, request, multiview=False):
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


def test_predict_on_label_csv_multiview(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on a CSV file.
    model.predict_on_label_csv(Path(toy_mdata_dir) / "top.csv")

    assert (model.image_preds_dir() / "top.csv" / "predictions.csv").is_file()
    assert (
        model.image_preds_dir() / "top.csv" / "predictions_pixel_error.csv"
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
        Path(toy_data_dir) / "videos" / "test_vid.mp4", generate_labeled_video=True
    )
    assert (model.labeled_videos_dir() / "test_vid_labeled.mp4").is_file()


def test_predict_on_video_file_method_multiview_model(tmp_path, request, toy_mdata_dir):
    model = _setup_test_model(tmp_path, request, multiview=True)

    # Test prediction on a test video.
    model.predict_on_video_file(
        Path(toy_mdata_dir) / "videos" / "test_vid_top.mp4", generate_labeled_video=True
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
