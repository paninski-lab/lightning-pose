"""Shared, non-fixture test helpers for lightning_pose.api.model tests.

Split across two test files that mirror the source split between `model.py` and
`model_runtime.py`: `test_model.py` (construction/predict_*) and
`test_model_runtime.py` (compile/export/onnx/tensorrt). `_setup_test_model` is used
by tests in both files, so it lives here rather than in either test module.
"""

import shutil
from pathlib import Path

from lightning_pose.api import Model
from lightning_pose.api.model import _Precision
from tests.fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(
    tmp_path: Path, request, multiview: bool = False, precision: _Precision = "fp32"
) -> Model:
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

    model = Model.from_dir(tmp_model_dir, precision=precision)

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
