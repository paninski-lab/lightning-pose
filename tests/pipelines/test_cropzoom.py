"""Tests for lightning_pose.pipelines.cropzoom."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lightning_pose.pipelines import CropZoomPipeline, KeypointBboxStage


def _fake_model(model_type: str = 'heatmap', multiview: bool = False) -> MagicMock:
    model = MagicMock()
    model.config.is_multi_view.return_value = multiview
    model.cfg.model.model_type = model_type
    model.model_dir = Path('/fake/model')
    return model


def _stage() -> KeypointBboxStage:
    return KeypointBboxStage(['a', 'b'], crop_ratio=1.0)


class TestCropZoomPipelineInit:
    """Test CropZoomPipeline construction."""

    def test_accepts_singleview_noncontext_models(self):
        pipeline = CropZoomPipeline(_fake_model(), _fake_model(), _stage())
        assert pipeline.bbox_stage is not None

    def test_rejects_context_pose_estimator(self):
        with pytest.raises(NotImplementedError, match='context.*pose estimator'):
            CropZoomPipeline(_fake_model(), _fake_model('heatmap_mhcrnn'), _stage())

    def test_rejects_context_detector(self):
        with pytest.raises(NotImplementedError, match='context.*detector'):
            CropZoomPipeline(_fake_model('heatmap_mhcrnn'), _fake_model(), _stage())

    def test_rejects_multiview(self):
        with pytest.raises(NotImplementedError, match='multi-view'):
            CropZoomPipeline(_fake_model(), _fake_model(multiview=True), _stage())


class TestPredictOnVideoFile:
    """Test CropZoomPipeline.predict_on_video_file."""

    def test_not_implemented_yet(self, tmp_path):
        pipeline = CropZoomPipeline(_fake_model(), _fake_model(), _stage())
        with pytest.raises(NotImplementedError):
            pipeline.predict_on_video_file('v.mp4', tmp_path)
