"""Tests for lightning_pose.pipelines.cropzoom."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from lightning_pose.pipelines import CropZoomPipeline, KeypointBboxStage
from tests.pipelines.conftest import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_FRAMES,
    RESIZE_DETECTOR,
    RESIZE_POSE,
    FakeNet,
    make_fake_model,
)


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
    """Test CropZoomPipeline.predict_on_video_file with fake models on CPU."""

    # detector keypoints in model space (32x48 input, i.e. half the 64x96 frame), which map to
    # frame-space (20, 12) and (40, 28): span 20 in x, 16 in y, so with crop_ratio=1 the box is
    # 20x20 centred on (30, 20), i.e. x=20, y=10 at frame 0 (x then drifts +1 per frame)
    DETECTOR_KP = torch.tensor([[10.0, 6.0], [20.0, 14.0]])

    @pytest.fixture
    def pipeline(self, tmp_path, monkeypatch):
        # never touch a GPU in tests
        monkeypatch.setattr('lightning_pose.pipelines.cropzoom.get_accelerator', lambda: 'cpu')
        detector_net = FakeNet(self.DETECTOR_KP, drift_x=1.0)
        detector = make_fake_model(detector_net, RESIZE_DETECTOR, 2, tmp_path)
        # pose model reports the centre of its input crop
        pose_kp = torch.tensor([[RESIZE_POSE[1] / 2, RESIZE_POSE[0] / 2]])
        pose = make_fake_model(FakeNet(pose_kp), RESIZE_POSE, 1, tmp_path)
        stage = KeypointBboxStage(['kp0', 'kp1'], crop_ratio=1.0)
        return CropZoomPipeline(detector, pose, stage)

    def test_predict_on_video_file_outputs(self, pipeline, video_file, tmp_path):
        out = tmp_path / 'out'
        result = pipeline.predict_on_video_file(video_file, out, reader='opencv', batch_size=8)

        assert (out / 'video.csv').is_file()
        assert (out / 'video_bbox.csv').is_file()
        # 20 frames in batches of 8 -> last batch padded; padding must be trimmed
        assert len(result.predictions) == NUM_FRAMES
        assert len(result.bboxes) == NUM_FRAMES
        assert list(result.bboxes.columns) == ['x', 'y', 'h', 'w']
        assert result.bboxes['x'].tolist() == [20 + i for i in range(NUM_FRAMES)]
        assert (result.bboxes[['y', 'h', 'w']] == [10, 20, 20]).all(axis=None)

    def test_predict_on_video_file_keypoints_in_original_frame_coords(
        self, pipeline, video_file, tmp_path,
    ):
        result = pipeline.predict_on_video_file(
            video_file, tmp_path / 'out', reader='opencv', batch_size=8,
        )
        # the pose model reports its crop centre, which must map back to the bbox centre
        # (bbox centre: x = 30 + frame idx, y = 20)
        x = result.predictions.xs('x', level='coords', axis=1).to_numpy().ravel()
        y = result.predictions.xs('y', level='coords', axis=1).to_numpy().ravel()
        np.testing.assert_allclose(x, 30 + np.arange(NUM_FRAMES), atol=1e-4)
        np.testing.assert_allclose(y, 20, atol=1e-4)

    def test_predict_on_video_file_models_see_expected_inputs(
        self, pipeline, video_file, tmp_path,
    ):
        pipeline.predict_on_video_file(
            video_file, tmp_path / 'out', reader='opencv', batch_size=8,
        )
        detector_batch = pipeline.detector.model.batches[0]
        pose_batch = pipeline.pose_estimator.model.batches[0]
        assert tuple(detector_batch['frames'].shape) == (8, 3, *RESIZE_DETECTOR)
        assert tuple(pose_batch['frames'].shape) == (8, 3, *RESIZE_POSE)
        # detector bbox is the full original frame; pose bbox is the crop
        assert detector_batch['bbox'][0].tolist() == [0, 0, FRAME_HEIGHT, FRAME_WIDTH]
        assert pose_batch['bbox'][0].tolist() == [20, 10, 20, 20]

    def test_predict_on_video_file_missing_detection_reuses_previous_bbox(
        self, pipeline, video_file, tmp_path,
    ):
        pipeline.detector.model.nan_frames = (0, 5)
        result = pipeline.predict_on_video_file(
            video_file, tmp_path / 'out', reader='opencv', batch_size=8,
        )
        x = result.bboxes['x'].tolist()
        assert x[5] == x[4]
        assert x[6] == 20 + 6
        # frame 0 has no previous bbox: full frame
        assert result.bboxes.iloc[0].tolist() == [0, 0, FRAME_HEIGHT, FRAME_WIDTH]
        assert pipeline.bbox_stage.num_fallback == 2

    def test_predict_on_video_file_bbox_csv_roundtrips_through_loader(
        self, pipeline, video_file, tmp_path,
    ):
        # the saved bbox CSV must be readable by the existing `--bbox_dir` machinery
        from lightning_pose.utils.predictions import _load_bbox_dfs

        out = tmp_path / 'out'
        result = pipeline.predict_on_video_file(video_file, out, reader='opencv', batch_size=8)
        loaded = _load_bbox_dfs(str(video_file), out / 'video_bbox.csv')
        assert isinstance(loaded, pd.DataFrame)
        assert loaded.to_numpy().tolist() == result.bboxes.to_numpy().tolist()

    def test_predict_on_video_file_writes_progress(self, pipeline, video_file, tmp_path):
        progress_file = tmp_path / 'progress' / 'progress.json'
        pipeline.predict_on_video_file(
            video_file, tmp_path / 'out', reader='opencv', batch_size=8,
            progress_file=progress_file,
        )
        progress = json.loads(progress_file.read_text())
        assert progress['completed'] == progress['total'] == 3

    def test_predict_on_video_file_resets_stage_between_videos(
        self, pipeline, video_file, tmp_path,
    ):
        pipeline.detector.model.nan_frames = (0,)
        pipeline.predict_on_video_file(
            video_file, tmp_path / 'a', reader='opencv', batch_size=8,
        )
        pipeline.detector.model.num_frames_seen = 0
        result = pipeline.predict_on_video_file(
            video_file, tmp_path / 'b', reader='opencv', batch_size=8,
        )
        assert result.bboxes.iloc[0].tolist() == [0, 0, FRAME_HEIGHT, FRAME_WIDTH]
        assert pipeline.bbox_stage.num_fallback == 1

    def test_predict_on_video_file_labeled_video_only_when_requested(
        self, pipeline, video_file, tmp_path,
    ):
        with patch('lightning_pose.pipelines.cropzoom.generate_labeled_video_fn') as mock_fn:
            pipeline.predict_on_video_file(
                video_file, tmp_path / 'a', reader='opencv', batch_size=8,
            )
            mock_fn.assert_not_called()
            result = pipeline.predict_on_video_file(
                video_file, tmp_path / 'b', reader='opencv', batch_size=8,
                generate_labeled_video=True,
            )
        kwargs = mock_fn.call_args.kwargs
        assert kwargs['video_file'] == str(video_file)
        assert kwargs['output_mp4_file'] == str(
            tmp_path / 'b' / 'labeled_videos' / 'video_labeled.mp4'
        )
        assert kwargs['preds_df'] is result.predictions
        assert kwargs['confidence_thresh_for_vid'] == 0.3
        assert kwargs['colormap'] == 'viridis'
