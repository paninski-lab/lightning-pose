"""Tests for lightning_pose.pipelines.cropzoom."""

import gc
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

import lightning_pose.utils.cropzoom as cz
from lightning_pose.api import Model
from lightning_pose.pipelines import CropZoomPipeline, KeypointBboxStage
from lightning_pose.utils.inference_types import _Reader
from tests.pipelines.conftest import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_CLIP_FRAMES,
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
        assert kwargs['bbox_df'] is result.bboxes
        assert kwargs['confidence_thresh_for_vid'] == 0.3
        assert kwargs['colormap'] == 'viridis'


ANCHOR_KEYPOINTS = ['nose_top', 'tailBase_top']
CROP_RATIO = 2.0
BATCH_SIZE = 8


def _run_pipeline(model_dir: Path, clip: Path, out: Path, reader: _Reader):
    """Run CropZoomPipeline with the same checkpoint as detector and pose estimator."""
    pipeline = CropZoomPipeline.from_dirs(
        detector_dir=model_dir,
        pose_estimator_dir=model_dir,
        anchor_keypoints=ANCHOR_KEYPOINTS,
        crop_ratio=CROP_RATIO,
    )
    return pipeline, pipeline.predict_on_video_file(
        clip, out, reader=reader, batch_size=BATCH_SIZE,
    )


def _run_two_step(model_dir: Path, clip: Path, out: Path, reader: _Reader):
    """The multi-command flow: predict -> create_bbox -> predict --bbox_dir.

    Returns:
        (predictions DataFrame, bbox DataFrame), the reference for the pipeline's outputs.
    """
    model = Model.from_dir(model_dir)
    model.predict_on_video_file(
        clip, output_dir=out / 'detector', compute_metrics=False, reader=reader,
    )
    bbox_file = out / 'detector' / f'{clip.stem}_bbox.csv'
    cz.generate_bbox(
        input_preds_file=out / 'detector' / f'{clip.stem}.csv',
        detector_cfg=OmegaConf.create(
            {'crop_ratio': CROP_RATIO, 'anchor_keypoints': ANCHOR_KEYPOINTS}
        ),
        output_bbox_file=bbox_file,
    )
    result = model.predict_on_video_file(
        clip, output_dir=out / 'pose', compute_metrics=False, reader=reader,
        bbox_file=bbox_file,
    )
    return result.predictions, pd.read_csv(bbox_file, index_col=0)


def _assert_matches_reference(
    result, ref_preds: pd.DataFrame, ref_bboxes: pd.DataFrame, bbox_atol: float, kp_atol: float,
) -> None:
    """Pipeline outputs must match the two-step flow (tolerances are in pixels)."""
    assert result.predictions.shape == ref_preds.shape == (NUM_CLIP_FRAMES, ref_preds.shape[1])
    assert list(result.predictions.columns) == list(ref_preds.columns)
    assert list(result.bboxes.columns) == list(ref_bboxes.columns) == ['x', 'y', 'h', 'w']
    assert len(result.bboxes) == len(ref_bboxes) == NUM_CLIP_FRAMES
    bbox_diff = np.abs(result.bboxes.to_numpy() - ref_bboxes.to_numpy()).max()
    assert bbox_diff <= bbox_atol, f'max bbox difference {bbox_diff} px > {bbox_atol}'
    kp_diff = np.abs(result.predictions.to_numpy() - ref_preds.to_numpy()).max()
    assert kp_diff <= kp_atol, f'max prediction difference {kp_diff} > {kp_atol}'


class TestCropZoomPipelineToyModel:
    """End-to-end CropZoomPipeline runs with the real toy checkpoint (CPU, no GPU needed)."""

    @pytest.fixture(autouse=True)
    def force_cpu(self, monkeypatch):
        # the pipeline and Model.predict_on_video_file's Trainer both pick the GPU when visible
        monkeypatch.setattr('lightning_pose.pipelines.cropzoom.get_accelerator', lambda: 'cpu')
        monkeypatch.setattr('lightning_pose.utils.predictions.get_accelerator', lambda: 'cpu')

    def test_matches_two_step_flow(self, toy_model_dir, toy_clip, tmp_path):
        """One in-memory pass equals predict -> create_bbox -> predict --bbox_dir."""
        ref_preds, ref_bboxes = _run_two_step(toy_model_dir, toy_clip, tmp_path / 'ref', 'opencv')
        _, result = _run_pipeline(toy_model_dir, toy_clip, tmp_path / 'pipe', 'opencv')
        _assert_matches_reference(result, ref_preds, ref_bboxes, bbox_atol=1, kp_atol=1e-2)

    def test_writes_standard_outputs(self, toy_model_dir, toy_clip, tmp_path):
        out = tmp_path / 'pipe'
        _, result = _run_pipeline(toy_model_dir, toy_clip, out, 'opencv')
        assert (out / f'{toy_clip.stem}.csv').is_file()
        assert (out / f'{toy_clip.stem}_bbox.csv').is_file()
        assert np.isfinite(result.predictions.to_numpy()).all()
        assert (result.bboxes[['h', 'w']].to_numpy() > 0).all()


def _skip_unless_reader_usable(reader: str, video: Path) -> None:
    """Skip the test if the GPU or the requested reader is unavailable on this machine.

    Probes are done lazily here, not at import: PyNvVideoCodec's probe aborts natively when
    there is no CUDA device, so it must never run during CPU-only collection.
    """
    if not torch.cuda.is_available():
        pytest.skip('no CUDA device')
    if reader == 'dali':
        pytest.importorskip('nvidia.dali', reason='nvidia-dali not installed')
    elif reader == 'pynvvc':
        from lightning_pose.data.video.pynvvc import is_pynvvc_available

        if not is_pynvvc_available(str(video)):
            pytest.skip('PyNvVideoCodec cannot decode on this GPU/driver')


@pytest.mark.gpu
class TestCropZoomPipelineToyModelGpu:
    """CropZoomPipeline on a GPU with each video reader: does it run, and clean up after itself.

    The CPU tests cannot see device-mismatch bugs or the DALI / PyNvVideoCodec
    full-resolution passthrough paths; these do. They check that the outputs are well-formed,
    not that they are numerically right: correctness against the two-step flow is covered on CPU
    (``TestCropZoomPipelineToyModel``), and different readers' decoders legitimately differ.
    """

    @pytest.mark.parametrize('reader', ['opencv', 'pynvvc', 'dali'])
    def test_runs_with_reader(self, reader, toy_model_dir, toy_clip, tmp_path):
        _skip_unless_reader_usable(reader, toy_clip)

        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        out = tmp_path / 'pipe'
        pipeline, result = _run_pipeline(toy_model_dir, toy_clip, out, reader)

        assert (out / f'{toy_clip.stem}.csv').is_file()
        assert (out / f'{toy_clip.stem}_bbox.csv').is_file()
        assert len(result.predictions) == len(result.bboxes) == NUM_CLIP_FRAMES
        assert np.isfinite(result.predictions.to_numpy()).all()
        assert (result.bboxes[['h', 'w']].to_numpy() > 0).all()

        # the pipeline leaves the GPU as it found it: weights back on CPU, no leaked tensors
        for model in (pipeline.detector, pipeline.pose_estimator):
            assert model.model is not None
            assert next(model.model.parameters()).device.type == 'cpu'
        gc.collect()
        assert torch.cuda.memory_allocated() - baseline < 64 * 2**20
