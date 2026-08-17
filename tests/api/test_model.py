import copy
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from lightning_pose.api import Model
from lightning_pose.api.model import _build_datamodule_pred, _Precision
from tests.fetch_test_data import fetch_test_data_if_needed


def _setup_test_model(
    tmp_path, request, multiview=False, precision: _Precision = "fp32"
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


class TestPredictOnLabelCsv:
    """Test the predict_on_label_csv method."""

    pytestmark = pytest.mark.gpu

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

    def test_predict_on_label_csv_struct_mode_missing_bbox_file(
        self, tmp_path, request, toy_data_dir
    ):
        """Reproduces a real crash report: the test fixture's config.yaml predates the
        data.bbox_file key (added for the bbox/cropzoom feature). Model.from_dir2 with
        hydra_overrides set (even an empty list, as `litpose predict --overrides` with no
        values produces) puts the loaded cfg in struct mode via hydra.compose. Merging in
        cfg_overrides, which always sets data.bbox_file, then raised ConfigAttributeError
        before predict_on_label_csv's OmegaConf.merge was wrapped in open_dict.
        """
        dataset_name = "test_model_mirror_mouse"
        fetch_test_data_if_needed(request.path.parent, dataset_name)
        tmp_model_dir = tmp_path / dataset_name
        shutil.copytree(request.path.parent / dataset_name, tmp_model_dir)
        assert "bbox_file" not in Model.from_dir(tmp_model_dir).cfg.data

        model = Model.from_dir2(tmp_model_dir, hydra_overrides=[])
        assert OmegaConf.is_struct(model.cfg)

        model.predict_on_label_csv(Path(toy_data_dir) / "CollectedData.csv")

        assert (model.image_preds_dir() / "CollectedData.csv" / "predictions.csv").is_file()


class TestPredictOnVideoFile:
    """Test the predict_on_video_file method."""

    @pytest.mark.gpu
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

    @pytest.mark.gpu
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

    def test_predict_on_video_file_bbox_file_forwarded(self, tmp_path):
        """bbox_file is forwarded to predict_video when provided."""
        model = Model(tmp_path, MagicMock())
        bbox_file = tmp_path / 'vid_bbox.csv'

        with (
            patch.object(model, '_load'),
            patch(
                'lightning_pose.api.model.predict_video', return_value=pd.DataFrame(),
            ) as mock_pv,
        ):
            model.predict_on_video_file(
                video_file=tmp_path / 'vid.mp4',
                compute_metrics=False,
                bbox_file=bbox_file,
            )

        assert mock_pv.call_args.kwargs['bbox_file'] == bbox_file

    @pytest.mark.gpu
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

    pytestmark = pytest.mark.gpu

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

    def test_predict_frame_fp16_precision(self, tmp_path, request):
        """predict_frame actually enters torch.autocast(dtype=torch.float16) for fp16.

        Spies on torch.autocast (wraps the real implementation, so the forward
        pass still runs normally) to confirm the context manager is engaged with
        the correct dtype -- not just that the fp16 code path runs without error.
        """
        model = _setup_test_model(tmp_path, request, precision="fp16")

        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with patch("torch.autocast", wraps=torch.autocast) as mock_autocast:
            result = model.predict_frame(frame)

        mock_autocast.assert_called_once()
        assert mock_autocast.call_args.kwargs["dtype"] == torch.float16

        assert "keypoints" in result
        assert "confidence" in result
        assert result["keypoints"].dtype == np.float32
        assert result["confidence"].dtype == np.float32
        assert result["keypoints"].shape[0] > 0

    def test_predict_frame_fp32_precision_skips_autocast(self, tmp_path, request):
        """predict_frame does NOT enter torch.autocast for the default fp32 precision."""
        model = _setup_test_model(tmp_path, request)  # default precision="fp32"

        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        with patch("torch.autocast", wraps=torch.autocast) as mock_autocast:
            model.predict_frame(frame)

        mock_autocast.assert_not_called()

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

    pytestmark = pytest.mark.gpu

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


class TestBuildDatamodulePred:
    """Test the _build_datamodule_pred helper."""

    pytestmark = pytest.mark.gpu

    def test_build_datamodule_pred_imgaug_reset_to_default(self, cfg):
        """imgaug pipeline is resize-only regardless of the training config."""
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.training.imgaug = 'dlc'
        data_module = _build_datamodule_pred(cfg_copy)
        # pipeline always has exactly one element: the final resize transform
        assert data_module.dataset.imgaug_transform is not None
        assert len(data_module.dataset.imgaug_transform) == 1

    def test_build_datamodule_pred_imgaug_hflip_cleared(self, cfg):
        """imgaug_hflip is False on the prediction dataset even when True in the config."""
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.training.imgaug_hflip = True
        data_module = _build_datamodule_pred(cfg_copy)
        assert data_module.dataset.imgaug_hflip is False

    def test_build_datamodule_pred_does_not_mutate_cfg(self, cfg):
        """the original config object is not modified."""
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.training.imgaug = 'dlc'
        cfg_copy.training.imgaug_hflip = True
        _build_datamodule_pred(cfg_copy)
        assert cfg_copy.training.imgaug == 'dlc'
        assert cfg_copy.training.imgaug_hflip is True

    def test_build_datamodule_pred_struct_mode_missing_imgaug_hflip(self, cfg):
        """Reproduces a real crash: an older model's config.yaml predates the
        training.imgaug_hflip key (config_mirror-mouse-example.yaml, loaded by the `cfg`
        fixture, is one such config), and the loaded cfg is struct-mode (e.g. because it
        was composed via Model.from_dir2's hydra_overrides). Without open_dict, setting
        cfg_pred.training.imgaug_hflip raised ConfigAttributeError.
        """
        cfg_copy = copy.deepcopy(cfg)
        assert 'imgaug_hflip' not in cfg_copy.training
        OmegaConf.set_struct(cfg_copy, True)
        data_module = _build_datamodule_pred(cfg_copy)
        assert data_module.dataset.imgaug_hflip is False


def _torch_compile_supported() -> bool:
    """Whether torch.compile's inductor backend can actually run on this machine.

    Inductor generates Triton kernels, which require a GPU of CUDA compute
    capability >= 7.0 (Volta and newer).
    """
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (7, 0)


requires_torch_compile = pytest.mark.skipif(
    not _torch_compile_supported(),
    reason=(
        "torch.compile's inductor backend requires a GPU of CUDA capability >= 7.0 "
        "(Triton). Compilation is lazy, so only tests that run inference on a "
        "compiled model are affected."
    ),
)


class TestCompile:
    """Test the compile method."""

    pytestmark = pytest.mark.gpu

    def test_compile_loads_model(self, tmp_path, request):
        """compile() loads the checkpoint, so it can be called before any prediction."""
        model = _setup_test_model(tmp_path, request)
        assert model.model is None
        # Capability is mocked so this runs on any GPU: compilation is lazy, so no
        # Triton kernels are generated here -- only the forward-pass wrapping.
        with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            model.compile()
        assert model.model is not None
        assert model._compiled

    def test_compile_is_idempotent(self, tmp_path, request):
        """Calling compile() twice wraps forward only once.

        Spies on torch.compile (wrapping the real implementation, so the model is
        still genuinely compiled) to confirm the second call is a no-op rather
        than double-wrapping the already-compiled forward.
        """
        model = _setup_test_model(tmp_path, request)
        with (
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("torch.compile", wraps=torch.compile) as mock_compile,
        ):
            model.compile()
            model.compile()
        mock_compile.assert_called_once()
        assert model._compiled

    def test_compile_raises_on_old_gpu(self, tmp_path, request):
        """compile() fails fast with a readable error on pre-Volta GPUs.

        Mocked rather than skipped, so the check itself is covered on any machine.
        """
        model = _setup_test_model(tmp_path, request)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(6, 1)),
            patch(
                "torch.cuda.get_device_name",
                return_value="NVIDIA GeForce GTX 1080 Ti",
            ),
            pytest.raises(RuntimeError, match="CUDA compute capability"),
        ):
            model.compile()
        assert not model._compiled

    @requires_torch_compile
    def test_compile_then_predict_on_label_csv(self, tmp_path, request, toy_data_dir):
        """A compiled model predicts on a labeled CSV and writes the usual outputs."""
        model = _setup_test_model(tmp_path, request)
        model.compile()
        result = model.predict_on_label_csv(Path(toy_data_dir) / "CollectedData.csv")
        assert (
            model.image_preds_dir() / "CollectedData.csv" / "predictions.csv"
        ).is_file()
        assert result.predictions.shape[0] > 0
        assert model.model is not None
        xy_cols = [c for c in result.predictions.columns if c[-1] in ("x", "y")]
        assert len(xy_cols) == 2 * model.model.num_keypoints

    @requires_torch_compile
    def test_compile_then_predict_on_video_file(self, tmp_path, request, toy_data_dir):
        """A compiled model predicts on a video file."""
        model = _setup_test_model(tmp_path, request)
        model.compile()
        model.predict_on_video_file(Path(toy_data_dir) / "videos" / "test_vid.mp4")
        assert (model.video_preds_dir() / "test_vid.csv").is_file()

    @requires_torch_compile
    def test_compile_matches_eager_predictions(self, tmp_path, request, toy_data_dir):
        """Compiled predictions match eager ones to well under a pixel.

        Guards against a future PyTorch release changing compile behavior in a way
        that actually moves keypoints. Separate tmp_path subdirectories because
        _setup_test_model copies the model tree and asserts no prediction outputs
        exist yet.
        """
        csv_file = Path(toy_data_dir) / "CollectedData.csv"
        eager_model = _setup_test_model(tmp_path / "eager", request)
        compiled_model = _setup_test_model(tmp_path / "compiled", request)
        compiled_model.compile()

        eager_preds = eager_model.predict_on_label_csv(csv_file).predictions
        compiled_preds = compiled_model.predict_on_label_csv(csv_file).predictions

        xy_cols = [c for c in eager_preds.columns if c[-1] in ("x", "y")]
        deviation = np.abs(
            eager_preds[xy_cols].to_numpy(dtype=float)
            - compiled_preds[xy_cols].to_numpy(dtype=float)
        )
        max_deviation = np.nanmax(deviation)
        assert max_deviation < 0.1, f"max pixel deviation {max_deviation:.4f} >= 0.1"

    @requires_torch_compile
    def test_compile_handles_changing_input_shape(self, tmp_path, request, toy_data_dir):
        """Changing batch size after compiling triggers recompilation, not an error."""
        model = _setup_test_model(tmp_path, request)
        model.compile()
        # predict_frame runs a batch of 1 ...
        frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        model.predict_frame(frame)
        # ... and predict_on_label_csv runs the configured batch size.
        model.predict_on_label_csv(Path(toy_data_dir) / "CollectedData.csv")


def _onnxruntime_available() -> bool:
    """Whether the optional onnxruntime dependency is importable."""
    try:
        import onnxruntime  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True


requires_onnxruntime = pytest.mark.skipif(
    not _onnxruntime_available(),
    reason=(
        "onnxruntime is an optional dependency. Export and runtime tests need it "
        "installed; see docs/source/user_guide_advanced/increasing_inference_speed.rst."
    ),
)


class TestExport:
    """Test the export method."""

    pytestmark = pytest.mark.gpu

    def test_export_rejects_unknown_runtime(self, tmp_path, request):
        """export() rejects a runtime other than 'onnx' before loading anything."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(ValueError, match="Unsupported export runtime"):
            model.export("coreml")
        assert model.model is None

    def test_export_rejects_unknown_onnx_precision(self, tmp_path, request):
        """export() rejects bf16, which ONNX export does not support."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(ValueError, match="Unsupported onnx_precision"):
            model.export("onnx", onnx_precision="bf16")  # type: ignore[arg-type]
        assert model.model is None

    @requires_onnxruntime
    @pytest.mark.parametrize("onnx_precision", ["fp32", "fp16"])
    def test_export_writes_expected_path(self, tmp_path, request, onnx_precision):
        """export() writes {ckpt_stem}_{onnx_precision}.onnx into exports_onnx/."""
        model = _setup_test_model(tmp_path, request)
        output_path = model.export("onnx", onnx_precision=onnx_precision)

        assert output_path.is_file()
        assert output_path.parent == model.exports_onnx_dir()
        assert output_path.parent == model.model_dir / "exports_onnx"
        assert output_path.name == f"{model._ckpt_stem()}_{onnx_precision}.onnx"
        assert output_path.stat().st_size > 0

    @requires_onnxruntime
    def test_export_loads_model_lazily(self, tmp_path, request):
        """export() loads the checkpoint itself, so it works before any prediction."""
        model = _setup_test_model(tmp_path, request)
        assert model.model is None
        model.export("onnx", onnx_precision="fp32")
        assert model.model is not None

    @requires_onnxruntime
    def test_export_works_when_weights_already_loaded(self, tmp_path, request):
        """export() is unaffected by whether _load() already ran (lazy-load interaction)."""
        model = _setup_test_model(tmp_path, request)
        model._load()
        assert model.model is not None
        output_path = model.export("onnx", onnx_precision="fp32")
        assert output_path.is_file()

    @requires_onnxruntime
    def test_fp16_export_leaves_live_model_in_fp32(self, tmp_path, request):
        """An fp16 export must not half() the in-memory model.

        torch's .half() mutates in place and returns self, so tracing the live
        module would silently leave every subsequent eager call running in half
        precision. export() traces a copy instead.
        """
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")

        assert model.model is not None
        dtypes = {p.dtype for p in model.model.parameters()}
        assert dtypes == {torch.float32}, f"live model was mutated to {dtypes}"

    @requires_onnxruntime
    def test_export_multiview(self, tmp_path, request):
        """A multiview model exports with its per-view dummy input shape."""
        model = _setup_test_model(tmp_path, request, multiview=True)
        output_path = model.export("onnx", onnx_precision="fp32")
        assert output_path.is_file()
        assert output_path.stat().st_size > 0


class TestTensorRTExport:
    """Test the export('tensorrt', ...) method."""

    pytestmark = pytest.mark.gpu

    def test_export_tensorrt_requires_existing_onnx_export(self, tmp_path, request):
        """export('tensorrt') points at export('onnx') rather than silently exporting one."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(FileNotFoundError, match=r"Run model\.export\('onnx'"):
            model.export("tensorrt", onnx_precision="fp16")

    @requires_onnxruntime
    def test_export_tensorrt_writes_expected_path(self, tmp_path, request):
        """export('tensorrt') writes exports_trt/{ckpt_stem}_{onnx_precision}/."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")
        cache_dir = model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        assert cache_dir.is_dir()
        assert cache_dir.parent == model.exports_trt_dir()
        assert cache_dir.parent == model.model_dir / "exports_trt"
        assert cache_dir.name == f"{model._ckpt_stem()}_fp16"
        assert (cache_dir / "trt_metadata.json").is_file()

    @requires_onnxruntime
    def test_export_tensorrt_metadata_contents(self, tmp_path, request):
        """trt_metadata.json records exactly what the build used."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")
        cache_dir = model.export(
            "tensorrt", onnx_precision="fp16", max_batch_size=4, opt_batch_size=2
        )
        metadata = json.loads((cache_dir / "trt_metadata.json").read_text())
        assert metadata["onnx_precision"] == "fp16"
        assert metadata["max_batch_size"] == 4
        assert metadata["opt_batch_size"] == 2
        assert metadata["gpu_name"]
        assert metadata["tensorrt_version"]
        assert metadata["onnxruntime_version"]
        assert metadata["built_at"]

    @requires_onnxruntime
    def test_export_tensorrt_opt_batch_size_defaults_to_max(self, tmp_path, request):
        """opt_batch_size defaults to max_batch_size when not given separately."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")
        cache_dir = model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        metadata = json.loads((cache_dir / "trt_metadata.json").read_text())
        assert metadata["opt_batch_size"] == 4

    @requires_onnxruntime
    def test_export_tensorrt_does_not_load_torch_weights(self, tmp_path, request):
        """export('tensorrt') builds purely from the .onnx file.

        Verified empirically (T4, 2026-08-06) that the engine build needs no
        torch weights at all -- shape info comes from the ONNX graph's own
        input metadata, not self.cfg/self.model.do_context. A separate Model
        instance is used so this really tests _export_tensorrt rather than
        leftover state from the onnx export that had to run first.
        """
        onnx_source = _setup_test_model(tmp_path / "onnx_source", request)
        onnx_source.export("onnx", onnx_precision="fp16")
        trt_model = Model.from_dir(onnx_source.model_dir)
        assert trt_model.model is None
        trt_model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        assert trt_model.model is None

    @requires_onnxruntime
    def test_export_tensorrt_multiview(self, tmp_path, request):
        """A multiview model builds a TensorRT engine from its per-view ONNX export."""
        model = _setup_test_model(tmp_path, request, multiview=True)
        model.export("onnx", onnx_precision="fp16")
        cache_dir = model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        assert (cache_dir / "trt_metadata.json").is_file()


class TestOnnxRuntime:
    """Test loading a model with runtime='onnx'."""

    pytestmark = pytest.mark.gpu

    def test_from_dir_rejects_unknown_runtime(self, tmp_path, request):
        """from_dir() rejects a runtime other than 'eager' or 'onnx'."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(ValueError, match="Unsupported runtime"):
            Model.from_dir(model.model_dir, runtime="coreml")  # type: ignore[arg-type]

    def test_from_dir_raises_when_no_export_exists(self, tmp_path, request):
        """runtime='onnx' with no export points the user at export()."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(FileNotFoundError, match=r"Run model\.export\('onnx'\) first"):
            Model.from_dir(model.model_dir, runtime="onnx")

    @requires_onnxruntime
    def test_from_dir_auto_selects_sole_export(self, tmp_path, request):
        """onnx_precision may be omitted when exactly one export exists."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp32")

        onnx_model = Model.from_dir(model.model_dir, runtime="onnx")
        assert onnx_model._runtime == "onnx"

    @requires_onnxruntime
    def test_from_dir_raises_when_multiple_exports_exist(self, tmp_path, request):
        """Ambiguous exports raise rather than picking one arbitrarily."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp32")
        model.export("onnx", onnx_precision="fp16")

        with pytest.raises(ValueError, match="Multiple ONNX exports found"):
            Model.from_dir(model.model_dir, runtime="onnx")

    @requires_onnxruntime
    def test_from_dir_selects_requested_precision(self, tmp_path, request):
        """onnx_precision disambiguates when several exports exist."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp32")
        model.export("onnx", onnx_precision="fp16")

        onnx_model = Model.from_dir(
            model.model_dir, runtime="onnx", onnx_precision="fp16"
        )
        assert onnx_model._runtime == "onnx"

    @requires_onnxruntime
    def test_export_and_load_agree_on_checkpoint(self, tmp_path, request):
        """Loading finds exactly the file exporting produced.

        Both sides resolve the checkpoint through io_utils.ckpt_path_from_base_path
        via _ckpt_stem(); this pins that they cannot drift apart.
        """
        model = _setup_test_model(tmp_path, request)
        exported = model.export("onnx", onnx_precision="fp32")

        loaded = Model.from_dir(model.model_dir, runtime="onnx")
        found = sorted(loaded.exports_onnx_dir().glob(f"{loaded._ckpt_stem()}_*.onnx"))
        assert found == [exported]

    def test_provider_guard_raises_on_silent_cpu_fallback(self, tmp_path, request):
        """A CUDA machine that loses CUDAExecutionProvider must fail loudly.

        onnxruntime falls back to CPU without raising, which "works" but is
        drastically slower with no indication why. Mocked rather than skipped so
        the guard is covered even where CUDA is genuinely available.
        """
        pytest.importorskip("onnxruntime")
        model = _setup_test_model(tmp_path, request)

        # A stub export file is enough -- the session is mocked out below.
        model.exports_onnx_dir().mkdir(parents=True, exist_ok=True)
        (model.exports_onnx_dir() / f"{model._ckpt_stem()}_fp16.onnx").touch()

        fake_session = MagicMock()
        fake_session.get_providers.return_value = ["CPUExecutionProvider"]

        with (
            patch("onnxruntime.InferenceSession", return_value=fake_session),
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(RuntimeError, match="CUDAExecutionProvider"),
        ):
            Model.from_dir(model.model_dir, runtime="onnx")

    def test_missing_onnxruntime_raises_install_hint(self, tmp_path, request):
        """A missing optional dependency explains how to install it.

        Simulated rather than skipped, so the message is covered on machines
        that do have onnxruntime.
        """
        model = _setup_test_model(tmp_path, request)
        model.exports_onnx_dir().mkdir(parents=True, exist_ok=True)
        (model.exports_onnx_dir() / f"{model._ckpt_stem()}_fp16.onnx").touch()

        with (
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="increasing_inference_speed"),
        ):
            Model.from_dir(model.model_dir, runtime="onnx")

    @requires_onnxruntime
    def test_compile_raises_on_onnx_runtime(self, tmp_path, request):
        """compile() is rejected on an ONNX-backed model rather than silently no-op."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp32")

        onnx_model = Model.from_dir(model.model_dir, runtime="onnx")
        with pytest.raises(RuntimeError, match="only supported for runtime='eager'"):
            onnx_model.compile()
        assert not onnx_model._compiled

    @requires_onnxruntime
    def test_onnx_matches_eager_predictions(self, tmp_path, request, toy_data_dir):
        """fp32 ONNX predictions match eager ones to well under a pixel.

        Separate tmp_path subdirectories because _setup_test_model copies the
        model tree and asserts no prediction outputs exist yet.
        """
        csv_file = Path(toy_data_dir) / "CollectedData.csv"

        eager_model = _setup_test_model(tmp_path / "eager", request)
        onnx_source = _setup_test_model(tmp_path / "onnx", request)
        onnx_source.export("onnx", onnx_precision="fp32")
        onnx_model = Model.from_dir(onnx_source.model_dir, runtime="onnx")

        eager_preds = eager_model.predict_on_label_csv(csv_file).predictions
        onnx_preds = onnx_model.predict_on_label_csv(csv_file).predictions

        xy_cols = [c for c in eager_preds.columns if c[-1] in ("x", "y")]
        deviation = np.abs(
            eager_preds[xy_cols].to_numpy(dtype=float)
            - onnx_preds[xy_cols].to_numpy(dtype=float)
        )
        max_deviation = np.nanmax(deviation)
        assert max_deviation < 0.1, f"max pixel deviation {max_deviation:.4f} >= 0.1"


class TestTensorRTRuntime:
    """Test loading a model with runtime='tensorrt'."""

    pytestmark = pytest.mark.gpu

    def test_from_dir_raises_when_no_trt_export_exists(self, tmp_path, request):
        """runtime='tensorrt' with no engine cache points the user at export()."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(
            FileNotFoundError, match=r"Run model\.export\('tensorrt', \.\.\.\) first"
        ):
            Model.from_dir(model.model_dir, runtime="tensorrt")

    def test_from_dir_tensorrt_raises_without_cuda(self, tmp_path, request):
        """No CPU fallback for TensorRT -- fails immediately, not after a slow attempt."""
        model = _setup_test_model(tmp_path, request)
        with (
            patch("torch.cuda.is_available", return_value=False),
            pytest.raises(RuntimeError, match="requires a CUDA-capable GPU"),
        ):
            Model.from_dir(model.model_dir, runtime="tensorrt")

    @requires_onnxruntime
    def test_from_dir_tensorrt_sets_runtime(self, tmp_path, request):
        """A real end-to-end export + load correctly marks _runtime."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")
        model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        trt_model = Model.from_dir(
            model.model_dir, runtime="tensorrt", onnx_precision="fp16"
        )
        assert trt_model._runtime == "tensorrt"

    @requires_onnxruntime
    def test_compile_raises_on_tensorrt_runtime(self, tmp_path, request):
        """compile() is rejected on a TensorRT-backed model rather than silently no-op."""
        model = _setup_test_model(tmp_path, request)
        model.export("onnx", onnx_precision="fp16")
        model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        trt_model = Model.from_dir(
            model.model_dir, runtime="tensorrt", onnx_precision="fp16"
        )
        with pytest.raises(RuntimeError, match="only supported for runtime='eager'"):
            trt_model.compile()
        assert not trt_model._compiled

    @requires_onnxruntime
    def test_tensorrt_fp32_matches_eager_predictions(
        self, tmp_path, request, toy_data_dir
    ):
        """TensorRT fp32 predictions match eager fp32 almost exactly.

        Isolates the export/engine-build pipeline's own correctness from
        precision-reduction noise: fp32 uses the same numerics as eager, so
        a large deviation here would mean a real bug, not fp16 rounding.
        """
        csv_file = Path(toy_data_dir) / "CollectedData.csv"
        eager_model = _setup_test_model(tmp_path / "eager", request)
        trt_source = _setup_test_model(tmp_path / "trt", request)
        trt_source.export("onnx", onnx_precision="fp32")
        trt_source.export("tensorrt", onnx_precision="fp32", max_batch_size=4)
        trt_model = Model.from_dir(
            trt_source.model_dir, runtime="tensorrt", onnx_precision="fp32"
        )
        eager_preds = eager_model.predict_on_label_csv(csv_file).predictions
        trt_preds = trt_model.predict_on_label_csv(csv_file).predictions
        xy_cols = [c for c in eager_preds.columns if c[-1] in ("x", "y")]
        deviation = np.abs(
            eager_preds[xy_cols].to_numpy(dtype=float)
            - trt_preds[xy_cols].to_numpy(dtype=float)
        )
        max_deviation = np.nanmax(deviation)
        assert max_deviation < 0.01, f"max pixel deviation {max_deviation:.4f} >= 0.01"

    @requires_onnxruntime
    def test_tensorrt_fp16_matches_eager_predictions(
        self, tmp_path, request, toy_data_dir
    ):
        """TensorRT fp16 mean deviation from eager fp32 stays small on confident keypoints.

        fp16 quantization can shift the heatmap argmax by several pixels on
        keypoints the model has no real opinion about (near-uniform heatmap,
        likelihood near the ~1/n_pixels noise floor). This peak-flipping
        happens under any precision or kernel change -- including the
        already-merged ONNX fp16 path on the full (non-toy) dataset -- and
        is not specific to TensorRT, so keypoints below likelihood 0.1 (no
        real peak) are excluded here.

        Even among the remaining confidently-tracked keypoints, TensorRT's
        kernel autotuning makes engine builds non-deterministic: repeated
        builds of the identical export measured max deviation anywhere from
        ~0.5px to ~2.0px, i.e. the max is not a stable statistic on this
        small (~18-value) toy sample. Mean deviation was stable across the
        same repeated builds (~0.2-0.3px), so this test asserts on the mean
        instead -- 1.0px leaves several-fold headroom while still catching
        real breakage (a broken export/engine blows up the mean, not just
        one keypoint's max -- see test_tensorrt_fp32_matches_eager_predictions
        for a tight, precision-independent correctness check).
        """
        csv_file = Path(toy_data_dir) / "CollectedData.csv"
        eager_model = _setup_test_model(tmp_path / "eager", request)
        trt_source = _setup_test_model(tmp_path / "trt", request)
        trt_source.export("onnx", onnx_precision="fp16")
        trt_source.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        trt_model = Model.from_dir(
            trt_source.model_dir, runtime="tensorrt", onnx_precision="fp16"
        )
        eager_preds = eager_model.predict_on_label_csv(csv_file).predictions
        trt_preds = trt_model.predict_on_label_csv(csv_file).predictions
        xy_cols = [c for c in eager_preds.columns if c[-1] in ("x", "y")]
        deviations = []
        for col in xy_cols:
            bp = col[-2]
            lik_col = [
                c
                for c in eager_preds.columns
                if c[-2] == bp and c[-1] == "likelihood"
            ][0]
            mask = eager_preds[lik_col] >= 0.1
            dev = (eager_preds.loc[mask, col] - trt_preds.loc[mask, col]).abs()
            deviations.extend(dev.tolist())
        deviations = np.array(deviations, dtype=float)
        assert len(deviations) > 0, "no confidently-tracked keypoints in toy data"
        mean_deviation = np.nanmean(deviations)
        assert mean_deviation < 1.0, f"mean pixel deviation {mean_deviation:.4f} >= 1.0"


def _make_mock_model(tmp_path, *, multiview=False, context=False, num_views=2):
    """A Model with mocked config and weights, usable without a GPU or onnxruntime.

    ``model`` is set directly so ``_load()`` short-circuits, letting the export
    and runtime code paths be exercised without a real checkpoint.
    """
    config = MagicMock()
    config.is_multi_view.return_value = multiview
    config.cfg.data.image_resize_dims.height = 128
    config.cfg.data.image_resize_dims.width = 128
    config.cfg.data.view_names = [f"view{i}" for i in range(num_views)]
    config.cfg.model.model_name = "test_model"

    model = Model(tmp_path, config)
    model.model = MagicMock()
    model.model.device = torch.device("cpu")
    model.model.do_context = context
    return model


def _fake_ort(providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
              input_type="tensor(float)", output=None):
    """Stand-in onnxruntime module whose InferenceSession returns a MagicMock.

    Lets the ONNX runtime code path be tested on machines without onnxruntime
    installed -- which includes CI, where the real dependency is absent and the
    @requires_onnxruntime tests above are skipped.
    """
    session = MagicMock()
    session.get_providers.return_value = list(providers)

    inp = MagicMock()
    inp.name = "images"
    inp.type = input_type
    session.get_inputs.return_value = [inp]

    out = MagicMock()
    out.name = "heatmaps"
    session.get_outputs.return_value = [out]

    if output is None:
        output = np.zeros((1, 2, 4, 4), dtype=np.float32)
    session.run.return_value = [output]

    module = MagicMock()
    module.InferenceSession.return_value = session
    return module, session


@pytest.fixture()
def mock_ckpt():
    """Pin checkpoint resolution to a fixed stem without touching the filesystem."""
    with patch(
        "lightning_pose.api.model.io_utils.ckpt_path_from_base_path",
        return_value="/some/dir/epoch=1-step=2-best.ckpt",
    ):
        yield


class TestExportUnit:
    """Export tests that need neither a GPU nor onnxruntime.

    The @requires_onnxruntime tests above are skipped wherever the optional
    dependency isn't installed, which leaves export() uncovered on CI. These
    mock torch.onnx.export instead so the logic around it stays covered.
    """

    def test_ckpt_stem_raises_when_no_checkpoint(self, tmp_path):
        """An untrained model directory gets a clear error, not a TypeError."""
        model = _make_mock_model(tmp_path)
        with (
            patch(
                "lightning_pose.api.model.io_utils.ckpt_path_from_base_path",
                return_value=None,
            ),
            pytest.raises(FileNotFoundError, match="Checkpoint file not found"),
        ):
            model._ckpt_stem()

    @pytest.mark.parametrize(
        "kwargs,expected_shape",
        [
            ({}, (1, 3, 128, 128)),
            ({"multiview": True}, (1, 2, 3, 128, 128)),
            ({"context": True}, (1, 5, 3, 128, 128)),
        ],
        ids=["singleview", "multiview", "context"],
    )
    def test_export_dummy_input_shape(self, tmp_path, mock_ckpt, kwargs, expected_shape):
        """Dummy input shape differs across singleview, multiview and context models."""
        model = _make_mock_model(tmp_path, **kwargs)
        with patch("torch.onnx.export") as mock_export:
            model.export("onnx", onnx_precision="fp32")
        dummy = mock_export.call_args.args[1][0]
        assert tuple(dummy.shape) == expected_shape

    def test_export_returns_path_under_exports_onnx_dir(self, tmp_path, mock_ckpt):
        """The returned path is {ckpt_stem}_{onnx_precision}.onnx inside exports_onnx/."""
        model = _make_mock_model(tmp_path)
        with patch("torch.onnx.export"):
            output_path = model.export("onnx", onnx_precision="fp32")
        assert output_path == model.exports_onnx_dir() / "epoch=1-step=2-best_fp32.onnx"
        assert model.exports_onnx_dir().is_dir()

    def test_export_fp16_traces_a_copy_not_the_live_model(self, tmp_path, mock_ckpt):
        """fp16 export halves a deepcopy, leaving the in-memory model alone."""
        model = _make_mock_model(tmp_path)
        original = model.model
        with patch("torch.onnx.export") as mock_export:
            model.export("onnx", onnx_precision="fp16")
        traced = mock_export.call_args.args[0]
        dummy = mock_export.call_args.args[1][0]
        assert dummy.dtype == torch.float16
        assert traced is not original
        assert model.model is original

    def test_export_fp32_traces_the_live_model(self, tmp_path, mock_ckpt):
        """fp32 export needs no copy, so it traces the module directly."""
        model = _make_mock_model(tmp_path)
        with patch("torch.onnx.export") as mock_export:
            model.export("onnx", onnx_precision="fp32")
        assert mock_export.call_args.args[0] is model.model
        assert mock_export.call_args.args[1][0].dtype == torch.float32

    def test_export_raises_when_model_fails_to_load(self, tmp_path, mock_ckpt):
        """A model that stays None after _load() raises rather than crashing later."""
        model = _make_mock_model(tmp_path)
        model.model = None
        with (
            patch.object(model, "_load"),
            pytest.raises(RuntimeError, match="model failed to load"),
        ):
            model.export("onnx")


class TestTensorRTExportUnit:
    """export('tensorrt', ...) tests that need neither a GPU nor onnxruntime."""

    def _prepare_onnx_export(self, model, precision="fp16"):
        """Create a stub .onnx file matching the mocked checkpoint stem."""
        model.exports_onnx_dir().mkdir(parents=True, exist_ok=True)
        path = model.exports_onnx_dir() / f"epoch=1-step=2-best_{precision}.onnx"
        path.touch()
        return path

    def test_export_tensorrt_requires_existing_onnx_export(self, tmp_path, mock_ckpt):
        """No matching .onnx export raises before importing onnxruntime."""
        model = _make_mock_model(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"Run model\.export\('onnx'"):
            model.export("tensorrt", onnx_precision="fp16")

    def test_export_tensorrt_reads_shape_from_onnx_graph(self, tmp_path, mock_ckpt):
        """The batch profile is built from the ONNX graph's own input shape,
        not self.cfg -- see _export_tensorrt's docstring for why.
        """
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        fake_module, probe_session = _fake_ort()
        fake_module.__version__ = "1.28.0"
        probe_session.get_inputs.return_value[0].shape = ["batch", 3, 256, 256]
        build_session = MagicMock()
        build_session.get_providers.return_value = ["TensorrtExecutionProvider"]
        fake_module.InferenceSession.side_effect = [probe_session, build_session]
        with patch.dict(sys.modules, {"onnxruntime": fake_module}):
            model.export(
                "tensorrt", onnx_precision="fp16", max_batch_size=4, opt_batch_size=2
            )
        build_kwargs = fake_module.InferenceSession.call_args_list[1]
        trt_options = build_kwargs.kwargs["providers"][0][1]
        assert trt_options["trt_profile_min_shapes"] == "images:1x3x256x256"
        assert trt_options["trt_profile_opt_shapes"] == "images:2x3x256x256"
        assert trt_options["trt_profile_max_shapes"] == "images:4x3x256x256"

    def test_export_tensorrt_writes_metadata(self, tmp_path, mock_ckpt):
        """trt_metadata.json is written with the expected keys."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        fake_module, probe_session = _fake_ort()
        fake_module.__version__ = "1.28.0"
        build_session = MagicMock()
        build_session.get_providers.return_value = ["TensorrtExecutionProvider"]
        fake_module.InferenceSession.side_effect = [probe_session, build_session]
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
        ):
            cache_dir = model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)
        metadata = json.loads((cache_dir / "trt_metadata.json").read_text())
        assert metadata["gpu_name"] == "Fake GPU"
        assert metadata["onnx_precision"] == "fp16"
        assert metadata["max_batch_size"] == 4
        assert metadata["opt_batch_size"] == 4

    def test_export_tensorrt_provider_guard_raises(self, tmp_path, mock_ckpt):
        """A build that doesn't engage TensorrtExecutionProvider raises."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        fake_module, probe_session = _fake_ort()
        build_session = MagicMock()
        build_session.get_providers.return_value = ["CUDAExecutionProvider"]
        fake_module.InferenceSession.side_effect = [probe_session, build_session]
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            pytest.raises(RuntimeError, match="TensorrtExecutionProvider"),
        ):
            model.export("tensorrt", onnx_precision="fp16", max_batch_size=4)


class TestOnnxRuntimeUnit:
    """Runtime-attach tests using a fake onnxruntime module.

    Same rationale as TestExportUnit: keeps _attach_onnx_runtime covered where
    the real dependency isn't installed.
    """

    def _prepare_export(self, model, precision="fp16"):
        """Create a stub .onnx file matching the mocked checkpoint stem."""
        model.exports_onnx_dir().mkdir(parents=True, exist_ok=True)
        path = model.exports_onnx_dir() / f"epoch=1-step=2-best_{precision}.onnx"
        path.touch()
        return path

    def test_attach_raises_when_no_export(self, tmp_path, mock_ckpt):
        """Missing export points at export(), without importing onnxruntime."""
        model = _make_mock_model(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"Run model\.export\('onnx'\) first"):
            model._attach_onnx_runtime(None)

    def test_attach_raises_when_multiple_exports(self, tmp_path, mock_ckpt):
        """Ambiguous exports raise instead of picking one arbitrarily."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model, "fp16")
        self._prepare_export(model, "fp32")
        with pytest.raises(ValueError, match="Multiple ONNX exports found"):
            model._attach_onnx_runtime(None)

    def test_attach_missing_dependency_explains_install(self, tmp_path, mock_ckpt):
        """A missing optional dependency explains how to install it."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model)
        with (
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="increasing_inference_speed"),
        ):
            model._attach_onnx_runtime(None)

    def test_attach_provider_guard_fires_on_cpu_fallback(self, tmp_path, mock_ckpt):
        """Losing CUDAExecutionProvider on a CUDA machine raises, not silently CPU."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model)
        fake_module, _ = _fake_ort(providers=["CPUExecutionProvider"])
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(RuntimeError, match="CUDAExecutionProvider"),
        ):
            model._attach_onnx_runtime(None)

    def test_attach_sets_runtime_and_rebinds_forward(self, tmp_path, mock_ckpt):
        """Successful attach marks _runtime and replaces forward."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model)
        assert model.model is not None
        original_forward = model.model.forward
        fake_module, _ = _fake_ort()
        with patch.dict(sys.modules, {"onnxruntime": fake_module}):
            model._attach_onnx_runtime(None)
        assert model._runtime == "onnx"
        assert model.model.forward is not original_forward

    def test_onnx_forward_runs_through_session(self, tmp_path, mock_ckpt):
        """The rebound forward feeds the session and returns a tensor."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model)
        fake_module, session = _fake_ort()
        with patch.dict(sys.modules, {"onnxruntime": fake_module}):
            model._attach_onnx_runtime(None)

        assert model.model is not None
        result = model.model.forward(torch.randn(1, 3, 128, 128))  # type: ignore[arg-type]
        session.run.assert_called_once()
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 2, 4, 4)

    def test_onnx_forward_casts_input_to_export_dtype(self, tmp_path, mock_ckpt):
        """An fp16 export receives fp16 buffers, not hardcoded fp32."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model)
        fake_module, session = _fake_ort(
            input_type="tensor(float16)",
            output=np.zeros((1, 2, 4, 4), dtype=np.float16),
        )
        with patch.dict(sys.modules, {"onnxruntime": fake_module}):
            model._attach_onnx_runtime(None)

        assert model.model is not None
        model.model.forward(torch.randn(1, 3, 128, 128))  # type: ignore[arg-type]
        fed = session.run.call_args.args[1]["images"]
        assert fed.dtype == np.float16

    def test_attach_selects_requested_precision(self, tmp_path, mock_ckpt):
        """onnx_precision picks a specific file when several exist."""
        model = _make_mock_model(tmp_path)
        self._prepare_export(model, "fp16")
        fp32_path = self._prepare_export(model, "fp32")
        fake_module, _ = _fake_ort()
        with patch.dict(sys.modules, {"onnxruntime": fake_module}):
            model._attach_onnx_runtime("fp32")
        assert fake_module.InferenceSession.call_args.args[0] == str(fp32_path)


class TestTensorRTRuntimeUnit:
    """runtime='tensorrt' attach tests using a fake onnxruntime module."""

    def _prepare_onnx_export(self, model, precision="fp16"):
        model.exports_onnx_dir().mkdir(parents=True, exist_ok=True)
        path = model.exports_onnx_dir() / f"epoch=1-step=2-best_{precision}.onnx"
        path.touch()
        return path

    def _prepare_trt_cache(self, model, precision="fp16", gpu_name="Fake GPU"):
        cache_dir = model.exports_trt_dir() / f"epoch=1-step=2-best_{precision}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "gpu_name": gpu_name,
            "tensorrt_version": "10.0.0",
            "onnxruntime_version": "1.28.0",
            "onnx_precision": precision,
            "max_batch_size": 4,
            "opt_batch_size": 4,
            "built_at": "2026-01-01T00:00:00+00:00",
        }
        (cache_dir / "trt_metadata.json").write_text(json.dumps(metadata))
        return cache_dir

    def test_attach_tensorrt_raises_without_cuda(self, tmp_path, mock_ckpt):
        """No CPU fallback for TensorRT."""
        model = _make_mock_model(tmp_path)
        with (
            patch("torch.cuda.is_available", return_value=False),
            pytest.raises(RuntimeError, match="requires a CUDA-capable GPU"),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_raises_when_no_cache(self, tmp_path, mock_ckpt):
        """Missing engine cache points at export('tensorrt', ...)."""
        model = _make_mock_model(tmp_path)
        with (
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(
                FileNotFoundError, match=r"Run model\.export\('tensorrt', \.\.\.\) first"
            ),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_raises_when_multiple_caches(self, tmp_path, mock_ckpt):
        """Ambiguous caches raise instead of picking one arbitrarily."""
        model = _make_mock_model(tmp_path)
        self._prepare_trt_cache(model, "fp16")
        self._prepare_trt_cache(model, "fp32")
        with (
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(ValueError, match="Multiple TensorRT exports found"),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_gpu_mismatch_warns(self, tmp_path, mock_ckpt):
        """A GPU-name mismatch warns rather than silently rebuilding unnoticed."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        self._prepare_trt_cache(model, gpu_name="Some Other GPU")
        fake_module, _ = _fake_ort(providers=["TensorrtExecutionProvider"])
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
            pytest.warns(UserWarning, match="was built on"),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_provider_guard_raises_on_cuda_fallback(self, tmp_path, mock_ckpt):
        """Landing on CUDAExecutionProvider instead of TensorRT raises, not warns.

        Unlike ONNX's CPU fallback, there is no acceptable degraded tier here.
        """
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        self._prepare_trt_cache(model)
        fake_module, _ = _fake_ort(providers=["CUDAExecutionProvider"])
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
            pytest.raises(RuntimeError, match="TensorrtExecutionProvider"),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_missing_dependency_explains_install(self, tmp_path, mock_ckpt):
        """A missing optional dependency explains how to install it."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        self._prepare_trt_cache(model)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
            patch.dict(sys.modules, {"onnxruntime": None}),
            pytest.raises(ImportError, match="increasing_inference_speed"),
        ):
            model._attach_tensorrt_runtime(None)

    def test_attach_tensorrt_sets_runtime_and_rebinds_forward(self, tmp_path, mock_ckpt):
        """Successful attach marks _runtime and replaces forward."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model)
        self._prepare_trt_cache(model)
        assert model.model is not None
        original_forward = model.model.forward
        fake_module, _ = _fake_ort(providers=["TensorrtExecutionProvider"])
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
        ):
            model._attach_tensorrt_runtime(None)
        assert model._runtime == "tensorrt"
        assert model.model.forward is not original_forward

    def test_attach_tensorrt_selects_requested_precision(self, tmp_path, mock_ckpt):
        """onnx_precision disambiguates when several caches exist."""
        model = _make_mock_model(tmp_path)
        self._prepare_onnx_export(model, "fp16")
        self._prepare_onnx_export(model, "fp32")
        self._prepare_trt_cache(model, "fp16")
        self._prepare_trt_cache(model, "fp32")
        fake_module, _ = _fake_ort(providers=["TensorrtExecutionProvider"])
        with (
            patch.dict(sys.modules, {"onnxruntime": fake_module}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
        ):
            model._attach_tensorrt_runtime("fp32")
        onnx_call = fake_module.InferenceSession.call_args
        assert "fp32" in onnx_call.args[0]
