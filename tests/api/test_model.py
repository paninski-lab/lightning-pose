import copy
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

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
        import onnxruntime  # noqa: F401
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
            model.export("tensorrt")
        assert model.model is None

    def test_export_rejects_unknown_onnx_precision(self, tmp_path, request):
        """export() rejects bf16, which ONNX export does not support."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(ValueError, match="Unsupported onnx_precision"):
            model.export("onnx", onnx_precision="bf16")
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


class TestOnnxRuntime:
    """Test loading a model with runtime='onnx'."""

    pytestmark = pytest.mark.gpu

    def test_from_dir_rejects_unknown_runtime(self, tmp_path, request):
        """from_dir() rejects a runtime other than 'eager' or 'onnx'."""
        model = _setup_test_model(tmp_path, request)
        with pytest.raises(ValueError, match="Unsupported runtime"):
            Model.from_dir(model.model_dir, runtime="tensorrt")

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
