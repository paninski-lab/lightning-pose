"""Test the export CLI command argument parsing."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.cli.commands.export import get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_export(self):
        """Returned parser has prog set to 'litpose export'."""
        assert get_parser().prog == 'litpose export'


class TestExportParser:
    """Test the export subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir)])
        assert args.model_dir == model_dir

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(['export', str(tmp_path / 'missing')])

    def test_runtime_default_is_onnx(self, parser, tmp_path):
        """--runtime defaults to onnx."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir)])
        assert args.runtime == 'onnx'

    def test_runtime_accepts_tensorrt(self, parser, tmp_path):
        """--runtime tensorrt is parsed."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir), '--runtime', 'tensorrt'])
        assert args.runtime == 'tensorrt'

    def test_runtime_rejects_unknown_value(self, parser, tmp_path):
        """A runtime other than onnx/tensorrt exits."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(['export', str(model_dir), '--runtime', 'coreml'])

    def test_onnx_precision_default_is_fp16(self, parser, tmp_path):
        """--onnx-precision defaults to fp16."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir)])
        assert args.onnx_precision == 'fp16'

    def test_onnx_precision_fp32(self, parser, tmp_path):
        """--onnx-precision fp32 is parsed into onnx_precision."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['export', str(model_dir), '--onnx-precision', 'fp32']
        )
        assert args.onnx_precision == 'fp32'

    def test_onnx_precision_rejects_bf16(self, parser, tmp_path):
        """bf16 is valid for `predict --precision` but not for ONNX export."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['export', str(model_dir), '--onnx-precision', 'bf16']
            )

    def test_max_batch_size_default_is_8(self, parser, tmp_path):
        """--max-batch-size defaults to 8."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir)])
        assert args.max_batch_size == 8

    def test_max_batch_size_custom(self, parser, tmp_path):
        """--max-batch-size is parsed as an int."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['export', str(model_dir), '--max-batch-size', '32']
        )
        assert args.max_batch_size == 32

    def test_opt_batch_size_default_is_none(self, parser, tmp_path):
        """--opt-batch-size defaults to None (falls back to max_batch_size in export())."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['export', str(model_dir)])
        assert args.opt_batch_size is None

    def test_opt_batch_size_custom(self, parser, tmp_path):
        """--opt-batch-size is parsed as an int."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['export', str(model_dir), '--opt-batch-size', '4']
        )
        assert args.opt_batch_size == 4


class TestHandle:
    """Test the handle function."""

    @pytest.fixture
    def mock_model(self):
        """Mock Model returned by Model.from_dir."""
        return MagicMock()

    def _make_args(
        self, tmp_path, runtime='onnx', onnx_precision='fp16',
        max_batch_size=8, opt_batch_size=None,
    ):
        return argparse.Namespace(
            model_dir=tmp_path / 'model',
            runtime=runtime,
            onnx_precision=onnx_precision,
            max_batch_size=max_batch_size,
            opt_batch_size=opt_batch_size,
        )

    def test_handle_calls_export(self, tmp_path, mock_model):
        """handle() forwards runtime, onnx_precision, and batch sizes to Model.export."""
        args = self._make_args(tmp_path)
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_model.export.assert_called_once_with(
            'onnx', onnx_precision='fp16', max_batch_size=8, opt_batch_size=None,
        )

    def test_handle_passes_fp32(self, tmp_path, mock_model):
        """handle() honors --onnx-precision fp32."""
        args = self._make_args(tmp_path, onnx_precision='fp32')
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_model.export.assert_called_once_with(
            'onnx', onnx_precision='fp32', max_batch_size=8, opt_batch_size=None,
        )

    def test_handle_passes_tensorrt_runtime(self, tmp_path, mock_model):
        """handle() forwards --runtime tensorrt to Model.export."""
        args = self._make_args(tmp_path, runtime='tensorrt')
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_model.export.assert_called_once_with(
            'tensorrt', onnx_precision='fp16', max_batch_size=8, opt_batch_size=None,
        )

    def test_handle_passes_custom_batch_sizes(self, tmp_path, mock_model):
        """handle() forwards --max-batch-size and --opt-batch-size to Model.export."""
        args = self._make_args(
            tmp_path, runtime='tensorrt', max_batch_size=32, opt_batch_size=16,
        )
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        mock_model.export.assert_called_once_with(
            'tensorrt', onnx_precision='fp16', max_batch_size=32, opt_batch_size=16,
        )

    def test_handle_loads_model_eagerly(self, tmp_path, mock_model):
        """handle() loads the model with the default eager runtime, not runtime=onnx.

        Exporting reads the trained checkpoint; loading through an ONNX session
        here would be circular.
        """
        args = self._make_args(tmp_path)
        with patch('lightning_pose.api.Model') as MockModel:
            MockModel.from_dir.return_value = mock_model
            handle(args)
        MockModel.from_dir.assert_called_once_with(args.model_dir)
