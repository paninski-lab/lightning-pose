"""Test the predict CLI command argument parsing."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from lightning_pose.cli.commands.predict import _predict_multi_type, get_parser, handle


class TestGetParser:
    """Test the get_parser function."""

    def test_returns_argument_parser(self):
        """Returns an ArgumentParser instance."""
        assert isinstance(get_parser(), argparse.ArgumentParser)

    def test_prog_is_litpose_predict(self):
        """Returned parser has prog set to 'litpose predict'."""
        assert get_parser().prog == 'litpose predict'


class TestPredictParser:
    """Test the predict subcommand argument parsing."""

    def test_valid_args(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video = tmp_path / 'video.mp4'
        args = parser.parse_args(['predict', str(model_dir), str(video)])
        assert args.model_dir == model_dir
        assert args.input_path == [video]
        assert not args.skip_viz
        assert not args.overwrite

    def test_multiple_input_paths(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        video1 = tmp_path / 'video1.mp4'
        video2 = tmp_path / 'video2.mp4'
        args = parser.parse_args(['predict', str(model_dir), str(video1), str(video2)])
        assert args.input_path == [video1, video2]

    def test_missing_model_dir_exits(self, parser, tmp_path):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['predict', str(tmp_path / 'missing'), str(tmp_path / 'video.mp4')]
            )

    def test_skip_viz_flag(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--skip_viz'])
        assert args.skip_viz

    def test_overwrite_flag(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--overwrite'])
        assert args.overwrite

    def test_compile_flag(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--compile'])
        assert args.compile


    def test_runtime_default_is_eager(self, parser, tmp_path):
        """--runtime defaults to eager."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.runtime == 'eager'

    def test_runtime_onnx(self, parser, tmp_path):
        """--runtime onnx is parsed."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['predict', str(model_dir), 'video.mp4', '--runtime', 'onnx']
        )
        assert args.runtime == 'onnx'

    def test_runtime_tensorrt(self, parser, tmp_path):
        """--runtime tensorrt is parsed."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['predict', str(model_dir), 'video.mp4', '--runtime', 'tensorrt']
        )
        assert args.runtime == 'tensorrt'

    def test_runtime_rejects_unknown_value(self, parser, tmp_path):
        """A runtime other than eager/onnx/tensorrt exits."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['predict', str(model_dir), 'video.mp4', '--runtime', 'coreml']
            )

    def test_decoder_default_is_none(self, parser, tmp_path):
        """--decoder defaults to None so predict_video() auto-selects."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.decoder is None

    def test_decoder_pynvvc(self, parser, tmp_path):
        """--decoder pynvvc is parsed."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['predict', str(model_dir), 'video.mp4', '--decoder', 'pynvvc']
        )
        assert args.decoder == 'pynvvc'

    def test_decoder_dali(self, parser, tmp_path):
        """--decoder dali is parsed."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['predict', str(model_dir), 'video.mp4', '--decoder', 'dali']
        )
        assert args.decoder == 'dali'

    def test_decoder_rejects_unknown_value(self, parser, tmp_path):
        """--decoder nvdec exits; only dali/pynvvc are valid."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['predict', str(model_dir), 'video.mp4', '--decoder', 'nvdec']
            )

    def test_onnx_precision_default_is_none(self, parser, tmp_path):
        """--onnx-precision defaults to None so a sole export is auto-selected."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.onnx_precision is None

    def test_onnx_precision_arg(self, parser, tmp_path):
        """--onnx-precision is parsed into onnx_precision."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args([
            'predict', str(model_dir), 'video.mp4',
            '--runtime', 'onnx', '--onnx-precision', 'fp16',
        ])
        assert args.onnx_precision == 'fp16'

    def test_onnx_precision_rejects_bf16(self, parser, tmp_path):
        """bf16 is valid for --precision but not for --onnx-precision."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        with pytest.raises(SystemExit):
            parser.parse_args([
                'predict', str(model_dir), 'video.mp4', '--onnx-precision', 'bf16',
            ])

    def test_overrides(self, parser, tmp_path):
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        override = 'dali.base.predict.batch_size=4'
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4', '--overrides', override])
        assert args.overrides == [override]

    def test_bbox_dir_default_is_none(self, parser, tmp_path):
        """bbox_dir defaults to None when not provided."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.bbox_dir is None

    def test_bbox_dir_arg(self, parser, tmp_path):
        """--bbox_dir is parsed as a Path."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        bbox_dir = tmp_path / 'bboxes'
        args = parser.parse_args([
            'predict', str(model_dir), 'labels.csv', '--bbox_dir', str(bbox_dir),
        ])
        assert args.bbox_dir == bbox_dir

    def test_batch_size_default_is_none(self, parser, tmp_path):
        """--batch_size defaults to None when not provided."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(['predict', str(model_dir), 'video.mp4'])
        assert args.batch_size is None

    def test_batch_size_arg(self, parser, tmp_path):
        """--batch_size is parsed as an int."""
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        args = parser.parse_args(
            ['predict', str(model_dir), 'video.mp4', '--batch_size', '8']
        )
        assert args.batch_size == 8


class TestPredictMultiType:
    """Test _predict_multi_type bbox threading."""

    @pytest.fixture
    def mock_model(self, tmp_path):
        """Mock Model with path methods returning real paths."""
        model = MagicMock()
        model.image_preds_dir.return_value = tmp_path / 'image_preds'
        model.video_preds_dir.return_value = tmp_path / 'video_preds'
        return model

    def test_csv_passes_bbox_file_when_bbox_dir_given(self, tmp_path, mock_model):
        """CSV input with bbox_dir passes <bbox_dir>/bbox.csv to predict_on_label_csv."""
        bbox_dir = tmp_path / 'bboxes'
        csv_path = tmp_path / 'labels.csv'
        _predict_multi_type(
            mock_model, csv_path, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_label_csv.assert_called_once()
        call_kwargs = mock_model.predict_on_label_csv.call_args.kwargs
        assert call_kwargs['bbox_file'] == bbox_dir / 'bbox.csv'

    def test_csv_passes_none_bbox_file_when_no_bbox_dir(self, tmp_path, mock_model):
        """CSV input without bbox_dir passes bbox_file=None to predict_on_label_csv."""
        csv_path = tmp_path / 'labels.csv'
        _predict_multi_type(
            mock_model, csv_path, skip_viz=True, skip_existing=False, bbox_dir=None,
        )
        call_kwargs = mock_model.predict_on_label_csv.call_args.kwargs
        assert call_kwargs['bbox_file'] is None

    def test_mp4_passes_bbox_file_when_bbox_dir_given(self, tmp_path, mock_model):
        """MP4 input with bbox_dir passes <bbox_dir>/<stem>_bbox.csv to predict_on_video_file."""
        bbox_dir = tmp_path / 'bboxes'
        video_path = tmp_path / 'vid.mp4'
        _predict_multi_type(
            mock_model, video_path, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_video_file.assert_called_once()
        call_kwargs = mock_model.predict_on_video_file.call_args.kwargs
        assert call_kwargs['bbox_file'] == bbox_dir / 'vid_bbox.csv'

    def test_mp4_passes_none_bbox_file_when_no_bbox_dir(self, tmp_path, mock_model):
        """MP4 input without bbox_dir passes bbox_file=None to predict_on_video_file."""
        video_path = tmp_path / 'vid.mp4'
        _predict_multi_type(
            mock_model, video_path, skip_viz=True, skip_existing=False, bbox_dir=None,
        )
        call_kwargs = mock_model.predict_on_video_file.call_args.kwargs
        assert call_kwargs['bbox_file'] is None

    def test_directory_input_with_bbox_dir_recurses_into_mp4s(self, tmp_path, mock_model):
        """A directory input with bbox_dir passes through to each recursive mp4 call."""
        video_dir = tmp_path / 'videos'
        video_dir.mkdir()
        (video_dir / 'a.mp4').touch()
        bbox_dir = tmp_path / 'bboxes'
        _predict_multi_type(
            mock_model, video_dir, skip_viz=True, skip_existing=False, bbox_dir=bbox_dir,
        )
        mock_model.predict_on_video_file.assert_called_once()


class TestHandle:
    """Test the handle function."""

    @pytest.fixture
    def mock_model(self):
        """Mock Model returned by Model.from_dir2."""
        model = MagicMock()
        model.config.is_multi_view.return_value = False
        return model

    def _make_args(self, tmp_path, video, bbox_dir=None):
        return argparse.Namespace(
            model_dir=tmp_path / 'model',
            input_path=[video],
            skip_viz=False,
            overwrite=False,
            overrides=None,
            progress_file=None,
            bbox_dir=bbox_dir,
            precision='fp32',
            compile=False,
            runtime='eager',
            onnx_precision=None,
            decoder=None,
            batch_size=None,
        )

    def test_handle_threads_bbox_dir_to_predict_multi_type(self, tmp_path, mock_model):
        """handle() forwards args.bbox_dir to _predict_multi_type."""
        bbox_dir = tmp_path / 'bboxes'
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', bbox_dir=bbox_dir)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch(
                'lightning_pose.cli.commands.predict._predict_multi_type',
            ) as mock_predict,
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        assert mock_predict.call_args.kwargs['bbox_dir'] == bbox_dir

    def test_handle_threads_none_bbox_dir_when_not_provided(self, tmp_path, mock_model):
        """handle() passes bbox_dir=None to _predict_multi_type when not provided."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4', bbox_dir=None)
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch(
                'lightning_pose.cli.commands.predict._predict_multi_type',
            ) as mock_predict,
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        assert mock_predict.call_args.kwargs['bbox_dir'] is None

    def test_handle_compiles_model_when_flag_set(self, tmp_path, mock_model):
        """handle() calls model.compile() when --compile is passed."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.compile = True
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        mock_model.compile.assert_called_once_with()

    def test_handle_does_not_compile_by_default(self, tmp_path, mock_model):
        """handle() leaves the model uncompiled when --compile is not passed."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        mock_model.compile.assert_not_called()

    def test_handle_compiles_multiview_model(self, tmp_path, mock_model):
        """--compile is applied before the multiview branch, so it covers both paths."""
        mock_model.config.is_multi_view.return_value = True
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.compile = True
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch(
                'lightning_pose.cli.commands.predict._predict_multi_type_multi_view',
            ),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        mock_model.compile.assert_called_once_with()

    def test_handle_threads_runtime_to_from_dir2(self, tmp_path, mock_model):
        """handle() forwards --runtime and --onnx-precision to Model.from_dir2."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.runtime = 'onnx'
        args.onnx_precision = 'fp16'
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['runtime'] == 'onnx'
        assert call_kwargs['onnx_precision'] == 'fp16'

    def test_handle_defaults_to_eager_runtime(self, tmp_path, mock_model):
        """handle() passes runtime='eager' and onnx_precision=None by default."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['runtime'] == 'eager'
        assert call_kwargs['onnx_precision'] is None

    def test_handle_rejects_compile_with_onnx_runtime(self, tmp_path, mock_model):
        """--compile with --runtime onnx fails before the model is constructed.

        Checked in the CLI so the user sees this message rather than the
        equivalent RuntimeError from Model.compile(), which would surface as a
        raw traceback after the ONNX session had already been built.
        """
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.compile = True
        args.runtime = 'onnx'
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            with pytest.raises(ValueError, match='only supported with --runtime eager'):
                handle(args)
        MockModel.from_dir2.assert_not_called()

    def test_handle_rejects_compile_with_tensorrt_runtime(self, tmp_path, mock_model):
        """--compile with --runtime tensorrt fails the same way as --runtime onnx."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.compile = True
        args.runtime = 'tensorrt'
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            with pytest.raises(ValueError, match='only supported with --runtime eager'):
                handle(args)
        MockModel.from_dir2.assert_not_called()

    def test_handle_builds_overrides_from_batch_size(self, tmp_path, mock_model):
        """--batch_size expands into the three relevant hydra overrides."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.batch_size = 8
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['hydra_overrides'] == [
            'training.val_batch_size=8',
            'dali.base.predict.sequence_length=8',
            'dali.context.predict.sequence_length=8',
        ]

    def test_handle_combines_batch_size_with_explicit_overrides(self, tmp_path, mock_model):
        """--batch_size overrides are appended after any --overrides the user passed."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.overrides = ['model.model_type=heatmap']
        args.batch_size = 4
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['hydra_overrides'] == [
            'model.model_type=heatmap',
            'training.val_batch_size=4',
            'dali.base.predict.sequence_length=4',
            'dali.context.predict.sequence_length=4',
        ]

    def test_handle_passes_none_overrides_when_batch_size_not_set(self, tmp_path, mock_model):
        """hydra_overrides stays None when neither --overrides nor --batch_size is given."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['hydra_overrides'] is None

    def test_handle_threads_tensorrt_runtime_to_from_dir2(self, tmp_path, mock_model):
        """handle() forwards --runtime tensorrt to Model.from_dir2 same as onnx."""
        args = self._make_args(tmp_path, tmp_path / 'vid.mp4')
        args.runtime = 'tensorrt'
        args.onnx_precision = 'fp16'
        with (
            patch('lightning_pose.api.Model') as MockModel,
            patch('lightning_pose.cli.commands.predict._predict_multi_type'),
        ):
            MockModel.from_dir2.return_value = mock_model
            handle(args)
        call_kwargs = MockModel.from_dir2.call_args.kwargs
        assert call_kwargs['runtime'] == 'tensorrt'
        assert call_kwargs['onnx_precision'] == 'fp16'
