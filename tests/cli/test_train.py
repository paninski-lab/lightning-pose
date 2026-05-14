"""Test the train CLI command argument parsing."""

import pytest


class TestTrainParser:
    """Test the train subcommand argument parsing."""

    def test_valid_config_file(self, parser, tmp_path):
        config = tmp_path / 'config.yaml'
        config.write_text('')
        args = parser.parse_args(['train', str(config)])
        assert args.config_file == config
        assert args.output_dir is None
        assert args.overrides is None
        assert args.detector_model is None

    def test_missing_config_file_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(['train', '/nonexistent/config.yaml'])

    def test_non_yaml_config_file_exits(self, parser, tmp_path):
        config = tmp_path / 'config.txt'
        config.write_text('')
        with pytest.raises(SystemExit):
            parser.parse_args(['train', str(config)])

    def test_output_dir(self, parser, tmp_path):
        config = tmp_path / 'config.yaml'
        config.write_text('')
        output_dir = tmp_path / 'output'
        args = parser.parse_args(['train', str(config), '--output_dir', str(output_dir)])
        assert args.output_dir == output_dir

    def test_overrides(self, parser, tmp_path):
        config = tmp_path / 'config.yaml'
        config.write_text('')
        args = parser.parse_args(
            ['train', str(config), '--overrides', 'model.backbone=resnet50', 'training.lr=1e-4']
        )
        assert args.overrides == ['model.backbone=resnet50', 'training.lr=1e-4']

    def test_detector_model_valid(self, parser, tmp_path):
        config = tmp_path / 'config.yaml'
        config.write_text('')
        detector = tmp_path / 'detector'
        detector.mkdir()
        args = parser.parse_args(['train', str(config), '--detector_model', str(detector)])
        assert args.detector_model == detector

    def test_detector_model_missing_exits(self, parser, tmp_path):
        config = tmp_path / 'config.yaml'
        config.write_text('')
        with pytest.raises(SystemExit):
            parser.parse_args(
                ['train', str(config), '--detector_model', str(tmp_path / 'missing')]
            )
