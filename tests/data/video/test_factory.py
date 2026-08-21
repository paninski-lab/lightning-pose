"""Test the video reader factory (backend resolution + construction dispatch)."""
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lightning_pose.data.video.factory import build_video_reader


def _fake_dali_module(prepare_dali: MagicMock) -> MagicMock:
    """A stand-in for the real lightning_pose.data.video.dali module.

    Installed into sys.modules (rather than patching PrepareDALI via
    unittest.mock.patch's dotted-path target) so these tests don't require
    nvidia-dali to actually be importable in the test environment --
    unittest.mock.patch('lightning_pose.data.video.dali.PrepareDALI', ...) has to
    import the real module to resolve that target, which fails outright on a
    machine without nvidia-dali installed (its own top-level import re-raises).
    Mirrors the sys.modules-substitution trick test_pynvvc.py's TestIsPynvvcAvailable
    already uses for PyNvVideoCodec, for the same reason.
    """
    fake_module = MagicMock()
    fake_module.PrepareDALI = prepare_dali
    return fake_module


class TestBuildVideoReader:
    """Test the build_video_reader function.

    All tests mock at the boundary functions (is_pynvvc_available, is_opencv_available,
    dali's own importability) and the Prepare* classes themselves, same discipline as
    test_pynvvc.py's TestIsPynvvcAvailable -- deterministic regardless of what's
    actually installed in the test environment, and no real decoder/GPU/file I/O.
    """

    @pytest.fixture
    def dali_config(self):
        return {
            'base': {'predict': {'sequence_length': 8}},
            'context': {'predict': {'sequence_length': 9}},
        }

    # -- auto-select (reader=None) --

    def test_auto_select_picks_pynvvc_when_available(self, dali_config):
        mock_prep = MagicMock()
        with (
            patch('lightning_pose.data.video.pynvvc.is_pynvvc_available', return_value=True),
            patch('lightning_pose.data.video.pynvvc.PreparePynvvc', mock_prep),
        ):
            build_video_reader(
                None, '/fake/video.mp4', 'base', dali_config, ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    def test_auto_select_falls_back_to_dali_when_pynvvc_unavailable(self, dali_config):
        mock_prep = MagicMock()
        fake_module = {'lightning_pose.data.video.dali': _fake_dali_module(mock_prep)}
        with (
            patch('lightning_pose.data.video.pynvvc.is_pynvvc_available', return_value=False),
            patch.dict(sys.modules, fake_module),
        ):
            build_video_reader(
                None, '/fake/video.mp4', 'base', dali_config, ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    def test_auto_select_falls_back_to_opencv_when_neither_available(self, dali_config):
        """Simulates a machine with neither pynvvc usable nor dali installed at all
        (e.g. macOS/Windows) -- opencv must still be reachable."""
        mock_prep = MagicMock()
        with (
            patch('lightning_pose.data.video.pynvvc.is_pynvvc_available', return_value=False),
            patch.dict(sys.modules, {'lightning_pose.data.video.dali': None}),
            patch('lightning_pose.data.video.opencv.PrepareOpenCV', mock_prep),
        ):
            build_video_reader(
                None, '/fake/video.mp4', 'base', dali_config, ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    # -- explicit reader: validation + successful construction --

    def test_explicit_dali_raises_when_uninstalled(self, dali_config):
        with patch.dict(sys.modules, {'lightning_pose.data.video.dali': None}):
            with pytest.raises(RuntimeError, match="reader='dali'"):
                build_video_reader(
                    'dali', '/fake/video.mp4', 'base', dali_config,
                    ['/fake/video.mp4'], [64, 64], None,
                )

    def test_explicit_dali_succeeds_when_installed(self, dali_config):
        mock_prep = MagicMock()
        fake_module = {'lightning_pose.data.video.dali': _fake_dali_module(mock_prep)}
        with patch.dict(sys.modules, fake_module):
            build_video_reader(
                'dali', '/fake/video.mp4', 'base', dali_config,
                ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    def test_explicit_pynvvc_raises_when_unavailable(self, dali_config):
        with patch('lightning_pose.data.video.pynvvc.is_pynvvc_available', return_value=False):
            with pytest.raises(RuntimeError, match="reader='pynvvc'"):
                build_video_reader(
                    'pynvvc', '/fake/video.mp4', 'base', dali_config,
                    ['/fake/video.mp4'], [64, 64], None,
                )

    def test_explicit_pynvvc_succeeds_when_available(self, dali_config):
        mock_prep = MagicMock()
        with (
            patch('lightning_pose.data.video.pynvvc.is_pynvvc_available', return_value=True),
            patch('lightning_pose.data.video.pynvvc.PreparePynvvc', mock_prep),
        ):
            build_video_reader(
                'pynvvc', '/fake/video.mp4', 'base', dali_config,
                ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    def test_explicit_opencv_raises_when_unusable(self, dali_config):
        with patch('lightning_pose.data.video.opencv.is_opencv_available', return_value=False):
            with pytest.raises(RuntimeError, match="reader='opencv'"):
                build_video_reader(
                    'opencv', '/fake/video.mp4', 'base', dali_config,
                    ['/fake/video.mp4'], [64, 64], None,
                )

    def test_explicit_opencv_succeeds_when_usable(self, dali_config):
        mock_prep = MagicMock()
        with (
            patch('lightning_pose.data.video.opencv.is_opencv_available', return_value=True),
            patch('lightning_pose.data.video.opencv.PrepareOpenCV', mock_prep),
        ):
            build_video_reader(
                'opencv', '/fake/video.mp4', 'base', dali_config,
                ['/fake/video.mp4'], [64, 64], None,
            )
        assert mock_prep.called

    # -- construction args forwarded unchanged --

    def test_forwards_construction_args(self, dali_config):
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [10], 'w': [10]})
        mock_prep = MagicMock()
        with (
            patch('lightning_pose.data.video.opencv.is_opencv_available', return_value=True),
            patch('lightning_pose.data.video.opencv.PrepareOpenCV', mock_prep),
        ):
            build_video_reader(
                'opencv', '/fake/video.mp4', 'context', dali_config,
                ['/fake/video.mp4'], [128, 128], bbox_df,
            )
        call_kwargs = mock_prep.call_args.kwargs
        assert call_kwargs['model_type'] == 'context'
        assert call_kwargs['dali_config'] is dali_config
        assert call_kwargs['filenames'] == ['/fake/video.mp4']
        assert call_kwargs['resize_dims'] == [128, 128]
        assert call_kwargs['bbox_df'] is bbox_df
