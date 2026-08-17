"""Test pynvvc dataloading functionality."""
import os
import shutil
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data import pynvvc as pynvvc_module
from lightning_pose.data.pynvvc import LitPynvvcWrapper, PreparePynvvc, is_pynvvc_available


class TestIsPynvvcAvailable:
    """Test the is_pynvvc_available function.

    All three tests mock at the sys.modules level rather than relying on whatever
    PyNvVideoCodec happens to be installed in the test environment (it IS installed on
    the T4 GPU studio but not necessarily elsewhere), so behavior is deterministic
    regardless of where these run.
    """

    def test_returns_true_when_decoder_constructs(self):
        mock_nvc = MagicMock()
        mock_nvc.SimpleDecoder.return_value = MagicMock()
        with patch.dict(sys.modules, {'PyNvVideoCodec': mock_nvc}):
            assert is_pynvvc_available('/fake/video.mp4') is True

    def test_returns_false_when_decoder_construction_raises(self):
        """Closes the gap flagged in the docstring: the 'fails safely' path was untested."""
        mock_nvc = MagicMock()
        mock_nvc.SimpleDecoder.side_effect = RuntimeError('unsupported GPU generation')
        with patch.dict(sys.modules, {'PyNvVideoCodec': mock_nvc}):
            assert is_pynvvc_available('/fake/video.mp4') is False

    def test_returns_false_when_package_not_installed(self):
        """sys.modules[name] = None forces the same ImportError a real missing install would."""
        with patch.dict(sys.modules, {'PyNvVideoCodec': None}):
            assert is_pynvvc_available('/fake/video.mp4') is False


class TestPreparePynvvc:
    """Test the PreparePynvvc class."""

    def test_single_view_raises_on_nonexistent_file(self, cfg, video_list):
        with pytest.raises(FileNotFoundError):
            PreparePynvvc(
                model_type='base',
                filenames=[video_list[0] + '_bad-id.mp4'],
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_single_view_raises_when_path_is_directory(self, cfg, video_list):
        with pytest.raises(FileNotFoundError):
            PreparePynvvc(
                model_type='base',
                filenames=[os.path.dirname(video_list[0])],
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_raises_on_more_than_one_video_per_view(self, cfg, video_list):
        """pynvvc has no multi-session-batching concept, unlike DALI's train pipeline."""
        vid = video_list[0]
        with pytest.raises(NotImplementedError, match='exactly one video per view'):
            PreparePynvvc(
                model_type='base',
                filenames=[[vid, vid]],
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_multiview_raises_on_unequal_frame_counts(
        self, cfg_multiview, video_list, tmp_path, monkeypatch,
    ):
        vid = video_list[0]
        vid_copy = str(tmp_path / 'view1.mp4')
        shutil.copy(vid, vid_copy)
        monkeypatch.setattr(
            pynvvc_module, 'count_frames', lambda p: {vid: 100, vid_copy: 90}[p],
        )
        with pytest.raises(ValueError, match='frame counts across views'):
            PreparePynvvc(
                model_type='base',
                filenames=[[vid], [vid_copy]],
                resize_dims=[256, 256],
                dali_config=cfg_multiview.dali,
            )

    def test_unknown_model_type_raises(self, cfg, video_list):
        with pytest.raises(ValueError, match='unknown model_type'):
            PreparePynvvc(
                model_type='bogus',  # type: ignore[arg-type]
                filenames=video_list,
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_num_iters_base(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(pynvvc_module, 'count_frames', lambda p: 100)
        cfg.dali.base.predict.sequence_length = 16
        prep = PreparePynvvc(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        assert prep.num_iters == 7  # ceil(100 / 16) = 7

    def test_num_iters_context(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(pynvvc_module, 'count_frames', lambda p: 100)
        cfg.dali.context.predict.sequence_length = 9
        prep = PreparePynvvc(
            model_type='context',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        # step = 9 - 4 = 5; ceil((100 - 9) / 5) + 1 = ceil(91 / 5) + 1 = 19 + 1 = 20
        assert prep.num_iters == 20

    def test_context_num_iters_raises_on_invalid_step(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(pynvvc_module, 'count_frames', lambda p: 100)
        cfg.dali.context.predict.sequence_length = 4  # step = 4 - 4 = 0
        prep = PreparePynvvc(
            model_type='context',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        with pytest.raises(ValueError, match='step cannot be 0'):
            _ = prep.num_iters

    def test_bbox_df_sets_decode_resize_dims_none(self, cfg, video_list):
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [10], 'w': [10]})
        prep = PreparePynvvc(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
            bbox_df=bbox_df,
        )
        assert prep._decode_resize_dims is None

    def test_no_bbox_df_sets_decode_resize_dims_to_resize_dims(self, cfg, video_list):
        prep = PreparePynvvc(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        assert prep._decode_resize_dims == [256, 256]

    def test_call_builds_wrapper_with_correct_params(self, cfg, video_list, monkeypatch):
        """Integration check: PreparePynvvc() actually produces a correctly-configured
        LitPynvvcWrapper, with the real PyNvVideoCodec import mocked out."""
        monkeypatch.setattr(pynvvc_module, 'count_frames', lambda p: 100)
        prep = PreparePynvvc(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        mock_nvc = MagicMock()
        fake_decoder = MagicMock()
        fake_decoder.__len__.return_value = 100
        mock_nvc.SimpleDecoder.return_value = fake_decoder
        with patch.dict(sys.modules, {'PyNvVideoCodec': mock_nvc}):
            wrapper = prep()
        assert isinstance(wrapper, LitPynvvcWrapper)
        assert wrapper.resize_dims == [256, 256]
        assert wrapper.sequence_length == prep.sequence_length
        assert wrapper.multiview is False
        assert wrapper.num_iters == prep.num_iters

    def test_multiview_synchronized_frames(self, cfg_multiview, video_list, monkeypatch):
        """Every view's decoder is read from the same shared cursor each iteration
        (see LitPynvvcWrapper.__next__), so per-view frame content stays
        synchronized across the whole video -- the pynvvc analogue of
        test_dali.py's TestPrepareDALI.test_multiview_synchronized_frames.

        Unlike DALI (independent per-view readers kept in lockstep only via a
        shared reader seed under random shuffle), pynvvc uses a single shared
        self._cursor for every decoder, so synchronization holds by construction
        today -- this test is a regression guard in case a future change ever
        gives each view its own cursor.

        Each mocked SimpleDecoder returns frames whose content encodes the frame
        index (not real decoded pixels), so cross-view content equality actually
        proves matching windows, not just matching shape.
        """
        monkeypatch.setattr(pynvvc_module, 'count_frames', lambda p: 100)
        num_views = 3
        vid = video_list[0]
        filenames = [[vid]] * num_views
        total_frames = 100
        cfg_multiview.dali.base.predict.sequence_length = 4

        def make_fake_decoder(*args, **kwargs):
            return _FakeSimpleDecoder(
                [torch.full((3, 8, 8), float(k)) for k in range(total_frames)]
            )

        mock_nvc = MagicMock()
        mock_nvc.SimpleDecoder.side_effect = make_fake_decoder

        prep = PreparePynvvc(
            model_type='base',
            filenames=filenames,
            resize_dims=[8, 8],
            dali_config=cfg_multiview.dali,
        )
        with patch.dict(sys.modules, {'PyNvVideoCodec': mock_nvc}):
            wrapper = prep()

        with patch('torch.cuda.current_stream'):
            for _ in range(4):
                batch = next(wrapper)
                frames = batch['frames']  # (seq_len, num_views, C, H, W)
                for view in range(1, num_views):
                    assert torch.equal(frames[:, 0], frames[:, view])
                # sanity: frames within a window are still distinct, not a
                # degenerate constant-value fake that would pass trivially
                assert not torch.equal(frames[0, 0], frames[-1, 0])


class _FakeSimpleDecoder:
    """Minimal fake matching the PyNvVideoCodec 2.1.0 access API."""

    def __init__(self, frames: list[torch.Tensor]) -> None:
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        if not isinstance(index, int):
            raise TypeError("SimpleDecoder batch reads require get_batch_frames_by_index()")
        return self.frames[index]

    def get_batch_frames_by_index(self, indices: list[int]) -> list[torch.Tensor]:
        return [self.frames[index] for index in indices]


def _make_fake_decoder(
    total_frames: int,
    h: int = 100,
    w: int = 120,
) -> _FakeSimpleDecoder:
    """Build a decoder with CHW CPU tensors that support DLPack conversion.

    torch.from_dlpack() accepts anything implementing __dlpack__, which torch.Tensor
    already does (including CPU tensors) -- so _read_window's real
    `torch.from_dlpack(f)` call works unmodified against this fake, no mocking needed
    for that part.
    """
    return _FakeSimpleDecoder(
        [torch.rand(3, h, w) for _ in range(total_frames)]
    )


class TestLitPynvvcWrapper:
    """Test the LitPynvvcWrapper class.

    All tests bypass __init__ (which needs a real/mocked PyNvVideoCodec import) via
    object.__new__, same discipline as test_dali.py's TestLitDaliWrapper. The one
    piece of real hardware LitPynvvcWrapper touches outside decoding itself --
    torch.cuda.current_stream().synchronize() inside __next__ -- is mocked out so
    these run on CPU-only machines too.
    """

    @pytest.fixture
    def bbox_df(self):
        """Sample bbox DataFrame with 10 rows (same values as test_dali.py's fixture)."""
        return pd.DataFrame({
            'x': [10] * 10,
            'y': [20] * 10,
            'h': [50] * 10,
            'w': [60] * 10,
        })

    def _make_wrapper(
        self,
        total_frames: int,
        resize_dims: list[int] | None = None,
        decode_resize_dims: list[int] | None = None,
        sequence_length: int = 4,
        do_context: bool = False,
        multiview: bool = False,
        bbox_df: pd.DataFrame | None = None,
        frame_idx: int = 0,
        num_views: int = 1,
        frame_h: int = 100,
        frame_w: int = 120,
    ) -> LitPynvvcWrapper:
        """Create a LitPynvvcWrapper without a real PyNvVideoCodec decoder."""
        wrapper = object.__new__(LitPynvvcWrapper)
        wrapper._decoders = [  # type: ignore[assignment]
            _make_fake_decoder(total_frames, frame_h, frame_w) for _ in range(num_views)
        ]
        wrapper._total_frames = total_frames
        # resize_dims is always a concrete list on the real class (unlike
        # decode_resize_dims, which is the actually-optional one) -- default to a
        # harmless placeholder when a test doesn't care about its value.
        wrapper.resize_dims = resize_dims if resize_dims is not None else [64, 64]
        wrapper.decode_resize_dims = decode_resize_dims
        wrapper.sequence_length = sequence_length
        wrapper.step = sequence_length - 4 if do_context else sequence_length
        wrapper.do_context = do_context
        wrapper.num_iters = 1000  # overridden per-test where StopIteration timing matters
        wrapper.multiview = multiview
        wrapper.bbox_df = bbox_df
        wrapper._cursor = 0
        wrapper._iters_done = 0
        wrapper._frame_idx = frame_idx
        return wrapper

    # -- _read_window --

    def test_read_window_full_batch_no_padding(self):
        wrapper = self._make_wrapper(total_frames=10, sequence_length=4)
        window = wrapper._read_window(wrapper._decoders[0])
        assert window.shape == (4, 3, 100, 120)

    def test_read_window_pads_last_partial_batch(self):
        wrapper = self._make_wrapper(total_frames=10, sequence_length=4)
        wrapper._cursor = 8  # only frames 8, 9 are real; need 2 padding frames
        window = wrapper._read_window(wrapper._decoders[0])
        assert window.shape == (4, 3, 100, 120)
        assert torch.equal(window[1], window[2])
        assert torch.equal(window[2], window[3])
        assert not torch.equal(window[0], window[1])

    def test_read_window_fully_out_of_bounds(self):
        """Closes the gap flagged in _read_window's docstring: cursor already past the
        end of the video (untested against the real SimpleDecoder API before this)."""
        wrapper = self._make_wrapper(total_frames=10, sequence_length=4)
        wrapper._cursor = 10  # nothing left to read
        window = wrapper._read_window(wrapper._decoders[0])
        assert window.shape == (4, 3, 100, 120)
        # every frame should be the single last real frame (index 9), repeated
        for i in range(1, 4):
            assert torch.equal(window[0], window[i])

    # -- _resize_normalize --

    def test_resize_normalize_no_resize_when_decode_resize_dims_none(self):
        wrapper = self._make_wrapper(total_frames=5, decode_resize_dims=None)
        frames = torch.randint(0, 256, (4, 3, 50, 60), dtype=torch.uint8).float()
        out = wrapper._resize_normalize(frames)
        assert out.shape == (4, 3, 50, 60)

    def test_resize_normalize_resizes_when_decode_resize_dims_set(self):
        wrapper = self._make_wrapper(total_frames=5, decode_resize_dims=[32, 32])
        frames = torch.rand(4, 3, 50, 60) * 255
        out = wrapper._resize_normalize(frames)
        assert out.shape == (4, 3, 32, 32)

    def test_resize_normalize_applies_imagenet_normalization(self):
        """Exact-value check against the same formula/order of operations the method
        itself uses, so this fails if the scale-then-normalize order ever changes."""
        wrapper = self._make_wrapper(total_frames=5, decode_resize_dims=None)
        frames = torch.full((1, 3, 4, 4), 127.5)
        out = wrapper._resize_normalize(frames)
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        expected = (127.5 / 255.0 - mean) / std
        assert torch.allclose(out, expected.expand_as(out), atol=1e-5)

    # -- __next__ --

    def test_next_output_frames_shape_no_bbox(self):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, decode_resize_dims=[64, 64],
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            batch = next(wrapper)
        assert batch['frames'].shape == (4, 3, 64, 64)
        assert batch['is_multiview'] is False
        assert batch['bbox'].shape == (4, 4)

    def test_next_bbox_crop_output_shape(self, bbox_df):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=bbox_df,
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            batch = next(wrapper)
        assert batch['frames'].shape == (4, 3, 64, 64)
        assert batch['is_multiview'] is False

    def test_next_advances_frame_idx_base(self, bbox_df):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=bbox_df, do_context=False,
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            next(wrapper)
        assert wrapper._frame_idx == 4

    def test_next_advances_frame_idx_context(self, bbox_df):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=5, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=bbox_df, do_context=True,
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            next(wrapper)
        assert wrapper._frame_idx == 1  # step = seq_len - 4 = 1

    def test_next_pads_bbox_rows_on_last_batch(self):
        """Mirrors test_dali.py's TestLitDaliWrapper.test_pads_last_partial_batch --
        same padding logic, ported to the pynvvc backend."""
        two_row_df = pd.DataFrame({
            'x': [10, 20], 'y': [10, 20], 'h': [50, 60], 'w': [50, 60],
        })
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=two_row_df, frame_idx=1,
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            batch = next(wrapper)
        assert batch['bbox'].shape == (4, 4)
        expected_last = torch.tensor([20, 20, 60, 60], dtype=torch.float32)
        for i in range(4):
            assert torch.allclose(batch['bbox'][i], expected_last)

    def test_next_stops_iteration_after_num_iters(self):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, decode_resize_dims=[32, 32],
        )
        wrapper.num_iters = 2
        with patch('torch.cuda.current_stream'):
            next(wrapper)
            next(wrapper)
            with pytest.raises(StopIteration):
                next(wrapper)

    def test_next_multiview_output_shape(self):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, decode_resize_dims=[64, 64],
            multiview=True, num_views=2,
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream'):
            batch = next(wrapper)
        assert batch['frames'].shape == (4, 2, 3, 64, 64)
        assert batch['is_multiview'] is True
        assert batch['bbox'].shape == (4, 2 * 4)
        assert batch['transforms'].shape == (2, 1, 1)

    def test_next_calls_cuda_stream_synchronize(self):
        """Regression guard for the documented 'safe but not optimal' CUDA sync fix
        (see module docstring) -- fails loudly if this guard is ever accidentally
        removed, since that would silently corrupt predictions rather than error."""
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, decode_resize_dims=[32, 32],
        )
        wrapper.num_iters = 1
        with patch('torch.cuda.current_stream') as mock_stream:
            next(wrapper)
        mock_stream.assert_called_once()
        mock_stream.return_value.synchronize.assert_called_once()
