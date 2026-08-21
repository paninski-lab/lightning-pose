"""Test opencv dataloading functionality."""
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

import lightning_pose as lp
from lightning_pose.data import _IMAGENET_MEAN, _IMAGENET_STD
from lightning_pose.data.video import opencv as opencv_module
from lightning_pose.data.video.opencv import (
    LitOpenCVWrapper,
    PrepareOpenCV,
    is_opencv_available,
)

# real toy video used by the real-decode tests below -- opencv-python-headless is an
# unconditional dependency (see module docstring), so unlike pynvvc's requires_pynvvc
# gate, no skip marker is needed: these run on every CI runner, CPU included.
_TOY_VIDEO = str(lp.LP_ROOT_PATH / "data" / "mirror-mouse-example" / "videos" / "test_vid.mp4")


class TestIsOpenCVAvailable:
    """Test the is_opencv_available function."""

    def test_returns_true_for_real_video(self):
        assert is_opencv_available(_TOY_VIDEO) is True

    def test_returns_false_for_nonexistent_file(self):
        assert is_opencv_available('/does/not/exist.mp4') is False

    def test_returns_false_for_non_video_file(self, tmp_path):
        bad_file = tmp_path / 'not_a_video.mp4'
        bad_file.write_text('this is not video data')
        assert is_opencv_available(str(bad_file)) is False


class TestPrepareOpenCV:
    """Test the PrepareOpenCV class."""

    def test_single_view_raises_on_nonexistent_file(self, cfg, video_list):
        with pytest.raises(FileNotFoundError):
            PrepareOpenCV(
                model_type='base',
                filenames=[video_list[0] + '_bad-id.mp4'],
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_single_view_raises_when_path_is_directory(self, cfg, video_list):
        with pytest.raises(FileNotFoundError):
            PrepareOpenCV(
                model_type='base',
                filenames=[os.path.dirname(video_list[0])],
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_raises_on_more_than_one_video_per_view(self, cfg, video_list):
        """opencv has no multi-session-batching concept, same as pynvvc."""
        vid = video_list[0]
        with pytest.raises(NotImplementedError, match='exactly one video per view'):
            PrepareOpenCV(
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
            opencv_module, 'count_frames', lambda p: {vid: 100, vid_copy: 90}[p],
        )
        with pytest.raises(ValueError, match='frame counts across views'):
            PrepareOpenCV(
                model_type='base',
                filenames=[[vid], [vid_copy]],
                resize_dims=[256, 256],
                dali_config=cfg_multiview.dali,
            )

    def test_unknown_model_type_raises(self, cfg, video_list):
        with pytest.raises(ValueError, match='unknown model_type'):
            PrepareOpenCV(
                model_type='bogus',  # type: ignore[arg-type]
                filenames=video_list,
                resize_dims=[256, 256],
                dali_config=cfg.dali,
            )

    def test_num_iters_base(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(opencv_module, 'count_frames', lambda p: 100)
        cfg.dali.base.predict.sequence_length = 16
        prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        assert prep.num_iters == 7  # ceil(100 / 16) = 7

    def test_num_iters_context(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(opencv_module, 'count_frames', lambda p: 100)
        cfg.dali.context.predict.sequence_length = 9
        prep = PrepareOpenCV(
            model_type='context',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        # step = 9 - 4 = 5; ceil((100 - 9) / 5) + 1 = ceil(91 / 5) + 1 = 19 + 1 = 20
        assert prep.num_iters == 20

    def test_context_num_iters_raises_on_invalid_step(self, cfg, video_list, monkeypatch):
        monkeypatch.setattr(opencv_module, 'count_frames', lambda p: 100)
        cfg.dali.context.predict.sequence_length = 4  # step = 4 - 4 = 0
        prep = PrepareOpenCV(
            model_type='context',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        with pytest.raises(ValueError, match='step cannot be 0'):
            _ = prep.num_iters

    def test_bbox_df_sets_decode_resize_dims_none(self, cfg, video_list):
        bbox_df = pd.DataFrame({'x': [0], 'y': [0], 'h': [10], 'w': [10]})
        prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
            bbox_df=bbox_df,
        )
        assert prep._decode_resize_dims is None

    def test_no_bbox_df_sets_decode_resize_dims_to_resize_dims(self, cfg, video_list):
        prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        assert prep._decode_resize_dims == [256, 256]

    def test_call_builds_wrapper_with_correct_params(self, cfg, video_list, monkeypatch):
        """Integration check: PrepareOpenCV() actually produces a correctly-configured
        LitOpenCVWrapper. Unlike pynvvc/dali, no mocking is needed here -- cv2.VideoCapture
        against the real toy video is cheap and requires no proprietary package/hardware."""
        prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[256, 256],
            dali_config=cfg.dali,
        )
        wrapper = prep()
        assert isinstance(wrapper, LitOpenCVWrapper)
        assert wrapper.resize_dims == [256, 256]
        assert wrapper.sequence_length == prep.sequence_length
        assert wrapper.multiview is False
        assert wrapper.num_iters == prep.num_iters


class _FakeCapture:
    """Minimal fake matching the cv2.VideoCapture.read() interface used by
    LitOpenCVWrapper -- lets tests exercise the ring-buffer/padding logic without
    depending on the real toy video's specific frame count or content."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self._idx = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._idx >= len(self.frames):
            return False, None
        frame = self.frames[self._idx]
        self._idx += 1
        return True, frame


def _make_fake_frames(n: int, h: int = 20, w: int = 24) -> list[np.ndarray]:
    """HWC BGR uint8 frames, each filled with a distinct value so identity (not just
    shape) can be checked -- e.g. that overlap frames across windows really are the
    same physical frame, not just the same shape."""
    return [np.full((h, w, 3), fill_value=i, dtype=np.uint8) for i in range(n)]


class TestLitOpenCVWrapper:
    """Test the LitOpenCVWrapper class.

    All tests bypass __init__ (which needs real cv2.VideoCapture handles) via
    object.__new__, same discipline as test_pynvvc.py's TestLitPynvvcWrapper --
    swapping in _FakeCapture instances for self._caps.
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
    ) -> LitOpenCVWrapper:
        """Create a LitOpenCVWrapper without real cv2.VideoCapture handles."""
        wrapper = object.__new__(LitOpenCVWrapper)
        wrapper._caps = [  # type: ignore[assignment]
            _FakeCapture(_make_fake_frames(total_frames, frame_h, frame_w))
            for _ in range(num_views)
        ]
        wrapper._tail = [[] for _ in range(num_views)]
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
        wrapper._iters_done = 0
        wrapper._frame_idx = frame_idx
        return wrapper

    # -- _next_window --

    def test_next_window_full_batch_no_padding(self):
        wrapper = self._make_wrapper(total_frames=10, sequence_length=4)
        window = wrapper._next_window(0)
        assert window.shape == (4, 3, 100, 120)

    def test_next_window_pads_last_partial_batch(self):
        wrapper = self._make_wrapper(total_frames=6, sequence_length=4)
        wrapper._next_window(0)  # consume frames 0-3
        window = wrapper._next_window(0)  # only frames 4, 5 remain; need 2 padding frames
        assert window.shape == (4, 3, 100, 120)
        assert torch.equal(window[1], window[2])
        assert torch.equal(window[2], window[3])
        assert not torch.equal(window[0], window[1])

    def test_next_window_fully_out_of_bounds_raises(self):
        """Unlike pynvvc (which can re-fetch the last frame by index), opencv can't
        seek backward once the capture is exhausted -- an empty window (tail empty,
        capture exhausted) is a real invariant violation, not a case to pad through."""
        wrapper = self._make_wrapper(total_frames=0, sequence_length=4)
        with pytest.raises(RuntimeError, match='exhausted view 0'):
            wrapper._next_window(0)

    def test_next_window_context_caches_tail(self):
        wrapper = self._make_wrapper(total_frames=20, sequence_length=5, do_context=True)
        wrapper._next_window(0)
        assert len(wrapper._tail[0]) == 4

    def test_next_window_context_overlap_matches_previous_tail(self):
        """The core correctness property of the ring-buffer design: window i's last 4
        frames are byte-identical to window i+1's first 4 frames."""
        wrapper = self._make_wrapper(total_frames=20, sequence_length=5, do_context=True)
        w1 = wrapper._next_window(0)
        w2 = wrapper._next_window(0)
        assert torch.equal(w1[-4:], w2[:4])

    def test_next_window_base_model_has_no_overlap(self):
        wrapper = self._make_wrapper(total_frames=20, sequence_length=4, do_context=False)
        w1 = wrapper._next_window(0)
        w2 = wrapper._next_window(0)
        assert not torch.equal(w1[-1], w2[0])
        assert wrapper._tail[0] == []

    def test_next_window_converts_bgr_to_rgb(self):
        """frames are stored BGR (channel order [B, G, R] with distinct per-channel
        values); _next_window must return RGB ([R, G, B])."""
        wrapper = object.__new__(LitOpenCVWrapper)
        bgr_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr_frame[..., 0] = 10   # B
        bgr_frame[..., 1] = 20   # G
        bgr_frame[..., 2] = 30   # R
        wrapper._caps = [_FakeCapture([bgr_frame])]  # type: ignore[assignment]
        wrapper._tail = [[]]
        wrapper.sequence_length = 1
        wrapper.do_context = False

        window = wrapper._next_window(0)  # (1, C, H, W)
        assert window[0, 0].unique().item() == 30  # R channel first
        assert window[0, 1].unique().item() == 20  # G channel second
        assert window[0, 2].unique().item() == 10  # B channel last

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
        batch = next(wrapper)
        assert batch['frames'].shape == (4, 3, 64, 64)
        assert batch['is_multiview'] is False

    def test_next_advances_frame_idx_base(self, bbox_df):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=bbox_df, do_context=False,
        )
        wrapper.num_iters = 1
        next(wrapper)
        assert wrapper._frame_idx == 4

    def test_next_advances_frame_idx_context(self, bbox_df):
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=5, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=bbox_df, do_context=True,
        )
        wrapper.num_iters = 1
        next(wrapper)
        assert wrapper._frame_idx == 1  # step = seq_len - 4 = 1

    def test_next_pads_bbox_rows_on_last_batch(self):
        """Mirrors test_pynvvc.py's equivalent -- same padding logic, ported here."""
        two_row_df = pd.DataFrame({
            'x': [10, 20], 'y': [10, 20], 'h': [50, 60], 'w': [50, 60],
        })
        wrapper = self._make_wrapper(
            total_frames=20, sequence_length=4, resize_dims=[64, 64],
            decode_resize_dims=None, bbox_df=two_row_df, frame_idx=1,
        )
        wrapper.num_iters = 1
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
        batch = next(wrapper)
        assert batch['frames'].shape == (4, 2, 3, 64, 64)
        assert batch['is_multiview'] is True
        assert batch['bbox'].shape == (4, 2 * 4)
        assert batch['transforms'].shape == (2, 1, 1)


class TestOpenCVRealDecode:
    """Real, unmocked cv2.VideoCapture decode tests against the real toy video.

    Unlike pynvvc's TestPynvvcRealDecoder, none of this needs @pytest.mark.gpu --
    opencv decode needs no GPU, so these run on the CPU CI workflow directly.
    """

    def test_single_view_base_real_decode(self, cfg, video_list):
        """Base model predict loader yields correctly-shaped, finite batches from a real
        decoder, exercising every batch including the padded final one."""
        im_height, im_width = 64, 64
        prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[im_height, im_width],
            dali_config=cfg.dali,
        )
        loader = prep()
        num_iters = prep.num_iters

        batch_idx = -1
        for batch in loader:
            assert batch['frames'].shape == (
                cfg.dali.base.predict.sequence_length, 3, im_height, im_width,
            )
            assert torch.isfinite(batch['frames']).all()
            batch_idx += 1
        assert batch_idx == num_iters - 1

    def test_single_view_context_real_decode(self, cfg, video_list):
        """Context model predict loader handles the overlapping-window step correctly
        against a real decoder -- the trickiest path in this module, previously only
        covered by mocks and the Part 3 scratch verification."""
        im_height, im_width = 64, 64
        prep = PrepareOpenCV(
            model_type='context',
            filenames=video_list,
            resize_dims=[im_height, im_width],
            dali_config=cfg.dali,
        )
        loader = prep()
        num_iters = prep.num_iters

        batch_idx = -1
        for batch in loader:
            assert batch['frames'].shape == (
                cfg.dali.context.predict.sequence_length, 3, im_height, im_width,
            )
            batch_idx += 1
        assert batch_idx == num_iters - 1

    @pytest.mark.gpu
    def test_matches_dali_decode(self, cfg, video_list):
        """opencv's decode + resize + normalize output roughly matches DALI's on the
        same real video -- catches a wrong color order, a wrong normalization
        order/scale, or an off-by-one in the frame window that a shape-only check
        would miss. GPU-marked because the DALI side of the comparison needs one, not
        because opencv does.

        See test_pynvvc.py's test_matches_dali_decode for why this asserts on the
        mean deviation (loosely bounding the max as a backstop) rather than a strict
        max-deviation threshold: two independent decode/resize paths legitimately
        disagree at hard edges.
        """
        pytest.importorskip('nvidia.dali', reason='nvidia-dali not installed')
        from lightning_pose.data.video.dali import PrepareDALI

        im_height, im_width = 256, 256

        opencv_prep = PrepareOpenCV(
            model_type='base',
            filenames=video_list,
            resize_dims=[im_height, im_width],
            dali_config=cfg.dali,
        )
        opencv_batch = next(iter(opencv_prep()))

        dali_prep = PrepareDALI(
            train_stage='predict',
            model_type='base',
            filenames=video_list,
            dali_config=cfg.dali,
            resize_dims=[im_height, im_width],
        )
        dali_batch = next(iter(dali_prep()))

        opencv_frames = opencv_batch['frames'].cpu()
        dali_frames = dali_batch['frames'].cpu()
        assert opencv_frames.shape == dali_frames.shape
        deviation = (opencv_frames - dali_frames).abs()
        mean_deviation = deviation.mean().item()
        assert mean_deviation < 0.05, (
            f"opencv decode deviates from DALI decode by a mean of {mean_deviation:.4f} "
            "(normalized units) -- check BGR->RGB color order and resize/normalize order "
            "in LitOpenCVWrapper._resize_normalize"
        )
        max_deviation = deviation.max().item()
        assert max_deviation < 3.0, (
            f"opencv decode deviates from DALI decode by up to {max_deviation:.4f} "
            "(normalized units) even after allowing for per-pixel edge disagreement -- "
            "this is large enough to suggest a real decode bug, not resize-kernel noise"
        )
