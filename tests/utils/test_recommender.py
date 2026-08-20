"""Test the recommender module."""

from pathlib import Path

import pytest
from PIL import Image

from lightning_pose.utils.recommender import (
    DatasetAnalysis,
    GpuInfo,
    _apply_min_iterations_floor,
    _derive_view_names,
    _recommend_resize_dim,
    _select_batch_size,
    analyze_dataset,
    build_config,
    format_report,
    recommend,
)


def _write_label_csv(
    csv_path: Path,
    keypoint_names: list[str],
    image_names: list[str],
    scorer: str = 'scorer',
) -> None:
    """write a minimal 3-row-header DLC-format label csv with random-ish coordinates"""
    header1 = ['scorer'] + [scorer] * (2 * len(keypoint_names))
    header2 = ['bodyparts'] + [kp for kp in keypoint_names for _ in range(2)]
    header3 = ['coords'] + ['x', 'y'] * len(keypoint_names)
    lines = [','.join(header1), ','.join(header2), ','.join(header3)]
    for i, image_name in enumerate(image_names):
        row = [image_name] + [str(10.0 + i)] * (2 * len(keypoint_names))
        lines.append(','.join(row))
    csv_path.write_text('\n'.join(lines) + '\n')


def _make_dataset(
    tmp_path: Path,
    n_frames: int = 10,
    n_keypoints: int = 3,
    image_size: tuple[int, int] = (200, 150),
    csv_name: str = 'CollectedData.csv',
    with_videos: bool = False,
) -> Path:
    """build a minimal single-view dataset dir with a csv and matching labeled images"""
    data_dir = tmp_path / 'dataset'
    (data_dir / 'labeled-data' / 'sess0').mkdir(parents=True)
    keypoint_names = [f'kp{i}' for i in range(n_keypoints)]
    image_names = []
    for i in range(n_frames):
        rel_path = f'labeled-data/sess0/img{i:03d}.png'
        image_names.append(rel_path)
        Image.new('RGB', image_size).save(data_dir / rel_path)
    _write_label_csv(data_dir / csv_name, keypoint_names, image_names)
    if with_videos:
        video_dir = data_dir / 'videos'
        video_dir.mkdir()
        (video_dir / 'vid0.mp4').touch()
    return data_dir


class TestDeriveViewNames:
    """Test the function _derive_view_names."""

    def test_strips_common_prefix_and_suffix(self):
        paths = [Path('CollectedData_cam0.csv'), Path('CollectedData_cam1.csv')]
        assert _derive_view_names(paths) == ['cam0', 'cam1']

    def test_falls_back_to_full_stem_on_collision(self):
        paths = [Path('a_top.csv'), Path('a_top.csv')]
        assert _derive_view_names(paths) == ['a_top', 'a_top']

    def test_no_common_prefix(self):
        paths = [Path('top.csv'), Path('side.csv')]
        assert _derive_view_names(paths) == ['top', 'side']


class TestRecommendResizeDim:
    """Test the function _recommend_resize_dim."""

    def test_default_is_256(self):
        assert _recommend_resize_dim(500, n_frames=10) == 256

    def test_short_side_is_128(self):
        assert _recommend_resize_dim(191, n_frames=10) == 128

    def test_at_small_threshold_is_default(self):
        assert _recommend_resize_dim(192, n_frames=10) == 256

    def test_long_side_with_enough_frames_is_384(self):
        assert _recommend_resize_dim(1025, n_frames=501) == 384

    def test_long_side_without_enough_frames_falls_back_to_default(self):
        assert _recommend_resize_dim(1025, n_frames=500) == 256

    def test_at_large_threshold_is_default(self):
        assert _recommend_resize_dim(1024, n_frames=1000) == 256


class TestSelectBatchSize:
    """Test the function _select_batch_size."""

    def test_no_gpu_returns_default(self):
        assert _select_batch_size(None, 256) == 4

    def test_24gb_gpu_256px(self):
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        assert _select_batch_size(gpu, 256) == 232

    def test_24gb_gpu_384px(self):
        gpu = GpuInfo(name='A100', vram_gb=24.0)
        assert _select_batch_size(gpu, 384) == 100

    def test_8gb_gpu_384px(self):
        gpu = GpuInfo(name='RTX 2070', vram_gb=8.0)
        assert _select_batch_size(gpu, 384) == 32

    def test_below_smallest_tier_falls_back_to_default(self):
        gpu = GpuInfo(name='GTX 1050', vram_gb=4.0)
        assert _select_batch_size(gpu, 256) == 4


class TestApplyMinIterationsFloor:
    """Test the function _apply_min_iterations_floor."""

    def test_rounds_down_to_multiple_of_8(self):
        # plenty of frames so the iteration-count loop doesn't also kick in
        assert _apply_min_iterations_floor(12, n_frames=100_000) == 8

    def test_already_a_multiple_of_8_is_unchanged_when_iterations_suffice(self):
        assert _apply_min_iterations_floor(32, n_frames=100_000) == 32

    def test_shrinks_further_when_too_few_iterations(self):
        # n_train = 200*0.95 = 190; at bs=40, 190/40=4.75 < 10 -> shrink to 32: 190/32=5.9 < 10
        # -> shrink to 24: 190/24=7.9 < 10 -> shrink to 16: 190/16=11.9 >= 10 -> stop
        assert _apply_min_iterations_floor(40, n_frames=200) == 16

    def test_never_goes_below_8(self):
        assert _apply_min_iterations_floor(16, n_frames=1) == 8

    def test_small_suggestion_floors_up_to_8(self):
        assert _apply_min_iterations_floor(4, n_frames=100_000) == 8


class TestAnalyzeDataset:
    """Test the function analyze_dataset."""

    def test_single_view_directory(self, tmp_path):
        data_dir = _make_dataset(tmp_path, n_frames=10, n_keypoints=3, image_size=(200, 150))
        analysis = analyze_dataset(data_dir)
        assert analysis.data_dir == data_dir
        assert analysis.view_names is None
        assert analysis.n_frames == 10
        assert analysis.num_keypoints == 3
        assert analysis.image_width == 200
        assert analysis.image_height == 150
        assert analysis.has_videos is False

    def test_single_csv_file_path(self, tmp_path):
        data_dir = _make_dataset(tmp_path)
        analysis = analyze_dataset(data_dir / 'CollectedData.csv')
        assert analysis.data_dir == data_dir
        assert analysis.csv_paths == [data_dir / 'CollectedData.csv']

    def test_detects_videos(self, tmp_path):
        data_dir = _make_dataset(tmp_path, with_videos=True)
        analysis = analyze_dataset(data_dir)
        assert analysis.has_videos is True
        assert analysis.video_dir == data_dir / 'videos'

    def test_multiview_directory(self, tmp_path):
        data_dir = tmp_path / 'dataset'
        (data_dir / 'labeled-data' / 'sess0').mkdir(parents=True)
        keypoint_names = ['kp0', 'kp1']
        image_names = []
        for i in range(4):
            rel_path = f'labeled-data/sess0/img{i:03d}.png'
            image_names.append(rel_path)
            Image.new('RGB', (128, 128)).save(data_dir / rel_path)
        _write_label_csv(data_dir / 'CollectedData_cam0.csv', keypoint_names, image_names)
        _write_label_csv(data_dir / 'CollectedData_cam1.csv', keypoint_names, image_names)

        analysis = analyze_dataset(data_dir)
        assert analysis.view_names == ['cam0', 'cam1']
        assert len(analysis.csv_paths) == 2

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            analyze_dataset(tmp_path / 'does_not_exist')

    def test_no_csv_files_raises(self, tmp_path):
        empty_dir = tmp_path / 'empty'
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match='no label CSV files'):
            analyze_dataset(empty_dir)

    def test_missing_labeled_image_raises(self, tmp_path):
        data_dir = tmp_path / 'dataset'
        data_dir.mkdir()
        _write_label_csv(
            data_dir / 'CollectedData.csv', ['kp0'], ['labeled-data/sess0/missing.png']
        )
        with pytest.raises(FileNotFoundError, match='labeled image not found'):
            analyze_dataset(data_dir)


class TestRecommend:
    """Test the function recommend."""

    def _analysis(self, **overrides) -> DatasetAnalysis:
        defaults = dict(
            dataset_path=Path('/data'),
            data_dir=Path('/data'),
            csv_paths=[Path('/data/CollectedData.csv')],
            view_names=None,
            n_frames=200,
            num_keypoints=5,
            keypoint_names=['kp0', 'kp1', 'kp2', 'kp3', 'kp4'],
            image_height=150,
            image_width=200,
            video_dir=Path('/data/videos'),
            has_videos=False,
        )
        defaults.update(overrides)
        return DatasetAnalysis(**defaults)

    def test_single_view_model_type(self):
        rec = recommend(self._analysis(), gpu=None)
        assert rec.model_type == 'heatmap'

    def test_multiview_model_type(self):
        analysis = self._analysis(
            view_names=['cam0', 'cam1'],
            csv_paths=[Path('/data/a.csv'), Path('/data/b.csv')],
        )
        rec = recommend(analysis, gpu=None)
        assert rec.model_type == 'heatmap_multiview_transformer'

    def test_backbone_always_vits_dino(self):
        rec = recommend(self._analysis(), gpu=None)
        assert rec.backbone == 'vits_dino'

    def test_optimizer_is_adamw_for_vit_backbone(self):
        rec = recommend(self._analysis(), gpu=None)
        assert rec.optimizer == 'AdamW'

    def test_no_gpu_batch_size(self):
        # _NO_GPU_BATCH_SIZE=4 is floored up to the _MIN_TRAIN_BATCH_SIZE=8 minimum
        rec = recommend(self._analysis(), gpu=None)
        assert rec.train_batch_size == 8

    def test_gpu_batch_size(self):
        # n_frames large enough that the min-iterations floor doesn't kick in, so this
        # isolates the gpu-vram x image-size table lookup (see TestApplyMinIterationsFloor
        # for the floor behavior itself)
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        analysis = self._analysis(image_height=150, image_width=200, n_frames=3000)
        rec = recommend(analysis, gpu=gpu)
        assert rec.train_batch_size == 232

    def test_max_epochs_default(self):
        rec = recommend(self._analysis(n_frames=200), gpu=None)
        assert rec.max_epochs == 300

    def test_max_epochs_few_frames(self):
        rec = recommend(self._analysis(n_frames=30), gpu=None)
        assert rec.max_epochs == 500

    def test_max_epochs_top_down(self):
        rec = recommend(self._analysis(n_frames=200), gpu=None, top_down_freely_moving=True)
        assert rec.max_epochs == 500
        assert rec.imgaug == 'dlc-top-down'

    def test_max_epochs_few_frames_and_top_down(self):
        rec = recommend(self._analysis(n_frames=30), gpu=None, top_down_freely_moving=True)
        assert rec.max_epochs == 1000

    def test_imgaug_default(self):
        rec = recommend(self._analysis(), gpu=None)
        assert rec.imgaug == 'dlc'

    def test_losses_no_videos(self):
        rec = recommend(self._analysis(has_videos=False), gpu=None)
        assert rec.losses_to_use == []

    def test_losses_with_videos_enough_frames(self):
        analysis = self._analysis(has_videos=True, n_frames=200, num_keypoints=5)
        rec = recommend(analysis, gpu=None)
        assert rec.losses_to_use == ['temporal', 'pca_singleview']

    def test_losses_with_videos_too_few_frames_for_pca(self):
        analysis = self._analysis(has_videos=True, n_frames=5, num_keypoints=5)
        rec = recommend(analysis, gpu=None)
        assert rec.losses_to_use == ['temporal']

    def test_losses_multiview_with_videos_not_auto_recommended(self):
        analysis = self._analysis(
            has_videos=True,
            view_names=['cam0', 'cam1'],
            csv_paths=[Path('/data/a.csv'), Path('/data/b.csv')],
        )
        rec = recommend(analysis, gpu=None)
        assert rec.losses_to_use == []

    def test_multiview_batch_size_divided_by_num_views(self):
        # 232 // 2 views = 116, floored to a multiple of 8 -> 112
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        analysis = self._analysis(
            view_names=['cam0', 'cam1'],
            csv_paths=[Path('/data/a.csv'), Path('/data/b.csv')],
            image_height=200,
            image_width=200,
            n_frames=3000,
        )
        rec = recommend(analysis, gpu=gpu)
        assert rec.train_batch_size == 112

    def test_semi_supervised_halves_batch_and_sets_dali_sequence_length(self):
        # 232 // 2 = 116 for both train_batch_size and dali_train_sequence_length, but the
        # final min-iterations floor only touches train_batch_size (116 -> 112)
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        analysis = self._analysis(
            has_videos=True,
            n_frames=1500,
            num_keypoints=5,
            image_height=200,
            image_width=200,
        )
        rec = recommend(analysis, gpu=gpu)
        assert rec.train_batch_size == 112
        assert rec.dali_train_sequence_length == 116

    def test_fully_supervised_has_no_dali_sequence_length(self):
        rec = recommend(self._analysis(has_videos=False), gpu=None)
        assert rec.dali_train_sequence_length is None

    def test_square_resize_dims_for_vit_backbone(self):
        # height rounds to 128 (<192), width rounds to 256 (192-1024); vits_dino requires
        # square input, so both should end up at the larger value
        analysis = self._analysis(image_height=150, image_width=200)
        rec = recommend(analysis, gpu=None)
        assert rec.image_resize_height == rec.image_resize_width == 256

    def test_rationale_covers_all_fields(self):
        rec = recommend(self._analysis(), gpu=None)
        for name in (
            'model_type',
            'backbone',
            'image_resize_dims',
            'train_batch_size',
            'imgaug',
            'max_epochs',
            'losses_to_use',
            'optimizer',
        ):
            assert name in rec.rationale


class TestBuildConfig:
    """Test the function build_config."""

    def _analysis(self, **overrides) -> DatasetAnalysis:
        defaults = dict(
            dataset_path=Path('/data'),
            data_dir=Path('/data'),
            csv_paths=[Path('/data/CollectedData.csv')],
            view_names=None,
            n_frames=200,
            num_keypoints=2,
            keypoint_names=['kp0', 'kp1'],
            image_height=150,
            image_width=200,
            video_dir=Path('/data/videos'),
            has_videos=False,
        )
        defaults.update(overrides)
        return DatasetAnalysis(**defaults)

    def test_single_view_config_structure(self):
        analysis = self._analysis()
        rec = recommend(analysis, gpu=None)
        cfg = build_config(rec, analysis)
        assert cfg.data.data_dir == str(analysis.data_dir)
        assert cfg.data.csv_file == str(analysis.csv_paths[0])
        assert cfg.data.num_keypoints == 2
        assert cfg.model.model_type == 'heatmap'
        assert cfg.model.backbone == 'vits_dino'
        assert cfg.training.train_batch_size == rec.train_batch_size
        assert cfg.training.max_epochs == rec.max_epochs
        assert cfg.training.min_epochs == rec.max_epochs
        assert 'imgaug_3d' not in cfg.training
        assert cfg.losses.temporal.log_weight == 11.0

    def test_multiview_config_structure(self):
        analysis = self._analysis(
            view_names=['cam0', 'cam1'],
            csv_paths=[Path('/data/a.csv'), Path('/data/b.csv')],
        )
        rec = recommend(analysis, gpu=None)
        cfg = build_config(rec, analysis)
        assert cfg.data.csv_file == [str(p) for p in analysis.csv_paths]
        assert cfg.data.view_names == ['cam0', 'cam1']
        assert cfg.model.model_type == 'heatmap_multiview_transformer'
        assert cfg.training.imgaug_3d is True
        assert 'supervised_reprojection_heatmap_mse' in cfg.losses

    def test_val_and_test_batch_size_are_2x_train_batch_size(self):
        analysis = self._analysis()
        rec = recommend(analysis, gpu=None)
        cfg = build_config(rec, analysis)
        assert cfg.training.val_batch_size == 2 * rec.train_batch_size
        assert cfg.training.test_batch_size == 2 * rec.train_batch_size

    def test_semi_supervised_sets_dali_base_sequence_length(self):
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        analysis = self._analysis(has_videos=True, n_frames=200, num_keypoints=2)
        rec = recommend(analysis, gpu=gpu)
        cfg = build_config(rec, analysis)
        assert cfg.dali.base.train.sequence_length == rec.dali_train_sequence_length

    def test_fully_supervised_leaves_dali_defaults_untouched(self):
        rec = recommend(self._analysis(has_videos=False), gpu=None)
        cfg = build_config(rec, analysis=self._analysis(has_videos=False))
        assert cfg.dali.base.train.sequence_length == 32
        # module-level default must not have been mutated by a previous build_config() call
        from lightning_pose.utils.recommender import _DALI_DEFAULTS
        assert _DALI_DEFAULTS['base']['train']['sequence_length'] == 32


class TestFormatReport:
    """Test the function format_report."""

    def test_report_contains_key_sections(self):
        analysis = DatasetAnalysis(
            dataset_path=Path('/data'),
            data_dir=Path('/data'),
            csv_paths=[Path('/data/CollectedData.csv')],
            view_names=None,
            n_frames=200,
            num_keypoints=5,
            keypoint_names=['kp0', 'kp1', 'kp2', 'kp3', 'kp4'],
            image_height=150,
            image_width=200,
            video_dir=Path('/data/videos'),
            has_videos=False,
        )
        rec = recommend(analysis, gpu=None)
        report = format_report(rec, analysis, gpu=None)
        assert 'Dataset summary' in report
        assert 'Recommendations' in report
        assert 'model_type' in report
        assert 'none detected' in report

    def test_report_with_gpu(self):
        analysis = DatasetAnalysis(
            dataset_path=Path('/data'),
            data_dir=Path('/data'),
            csv_paths=[Path('/data/CollectedData.csv')],
            view_names=None,
            n_frames=200,
            num_keypoints=5,
            keypoint_names=['kp0', 'kp1', 'kp2', 'kp3', 'kp4'],
            image_height=150,
            image_width=200,
            video_dir=Path('/data/videos'),
            has_videos=False,
        )
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        rec = recommend(analysis, gpu=gpu)
        report = format_report(rec, analysis, gpu=gpu)
        assert 'A100' in report
