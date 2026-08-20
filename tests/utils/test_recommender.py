"""Test the recommender module."""

from pathlib import Path

import pytest
from PIL import Image

from lightning_pose.utils.recommender import (
    DatasetAnalysis,
    GpuInfo,
    _derive_view_names,
    _round_resize_dim,
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


class TestRoundResizeDim:
    """Test the function _round_resize_dim."""

    def test_rounds_up_to_multiple_of_128(self):
        assert _round_resize_dim(200) == 256

    def test_exact_multiple_unchanged(self):
        assert _round_resize_dim(256) == 256

    def test_caps_at_384(self):
        assert _round_resize_dim(1000) == 384

    def test_floors_at_128(self):
        assert _round_resize_dim(50) == 128


class TestSelectBatchSize:
    """Test the function _select_batch_size."""

    def test_no_gpu_returns_default(self):
        assert _select_batch_size(None, 256) == 4

    def test_24gb_gpu_256px(self):
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        assert _select_batch_size(gpu, 256) == 32

    def test_24gb_gpu_384px(self):
        gpu = GpuInfo(name='A100', vram_gb=24.0)
        assert _select_batch_size(gpu, 384) == 16

    def test_8gb_gpu_384px(self):
        gpu = GpuInfo(name='RTX 2070', vram_gb=8.0)
        assert _select_batch_size(gpu, 384) == 4

    def test_below_smallest_tier_falls_back_to_default(self):
        gpu = GpuInfo(name='GTX 1050', vram_gb=4.0)
        assert _select_batch_size(gpu, 256) == 4


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
        rec = recommend(self._analysis(), gpu=None)
        assert rec.train_batch_size == 4

    def test_gpu_batch_size(self):
        gpu = GpuInfo(name='A100', vram_gb=40.0)
        rec = recommend(self._analysis(image_height=150, image_width=200), gpu=gpu)
        assert rec.train_batch_size == 32

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
