"""Tests for scripts/inference-timing/benchmark_single_config.py.

These are lightweight, GPU-free checks on the script's pure helper logic
(argument parsing, hydra overrides, CSV writing) -- not an end-to-end
benchmark run, which needs a real trained model and a real GPU. The goal is
just to catch this script drifting out of sync with the rest of the repo, in
the same spirit as tests/scripts/hyper_sweep.
"""
import pytest


def test_hydra_overrides_without_dataset_dir(benchmark_single_config):
    overrides = benchmark_single_config.hydra_overrides_for(None)

    assert overrides == ["++training.imgaug_hflip=false"]


def test_hydra_overrides_with_dataset_dir(benchmark_single_config):
    overrides = benchmark_single_config.hydra_overrides_for("/fake/dataset")

    assert "data.data_dir=/fake/dataset" in overrides
    assert "data.video_dir=/fake/dataset/videos" in overrides


def _base_argv(**overrides):
    argv = {
        "--model_dir": "/fake/model_dir",
        "--model_label": "resnet50",
        "--video_paths": "/fake/video.mp4",
        "--variant": "eager_fp32",
        "--decoder": "dali",
        "--output_csv": "/fake/out.csv",
    }
    argv.update(overrides)
    flat = []
    for k, v in argv.items():
        flat.extend([k, v])
    return flat


def test_parse_args_defaults(benchmark_single_config):
    args = benchmark_single_config.parse_args(_base_argv())

    assert args.variant == "eager_fp32"
    assert args.num_repeats == 3
    assert args.num_warmup == 1
    assert args.max_batch_size == 8


def test_parse_args_rejects_unknown_variant(benchmark_single_config):
    argv = _base_argv(**{"--variant": "not_a_real_variant"})

    with pytest.raises(SystemExit):
        benchmark_single_config.parse_args(argv)


def test_parse_args_rejects_unknown_decoder(benchmark_single_config):
    argv = _base_argv(**{"--decoder": "not_a_real_decoder"})

    with pytest.raises(SystemExit):
        benchmark_single_config.parse_args(argv)


def test_write_rows_appends_without_duplicating_header(benchmark_single_config, tmp_path):
    output_csv = tmp_path / "out.csv"
    row = dict.fromkeys(benchmark_single_config.CSV_FIELDS, "x")

    benchmark_single_config.write_rows(str(output_csv), [row])
    benchmark_single_config.write_rows(str(output_csv), [row])

    lines = output_csv.read_text().splitlines()
    assert lines[0] == ",".join(benchmark_single_config.CSV_FIELDS)
    assert len(lines) == 3  # header + 2 appended rows


def test_count_frames_is_imported_not_reimplemented(benchmark_single_config):
    # Regression test for the switch away from a locally-duplicated
    # count_frames: this script should no longer define its own copy, so it
    # can't drift from lightning_pose/data/utils.py's version.
    assert not hasattr(benchmark_single_config, "count_frames")
