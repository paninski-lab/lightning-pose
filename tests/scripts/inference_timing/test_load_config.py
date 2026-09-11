"""Tests for scripts/inference-timing/load_config.py.

Keeps load_config.py in sync with the timing_config.yaml schema: if someone
renames a key in one without updating the other, these tests catch it
without needing a GPU or a trained model.
"""
TIMING_CONFIG_YAML = """
panels:
  - label: test-panel
    model_dir: /fake/model_dir
    dataset_dir: null
    video_paths:
      - /fake/videos/cam1.mp4
      - /fake/videos/cam2.mp4
sweep:
  variants: [eager_fp32, tensorrt_fp16]
  decoders: [dali, opencv]
timing:
  num_warmup: 2
  num_repeats: 5
output:
  dir: /fake/output_dir
  gpu_label: A100
export:
  max_batch_size: 4
  opt_batch_size: 2
"""


def test_load_config_reads_yaml_file(load_config, tmp_path):
    config_path = tmp_path / "timing_config.yaml"
    config_path.write_text(TIMING_CONFIG_YAML)

    cfg = load_config.load_config(str(config_path))

    assert cfg["sweep"]["variants"] == ["eager_fp32", "tensorrt_fp16"]
    assert cfg["output"]["gpu_label"] == "A100"


def test_panel_lines_matches_run_benchmark_pipe_format(load_config):
    panels = [
        {
            "label": "test-panel",
            "model_dir": "/fake/model_dir",
            "dataset_dir": None,
            "video_paths": ["/fake/videos/cam1.mp4", "/fake/videos/cam2.mp4"],
        }
    ]

    lines = load_config.panel_lines(panels)

    assert lines == ["test-panel|/fake/model_dir||/fake/videos/cam1.mp4,/fake/videos/cam2.mp4"]


def test_main_emits_sourceable_shell_assignments(load_config, tmp_path, capsys):
    config_path = tmp_path / "timing_config.yaml"
    config_path.write_text(TIMING_CONFIG_YAML)

    load_config.main(["--config", str(config_path)])

    out = capsys.readouterr().out
    assert "MODEL_PANELS=(" in out
    assert "VARIANTS=(eager_fp32 tensorrt_fp16)" in out
    assert "DECODERS=(dali opencv)" in out
    assert "NUM_WARMUP=2" in out
    assert "NUM_REPEATS=5" in out
    assert "GPU_LABEL=A100" in out
    assert "MAX_BATCH_SIZE=4" in out
    assert "OPT_BATCH_SIZE=2" in out


def test_missing_gpu_label_defaults_to_empty_string(load_config, tmp_path):
    config_path = tmp_path / "timing_config.yaml"
    config_path.write_text(
        "panels: []\nsweep:\n  variants: []\n  decoders: []\n"
    )

    cfg = load_config.load_config(str(config_path))
    output = cfg.get("output", {})

    assert (output.get("gpu_label") or "") == ""
