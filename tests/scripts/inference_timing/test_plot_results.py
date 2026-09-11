"""Tests for scripts/inference-timing/plot_results.py.

Exercises the pure data-shaping helpers plus a full main() run against
synthetic CSVs, so a change to the CSV schema written by
benchmark_single_config.py that plot_results.py doesn't know about gets
caught without needing real benchmark data or a GPU.
"""
import csv

CSV_FIELDS = ["gpu_label", "model_label", "variant", "decoder", "is_warmup", "fps"]


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(variant, decoder, fps=100.0):
    return {
        "gpu_label": "A100",
        "model_label": "panel_a",
        "variant": variant,
        "decoder": decoder,
        "is_warmup": 0,
        "fps": fps,
    }


def test_ordered_variants_uses_preferred_order_then_extras(plot_results):
    present = ["onnx_fp16", "made_up_variant", "eager_fp32"]

    ordered = plot_results.ordered_variants(present)

    assert ordered == ["eager_fp32", "onnx_fp16", "made_up_variant"]


def test_load_panels_skips_empty_csvs(plot_results, tmp_path):
    _write_csv(tmp_path / "panel_a.csv", [_row("eager_fp32", "dali")])
    _write_csv(tmp_path / "panel_b.csv", [])

    panels = plot_results.load_panels(str(tmp_path))

    assert list(panels.keys()) == ["panel_a"]


def test_main_writes_a_plot_from_synthetic_csvs(plot_results, tmp_path):
    rows = [
        _row(variant, decoder)
        for variant in ["eager_fp32", "tensorrt_fp16"]
        for decoder in ["dali", "pynvvc"]
    ]
    _write_csv(tmp_path / "panel_a.csv", rows)
    output_path = tmp_path / "out.png"

    plot_results.main(["--input_dir", str(tmp_path), "--output", str(output_path)])

    assert output_path.exists()
