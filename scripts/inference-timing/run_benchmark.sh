#!/bin/bash
# Orchestrates a full inference-timing sweep: repeatedly calls
# benchmark_single_config.py, once per (model panel x variant x decoder)
# combination, each as its own fresh process, then calls plot_results.py at
# the end.
#
# Each combination runs in its own `python3` process (not a Python-level
# loop inside one long-lived process) because Lightning Pose has a known
# history of a rare CUDA device-side assert that corrupts the entire CUDA
# context (upstream issue #483 / PR #498). One process per combination means
# a crash only costs that one combination's data point; this script logs it
# and continues rather than aborting the whole sweep or silently poisoning
# later measurements.
#
# Usage:
#   bash run_benchmark.sh --config path/to/your_config.yaml [--dry_run]
#
# See timing_config.yaml for the config file format. Copy it outside the
# repo and edit your copy before running (same convention as
# scripts/hyper-sweep/sweep_config.yaml).

set -uo pipefail  # deliberately NOT `set -e`: see note above per-combo loop

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: bash run_benchmark.sh --config path/to/your_config.yaml [--dry_run]" >&2
  echo "See timing_config.yaml for a template." >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

# load_config.py parses the YAML config and prints MODEL_PANELS/VARIANTS/
# DECODERS/etc as shell array/variable assignments for this script to eval.
# The config file stays YAML (matching scripts/hyper-sweep/sweep_config.yaml)
# while the orchestration itself stays a plain bash loop.
eval "$(python3 "$SCRIPT_DIR/load_config.py" --config "$CONFIG")"

: "${MODEL_PANELS?config must set panels}"
: "${VARIANTS?config must set sweep.variants}"
: "${DECODERS?config must set sweep.decoders}"
: "${NUM_WARMUP:=1}"
: "${NUM_REPEATS:=3}"
: "${OUTPUT_DIR:=$HOME/inference_timing_results}"
: "${MAX_BATCH_SIZE:=8}"
: "${OPT_BATCH_SIZE:=1}"
: "${GPU_LABEL:=}"

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -z "$GPU_LABEL" ]]; then
  GPU_LABEL="$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
  if [[ -z "$GPU_LABEL" ]]; then
    echo "Could not auto-detect GPU label (is a CUDA GPU visible / is torch installed?). Set output.gpu_label in your config to override." >&2
    exit 1
  fi
fi
echo "GPU_LABEL=$GPU_LABEL"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "Panels: ${#MODEL_PANELS[@]}, Variants: ${VARIANTS[*]}, Decoders: ${DECODERS[*]}"
echo

FAILED=()
TOTAL=0
OK=0

for panel in "${MODEL_PANELS[@]}"; do
  IFS='|' read -r LABEL MODEL_DIR DATASET_DIR VIDEO_PATHS <<< "$panel"

  for VARIANT in "${VARIANTS[@]}"; do
    for DECODER in "${DECODERS[@]}"; do
      TOTAL=$((TOTAL + 1))
      OUTPUT_CSV="$OUTPUT_DIR/${LABEL}.csv"
      LOG_FILE="$LOG_DIR/${LABEL}__${VARIANT}__${DECODER}.log"

      echo "--- [$TOTAL] $LABEL | $VARIANT | $DECODER ---"

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "  (dry run) would run: python3 $SCRIPT_DIR/benchmark_single_config.py --model_dir $MODEL_DIR --model_label $LABEL --video_paths $VIDEO_PATHS --variant $VARIANT --decoder $DECODER --num_repeats $NUM_REPEATS --num_warmup $NUM_WARMUP --output_csv $OUTPUT_CSV --gpu_label \"$GPU_LABEL\""
        continue
      fi

      DATASET_ARG=()
      if [[ -n "$DATASET_DIR" ]]; then
        DATASET_ARG=(--dataset_dir "$DATASET_DIR")
      fi

      if python3 "$SCRIPT_DIR/benchmark_single_config.py" \
          --model_dir "$MODEL_DIR" \
          --model_label "$LABEL" \
          "${DATASET_ARG[@]}" \
          --video_paths "$VIDEO_PATHS" \
          --variant "$VARIANT" \
          --decoder "$DECODER" \
          --num_repeats "$NUM_REPEATS" \
          --num_warmup "$NUM_WARMUP" \
          --output_csv "$OUTPUT_CSV" \
          --gpu_label "$GPU_LABEL" \
          --max_batch_size "$MAX_BATCH_SIZE" \
          --opt_batch_size "$OPT_BATCH_SIZE" \
          > >(tee "$LOG_FILE") 2>&1; then
        OK=$((OK + 1))
      else
        echo "  FAILED (see $LOG_FILE)"
        FAILED+=("$LABEL | $VARIANT | $DECODER")
      fi
      echo
    done
  done
done

echo "=========================================="
echo "Done: $OK/$TOTAL combinations succeeded."
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed combinations (${#FAILED[@]}):"
  for f in "${FAILED[@]}"; do
    echo "  - $f"
  done
fi
echo "=========================================="

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete; skipping plot generation."
  exit 0
fi

echo
echo "Generating plot from $OUTPUT_DIR ..."
python3 "$SCRIPT_DIR/plot_results.py" --input_dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/inference_timing.png"

# Exit non-zero if anything failed, so this is detectable in CI/automation,
# but we still ran everything we could and still produced a plot from
# whatever succeeded.
if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
