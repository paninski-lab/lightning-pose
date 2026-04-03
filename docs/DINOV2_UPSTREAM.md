# Alignment with [PPWangyc/lightning-pose `dinov2`](https://github.com/PPWangyc/lightning-pose/tree/dinov2)

This tree tracks the upstream **dinov2** fork for backbones (DINOv2/DINOv3/BEAST3D), apps, configs, and training fixes.

## What to mirror from upstream

| Area | Upstream expectation |
|------|----------------------|
| **Backbones** | `lightning_pose/models/backbones/vits.py` — `VisionEncoder` / `VisionEncoderDino`, `vitl_*`, `beast3d` |
| **BEAST3D** | `models/backbones/beast3d/`, `models/backbones/layers/`, `models/configs/beast3d_chickadee.yaml` |
| **Multiview + ViT** | `heatmap_tracker_multiview.py` — custom `forward_vit` with view embeddings; **DINOv3** uses `.layer` + RoPE (not `.encoder`) |
| **Routing** | `base.py` — `beast3d` must use `vits.build_backbone` (same as `vit*` prefixes) |
| **Training** | `train.py` — multi-GPU: `DDPStrategy(find_unused_parameters=True)` when `num_gpus > 1` |
| **Transformers** | **DINOv3** needs `transformers>=4.57` (`dinov3_vit` in `AutoConfig`). Pin in `pyproject.toml`. |

## E-RayZer (BEAST3D) submodule

Upstream declares a git submodule:

```ini
[submodule "third_party/E-RayZer"]
  path = third_party/E-RayZer
  url = https://github.com/PPWangyc/E-RayZer.git
  branch = working
```

### Using pretrained E-RayZer inside Lightning Pose

1. **Clone / init the submodule** at the **lightning-pose repo root**:

   ```bash
   cd /path/to/lightning-pose
   git submodule update --init --recursive third_party/E-RayZer
   ```

   Or clone [PPWangyc/E-RayZer](https://github.com/PPWangyc/E-RayZer) into `lightning-pose/third_party/E-RayZer`.

2. **Optional:** set `E_RAYZER_PATH` if you keep the code elsewhere:

   ```bash
   export E_RAYZER_PATH=/path/to/E-RayZer
   ```

   `beast3d.py` prepends this path (or `third_party/E-RayZer` under the repo root) to `sys.path` so `erayzer_core` imports work.

3. **LP config** — set `model.backbone: beast3d` and `model.backbone_checkpoint: /path/to.pt` (see `models/configs/beast3d_chickadee.yaml` for model YAML).

### Training / pretraining E-RayZer itself

**E-RayZer pretraining** is run from the **E-RayZer** repository (its own scripts, configs, and datasets), not from Lightning Pose. Lightning Pose **consumes** E-RayZer weights as a **frozen or fine-tuned image backbone** via the BEAST3D wrapper.

- Follow the README and training entrypoints in the [E-RayZer](https://github.com/PPWangyc/E-RayZer) repo (branch `working` per submodule).
- After you have a checkpoint, point **`model.backbone_checkpoint`** at it when using `backbone: beast3d`.

## Version drift

If something works on upstream `dinov2` but not here, compare:

- `diff` of `lightning_pose/models/backbones/vits.py`
- `diff` of `lightning_pose/models/heatmap_tracker_multiview.py`
- `pyproject.toml` `transformers` version
