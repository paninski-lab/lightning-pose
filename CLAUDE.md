# Claude Development Guidelines

This file contains project-specific guidelines for Claude Code when working on this project.

## Code Style

### Comments
- Use docstrings for all functions, classes, and modules
- Follow Google-style docstrings
- Add inline comments for complex logic
- Avoid obvious comments
- Comments should start with lowercase letters

### Type Hints
- Use type hints for all function parameters and return values
- Import types from typing module when needed
- use X | Y, list[X], dict[K, V] syntax; avoid typing imports for these

### Imports
- Group imports: standard library, third-party, local
- Use absolute imports (e.g., `from <package_name>.<modlue_name> import func` not `from .<module_name> import func`)
- Sort imports alphabetically within groups
- Avoid wildcard imports

### Code Formatting
- Line length: 99 characters
- Use 4 spaces for indentation
- Follow PEP 8 conventions
- Use meaningful variable names
- Use idx for index
- Use adjectives after nouns, such as idx_train and idx_test rather than train_idx and test_idx
- Include newline at the end of every .py file
- Do not allow trailing whitespace
- Do not include whitespace for blank lines
- Use single quotes for strings
- Add a comma to the end of multi-line function arguments:
```python
foo(
    param1,
    param2,
)
```
not
```python
foo(
    param1,
    param2
)
```

### Misc
- Use `pathlib.Path` instead of `os` for path handling
- Use f-strings for all string interpolation, including `logger` calls and `print` statements — never `%s` formatting or `.format()`

## Testing

### Unit Tests
- Use pytest framework
- Test directory structure must mirror the source package structure exactly:
  strip the `<package_name>/` prefix and prepend `tests/` — the rest of the path is identical.
  `<package_name>/a/b/c.py` → `tests/a/b/test_c.py`, at every level of nesting without exception.
  - Each subdirectory under `tests/` must have an `__init__.py`
- Create test classes for each function
- Use fixtures for common test data; place fixtures in a `conftest.py` in the same directory as the tests that use them
- Test assets (e.g. sample images) live in an `assets/` subdirectory alongside the tests that use them
- Aim for high test coverage
- Test both success and failure cases

### Test Structure
```python
class Test<function_name>:
    """Test the function <function_name>."""

    def test_<function_name>_<scenario>(self):
        # Arrange
        # Act
        # Assert
```

### Mocking
- Use unittest.mock for external dependencies
- Mock at the boundary of your system
- Use dependency injection when possible

## Documentation

### Docstrings
```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```
If there are too many params for a single line, put one param per line, indented four spaces:
```python
def function_name(
    param1: Type1,
    param2: Type2,
    param3: Type3,
    param4: Type4,
) -> ReturnType:
```

### Module Documentation
- Every module should have a module-level docstring
- Describe the purpose and main functionality

## Error Handling

### Exceptions
- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Use try/except blocks judiciously

### Logging
- Use Python's logging module
- Include appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Log important events and errors
- Include context in log messages

## Project Structure

### Package Organization
- Keep modules focused and cohesive
- Use clear, descriptive module names
- Group related functionality together
- Avoid circular imports

### File Naming
- Use snake_case for Python files
- Use descriptive names
- Group related files in subdirectories

## CLI Guidelines

### Command Structure
- Use subcommands for different operations
- Provide clear help messages
- Validate input parameters
- Handle errors gracefully

### Configuration
- Use sensible defaults
- Validate configuration before execution

## Dependencies

### Adding Dependencies
- Only add dependencies that are truly needed
- Prefer well-maintained packages
- Do not pin versions
- Update pyproject.toml and document changes

### Version Management
- Use semantic versioning
- Update version in pyproject.toml
- Tag releases appropriately

## Git Workflow

### Commits
- Write clear, descriptive commit messages
- Make atomic commits
- Include tests with feature commits
- Run tests before committing

### Branches
- Use feature branches for development
- Keep branches focused and short-lived
- Merge with pull requests
- Delete merged branches

## Architecture

### Module docstring conventions

Every subpackage that contains both an `__init__.py` and a `factory.py` observes a strict
separation of concerns in their module docstrings:

- **`__init__.py`** — user-facing package map: what classes/functions are exported, how they
  relate, and where to find them.  No contributor recipes.
- **`factory.py`** — contributor guide: describes the dispatch logic and includes an explicit
  "Adding a new X" step-by-step recipe so extensions don't require reverse-engineering.

This applies to `data/`, `losses/`, `models/`, and `models/backbones/`.

### Backbone package (`lightning_pose/models/backbones/`)

**Package layout**:
- `__init__.py` — re-export façade only; defines nothing, imports everything from `factory.py`.
- `factory.py` — single source of truth for type constants and dispatch logic. Contains all
  `ALLOWED_*` Literal aliases, `_ALLOWED_BACKBONE_VALUES` frozenset, `build_backbone`,
  `_build_transformer_backbone`, `_build_convnet_backbone`, and `grab_layers_sequential`.
- `vit.py` — `VisionEncoder` (generic HuggingFace ViT wrapper) + `load_vit_backbone_checkpoint`.
- `vit_dino.py` — `VisionEncoderDino` (DINOv2/v3 with patch-size re-interpolation and gated-repo
  auth handling).
- `vit_sam.py` — `VisionEncoderSam` (SAM ViT-B with pos-embed resize for arbitrary input sizes).
- `vit_sam2.py` — `VisionEncoderSam2` (SAM2 Hiera backbone; pos-embed interpolated internally).

**Type hierarchy**:
```
ALLOWED_BACKBONES = ALLOWED_CONVNET_BACKBONES | ALLOWED_TRANSFORMER_BACKBONES
ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW ⊂ ALLOWED_TRANSFORMER_BACKBONES
```
SAM and SAM2 backbones (`vitb_sam`, `vitb_sam2`, `vits_sam2`, `vitt_sam2`) are present in
`ALLOWED_TRANSFORMER_BACKBONES` but absent from `ALLOWED_TRANSFORMER_BACKBONES_MULTIVIEW`.
They are blocked at construction time in `HeatmapTrackerMultiviewTransformer` with a
`ValueError` because they are architecturally incompatible with the multiview cross-attention
block (no `embeddings` module, NHWC tensor layout).

**Adding a new backbone**: add its identifier to the appropriate `ALLOWED_*` constant(s) in
`factory.py`, add an `elif` branch in `_build_transformer_backbone` or `_build_convnet_backbone`,
and create a `vit_<name>.py` wrapper module if needed. The `_ALLOWED_BACKBONE_VALUES` frozenset
and `ALLOWED_BACKBONES` union update automatically.

**Lazy imports in `factory.py`**: all `lightning_pose` backbone wrapper classes are imported
inside the helper functions, not at module level. `factory.py` is loaded during
`backbones/__init__.py` initialisation, so a top-level `lightning_pose` import there would
create a circular import.

### Dataset class hierarchy (`lightning_pose/data/datasets.py`)

Three dataset classes, with a clear inheritance structure:

- **`BaseTrackingDataset`** — base class. Loads images and (x, y) keypoints, applies the imgaug
  pipeline, and handles `imgaug_hflip`. Returns a `BaseLabeledExampleDict`.
- **`HeatmapDataset(BaseTrackingDataset)`** — adds `compute_heatmap` to convert keypoints to
  `(K, H, W)` Gaussian heatmap targets, and synthesizes `self.visibility` from NaN positions when
  the CSV has no `visible` column. Returns a `HeatmapLabeledExampleDict`.
- **`MultiviewHeatmapDataset`** — does **not** inherit from `BaseTrackingDataset`. Holds a
  `dict[str, HeatmapDataset]` at `self.dataset` (keyed by view name). `__getitem__` delegates to
  each child and stacks results. Shares `imgaug_transform` and `imgaug_hflip` attributes with
  child datasets so the data module can update them in one pass.

**`__getitem__` branching**: both `BaseTrackingDataset` and `HeatmapDataset` branch on
`self.do_context` at the top of `__getitem__`. The non-context branch loads a single PIL image;
the context branch loads a sequence of frames. All augmentation logic (imgaug pipeline + hflip) is
duplicated in both branches.

**Adding an augmentation that affects keypoints**:

1. Add any per-sample state (e.g. `do_hflip`) at the top of each branch in `__getitem__`, not
   outside the branch, since the two paths are structurally independent.
2. Apply the augmentation *after* the imgaug pipeline so keypoints are already in resized
   coordinate space.
3. For context mode, generate any random decision once before the per-frame loop and reuse it for
   all frames so the same transform is applied consistently across the sequence.
4. Set `self.imgaug_hflip = False` (or equivalent sentinel) on `MultiviewHeatmapDataset` so the
   data module can safely reset it without `AttributeError`.
5. Update the data module's `_setup` condition (see comment there) if the new augmentation must
   also be disabled for val/test subsets.
6. **Also set the flag to `False` in `_build_datamodule_pred`** (`lightning_pose/api/model.py`).
   That function deep-copies the training config and overrides `cfg_pred.training.imgaug =
   "default"`, but it does **not** automatically clear other training-only flags. Any augmentation
   flag that is not cleared here will remain active during prediction, causing randomly-augmented
   inference on labeled frames and silently wrong evaluation metrics. The fix pattern is:
   ```python
   cfg_pred.training.imgaug = "default"
   cfg_pred.training.<your_flag> = False   # ← must be explicit
   ```

### Data module split logic (`lightning_pose/data/datamodules.py` → `BaseDataModule._setup`)

The imgaug pipeline always contains at least one element: a final resize transform appended by
`BaseTrackingDataset.__init__`. `len(imgaug_transform) == 1` therefore means "resize only, no
augmentations." When this is true **and** `imgaug_hflip` is False, all three splits can share the
same underlying dataset object (cheap path). Otherwise, three deep-copied datasets are created and
the val/test copies have their pipeline replaced with resize-only and their `imgaug_hflip` reset to
`False`. Any new augmentation applied outside the pipeline (like `imgaug_hflip`) must be added to
this condition and explicitly stripped from val/test datasets in the `else` branch.

### DALI Pipeline (`lightning_pose/data/dali.py`)

**`PrepareDALI`** — two-phase construction:
- `__init__`: validates inputs and pre-computes pipe arguments for all four
  `{train,predict}×{base,context}` combinations. Raises early so invalid configs are caught
  before GPU allocation.
- `__call__`: builds and returns the GPU pipeline as a `LitDaliWrapper`.

**`LitDaliWrapper`** — wraps `DALIGenericIterator`; converts raw DALI batch dicts to
`(frames, transforms)` tensors expected by the model. Manages `_frame_idx`, a cursor into
`bbox_df` that advances by `seq_len` (base models) or `seq_len - 4` (context models, because
consecutive DALI windows overlap by 4 frames to share context).

### Bbox / Cropzoom (`docs/source/user_guide_advanced/cropzoom_pipeline.rst`)

**CSV format**: columns `x`, `y`, `h`, `w` (top-left corner and size in pixels), one row per
frame. The index column is ignored on read.

**Naming convention**:
- Videos: `{video_stem}_bbox.csv` (one file per video)
- Labeled frames: `bbox.csv`

**Two prediction modes** (selected by whether `bbox_df` is passed to `PrepareDALI`):
- *Standard*: DALI resizes frames on GPU before passing them to the model. `resize_dims` is set.
- *Bbox-crop*: DALI delivers full-resolution frames; `LitDaliWrapper._apply_bbox_crop` crops and
  resizes each frame in PyTorch using per-frame rows from `bbox_df`. `resize_dims=None` in the
  pipe dict for bbox predict entries.

**Data flow**: `litpose predict --bbox_dir` → `_predict_multi_type` →
`model.predict_on_video_file(bbox_file=...)` → `predict_video(bbox_file=...)` →
`PrepareDALI(bbox_df=...)` → `LitDaliWrapper._apply_bbox_crop`.

### Per-keypoint visibility (`lightning_pose/data/`)

Label CSV files support an optional `visible` column after each `x, y` pair
(`x, y, visible, x, y, visible, …`), following the COCO keypoint convention:

| Value | Meaning | Heatmap target |
|-------|---------|----------------|
| 0 | not labeled | all-zeros (excluded from loss) |
| 1 | occluded | uniform (encourages low-confidence output) |
| 2 | visible | Gaussian (standard supervised target) |

**Detection**: inline in `parse_label_csv` — the `coords` header row is checked for a `visible`
entry. Only the `[0, 1, 2]` (DLC) header format is supported; all other formats are treated as
having no visibility column.

**Parsing** (`BaseTrackingDataset.__init__`): when detected, the raw CSV is reshaped to
`(N, K, 3)` and split into `self.keypoints (N, K, 2)` (x, y) and
`self.visibility (N, K)` `int64` tensor. When absent, `self.visibility = None`.

**Synthesis** (`HeatmapDataset.__init__`): when `self.visibility is None` after the base class
init, visibility is synthesized from NaN positions: NaN keypoints get `vis=1` (uniform) if the
`uniform_heatmaps_for_nan_keypoints` config flag is set, else `vis=0` (zero); valid keypoints
get `vis=2`. This ensures `self.visibility` is always populated before training.

**`BaseLabeledExampleDict`**: always contains a `visibility` key. When
`BaseTrackingDataset.visibility is None`, the value is an empty tensor
`torch.zeros(0, dtype=torch.long)` (sentinel that collates cleanly); otherwise a `(K,)` int64
tensor. `compute_heatmap` checks `numel() > 0` to distinguish the two cases.

**Heatmap generation** (`generate_heatmaps` in `lightning_pose/data/utils.py`): accepts an
optional `visibility: Int[Tensor, "batch K"]` parameter. The Gaussian computation and
normalization are unchanged; only the filler block at the end branches on visibility.
`visibility=None` → OOB/NaN keypoints produce zeros (safe for loss callers, which never pass
NaN-coordinate predictions).

**Propagation**: `HeatmapDataset.compute_heatmap` reads `example_dict["visibility"]` directly
(set by `BaseTrackingDataset.__getitem__`) and passes it to `generate_heatmaps`. No changes to
loss functions are needed — zero heatmaps are already filtered by `HeatmapLoss.remove_nans`,
and uniform heatmaps are already included.

**Warnings**: a `logger.warning` is emitted if vis=1 keypoints have non-NaN coordinates
(the visibility flag wins; coordinates are ignored).

**Validation**: raises `ValueError` if any visibility value is outside `{0, 1, 2}`.

### Post-training evaluation (`lightning_pose/train.py`)

After training, `train()` calls `_evaluate_on_training_dataset(model, suffix=...)` three times:

1. `suffix=None` — runs inference on the base training CSV (`data.csv_file`). Passes
   `add_train_val_test_set=True` so metrics are split by train/val/test set membership.
2. `suffix='_new'` — looks for `{csv_stem}_new.csv`; intended for OOD labeled frames added after
   training.
3. `suffix='_test'` — looks for `{csv_stem}_test.csv`; intended for a held-out test set.

Calls with a suffix are silently skipped when the suffixed file does not exist.

**Output**: prediction CSVs are written under `image_preds/{csv_filename}/predictions*.csv` by
`model.predict_on_label_csv`, then copied to `model_dir/predictions[_{view}][{metric_suffix}][{suffix}].csv`
for backward compatibility.
