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
