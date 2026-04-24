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
