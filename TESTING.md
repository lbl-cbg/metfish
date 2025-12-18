# Testing Guide for MetFish

This document provides comprehensive information about testing the metfish package.

## Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run fast tests (recommended for development)
make test

# Run all tests
make test-all

# Run tests with coverage
make test-coverage
```

## Test Organization

### Test Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `test_utils.py` | Core utility functions (get_Pr, extract_seq, etc.) | Base |
| `test_commands.py` | CLI commands | Base |
| `test_loss.py` | Loss functions and differentiable operations | PyTorch |
| `test_msa_model.py` | MSA model configuration and components | PyTorch, OpenFold (optional) |
| `test_refinement_model.py` | Refinement model configuration | PyTorch (optional) |
| `test_analysis.py` | Analysis and visualization modules | Base |
| `test_integration.py` | End-to-end integration tests | All |
| `test_extract_seq.py` | Sequence extraction (legacy) | Base |
| `test_get_Pr.py` | P(r) calculation (legacy) | Base |

### Test Categories

Tests are organized using pytest markers:

- **`@pytest.mark.unit`**: Fast, isolated unit tests
- **`@pytest.mark.integration`**: Integration tests across modules
- **`@pytest.mark.slow`**: Tests that take >5 seconds
- **`@pytest.mark.requires_data`**: Tests needing test data files
- **`@pytest.mark.requires_torch`**: Tests needing PyTorch
- **`@pytest.mark.requires_openfold`**: Tests needing OpenFold

## Running Tests

### Command Line

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_utils.py

# Run specific test class
pytest tests/test_utils.py::TestGetPr

# Run specific test function
pytest tests/test_utils.py::TestGetPr::test_get_pr_from_pdb

# Run with verbose output
pytest tests/ -v

# Run with very verbose output (show all test names)
pytest tests/ -vv
```

### Using Markers

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Exclude slow tests
pytest tests/ -m "not slow"

# Run tests that don't require PyTorch
pytest tests/ -m "not requires_torch"

# Run tests that don't require PyTorch or OpenFold
pytest tests/ -m "not requires_torch and not requires_openfold"

# Run only tests that require data
pytest tests/ -m requires_data
```

### Using Make Commands

```bash
# Run fast tests (default, excludes slow/torch/openfold)
make test

# Run all tests including slow ones
make test-all

# Run only unit tests
make test-unit

# Run only integration tests
make test-integration

# Run tests with coverage report
make test-coverage
```

## Coverage Reports

### Generate Coverage

```bash
# Generate HTML coverage report
pytest tests/ --cov=src/metfish --cov-report=html

# Generate terminal coverage report
pytest tests/ --cov=src/metfish --cov-report=term-missing

# Generate XML coverage report (for CI)
pytest tests/ --cov=src/metfish --cov-report=xml
```

### View Coverage

```bash
# Open HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing Tests

### Test Structure

```python
"""
Tests for module_name
"""
import pytest
from metfish.module import function


class TestFunctionName:
    """Tests for function_name."""
    
    def test_basic_case(self):
        """Test basic functionality."""
        result = function()
        assert result is not None
    
    @pytest.mark.requires_data
    def test_with_data(self, test_data_dir):
        """Test with actual data."""
        data_path = test_data_dir / "file.pdb"
        result = function(data_path)
        assert result is not None
    
    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function(invalid_input)
```

### Using Fixtures

Common fixtures from `conftest.py`:

```python
def test_example(test_data_dir, temp_dir, sample_pdb_path):
    """Example using fixtures."""
    # test_data_dir: Path to tests/data/
    # temp_dir: Temporary directory for outputs
    # sample_pdb_path: Path to sample PDB file
    
    output = temp_dir / "output.txt"
    process_pdb(sample_pdb_path, output)
    assert output.exists()
```

### Best Practices

1. **One concept per test**: Each test should verify one specific behavior
2. **Descriptive names**: Use clear, descriptive test function names
3. **Arrange-Act-Assert**: Structure tests with setup, action, and verification
4. **Independent tests**: Tests should not depend on each other
5. **Use fixtures**: Reuse common setup code via fixtures
6. **Mark appropriately**: Use markers for test categorization
7. **Test edge cases**: Include boundary conditions and error cases

## Continuous Integration

Tests run automatically on GitHub Actions for:
- All pushes to `main` and `develop` branches
- All pull requests to `main` and `develop` branches

CI runs multiple test configurations:
- Python 3.8, 3.9, 3.10, 3.11 on Ubuntu
- With and without optional dependencies
- Code quality checks (linting)

## Troubleshooting

### Missing Dependencies

**Problem**: Tests fail with `ModuleNotFoundError`

**Solution**:
```bash
# Install base dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install training dependencies (PyTorch, OpenFold)
pip install -e ".[training]"
```

### Test Data Issues

**Problem**: Tests fail with file not found errors

**Solution**: Ensure `tests/data/` directory contains required test files:
- `3nir.pdb`, `3nir.cif`, `3nir.fasta`
- `3DB7_A.pdb`, `3DB7_A.fasta`
- `3M8J_A.pdb`, `3M8J_A.fasta`
- `5K1S_1.pdb`
- `3nir.pr.csv`

### PyTorch/OpenFold Tests Failing

**Problem**: Tests requiring PyTorch or OpenFold fail

**Solution**: Either install the dependencies or skip those tests:
```bash
# Option 1: Install dependencies
pip install -e ".[training]"

# Option 2: Skip those tests
pytest tests/ -m "not requires_torch and not requires_openfold"
```

### Slow Tests Taking Too Long

**Problem**: Test suite takes too long during development

**Solution**: Exclude slow tests:
```bash
# Use the fast test command
make test

# Or explicitly exclude slow tests
pytest tests/ -m "not slow"
```

## Test Metrics

Current test coverage (target: >80%):
- Core utilities: High coverage
- CLI commands: High coverage
- Model configurations: High coverage
- Analysis modules: High coverage
- Loss functions: Medium coverage (requires PyTorch)
- Data pipelines: Medium coverage (requires OpenFold)

## Adding New Tests

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Add to appropriate test file** or create new one
3. **Use appropriate markers** for categorization
4. **Update this documentation** if adding new test categories
5. **Ensure CI passes** before merging

## Performance Considerations

- Fast tests (<1s each) should be unmarked or marked as `unit`
- Slow tests (>5s) must be marked with `@pytest.mark.slow`
- Integration tests should be marked with `@pytest.mark.integration`
- Tests requiring external data should be marked with `@pytest.mark.requires_data`

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)