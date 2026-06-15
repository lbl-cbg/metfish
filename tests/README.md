# MetFish Test Suite

This directory contains comprehensive test cases for the metfish package.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_utils.py            # Tests for core utility functions
├── test_commands.py         # Tests for CLI commands
├── test_msa_model.py        # Tests for MSA model components
├── test_refinement_model.py # Tests for refinement model components
├── test_analysis.py         # Tests for analysis modules
├── test_integration.py      # Integration and end-to-end tests
├── test_extract_seq.py      # Existing sequence extraction tests
├── test_get_Pr.py          # Existing P(r) calculation tests
├── test_loss.py            # Existing loss function tests
└── data/                   # Test data files
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# Tests that don't require PyTorch
pytest tests/ -m "not requires_torch"

# Tests that don't require OpenFold
pytest tests/ -m "not requires_openfold"

# Exclude slow tests
pytest tests/ -m "not slow"
```

### Run tests with coverage
```bash
pytest tests/ --cov=src/metfish --cov-report=html --cov-report=term-missing
```

### Run specific test file
```bash
pytest tests/test_utils.py -v
```

### Run specific test function
```bash
pytest tests/test_utils.py::TestGetPr::test_get_pr_from_pdb -v
```

## Test Markers

Tests are marked with the following markers:

- `unit`: Unit tests for individual functions/methods
- `integration`: Integration tests across multiple modules
- `slow`: Tests that take significant time to run
- `requires_data`: Tests that require test data files
- `requires_torch`: Tests that require PyTorch
- `requires_openfold`: Tests that require OpenFold

## Test Coverage

Generate and view coverage report:
```bash
pytest tests/ --cov=src/metfish --cov-report=html
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Writing New Tests

### Test Structure
Each test file should follow this structure:
```python
"""
Tests for [module name]
"""
import pytest
from metfish.module import function_to_test


class TestClassName:
    """Tests for ClassName."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = function_to_test()
        assert result is not None
    
    @pytest.mark.requires_data
    def test_with_test_data(self, test_data_dir):
        """Test using test data."""
        data_path = test_data_dir / "data.file"
        result = function_to_test(data_path)
        assert result is not None
```

### Using Fixtures
Common fixtures are defined in `conftest.py`:
- `test_data_dir`: Path to test data directory
- `temp_dir`: Temporary directory for test outputs
- `sample_pdb_path`: Path to sample PDB file
- `sample_structure`: Loaded BioPython structure
- And more...

### Best Practices
1. Use descriptive test names that explain what is being tested
2. Use fixtures for common test data and setup
3. Mark tests appropriately (slow, requires_torch, etc.)
4. Include both positive and negative test cases
5. Test edge cases and error handling
6. Keep tests independent and isolated

## Continuous Integration

Tests are automatically run on:
- Push to main/develop branches
- Pull requests to main/develop branches

See `.github/workflows/tests.yml` for CI configuration.

## Troubleshooting

### Tests fail due to missing dependencies
Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Tests fail due to missing PyTorch/OpenFold
These are optional dependencies. Either:
1. Install them: `pip install -e ".[training]"`
2. Skip those tests: `pytest tests/ -m "not requires_torch and not requires_openfold"`

### Tests fail due to missing test data
Ensure the `tests/data/` directory contains all required test files.

### Permission errors on test outputs
The test suite uses `tmp_path` fixture which creates temporary directories.
These are automatically cleaned up after tests complete.

## Adding New Test Data

1. Add new test data files to `tests/data/`
2. Update `conftest.py` with new fixtures if needed
3. Reference the data using fixtures in your tests
4. Document the test data requirements