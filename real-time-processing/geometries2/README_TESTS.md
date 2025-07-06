# Geometry Detection System - Test Documentation

This directory contains a comprehensive test suite for the Geometry Detection System, ensuring all components work correctly and reliably.

## Test Coverage Overview

### 1. **test_integrated_geometry_system.py**
Tests for the main geometry detection system including:
- Configuration management (Config class)
- Shape type enumeration and data structures
- Camera backends (OpenCV and Pylon)
- Kalman filtering for temporal smoothing
- Core geometry detection algorithms
- Tube angle detection
- Performance monitoring
- Visualization components
- Complete system integration

**Total functions tested: ~45+**

### 2. **test_performance_benchmark_tool.py**
Tests for the performance benchmarking system:
- Benchmark result data structures
- Test scenario generation (Simple and Complex)
- Performance measurement and comparison
- Hardware information gathering
- Results visualization and export
- Quick benchmark functionality

**Total functions tested: ~15+**

### 3. **test_realtime_calibration_tool.py**
Tests for the interactive calibration system:
- Calibration configuration management
- GUI controls and parameter updates
- Real-time detection parameter tuning
- Shape filtering and display options
- ROI selection functionality
- Configuration save/load
- Batch parameter optimization

**Total functions tested: ~25+**

### 4. **test_example_application.py**
Tests for the example shape analysis dashboard:
- Frame processing pipeline
- Shape filtering (by type, area, confidence)
- Statistical analysis and tracking
- Alert generation system
- Data export functionality
- Visualization components
- Keyboard controls

**Total functions tested: ~15+**

### 5. **test_setup_installer.py**
Tests for the installation helper:
- System requirements checking
- Package installation handling
- Camera detection and testing
- Configuration wizard
- System component testing
- Platform-specific functionality

**Total functions tested: ~30+**

### 6. **test_uv_compatible_setup.py**
Tests for the UV package manager integration:
- UV availability checking
- Package installation via UV
- Import verification
- OpenCV functionality testing
- Test script generation

**Total functions tested: ~10+**

### 7. **test_python313_fix.py**
Tests for Python 3.13 OpenCV compatibility:
- PyPI version fetching
- Version filtering (yanked packages)
- Sequential installation attempts
- Error handling and fallbacks

**Total functions tested: ~5+**

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements_test.txt
```

### Running Individual Test Suites

Run a specific test module:
```bash
python test_integrated_geometry_system.py
python test_performance_benchmark_tool.py
# etc.
```

### Running All Tests

Use the master test runner:
```bash
python run_all_tests.py
```

### Validation

Check if all tests can be imported and dependencies are met:
```bash
python validate_tests.py
```

## Test Structure

Each test file follows a consistent structure:

1. **Import statements** - Standard library, third-party, and module imports
2. **Test classes** - Organized by component/functionality
3. **setUp/tearDown** - Test fixtures and cleanup
4. **Test methods** - Individual test cases with descriptive names
5. **Integration tests** - End-to-end scenario testing
6. **Main block** - Allows running tests directly

## Testing Best Practices

1. **Comprehensive Coverage**: Every public function and method has at least one test
2. **Edge Cases**: Tests include boundary conditions, error cases, and unusual inputs
3. **Mocking**: External dependencies are mocked to ensure isolated testing
4. **Assertions**: Clear, specific assertions with helpful failure messages
5. **Documentation**: Each test has a docstring explaining what it tests

## Common Test Patterns

### Mocking External Dependencies
```python
@patch('cv2.VideoCapture')
def test_camera_open(self, mock_capture):
    mock_cap = Mock()
    mock_cap.isOpened.return_value = True
    mock_capture.return_value = mock_cap
    # ... test code
```

### Testing Error Conditions
```python
def test_invalid_input(self):
    with self.assertRaises(ValueError):
        function_under_test(invalid_input)
```

### Integration Testing
```python
def test_full_pipeline(self):
    # Set up complete scenario
    # Run through entire workflow
    # Verify end results
```

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements_test.txt
    python run_all_tests.py
```

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements_txt.txt`
- Check Python version compatibility (requires Python 3.8+)

### Test Failures
- Run individual test files to isolate issues
- Check test output for detailed error messages
- Ensure test data/fixtures are properly set up

### Performance Issues
- Some tests may take longer due to image processing
- Use pytest markers to skip slow tests if needed
- Consider running tests in parallel with pytest-xdist

## Contributing

When adding new functionality:
1. Write tests FIRST (TDD approach)
2. Ensure all tests pass before committing
3. Maintain test coverage above 80%
4. Document any special test requirements

## Test Metrics

- **Total Test Files**: 7
- **Total Test Functions**: ~145+
- **Total Assertions**: ~500+
- **Code Coverage Goal**: >80%

## License

Tests are part of the main project and follow the same license terms.