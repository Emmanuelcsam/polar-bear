# Image Classifier Test Suite

This directory contains all test files and configurations for the Advanced Image Classifier.

## Contents

- `test_image_classifier.py` - Comprehensive test suite for all classifier features
- `test_knowledge_bank.pkl` - Test knowledge bank data (if exists)
- `test_config.json` - Test configuration file

## Running Tests

To run the comprehensive test suite:

```bash
cd ..
python classifier-tests/test_image_classifier.py
```

This will run:
1. Unit tests for KnowledgeBank class
2. Unit tests for AdvancedImageClassifier class
3. Integration tests for the complete workflow
4. Manual feature tests

## Test Coverage

The test suite covers:
- Hierarchical folder structure parsing (fc/50/core/clean, etc.)
- Dynamic folder creation based on classification
- Custom keyword support
- Reference image storage functionality
- Knowledge bank persistence
- Feature extraction (visual and deep learning)
- Image classification workflow
- Error handling and edge cases

## Test Results

The test suite will output:
- Detailed test results with pass/fail status
- Test summary with success rate
- Manual test output for visual verification

## Notes

- Tests use mocked models to avoid loading actual PyTorch models
- Temporary directories are created and cleaned up automatically
- Tests are designed to be run in isolation without affecting production data