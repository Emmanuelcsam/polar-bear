#!/usr/bin/env python3
"""
Unit tests for orchestrator.py
"""
import unittest
import os
import tempfile
import importlib
from unittest.mock import patch, MagicMock
import orchestrator

class TestOrchestrator(unittest.TestCase):

    @patch('orchestrator.os.path.dirname')
    @patch('orchestrator.os.path.abspath')
    @patch('orchestrator.os.listdir')
    @patch('orchestrator.importlib.import_module')
    def test_load_all_success(self, mock_import, mock_listdir, mock_abspath, mock_dirname):
        """Test load_all with successful module loading."""
        # Mock directory and file listing
        mock_dirname.return_value = "/test/dir"
        mock_abspath.return_value = "/test/dir/orchestrator.py"
        mock_listdir.return_value = [
            "module1.py",
            "module2.py",
            "orchestrator.py",  # Should be skipped
            "test_module.py",   # Should be skipped
            "not_python.txt"    # Should be skipped
        ]

        # Mock import_module
        mock_import.return_value = MagicMock()

        with patch('orchestrator.print') as mock_print:
            result = orchestrator.load_all()

            # Check that correct modules were loaded
            expected_modules = ["module1", "module2"]
            self.assertEqual(result, expected_modules)

            # Check that import_module was called correctly
            mock_import.assert_any_call("module1")
            mock_import.assert_any_call("module2")
            self.assertEqual(mock_import.call_count, 2)

            # Check print messages
            mock_print.assert_any_call("[Orch] Loaded module1")
            mock_print.assert_any_call("[Orch] Loaded module2")

    @patch('orchestrator.os.path.dirname')
    @patch('orchestrator.os.path.abspath')
    @patch('orchestrator.os.listdir')
    @patch('orchestrator.importlib.import_module')
    def test_load_all_with_errors(self, mock_import, mock_listdir, mock_abspath, mock_dirname):
        """Test load_all with import errors."""
        # Mock directory and file listing
        mock_dirname.return_value = "/test/dir"
        mock_abspath.return_value = "/test/dir/orchestrator.py"
        mock_listdir.return_value = ["good_module.py", "bad_module.py"]

        # Mock import_module to succeed for one and fail for another
        def import_side_effect(name):
            if name == "good_module":
                return MagicMock()
            elif name == "bad_module":
                raise ImportError("Test import error")

        mock_import.side_effect = import_side_effect

        with patch('orchestrator.print') as mock_print:
            result = orchestrator.load_all()

            # Check that only successful module was returned
            self.assertEqual(result, ["good_module"])

            # Check print messages
            mock_print.assert_any_call("[Orch] Loaded good_module")
            mock_print.assert_any_call("[Orch] Failed to load bad_module: Test import error")

    @patch('orchestrator.os.path.dirname')
    @patch('orchestrator.os.path.abspath')
    @patch('orchestrator.os.listdir')
    @patch('orchestrator.sys.modules')
    @patch('orchestrator.importlib.reload')
    def test_load_all_with_reload(self, mock_reload, mock_modules, mock_listdir, mock_abspath, mock_dirname):
        """Test load_all with module reloading."""
        # Mock directory and file listing
        mock_dirname.return_value = "/test/dir"
        mock_abspath.return_value = "/test/dir/orchestrator.py"
        mock_listdir.return_value = ["existing_module.py"]

        # Mock existing module in sys.modules
        mock_existing_module = MagicMock()
        mock_modules.__contains__ = lambda name: name == "existing_module"
        mock_modules.__getitem__ = lambda name: mock_existing_module
        mock_reload.return_value = mock_existing_module

        with patch('orchestrator.print') as mock_print:
            result = orchestrator.load_all()

            # Check that reload was called
            mock_reload.assert_called_once_with(mock_existing_module)

            # Check result
            self.assertEqual(result, ["existing_module"])
            mock_print.assert_called_with("[Orch] Loaded existing_module")

    @patch('orchestrator.os.path.dirname')
    @patch('orchestrator.os.path.abspath')
    @patch('orchestrator.os.listdir')
    def test_load_all_no_modules(self, mock_listdir, mock_abspath, mock_dirname):
        """Test load_all with no Python modules."""
        # Mock directory and file listing
        mock_dirname.return_value = "/test/dir"
        mock_abspath.return_value = "/test/dir/orchestrator.py"
        mock_listdir.return_value = ["orchestrator.py", "test_something.py", "readme.txt"]

        with patch('orchestrator.print') as mock_print:
            result = orchestrator.load_all()

            # Should return empty list
            self.assertEqual(result, [])

            # No load messages should be printed
            mock_print.assert_not_called()

if __name__ == "__main__":
    unittest.main()
