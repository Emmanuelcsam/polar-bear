import unittest
import os
import logging
from pathlib import Path
from unittest.mock import patch, mock_open

# Make sure the master script is in the python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from polar_bear_master import (
    LoggingManager,
    DependencyManager,
    ConfigurationManager,
    ConnectorManager,
    ScriptAnalyzer,
    TaskManager,
    PolarBearMaster,
)

class TestLoggingManager(unittest.TestCase):
    def setUp(self):
        self.log_file = "test.log"
        self.logger_manager = LoggingManager(log_file=self.log_file)

    def test_logger_creation(self):
        """Test that the logger is created correctly."""
        logger = self.logger_manager.get_logger()
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(len(logger.handlers), 2)

    def tearDown(self):
        logging.shutdown()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

class TestDependencyManager(unittest.TestCase):
    @patch('subprocess.check_call')
    def test_install_package(self, mock_check_call):
        """Test that the install_package method works as expected."""
        logger = logging.getLogger("TestLogger")
        dep_manager = DependencyManager(logger)

        with patch.dict('sys.modules', {'pkg_resources': unittest.mock.MagicMock()}):
            import pkg_resources
            
            # Define the exception on the mock
            pkg_resources.DistributionNotFound = Exception

            # Test case where package is already installed
            pkg_resources.get_distribution.return_value = True
            dep_manager.install_package("numpy")
            mock_check_call.assert_not_called()

            # Test case where package is not installed
            pkg_resources.get_distribution.side_effect = pkg_resources.DistributionNotFound
            dep_manager.install_package("requests")
            mock_check_call.assert_called_once()

class TestConfigurationManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestLogger")
        self.config_file = "test_config.json"
        self.config_manager = ConfigurationManager(self.logger, config_file=self.config_file)

    def tearDown(self):
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_interactive_setup(self):
        """Test the interactive setup of the configuration."""
        with patch('builtins.input', side_effect=['Test Project', 'DEBUG']):
            self.config_manager.interactive_setup()
        
        self.assertEqual(self.config_manager.get('project_name'), 'Test Project')
        self.assertEqual(self.config_manager.get('log_level'), 'DEBUG')

class TestConnectorManager(unittest.TestCase):
    def test_find_connectors(self):
        """Test the discovery of connector scripts."""
        logger = logging.getLogger("TestLogger")
        connector_manager = ConnectorManager(logger)
        
        # Create a dummy directory structure with connector files
        test_dir = Path("test_project")
        test_dir.mkdir()
        (test_dir / "connector.py").touch()
        (test_dir / "sub").mkdir()
        (test_dir / "sub" / "hivemind_connector.py").touch()

        connectors = connector_manager.find_connectors(str(test_dir))
        self.assertEqual(len(connectors), 2)

        # Clean up the dummy directory
        os.remove(test_dir / "connector.py")
        os.remove(test_dir / "sub" / "hivemind_connector.py")
        os.rmdir(test_dir / "sub")
        os.rmdir(test_dir)

class TestScriptAnalyzer(unittest.TestCase):
    def test_analyze_script(self):
        """Test the analysis of a single script."""
        logger = logging.getLogger("TestLogger")
        analyzer = ScriptAnalyzer(logger)
        
        # Create a dummy script file
        test_script_path = "test_script.py"
        with open(test_script_path, "w") as f:
            f.write("import os\n\ndef my_func():\n    pass\n\nclass MyClass:\n    pass")

        analysis = analyzer.analyze_script(test_script_path)
        self.assertEqual(analysis['path'], test_script_path)
        self.assertIn('os', analysis['imports'])
        self.assertIn('my_func', analysis['functions'])
        self.assertIn('MyClass', analysis['classes'])
        
        # Clean up the dummy script
        os.remove(test_script_path)

class TestTaskManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("TestLogger")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        self.test_log_file = "test_diagnostics.log"
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.addHandler(logging.FileHandler(self.test_log_file))

        self.config_manager = ConfigurationManager(self.logger)
        self.analyzer = ScriptAnalyzer(self.logger)
        self.connectors = ConnectorManager(self.logger)
        self.task_manager = TaskManager(self.logger, self.config_manager, self.analyzer, self.connectors)

    def tearDown(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)

    @patch('builtins.input', side_effect=['1', 'n'])
    @patch('pathlib.Path.rglob')
    @patch('polar_bear_master.ScriptAnalyzer.analyze_script')
    def test_analyze_project_flow(self, mock_analyze_script, mock_rglob, mock_input):
        """Test the full flow of the analyze_project task."""
        dummy_file = Path("dummy_test_file.py")
        mock_rglob.return_value = [dummy_file]
        mock_analyze_script.return_value = {
            'path': str(dummy_file), 'functions': ['a', 'b'], 'classes': ['C'],
            'imports': ['os', 'sys'], 'error': None
        }
        with self.assertLogs('TestLogger', level='INFO') as cm:
            task_func = self.task_manager.get_task()
            task_func()
            self.assertTrue(any("Project Analysis Summary" in s for s in cm.output))
        mock_analyze_script.assert_called_once_with(dummy_file)

    @patch('builtins.input', return_value='4')
    def test_get_task_exit(self, mock_input):
        """Test that the exit task is returned and raises SystemExit."""
        task_func = self.task_manager.get_task()
        with self.assertRaises(SystemExit):
            task_func()

    @patch('pathlib.Path.is_dir')
    @patch('pathlib.Path.exists')
    def test_run_diagnostics(self, mock_exists, mock_is_dir):
        """Test the diagnostics task runs and logs correctly."""
        mock_exists.return_value = True
        mock_is_dir.side_effect = [True, True, False, True]

        # Run the function that writes to the log
        self.task_manager.run_diagnostics()

        # Close the handlers to ensure the log file is written to disk
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()
        
        # Read the log file and check its contents
        with open(self.test_log_file, 'r') as f:
            log_output = f.read()

        self.assertIn("Diagnostics Summary", log_output)
        self.assertIn("[PASS] Configuration file found.", log_output)
        self.assertIn("[PASS] Log file found", log_output)
        self.assertIn("[PASS] Directory 'modules' found.", log_output)
        self.assertIn("[FAIL] Required directory 'docs' not found.", log_output)
        self.assertIn("Some diagnostic checks failed.", log_output)

    @patch('builtins.input', side_effect=['1', '2']) # List connectors, then exit
    @patch('polar_bear_master.ConnectorManager.find_connectors')
    def test_manage_connectors(self, mock_find_connectors, mock_input):
        """Test the connector management flow."""
        mock_find_connectors.return_value = [Path("conn1.py"), Path("conn2.py")]

        with self.assertLogs('TestLogger', level='INFO') as cm:
            self.task_manager.manage_connectors()
            log_output = "".join(cm.output)
            self.assertIn("Found 2 connector scripts.", log_output)
            self.assertIn("1. List all connector paths", log_output)
            self.assertIn("001: conn1.py", log_output)
            self.assertIn("002: conn2.py", log_output)
            self.assertIn("Exiting Connector Management.", log_output)

class TestPolarBearMaster(unittest.TestCase):
    @patch('polar_bear_master.TaskManager.display_menu')
    @patch('polar_bear_master.TaskManager.get_task')
    @patch('polar_bear_master.ConfigurationManager.load_config')
    @patch('polar_bear_master.DependencyManager.check_and_install')
    def test_run_interactive_mode(self, mock_check_install, mock_load_config, mock_get_task, mock_display_menu):
        """Test that the master script runs in interactive mode and can exit."""
        # Mock get_task to return the exit function after one loop
        mock_get_task.side_effect = [lambda: None, SystemExit]
        
        master = PolarBearMaster(test_mode=False)
        with self.assertRaises(SystemExit):
            master.run_interactive_mode()

        mock_display_menu.assert_called()
        self.assertGreaterEqual(mock_get_task.call_count, 1)

    @patch('polar_bear_master.TaskManager.run_diagnostics')
    @patch('polar_bear_master.TaskManager.analyze_project')
    @patch('polar_bear_master.TaskManager.manage_connectors')
    @patch('polar_bear_master.ConfigurationManager.load_config')
    @patch('polar_bear_master.DependencyManager.check_and_install')
    def test_run_test_mode(self, mock_check_install, mock_load_config, mock_manage, mock_analyze, mock_diagnostics):
        """Test that the master script runs the full test suite in test mode."""
        master = PolarBearMaster(test_mode=True)
        master.run_tests()

        mock_diagnostics.assert_called_once()
        mock_analyze.assert_called_once()
        mock_manage.assert_called_once()


if __name__ == "__main__":
    unittest.main()