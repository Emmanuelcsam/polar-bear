
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
        self.analyzer = ScriptAnalyzer(self.logger)
        self.connectors = ConnectorManager(self.logger)
        self.task_manager = TaskManager(self.logger, self.analyzer, self.connectors)

    @patch('builtins.input', return_value='1')
    def test_get_task(self, mock_input):
        """Test that the correct task is returned based on user input."""
        with patch.object(self.task_manager, 'analyze_project') as mock_analyze_project:
            task = self.task_manager.get_task()
            task()
            mock_analyze_project.assert_called_once()


class TestPolarBearMaster(unittest.TestCase):
    @patch('polar_bear_master.PolarBearMaster.run_interactive_mode')
    @patch('polar_bear_master.ConfigurationManager.interactive_setup')
    def test_run_interactive(self, mock_interactive_setup, mock_run_interactive):
        """Test that the master script runs in interactive mode."""
        master = PolarBearMaster(test_mode=False)
        master.run()
        mock_run_interactive.assert_called_once()

    @patch('polar_bear_master.PolarBearMaster.run_tests')
    def test_run_test_mode(self, mock_run_tests):
        """Test that the master script runs in test mode."""
        master = PolarBearMaster(test_mode=True)
        master.run()
        mock_run_tests.assert_called_once()

if __name__ == "__main__":
    unittest.main()
