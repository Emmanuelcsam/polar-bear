import unittest
import os
import json
import time
import tempfile
from unittest.mock import patch
from logger import Logger, quick_log

class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_file = 'system_log.json'
        self.backup_file = None
        
        # Backup existing log file if it exists
        if os.path.exists(self.log_file):
            self.backup_file = tempfile.mktemp(suffix='.json')
            with open(self.log_file, 'r') as f:
                self.backup_data = f.read()
            with open(self.backup_file, 'w') as f:
                f.write(self.backup_data)
    
    def tearDown(self):
        # Clean up log file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        
        # Restore backup if it existed
        if self.backup_file and os.path.exists(self.backup_file):
            with open(self.backup_file, 'r') as f:
                data = f.read()
            with open(self.log_file, 'w') as f:
                f.write(data)
            os.remove(self.backup_file)
    
    def test_logger_initialization(self):
        # Test logger initialization
        logger = Logger("TEST_MODULE")
        
        self.assertEqual(logger.module, "TEST_MODULE")
        self.assertEqual(logger.log_file, 'system_log.json')
        self.assertIsInstance(logger.logs, list)
    
    def test_log_basic(self):
        # Test basic logging
        logger = Logger("TEST")
        
        with patch('builtins.print') as mock_print:
            logger.log("Test message")
        
        # Check that message was printed
        mock_print.assert_called_with("[TEST] Test message")
        
        # Check that log was saved
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['module'], "TEST")
        self.assertEqual(logs[0]['level'], "INFO")
        self.assertEqual(logs[0]['message'], "Test message")
        self.assertIn('timestamp', logs[0])
    
    def test_log_levels(self):
        # Test different log levels
        logger = Logger("TEST")
        
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]['level'], "INFO")
        self.assertEqual(logs[1]['level'], "WARNING")
        self.assertEqual(logs[2]['level'], "ERROR")
    
    def test_log_persistence(self):
        # Test that logs persist across logger instances
        logger1 = Logger("MODULE1")
        logger1.log("Message 1")
        
        # Create new logger instance
        logger2 = Logger("MODULE2")
        logger2.log("Message 2")
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]['module'], "MODULE1")
        self.assertEqual(logs[1]['module'], "MODULE2")
    
    def test_log_limit(self):
        # Test that logs are limited to 1000 entries
        logger = Logger("TEST")
        
        # Add more than 1000 entries
        for i in range(1100):
            logger.logs.append({
                'timestamp': time.time(),
                'module': 'TEST',
                'level': 'INFO',
                'message': f'Message {i}'
            })
        
        # Trigger save by logging one more
        logger.log("Final message")
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        # Should be limited to 1000
        self.assertEqual(len(logs), 1000)
        
        # Check that we kept the last 1000 entries
        self.assertEqual(logs[-1]['message'], "Final message")
    
    def test_quick_log(self):
        # Test quick_log utility function
        with patch('builtins.print') as mock_print:
            quick_log("QUICK_TEST", "Quick message")
        
        mock_print.assert_called_with("[QUICK_TEST] Quick message")
        
        # Check that log was saved
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        last_log = logs[-1]
        self.assertEqual(last_log['module'], "QUICK_TEST")
        self.assertEqual(last_log['message'], "Quick message")
    
    def test_timestamp_order(self):
        # Test that timestamps are in order
        logger = Logger("TEST")
        
        logger.log("Message 1")
        time.sleep(0.01)  # Small delay
        logger.log("Message 2")
        time.sleep(0.01)
        logger.log("Message 3")
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        # Check timestamps are increasing
        for i in range(1, len(logs)):
            self.assertGreater(logs[i]['timestamp'], logs[i-1]['timestamp'])
    
    def test_multiple_modules(self):
        # Test logging from multiple modules
        # Clear the log file first
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            
        # Create loggers and log messages
        Logger("MODULE_A").info("From A")
        Logger("MODULE_B").warning("From B")
        Logger("MODULE_C").error("From C")
        
        # Read the log file
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        # Get last 3 entries (in case there are other logs)
        recent_logs = logs[-3:] if len(logs) >= 3 else logs
        modules = [log['module'] for log in recent_logs]
        
        # Check all modules are present
        self.assertIn("MODULE_A", modules)
        self.assertIn("MODULE_B", modules)
        self.assertIn("MODULE_C", modules)
        
        # Also check the levels
        levels = [log['level'] for log in recent_logs]
        self.assertIn("INFO", levels)
        self.assertIn("WARNING", levels)
        self.assertIn("ERROR", levels)
    
    def test_special_characters(self):
        # Test logging messages with special characters
        logger = Logger("TEST")
        
        special_messages = [
            "Message with 'quotes'",
            'Message with "double quotes"',
            "Message with\nnewline",
            "Message with\ttab",
            "Message with unicode: 🐻"
        ]
        
        for msg in special_messages:
            logger.log(msg)
        
        with open(self.log_file, 'r') as f:
            logs = json.load(f)
        
        # Check that all messages were saved correctly
        saved_messages = [log['message'] for log in logs[-len(special_messages):]]
        for msg in special_messages:
            self.assertIn(msg, saved_messages)

if __name__ == '__main__':
    unittest.main()