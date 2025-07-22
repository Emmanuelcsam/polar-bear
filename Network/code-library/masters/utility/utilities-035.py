import time
import json
import os

class Logger:
    def __init__(self, module_name):
        self.module = module_name
        self.log_file = 'system_log.json'
        self.load_log()
    
    def load_log(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.logs = json.load(f)
        else:
            self.logs = []
    
    def log(self, message, level='INFO'):
        entry = {
            'timestamp': time.time(),
            'module': self.module,
            'level': level,
            'message': message
        }
        
        # Print to terminal
        print(f"[{self.module}] {message}")
        
        # Add to log
        self.logs.append(entry)
        
        # Keep only last 1000 entries
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
        
        # Save log
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f)
    
    def error(self, message):
        self.log(message, 'ERROR')
    
    def warning(self, message):
        self.log(message, 'WARNING')
    
    def info(self, message):
        self.log(message, 'INFO')

# Utility function for quick logging
def quick_log(module, message):
    logger = Logger(module)
    logger.log(message)

if __name__ == "__main__":
    # Test logger
    logger = Logger("LOGGER_TEST")
    logger.info("Logger initialized")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    print(f"[LOGGER] Log file created: {logger.log_file}")