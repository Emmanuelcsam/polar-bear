#!/usr/bin/env python3
"""
Script Wrapper Base Class
Provides a standard interface for all scripts to work independently and with connectors
"""

import sys
import json
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

class ScriptWrapper(ABC):
    """Base class for wrapping scripts with connector interface"""
    
    def __init__(self, script_name: str):
        self.script_name = script_name
        self.parameters = {}
        self.variables = {}
        self.logger = self._setup_logger()
        self.is_standalone = __name__ == "__main__"
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the script"""
        logger = logging.getLogger(self.script_name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f'[{self.script_name}] %(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the script with parameters"""
        pass
        
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Main processing function"""
        pass
        
    def set_parameter(self, name: str, value: Any) -> bool:
        """Set a parameter value"""
        if name in self.parameters:
            old_value = self.parameters[name]
            self.parameters[name] = value
            self.logger.info(f"Parameter '{name}' changed from {old_value} to {value}")
            return True
        return False
        
    def get_parameter(self, name: str) -> Any:
        """Get a parameter value"""
        return self.parameters.get(name)
        
    def set_variable(self, name: str, value: Any) -> bool:
        """Set a variable value"""
        self.variables[name] = value
        return True
        
    def get_variable(self, name: str) -> Any:
        """Get a variable value"""
        return self.variables.get(name)
        
    def get_info(self) -> Dict[str, Any]:
        """Get script information"""
        return {
            'name': self.script_name,
            'parameters': self.parameters,
            'variables': self.variables,
            'methods': [method for method in dir(self) if not method.startswith('_') and callable(getattr(self, method))]
        }
        
    def collaborate(self, other_script: 'ScriptWrapper', data: Any) -> Any:
        """Collaborate with another script"""
        self.logger.info(f"Collaborating with {other_script.script_name}")
        # Process data and pass to other script
        processed = self.process(data)
        return other_script.process(processed)
        
    def run_standalone(self):
        """Run the script in standalone mode"""
        self.logger.info(f"Running {self.script_name} in standalone mode")
        self.initialize()
        
        # Example standalone execution
        if hasattr(self, 'main'):
            return self.main()
        else:
            self.logger.warning("No main() method defined for standalone execution")
            
    def run_integrated(self, action: Dict[str, Any]) -> Any:
        """Run the script in integrated mode (called by connector)"""
        action_type = action.get('type')
        
        if action_type == 'initialize':
            return self.initialize(**action.get('kwargs', {}))
        elif action_type == 'process':
            return self.process(action.get('data'))
        elif action_type == 'set_parameter':
            return self.set_parameter(action.get('name'), action.get('value'))
        elif action_type == 'get_parameter':
            return self.get_parameter(action.get('name'))
        elif action_type == 'set_variable':
            return self.set_variable(action.get('name'), action.get('value'))
        elif action_type == 'get_variable':
            return self.get_variable(action.get('name'))
        elif action_type == 'get_info':
            return self.get_info()
        else:
            return {'error': f'Unknown action type: {action_type}'}
            

class DataProcessor(ScriptWrapper):
    """Example data processor script wrapper"""
    
    def initialize(self, **kwargs):
        """Initialize with default parameters"""
        self.parameters = {
            'threshold': kwargs.get('threshold', 0.5),
            'max_iterations': kwargs.get('max_iterations', 100),
            'output_format': kwargs.get('output_format', 'json')
        }
        self.logger.info(f"Initialized with parameters: {self.parameters}")
        
    def process(self, data: Any) -> Any:
        """Process data according to parameters"""
        # Example processing
        if isinstance(data, (list, tuple)):
            threshold = self.parameters['threshold']
            return [item for item in data if item > threshold]
        return data
        
    def main(self):
        """Standalone main function"""
        # Example standalone usage
        test_data = [0.1, 0.5, 0.7, 0.3, 0.9]
        result = self.process(test_data)
        self.logger.info(f"Processed data: {result}")
        return result


def create_wrapper(script_class: type, script_name: str) -> ScriptWrapper:
    """Factory function to create script wrappers"""
    instance = script_class(script_name)
    
    # Check if running standalone or integrated
    if instance.is_standalone:
        instance.run_standalone()
    
    return instance