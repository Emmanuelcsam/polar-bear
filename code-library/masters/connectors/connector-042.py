#!/usr/bin/env python3
"""
Enhanced Connector - Integrates with Universal Connector and Polar Bear Brain
Provides backward compatibility while adding new capabilities
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the universal connector
try:
    from universal_connector import UniversalConnector
except ImportError:
    UniversalConnector = None

# Import the brain
try:
    from polar_bear_brain import PolarBearBrain
except ImportError:
    PolarBearBrain = None

class EnhancedConnector:
    """Enhanced connector with full integration capabilities"""
    
    def __init__(self):
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.universal = UniversalConnector() if UniversalConnector else None
        self.brain = None  # Brain is initialized on demand
        
        # State tracking
        self.initialized = False
        self.scripts_loaded = 0
        self.last_activity = time.time()
        
        self.logger.info("Enhanced Connector initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("EnhancedConnector")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = log_dir / f"connector_{time.strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def initialize(self):
        """Full system initialization"""
        if self.initialized:
            return True
        
        self.logger.info("Initializing Enhanced Connector System...")
        
        try:
            # Check components
            if not self.universal:
                self.logger.error("Universal Connector not available")
                return False
            
            # Get script information
            scripts = self.universal.get_script_info()
            self.scripts_loaded = len(scripts)
            
            self.logger.info(f"Loaded {self.scripts_loaded} scripts")
            
            # Display categories
            self._categorize_scripts(scripts)
            
            self.initialized = True
            self.logger.info("System initialization complete!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _categorize_scripts(self, scripts: Dict):
        """Categorize and display scripts"""
        categories = {
            'Detection': [],
            'Analysis': [],
            'AI/ML': [],
            'Preprocessing': [],
            'Utilities': [],
            'Other': []
        }
        
        for script_name in scripts:
            if 'detect' in script_name or 'defect' in script_name:
                categories['Detection'].append(script_name)
            elif 'analy' in script_name or 'report' in script_name:
                categories['Analysis'].append(script_name)
            elif 'ai' in script_name or 'neural' in script_name or 'learning' in script_name:
                categories['AI/ML'].append(script_name)
            elif 'preprocess' in script_name or 'enhance' in script_name:
                categories['Preprocessing'].append(script_name)
            elif 'util' in script_name or 'helper' in script_name:
                categories['Utilities'].append(script_name)
            else:
                categories['Other'].append(script_name)
        
        self.logger.info("\nScript Categories:")
        for category, script_list in categories.items():
            if script_list:
                self.logger.info(f"  {category}: {len(script_list)} scripts")
    
    def execute_script(self, script_name: str, function_name: str = 'main', 
                      *args, **kwargs) -> Dict[str, Any]:
        """Execute a function from a script"""
        self.last_activity = time.time()
        
        try:
            if not self.universal:
                return {'status': 'error', 'message': 'Universal Connector not available'}
            
            result = self.universal.execute_function(script_name, function_name, *args, **kwargs)
            
            return {
                'status': 'success',
                'result': result,
                'script': script_name,
                'function': function_name,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'script': script_name,
                'function': function_name
            }
    
    def get_parameter(self, script_name: str, param_name: str) -> Any:
        """Get a parameter value from a script"""
        try:
            if not self.universal:
                return None
            
            return self.universal.get_variable(script_name, param_name)
            
        except Exception as e:
            self.logger.error(f"Error getting parameter: {e}")
            return None
    
    def set_parameter(self, script_name: str, param_name: str, value: Any) -> bool:
        """Set a parameter value in a script"""
        try:
            if not self.universal:
                return False
            
            return self.universal.set_variable(script_name, param_name, value)
            
        except Exception as e:
            self.logger.error(f"Error setting parameter: {e}")
            return False
    
    def run_pipeline(self, task_type: str = 'full', **kwargs) -> Dict[str, Any]:
        """Run a predefined processing pipeline"""
        self.logger.info(f"Running {task_type} pipeline...")
        
        pipelines = {
            'full': [
                {'script': 'image-quality-enhancer', 'function': 'enhance_image'},
                {'script': 'fiber-finder-29', 'function': 'find_fiber', 'use_previous_result': True},
                {'script': 'defect-detection-ai-39', 'function': 'detect_defects', 'use_previous_result': True},
                {'script': 'report-generator-40', 'function': 'generate_report', 'use_previous_result': True}
            ],
            'quick': [
                {'script': 'fiber-finder-29', 'function': 'find_fiber'},
                {'script': 'defect-detector-35', 'function': 'detect_defects', 'use_previous_result': True}
            ],
            'ai': [
                {'script': 'image-segmentation-ai-39', 'function': 'segment'},
                {'script': 'defect-detection-ai-39', 'function': 'detect_defects', 'use_previous_result': True}
            ]
        }
        
        pipeline = pipelines.get(task_type, pipelines['quick'])
        
        # Add kwargs to first step
        if kwargs and pipeline:
            pipeline[0]['kwargs'] = kwargs
        
        try:
            if not self.universal:
                return {'status': 'error', 'message': 'Universal Connector not available'}
            
            results = self.universal.execute_pipeline(pipeline)
            
            return {
                'status': 'success',
                'pipeline': task_type,
                'results': results,
                'steps': len(pipeline)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'pipeline': task_type
            }
    
    def start_brain(self, mode: str = 'interactive') -> bool:
        """Start the Polar Bear Brain system"""
        try:
            if not PolarBearBrain:
                self.logger.error("Polar Bear Brain not available")
                return False
            
            if mode == 'background':
                # Start brain in background thread
                import threading
                self.brain = PolarBearBrain()
                brain_thread = threading.Thread(target=self.brain.run)
                brain_thread.daemon = True
                brain_thread.start()
                self.logger.info("Brain started in background mode")
            else:
                # Start brain in foreground
                self.brain = PolarBearBrain()
                self.brain.run()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start brain: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            'initialized': self.initialized,
            'scripts_loaded': self.scripts_loaded,
            'universal_connector': self.universal is not None,
            'brain_available': PolarBearBrain is not None,
            'brain_running': self.brain is not None,
            'last_activity': time.time() - self.last_activity,
            'uptime': time.time() - self.last_activity
        }
        
        if self.universal:
            # Add script categories
            scripts = self.universal.get_script_info()
            status['total_functions'] = sum(
                len(info['functions']) for info in scripts.values()
            )
            status['total_classes'] = sum(
                len(info['classes']) for info in scripts.values()
            )
        
        return status
    
    def list_available_functions(self, pattern: str = None) -> List[Dict]:
        """List all available functions across scripts"""
        functions = []
        
        if not self.universal:
            return functions
        
        scripts = self.universal.get_script_info()
        
        for script_name, info in scripts.items():
            for func_name, func_info in info['functions'].items():
                if pattern and pattern not in func_name:
                    continue
                
                functions.append({
                    'script': script_name,
                    'function': func_name,
                    'args': func_info.get('args', []),
                    'docstring': func_info.get('docstring', '')
                })
        
        return functions
    
    def interactive_mode(self):
        """Run in interactive mode"""
        self.logger.info("\n=== Enhanced Connector Interactive Mode ===\n")
        
        if not self.initialize():
            self.logger.error("Failed to initialize system")
            return
        
        while True:
            print("\nOptions:")
            print("1. Execute a script function")
            print("2. Run a pipeline")
            print("3. List available functions")
            print("4. Get/Set parameters")
            print("5. System status")
            print("6. Start Polar Bear Brain")
            print("7. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self._interactive_execute()
            elif choice == '2':
                self._interactive_pipeline()
            elif choice == '3':
                self._interactive_list_functions()
            elif choice == '4':
                self._interactive_parameters()
            elif choice == '5':
                self._show_status()
            elif choice == '6':
                self.start_brain()
            elif choice == '7':
                break
            else:
                print("Invalid choice")
        
        self.logger.info("Exiting interactive mode")
    
    def _interactive_execute(self):
        """Interactive function execution"""
        script = input("Enter script name: ").strip()
        function = input("Enter function name [main]: ").strip() or "main"
        
        # Check if function exists
        try:
            scripts = self.universal.get_script_info()
            if script in scripts and function in scripts[script]['functions']:
                func_info = scripts[script]['functions'][function]
                print(f"Function arguments: {func_info.get('args', [])}")
                
                # Get arguments
                args_str = input("Enter arguments (comma-separated): ").strip()
                args = [arg.strip() for arg in args_str.split(',')] if args_str else []
                
                result = self.execute_script(script, function, *args)
                print(f"\nResult: {result}")
            else:
                print(f"Function '{function}' not found in '{script}'")
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_pipeline(self):
        """Interactive pipeline execution"""
        print("\nAvailable pipelines:")
        print("1. full - Complete processing pipeline")
        print("2. quick - Quick detection pipeline")
        print("3. ai - AI-powered pipeline")
        
        pipeline = input("Select pipeline [quick]: ").strip() or "quick"
        
        # Get input image
        image_path = input("Enter image path: ").strip()
        
        if image_path and os.path.exists(image_path):
            result = self.run_pipeline(pipeline, image_path=image_path)
            print(f"\nPipeline result: {result}")
        else:
            print("Invalid image path")
    
    def _interactive_list_functions(self):
        """List functions interactively"""
        pattern = input("Enter search pattern (leave empty for all): ").strip()
        
        functions = self.list_available_functions(pattern)
        
        print(f"\nFound {len(functions)} functions:")
        for func in functions[:20]:  # Show first 20
            print(f"  - {func['script']}.{func['function']}({', '.join(func['args'])})")
        
        if len(functions) > 20:
            print(f"  ... and {len(functions) - 20} more")
    
    def _interactive_parameters(self):
        """Get/Set parameters interactively"""
        action = input("Get or Set parameter? [get/set]: ").strip().lower()
        
        if action == 'get':
            script = input("Enter script name: ").strip()
            param = input("Enter parameter name: ").strip()
            
            value = self.get_parameter(script, param)
            print(f"\n{script}.{param} = {value}")
            
        elif action == 'set':
            script = input("Enter script name: ").strip()
            param = input("Enter parameter name: ").strip()
            value = input("Enter new value: ").strip()
            
            # Try to parse value
            try:
                value = eval(value)
            except:
                pass  # Keep as string
            
            success = self.set_parameter(script, param, value)
            print(f"\nSet {'successful' if success else 'failed'}")
    
    def _show_status(self):
        """Show system status"""
        status = self.get_status()
        
        print("\n=== System Status ===")
        print(f"Initialized: {status['initialized']}")
        print(f"Scripts loaded: {status['scripts_loaded']}")
        print(f"Total functions: {status.get('total_functions', 'N/A')}")
        print(f"Total classes: {status.get('total_classes', 'N/A')}")
        print(f"Brain available: {status['brain_available']}")
        print(f"Brain running: {status['brain_running']}")
        print(f"Last activity: {status['last_activity']:.1f} seconds ago")

def main():
    """Main entry point"""
    # Create enhanced connector
    connector = EnhancedConnector()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'status':
            connector.initialize()
            status = connector.get_status()
            print(json.dumps(status, indent=2))
            
        elif command == 'execute' and len(sys.argv) >= 4:
            script = sys.argv[2]
            function = sys.argv[3]
            args = sys.argv[4:] if len(sys.argv) > 4 else []
            
            connector.initialize()
            result = connector.execute_script(script, function, *args)
            print(json.dumps(result, indent=2))
            
        elif command == 'brain':
            connector.start_brain()
            
        else:
            print("Usage:")
            print("  connector_enhanced.py                    # Interactive mode")
            print("  connector_enhanced.py status             # Show status")
            print("  connector_enhanced.py execute <script> <function> [args...]")
            print("  connector_enhanced.py brain              # Start brain")
    else:
        # Run in interactive mode
        connector.interactive_mode()

if __name__ == "__main__":
    main()