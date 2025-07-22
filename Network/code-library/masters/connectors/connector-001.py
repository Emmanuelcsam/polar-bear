
#!/usr/bin/env python3
"""
Enhanced Connector - Integrates with Mega Connector, Universal Connector, and Brain
Provides seamless integration and backward compatibility
"""

import os
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Try to import advanced connectors
try:
    from mega_connector import MegaConnector, get_mega_connector
    MEGA_AVAILABLE = True
except ImportError:
    MEGA_AVAILABLE = False

try:
    from universal_connector import UniversalConnector
    UNIVERSAL_AVAILABLE = True
except ImportError:
    UNIVERSAL_AVAILABLE = False

try:
    from polar_bear_brain import PolarBearBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

# --- Configuration ---
LOG_FILE = "connector.log"

# --- Setup Logging ---
# Ensure the logger is configured from scratch for each script
logger = logging.getLogger(os.path.abspath(__file__))
logger.setLevel(logging.INFO)

# Prevent logging from propagating to the root logger
logger.propagate = False

# Remove any existing handlers to avoid duplicate logs
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")

# File Handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
except (IOError, OSError) as e:
    # Fallback to console if file logging fails
    print(f"Could not write to log file {LOG_FILE}: {e}", file=sys.stderr)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class Connector:
    """Main connector class that integrates all systems"""
    
    def __init__(self):
        self.logger = logger
        self.mega = None
        self.universal = None
        self.brain = None
        
        # Initialize available systems
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize available connector systems"""
        self.logger.info("Initializing connector systems...")
        
        # Try Mega Connector first (most advanced)
        if MEGA_AVAILABLE:
            try:
                self.mega = get_mega_connector()
                self.logger.info("✓ Mega Connector initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Mega Connector: {e}")
        else:
            self.logger.info("Mega Connector not available")
        
        # Try Universal Connector
        if UNIVERSAL_AVAILABLE and not self.mega:
            try:
                self.universal = UniversalConnector()
                self.logger.info("✓ Universal Connector initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize Universal Connector: {e}")
        
        # Check Brain availability
        if BRAIN_AVAILABLE:
            self.logger.info("✓ Polar Bear Brain available")
        
        # Summary
        if self.mega:
            self.logger.info("Using Mega Connector (full capabilities)")
        elif self.universal:
            self.logger.info("Using Universal Connector (standard capabilities)")
        else:
            self.logger.info("Using basic connector (limited capabilities)")
    
    def execute(self, script: str, function: str = 'main', *args, **kwargs) -> Dict:
        """Execute a function from any script"""
        if self.mega:
            result = self.mega.execute_script(script, function, *args, **kwargs)
            return {
                'success': result.success,
                'output': result.output,
                'error': result.error,
                'duration': result.duration
            }
        elif self.universal:
            try:
                output = self.universal.execute_function(script, function, *args, **kwargs)
                return {'success': True, 'output': output, 'error': None}
            except Exception as e:
                return {'success': False, 'output': None, 'error': str(e)}
        else:
            return {'success': False, 'output': None, 'error': 'No advanced connector available'}
    
    def analyze_image(self, image_path: str, analysis_type: str = 'complete') -> Dict:
        """Analyze an image using the best available method"""
        if self.mega:
            return self.mega.analyze_image(image_path, analysis_type)
        else:
            # Fallback to basic analysis
            self.logger.warning("Advanced analysis not available, using basic method")
            return self.execute('defect-detector-35', 'detect_defects', image_path)
    
    def get_status(self) -> Dict:
        """Get system status"""
        status = {
            'connector_type': 'unknown',
            'capabilities': [],
            'systems_available': {
                'mega_connector': MEGA_AVAILABLE,
                'universal_connector': UNIVERSAL_AVAILABLE,
                'brain': BRAIN_AVAILABLE
            }
        }
        
        if self.mega:
            status['connector_type'] = 'mega'
            status['capabilities'] = ['full_analysis', 'pipelines', 'learning', 'ai_models']
            mega_status = self.mega.get_system_status()
            status.update(mega_status)
        elif self.universal:
            status['connector_type'] = 'universal'
            status['capabilities'] = ['script_execution', 'basic_pipelines']
            status['scripts_loaded'] = len(self.universal.scripts)
        else:
            status['connector_type'] = 'basic'
            status['capabilities'] = ['directory_listing']
        
        return status
    
    def list_scripts(self) -> List[str]:
        """List available scripts"""
        if self.mega:
            return list(self.mega.scripts.keys())
        elif self.universal:
            return list(self.universal.scripts.keys())
        else:
            # Basic listing
            return [f.stem for f in Path('.').glob('*.py') if f.stem != 'connector']
    
    def run_pipeline(self, pipeline_name: str, input_data: Any, config: Dict = None) -> Dict:
        """Run a processing pipeline"""
        if self.mega:
            return self.mega.run_pipeline(pipeline_name, input_data, config)
        else:
            self.logger.error("Pipeline execution requires Mega Connector")
            return {'success': False, 'error': 'Pipeline feature not available'}
    
    def interactive_mode(self):
        """Run interactive mode"""
        if self.mega:
            self.mega.interactive_mode()
        elif self.universal:
            # Basic interactive mode
            self._basic_interactive_mode()
        else:
            self.logger.error("Interactive mode not available with basic connector")
    
    def _basic_interactive_mode(self):
        """Basic interactive mode for universal connector"""
        print("\n=== Connector Interactive Mode ===\n")
        
        while True:
            print("\nOptions:")
            print("1. List scripts")
            print("2. Execute function")
            print("3. System status")
            print("4. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                scripts = self.list_scripts()
                print(f"\nFound {len(scripts)} scripts:")
                for script in scripts[:20]:
                    print(f"  - {script}")
                if len(scripts) > 20:
                    print(f"  ... and {len(scripts) - 20} more")
                    
            elif choice == '2':
                script = input("Enter script name: ").strip()
                function = input("Enter function name [main]: ").strip() or "main"
                result = self.execute(script, function)
                print(f"\nResult: {result}")
                
            elif choice == '3':
                status = self.get_status()
                print(f"\nStatus: {json.dumps(status, indent=2)}")
                
            elif choice == '4':
                break
            else:
                print("Invalid choice")


def main():
    """Main function for the connector script."""
    logger.info(f"--- Enhanced Connector Script Initialized ---")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Create connector instance
    connector = Connector()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'status':
            status = connector.get_status()
            print(json.dumps(status, indent=2))
            
        elif command == 'list':
            scripts = connector.list_scripts()
            print(f"Available scripts ({len(scripts)}):")
            for script in scripts:
                print(f"  - {script}")
                
        elif command == 'analyze' and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            analysis_type = sys.argv[3] if len(sys.argv) > 3 else 'complete'
            result = connector.analyze_image(image_path, analysis_type)
            print(json.dumps(result, indent=2))
            
        elif command == 'execute' and len(sys.argv) >= 4:
            script = sys.argv[2]
            function = sys.argv[3]
            args = sys.argv[4:] if len(sys.argv) > 4 else []
            result = connector.execute(script, function, *args)
            print(json.dumps(result, indent=2))
            
        elif command == 'interactive':
            connector.interactive_mode()
            
        else:
            print("Usage:")
            print("  python connector.py                           # Show this help")
            print("  python connector.py status                    # System status")
            print("  python connector.py list                      # List scripts")
            print("  python connector.py analyze <image> [type]    # Analyze image")
            print("  python connector.py execute <script> <func>   # Execute function")
            print("  python connector.py interactive               # Interactive mode")
    else:
        # Show basic info and usage
        logger.info("Connector is ready. Use command line arguments or import this module.")
        
        # List available capabilities
        status = connector.get_status()
        logger.info(f"Connector type: {status['connector_type']}")
        logger.info(f"Capabilities: {', '.join(status['capabilities'])}")
        
        # List files in directory (backward compatibility)
        try:
            files = os.listdir()
            if files:
                logger.info("Files in this directory:")
                for file in files[:10]:  # Show first 10
                    logger.info(f"- {file}")
                if len(files) > 10:
                    logger.info(f"... and {len(files) - 10} more files")
            else:
                logger.info("No files found in this directory.")
        except OSError as e:
            logger.error(f"Error listing files: {e}")
    
    logger.info("Connector script finished.")

# Global connector instance for module usage
_global_connector = None

def get_connector() -> Connector:
    """Get or create global connector instance"""
    global _global_connector
    if _global_connector is None:
        _global_connector = Connector()
    return _global_connector

# Convenience functions for module usage
def execute(script: str, function: str = 'main', *args, **kwargs) -> Dict:
    """Execute a function from any script"""
    return get_connector().execute(script, function, *args, **kwargs)

def analyze(image_path: str, analysis_type: str = 'complete') -> Dict:
    """Analyze an image"""
    return get_connector().analyze_image(image_path, analysis_type)

def status() -> Dict:
    """Get system status"""
    return get_connector().get_status()

if __name__ == "__main__":
    main()
