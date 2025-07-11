#!/usr/bin/env python3
"""
Comprehensive Troubleshooting Script for Monitoring Systems
Tests all components and their integration with connectors
"""

import sys
import os
import json
import socket
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import psutil
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringSystemTroubleshooter:
    """Comprehensive troubleshooting for all monitoring components"""
    
    def __init__(self):
        self.results = {
            'system_checks': {},
            'dependency_checks': {},
            'script_checks': {},
            'connector_checks': {},
            'integration_tests': {},
            'recommendations': []
        }
        
        self.scripts_to_test = [
            'advanced_monitoring_system.py',
            'realtime_fiber_system.py',
            'streaming_data_pipeline.py',
            'comprehensive_testing_suite.py',
            'main-application.py'
        ]
        
        self.required_modules = [
            'numpy', 'pandas', 'torch', 'cv2', 'asyncio',
            'faust', 'websockets', 'redis', 'firebase_admin',
            'google.cloud.bigquery', 'prometheus_client'
        ]
        
    def run_all_checks(self):
        """Run all troubleshooting checks"""
        logger.info("Starting comprehensive troubleshooting...")
        
        # System checks
        self.check_system_resources()
        
        # Dependency checks
        self.check_dependencies()
        
        # Script checks
        self.check_scripts()
        
        # Connector checks
        self.check_connectors()
        
        # Integration tests
        self.test_integration()
        
        # Generate report
        self.generate_report()
        
    def check_system_resources(self):
        """Check system resources"""
        logger.info("Checking system resources...")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        self.results['system_checks']['cpu_usage'] = f"{cpu_percent}%"
        
        # Memory
        memory = psutil.virtual_memory()
        self.results['system_checks']['memory'] = {
            'total': f"{memory.total / (1024**3):.2f} GB",
            'available': f"{memory.available / (1024**3):.2f} GB",
            'percent': f"{memory.percent}%"
        }
        
        # Disk
        disk = psutil.disk_usage('/')
        self.results['system_checks']['disk'] = {
            'total': f"{disk.total / (1024**3):.2f} GB",
            'free': f"{disk.free / (1024**3):.2f} GB",
            'percent': f"{disk.percent}%"
        }
        
        # Network
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(('8.8.8.8', 80))
            sock.close()
            self.results['system_checks']['internet'] = 'Connected'
        except:
            self.results['system_checks']['internet'] = 'No connection'
            self.results['recommendations'].append(
                "No internet connection detected. Some features may not work."
            )
            
    def check_dependencies(self):
        """Check required Python modules"""
        logger.info("Checking dependencies...")
        
        for module_name in self.required_modules:
            try:
                if module_name == 'cv2':
                    import cv2
                else:
                    importlib.import_module(module_name)
                self.results['dependency_checks'][module_name] = 'Installed'
            except ImportError:
                self.results['dependency_checks'][module_name] = 'Missing'
                self.results['recommendations'].append(
                    f"Install missing module: pip install {module_name}"
                )
                
    def check_scripts(self):
        """Check if scripts exist and have basic syntax"""
        logger.info("Checking scripts...")
        
        for script in self.scripts_to_test:
            script_path = Path(script)
            
            if not script_path.exists():
                self.results['script_checks'][script] = 'Missing'
                continue
                
            # Check syntax
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'py_compile', str(script_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.results['script_checks'][script] = 'Valid syntax'
                else:
                    self.results['script_checks'][script] = f'Syntax error: {result.stderr}'
                    self.results['recommendations'].append(
                        f"Fix syntax errors in {script}"
                    )
                    
            except Exception as e:
                self.results['script_checks'][script] = f'Check failed: {str(e)}'
                
    def check_connectors(self):
        """Check connector availability and functionality"""
        logger.info("Checking connectors...")
        
        # Check main connector
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(('localhost', 10130))
            
            # Send status request
            message = json.dumps({'command': 'status'})
            sock.send(message.encode())
            
            # Receive response
            response = sock.recv(4096)
            if response:
                data = json.loads(response.decode())
                self.results['connector_checks']['main_connector'] = {
                    'status': 'Active',
                    'response': data
                }
            else:
                self.results['connector_checks']['main_connector'] = 'No response'
                
            sock.close()
            
        except ConnectionRefusedError:
            self.results['connector_checks']['main_connector'] = 'Not running'
            self.results['recommendations'].append(
                "Start the main connector: python connector.py"
            )
        except Exception as e:
            self.results['connector_checks']['main_connector'] = f'Error: {str(e)}'
            
        # Check hivemind connector
        try:
            # Check if hivemind_connector.py exists
            if Path('hivemind_connector.py').exists():
                self.results['connector_checks']['hivemind_connector'] = 'Script exists'
            else:
                self.results['connector_checks']['hivemind_connector'] = 'Script missing'
                
        except Exception as e:
            self.results['connector_checks']['hivemind_connector'] = f'Error: {str(e)}'
            
    def test_integration(self):
        """Test integration between components"""
        logger.info("Testing integration...")
        
        # Test 1: Connector interface import
        try:
            from connector_interface import ConnectorInterface
            self.results['integration_tests']['connector_interface'] = 'Importable'
        except ImportError as e:
            self.results['integration_tests']['connector_interface'] = f'Import error: {str(e)}'
            self.results['recommendations'].append(
                "Ensure connector_interface.py is in the current directory"
            )
            
        # Test 2: Script registry functionality
        try:
            from connector_interface import ScriptRegistry
            registry = ScriptRegistry()
            self.results['integration_tests']['script_registry'] = 'Functional'
        except Exception as e:
            self.results['integration_tests']['script_registry'] = f'Error: {str(e)}'
            
        # Test 3: Parameter registration
        try:
            from connector_interface import ConnectorInterface
            test_interface = ConnectorInterface('test_script.py', port=10131)
            test_interface.register_parameter('test_param', 42, 'int', 'Test parameter')
            value = test_interface.get_parameter('test_param')
            if value == 42:
                self.results['integration_tests']['parameter_system'] = 'Working'
            else:
                self.results['integration_tests']['parameter_system'] = 'Value mismatch'
        except Exception as e:
            self.results['integration_tests']['parameter_system'] = f'Error: {str(e)}'
            
    def test_script_execution(self, script_name: str) -> Tuple[bool, str]:
        """Test if a script can be executed"""
        try:
            # Try to run the script with a timeout
            result = subprocess.run(
                [sys.executable, script_name, '--test'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return True, "Executed successfully"
            else:
                return False, f"Exit code {result.returncode}: {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            return True, "Script started (timeout reached)"
        except Exception as e:
            return False, str(e)
            
    def generate_report(self):
        """Generate comprehensive troubleshooting report"""
        logger.info("Generating report...")
        
        report = []
        report.append("=" * 60)
        report.append("MONITORING SYSTEMS TROUBLESHOOTING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System checks
        report.append("SYSTEM RESOURCES:")
        report.append("-" * 40)
        for key, value in self.results['system_checks'].items():
            if isinstance(value, dict):
                report.append(f"{key.upper()}:")
                for k, v in value.items():
                    report.append(f"  {k}: {v}")
            else:
                report.append(f"{key}: {value}")
        report.append("")
        
        # Dependencies
        report.append("DEPENDENCIES:")
        report.append("-" * 40)
        missing_deps = []
        for module, status in self.results['dependency_checks'].items():
            report.append(f"{module}: {status}")
            if status == 'Missing':
                missing_deps.append(module)
                
        if missing_deps:
            report.append(f"\nMissing dependencies: {', '.join(missing_deps)}")
            report.append("Install with: pip install " + ' '.join(missing_deps))
        report.append("")
        
        # Scripts
        report.append("SCRIPTS:")
        report.append("-" * 40)
        for script, status in self.results['script_checks'].items():
            report.append(f"{script}: {status}")
        report.append("")
        
        # Connectors
        report.append("CONNECTORS:")
        report.append("-" * 40)
        for connector, status in self.results['connector_checks'].items():
            if isinstance(status, dict):
                report.append(f"{connector}: {status.get('status', 'Unknown')}")
            else:
                report.append(f"{connector}: {status}")
        report.append("")
        
        # Integration
        report.append("INTEGRATION TESTS:")
        report.append("-" * 40)
        for test, result in self.results['integration_tests'].items():
            report.append(f"{test}: {result}")
        report.append("")
        
        # Recommendations
        if self.results['recommendations']:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for i, rec in enumerate(self.results['recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")
            
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        
        total_issues = len(self.results['recommendations'])
        missing_deps_count = len(missing_deps)
        
        if total_issues == 0 and missing_deps_count == 0:
            report.append("✓ All systems operational!")
        else:
            report.append(f"⚠ Found {total_issues} issues")
            report.append(f"⚠ Missing {missing_deps_count} dependencies")
            
        report.append("=" * 60)
        
        # Print report
        report_text = '\n'.join(report)
        print(report_text)
        
        # Save report
        report_file = f"troubleshooting_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
            
        logger.info(f"Report saved to: {report_file}")
        
        # Save JSON results
        json_file = f"troubleshooting_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"JSON results saved to: {json_file}")


def main():
    """Main entry point"""
    troubleshooter = MonitoringSystemTroubleshooter()
    
    try:
        troubleshooter.run_all_checks()
    except KeyboardInterrupt:
        logger.info("Troubleshooting interrupted by user")
    except Exception as e:
        logger.error(f"Troubleshooting failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()