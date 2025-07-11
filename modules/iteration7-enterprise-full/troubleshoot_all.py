#!/usr/bin/env python3
"""
Comprehensive Troubleshooting Script
Tests and troubleshoots all scripts in the directory
"""

import sys
import os
import subprocess
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import importlib.util
import ast


class ScriptTroubleshooter:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'directory': os.getcwd(),
            'python_version': sys.version,
            'scripts': {},
            'connectors': {},
            'dependencies': {},
            'summary': {}
        }
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("\n=== Checking Dependencies ===")
        
        required_packages = {
            'PIL': 'pillow',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'torch': 'torch',
            'tensorflow': 'tensorflow'
        }
        
        for import_name, package_name in required_packages.items():
            try:
                importlib.import_module(import_name)
                self.results['dependencies'][package_name] = {
                    'installed': True,
                    'import_name': import_name
                }
                print(f"✓ {package_name} is installed")
            except ImportError:
                self.results['dependencies'][package_name] = {
                    'installed': False,
                    'import_name': import_name
                }
                print(f"✗ {package_name} is NOT installed")
    
    def analyze_script_imports(self, script_path):
        """Analyze imports in a script"""
        imports = []
        try:
            with open(script_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module if node.module else '')
                    
        except Exception as e:
            print(f"  Error analyzing imports: {e}")
            
        return list(set(imports))
    
    def test_script_syntax(self, script_path):
        """Test if a script has valid syntax"""
        try:
            with open(script_path, 'r') as f:
                compile(f.read(), script_path, 'exec')
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def test_script_execution(self, script_path, timeout=30):
        """Test script execution"""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'executed': True,
                'return_code': result.returncode,
                'stdout': result.stdout[:1000],
                'stderr': result.stderr[:1000],
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'executed': False,
                'error': 'Timeout expired',
                'success': False
            }
        except Exception as e:
            return {
                'executed': False,
                'error': str(e),
                'success': False
            }
    
    def check_script_structure(self, script_path):
        """Check if script follows good practices"""
        issues = []
        
        try:
            with open(script_path, 'r') as f:
                content = f.read()
                
            # Check for main guard
            if 'if __name__ == "__main__":' not in content:
                issues.append("No main guard found")
                
            # Check for logging/print statements
            if 'print(' not in content and 'logging' not in content:
                issues.append("No output mechanism found")
                
            # Check for error handling
            if 'try:' not in content:
                issues.append("No error handling found")
                
            # Check for functions
            tree = ast.parse(content)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if not functions:
                issues.append("No functions defined")
                
            return {
                'has_main_guard': 'if __name__ == "__main__":' in content,
                'has_output': 'print(' in content or 'logging' in content,
                'has_error_handling': 'try:' in content,
                'functions': functions,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'issues': ['Failed to analyze structure']
            }
    
    def test_connector_integration(self, script_path):
        """Test if script can integrate with connector"""
        script_name = os.path.basename(script_path)
        
        try:
            # Import the connector
            from connector import get_connector
            
            connector = get_connector()
            connector.start()
            
            # Try to add the script
            connector.add_script(script_path)
            
            # Check if loaded
            script = connector.get_script(script_name)
            
            if script and script.module:
                return {
                    'loadable': True,
                    'functions': list(script.functions.keys()),
                    'attributes': list(script.attributes.keys()),
                    'classes': list(script.classes.keys())
                }
            else:
                return {
                    'loadable': False,
                    'error': 'Failed to load in connector'
                }
                
        except Exception as e:
            return {
                'loadable': False,
                'error': str(e)
            }
    
    def troubleshoot_all_scripts(self):
        """Troubleshoot all Python scripts in the directory"""
        print("\n=== Troubleshooting All Scripts ===")
        
        script_files = list(Path('.').glob('*.py'))
        excluded = {'troubleshoot_all.py', '__pycache__'}
        
        for script_path in script_files:
            if script_path.name in excluded:
                continue
                
            print(f"\n--- Testing {script_path.name} ---")
            
            script_result = {
                'path': str(script_path),
                'size': script_path.stat().st_size,
                'modified': datetime.fromtimestamp(script_path.stat().st_mtime).isoformat()
            }
            
            # Test syntax
            print("  Checking syntax...")
            syntax_ok, syntax_error = self.test_script_syntax(script_path)
            script_result['syntax'] = {
                'valid': syntax_ok,
                'error': syntax_error
            }
            
            if syntax_ok:
                print("  ✓ Syntax OK")
                
                # Analyze imports
                print("  Analyzing imports...")
                imports = self.analyze_script_imports(script_path)
                script_result['imports'] = imports
                
                # Check structure
                print("  Checking structure...")
                structure = self.check_script_structure(script_path)
                script_result['structure'] = structure
                
                # Test execution
                print("  Testing execution...")
                execution = self.test_script_execution(script_path, timeout=10)
                script_result['execution'] = execution
                
                if execution['success']:
                    print("  ✓ Execution successful")
                else:
                    print(f"  ✗ Execution failed: {execution.get('error', 'Return code ' + str(execution.get('return_code')))}")
                
                # Test connector integration
                if script_path.name not in ['connector.py', 'hivemind_connector.py']:
                    print("  Testing connector integration...")
                    integration = self.test_connector_integration(script_path)
                    script_result['connector_integration'] = integration
                    
                    if integration['loadable']:
                        print(f"  ✓ Connector integration OK ({len(integration['functions'])} functions)")
                    else:
                        print(f"  ✗ Connector integration failed: {integration.get('error')}")
                        
            else:
                print(f"  ✗ Syntax error: {syntax_error}")
            
            self.results['scripts'][script_path.name] = script_result
    
    def test_connectors(self):
        """Test the connector systems"""
        print("\n=== Testing Connectors ===")
        
        # Test enhanced connector
        print("\n--- Testing Enhanced Connector ---")
        try:
            from connector import get_connector
            conn = get_connector()
            conn.start()
            
            status = conn.get_status()
            self.results['connectors']['enhanced'] = {
                'available': True,
                'status': status,
                'scripts_loaded': status['scripts_loaded']
            }
            print(f"✓ Enhanced connector working ({status['scripts_loaded']} scripts loaded)")
            
        except Exception as e:
            self.results['connectors']['enhanced'] = {
                'available': False,
                'error': str(e)
            }
            print(f"✗ Enhanced connector failed: {e}")
        
        # Test hivemind connector
        print("\n--- Testing Hivemind Connector ---")
        try:
            import socket
            import json
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(('localhost', 10087))
            
            request = json.dumps({'command': 'status'})
            sock.send(request.encode())
            
            response = sock.recv(4096).decode()
            status = json.loads(response)
            
            self.results['connectors']['hivemind'] = {
                'available': True,
                'status': status
            }
            print("✓ Hivemind connector active")
            
            sock.close()
            
        except Exception as e:
            self.results['connectors']['hivemind'] = {
                'available': False,
                'error': 'Not running or not accessible'
            }
            print("✗ Hivemind connector not accessible")
    
    def generate_summary(self):
        """Generate summary of troubleshooting results"""
        total_scripts = len(self.results['scripts'])
        syntax_valid = sum(1 for s in self.results['scripts'].values() if s.get('syntax', {}).get('valid', False))
        execution_success = sum(1 for s in self.results['scripts'].values() if s.get('execution', {}).get('success', False))
        connector_ready = sum(1 for s in self.results['scripts'].values() if s.get('connector_integration', {}).get('loadable', False))
        
        self.results['summary'] = {
            'total_scripts': total_scripts,
            'syntax_valid': syntax_valid,
            'execution_success': execution_success,
            'connector_ready': connector_ready,
            'dependencies_missing': [pkg for pkg, info in self.results['dependencies'].items() if not info['installed']],
            'problematic_scripts': [
                name for name, info in self.results['scripts'].items() 
                if not info.get('syntax', {}).get('valid', False) or not info.get('execution', {}).get('success', False)
            ]
        }
    
    def save_report(self):
        """Save troubleshooting report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'troubleshooting_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nReport saved to: {report_file}")
        
        # Also create a human-readable summary
        summary_file = f'troubleshooting_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("TROUBLESHOOTING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Directory: {self.results['directory']}\n")
            f.write(f"Python Version: {sys.version.split()[0]}\n\n")
            
            summary = self.results['summary']
            f.write(f"Total Scripts: {summary['total_scripts']}\n")
            f.write(f"Syntax Valid: {summary['syntax_valid']}\n")
            f.write(f"Execution Success: {summary['execution_success']}\n")
            f.write(f"Connector Ready: {summary['connector_ready']}\n\n")
            
            if summary['dependencies_missing']:
                f.write("Missing Dependencies:\n")
                for dep in summary['dependencies_missing']:
                    f.write(f"  - {dep}\n")
                f.write("\n")
                
            if summary['problematic_scripts']:
                f.write("Problematic Scripts:\n")
                for script in summary['problematic_scripts']:
                    f.write(f"  - {script}\n")
                    
        print(f"Summary saved to: {summary_file}")
    
    def run(self):
        """Run complete troubleshooting"""
        print("=" * 60)
        print("COMPREHENSIVE SCRIPT TROUBLESHOOTER")
        print("=" * 60)
        
        # Check dependencies first
        self.check_dependencies()
        
        # Test all scripts
        self.troubleshoot_all_scripts()
        
        # Test connectors
        self.test_connectors()
        
        # Generate summary
        self.generate_summary()
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        summary = self.results['summary']
        print(f"\nTotal Scripts: {summary['total_scripts']}")
        print(f"Syntax Valid: {summary['syntax_valid']}/{summary['total_scripts']}")
        print(f"Execution Success: {summary['execution_success']}/{summary['total_scripts']}")
        print(f"Connector Ready: {summary['connector_ready']}/{summary['total_scripts']}")
        
        if summary['dependencies_missing']:
            print(f"\nMissing Dependencies: {', '.join(summary['dependencies_missing'])}")
            
        if summary['problematic_scripts']:
            print(f"\nProblematic Scripts: {', '.join(summary['problematic_scripts'])}")
        
        # Save report
        self.save_report()
        
        return self.results


def main():
    """Main function"""
    troubleshooter = ScriptTroubleshooter()
    results = troubleshooter.run()
    
    # Return exit code based on results
    if results['summary']['execution_success'] == results['summary']['total_scripts']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())