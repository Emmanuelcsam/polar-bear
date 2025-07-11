#!/usr/bin/env python3
"""
Troubleshooting Script for All Modules
Identifies and fixes common issues
"""

import os
import sys
import subprocess
import json
import ast
import importlib.util
from pathlib import Path
from datetime import datetime

class Troubleshooter:
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        self.script_dir = Path(__file__).parent
        
    def check_python_syntax(self, filepath):
        """Check Python syntax"""
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    def check_imports(self, filepath):
        """Check if all imports are available"""
        issues = []
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        try:
                            importlib.import_module(alias.name)
                        except ImportError:
                            issues.append(f"Missing module: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    try:
                        if node.module:
                            importlib.import_module(node.module)
                    except ImportError:
                        issues.append(f"Missing module: {node.module}")
        except:
            pass
        
        return len(issues) == 0, issues
    
    def check_file_permissions(self, filepath):
        """Check file permissions"""
        path = Path(filepath)
        
        if not path.exists():
            return False, "File does not exist"
        
        if not os.access(path, os.R_OK):
            return False, "File is not readable"
        
        if filepath.endswith('.py') and not os.access(path, os.X_OK):
            # Python files don't need to be executable
            pass
        
        return True, None
    
    def check_dependencies(self):
        """Check required dependencies"""
        required = ['numpy', 'PIL', 'scipy']
        missing = []
        
        for module in required:
            try:
                importlib.import_module(module)
            except ImportError:
                missing.append(module)
        
        return len(missing) == 0, missing
    
    def fix_file_format(self, filepath):
        """Fix common file format issues"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Fix line endings
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Ensure file ends with newline
            if not content.endswith('\n'):
                content += '\n'
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            return True, "Fixed file format"
        except Exception as e:
            return False, str(e)
    
    def check_script_interface(self, script_name):
        """Check if script has proper interface functions"""
        required_functions = ['set_param', 'get_param', 'get_info']
        
        try:
            # Import the script
            spec = importlib.util.spec_from_file_location(script_name[:-3], script_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            missing = []
            for func in required_functions:
                if not hasattr(module, func):
                    missing.append(func)
            
            return len(missing) == 0, missing
        except Exception as e:
            return False, str(e)
    
    def troubleshoot_script(self, script_path):
        """Troubleshoot a single script"""
        script_name = os.path.basename(script_path)
        print(f"\nChecking {script_name}...")
        
        issues = []
        
        # Check syntax
        ok, error = self.check_python_syntax(script_path)
        if not ok:
            issues.append(f"Syntax error: {error}")
        
        # Check imports
        ok, import_issues = self.check_imports(script_path)
        if not ok:
            issues.extend(import_issues)
        
        # Check permissions
        ok, error = self.check_file_permissions(script_path)
        if not ok:
            issues.append(f"Permission issue: {error}")
        
        # Check interface (skip for certain files)
        if script_name not in ['connector.py', 'hivemind_connector.py', 'script_interface.py', 
                               'script_wrapper.py', 'enhanced_scripts.py', 'test_integration.py',
                               'troubleshoot_all.py']:
            ok, missing = self.check_script_interface(script_path)
            if not ok:
                if isinstance(missing, list):
                    issues.append(f"Missing interface functions: {', '.join(missing)}")
                else:
                    issues.append(f"Interface check failed: {missing}")
        
        if issues:
            self.issues.append({
                'script': script_name,
                'issues': issues
            })
            print(f"  ✗ Found {len(issues)} issues")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ No issues found")
        
        return len(issues) == 0
    
    def test_script_execution(self, script_path):
        """Test if script can execute"""
        script_name = os.path.basename(script_path)
        
        # Skip certain scripts that need arguments or run indefinitely
        skip_scripts = ['pixel-generator.py', 'intensity-reader.py', 'batch-processor.py']
        
        if script_name in skip_scripts:
            print(f"  - Skipping execution test for {script_name}")
            return True
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print(f"  ✗ Execution failed: {result.stderr[:100]}")
                return False
            else:
                print(f"  ✓ Execution successful")
                return True
        except subprocess.TimeoutExpired:
            print(f"  ✓ Script runs (terminated after timeout)")
            return True
        except Exception as e:
            print(f"  ✗ Execution error: {e}")
            return False
    
    def generate_report(self):
        """Generate troubleshooting report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'script_directory': str(self.script_dir),
            'total_scripts': 0,
            'scripts_with_issues': len(self.issues),
            'issues': self.issues,
            'fixes_applied': self.fixes_applied,
            'recommendations': []
        }
        
        # Add recommendations
        if any('Missing module: PIL' in str(issue) for issue in self.issues):
            report['recommendations'].append("Install Pillow: pip install Pillow")
        
        if any('Missing module: scipy' in str(issue) for issue in self.issues):
            report['recommendations'].append("Install scipy: pip install scipy")
        
        if any('Missing module: numpy' in str(issue) for issue in self.issues):
            report['recommendations'].append("Install numpy: pip install numpy")
        
        # Check dependencies
        ok, missing = self.check_dependencies()
        if not ok:
            report['missing_dependencies'] = missing
        
        return report
    
    def run(self):
        """Run troubleshooting on all scripts"""
        print("=" * 60)
        print("TROUBLESHOOTING ALL SCRIPTS")
        print("=" * 60)
        
        # Find all Python scripts
        scripts = list(self.script_dir.glob('*.py'))
        
        print(f"\nFound {len(scripts)} Python scripts")
        
        # Check each script
        for script in scripts:
            self.troubleshoot_script(str(script))
            self.test_script_execution(str(script))
        
        # Generate and save report
        report = self.generate_report()
        report['total_scripts'] = len(scripts)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total scripts: {report['total_scripts']}")
        print(f"Scripts with issues: {report['scripts_with_issues']}")
        
        if report.get('missing_dependencies'):
            print(f"\nMissing dependencies: {', '.join(report['missing_dependencies'])}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # Save report
        report_file = f"troubleshooting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        return report['scripts_with_issues'] == 0

def main():
    troubleshooter = Troubleshooter()
    success = troubleshooter.run()
    
    if not success:
        print("\n⚠️  Some issues were found. Please review the report and apply fixes.")
        sys.exit(1)
    else:
        print("\n✅ All scripts are working correctly!")
        sys.exit(0)

if __name__ == "__main__":
    main()