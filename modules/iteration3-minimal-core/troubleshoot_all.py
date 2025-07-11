#!/usr/bin/env python3
"""
Comprehensive troubleshooting script for all components
"""
import subprocess
import sys
import os
import importlib.util
import json
from datetime import datetime
import socket
import time

class TroubleshootingReport:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {},
            'scripts': {},
            'connectors': {},
            'issues': [],
            'recommendations': []
        }
        
    def add_issue(self, category, issue):
        self.report['issues'].append({
            'category': category,
            'issue': issue,
            'timestamp': datetime.now().isoformat()
        })
        
    def add_recommendation(self, recommendation):
        self.report['recommendations'].append(recommendation)
        
    def save_report(self, filename='troubleshooting_report.json'):
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"\nReport saved to: {filename}")

def check_python_version():
    """Check Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        return False, "Python 3.6 or higher required"
    return True, "Python version OK"

def check_required_modules():
    """Check required Python modules"""
    print("\n2. Checking required modules...")
    required_modules = {
        'PIL': 'pillow',
        'numpy': 'numpy'
    }
    
    missing = []
    for module, package in required_modules.items():
        try:
            importlib.import_module(module)
            print(f"   ✓ {module} installed")
        except ImportError:
            print(f"   ✗ {module} not installed")
            missing.append(package)
            
    if missing:
        return False, f"Missing modules: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
    return True, "All required modules installed"

def check_script_files():
    """Check if all script files exist and are valid"""
    print("\n3. Checking script files...")
    scripts = [
        'connector.py',
        'hivemind_connector.py',
        'script_interface.py',
        'intensity_reader.py',
        'random_pixel_generator.py'
    ]
    
    all_valid = True
    issues = []
    
    for script in scripts:
        if os.path.exists(script):
            # Try to compile the script
            try:
                with open(script, 'r') as f:
                    compile(f.read(), script, 'exec')
                print(f"   ✓ {script} - exists and valid")
            except SyntaxError as e:
                print(f"   ✗ {script} - syntax error: {e}")
                issues.append(f"{script} has syntax errors")
                all_valid = False
        else:
            print(f"   ✗ {script} - not found")
            issues.append(f"{script} not found")
            all_valid = False
            
    if not all_valid:
        return False, f"Script issues: {'; '.join(issues)}"
    return True, "All scripts valid"

def test_script_execution():
    """Test individual script execution"""
    print("\n4. Testing script execution...")
    
    test_results = {}
    
    # Test random pixel generator
    print("   Testing random_pixel_generator.py...")
    try:
        result = subprocess.run(
            [sys.executable, "random_pixel_generator.py", "0", "255", "0", "1"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            print("   ✓ Random pixel generator works")
            test_results['random_pixel_generator'] = True
        else:
            print(f"   ✗ Random pixel generator failed: {result.stderr}")
            test_results['random_pixel_generator'] = False
    except Exception as e:
        print(f"   ✗ Error testing random pixel generator: {e}")
        test_results['random_pixel_generator'] = False
        
    # Test intensity reader
    print("   Testing intensity_reader.py...")
    try:
        # Create a test image
        from PIL import Image
        import numpy as np
        
        test_data = np.full((5, 5), 200, dtype=np.uint8)
        img = Image.fromarray(test_data, mode='L')
        img.save('_test_troubleshoot.png')
        
        result = subprocess.run(
            [sys.executable, "intensity_reader.py", "_test_troubleshoot.png"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        os.remove('_test_troubleshoot.png')
        
        if result.returncode == 0:
            print("   ✓ Intensity reader works")
            test_results['intensity_reader'] = True
        else:
            print(f"   ✗ Intensity reader failed: {result.stderr}")
            test_results['intensity_reader'] = False
            
    except Exception as e:
        print(f"   ✗ Error testing intensity reader: {e}")
        test_results['intensity_reader'] = False
        
    all_pass = all(test_results.values())
    if not all_pass:
        failed = [k for k, v in test_results.items() if not v]
        return False, f"Failed scripts: {', '.join(failed)}"
    return True, "All scripts execute correctly"

def test_connector_socket():
    """Test connector socket functionality"""
    print("\n5. Testing connector socket...")
    
    # Start connector
    connector_process = subprocess.Popen(
        [sys.executable, "connector.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    time.sleep(2)
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(('localhost', 10089))
        
        # Send test command
        test_cmd = {'type': 'list_scripts'}
        sock.send(json.dumps(test_cmd).encode())
        response = sock.recv(4096).decode()
        
        sock.close()
        connector_process.terminate()
        connector_process.wait(timeout=5)
        
        response_data = json.loads(response)
        if response_data.get('status') == 'success':
            print("   ✓ Connector socket communication works")
            return True, "Connector socket OK"
        else:
            print("   ✗ Connector socket failed to respond correctly")
            return False, "Connector socket communication failed"
            
    except Exception as e:
        connector_process.terminate()
        connector_process.wait(timeout=5)
        print(f"   ✗ Connector socket error: {e}")
        return False, f"Connector socket error: {str(e)}"

def check_port_availability():
    """Check if required ports are available"""
    print("\n6. Checking port availability...")
    
    port = 10089
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        sock.bind(('localhost', port))
        sock.close()
        print(f"   ✓ Port {port} is available")
        return True, "Required port available"
    except OSError:
        print(f"   ✗ Port {port} is already in use")
        return False, f"Port {port} is already in use"

def generate_recommendations(issues):
    """Generate recommendations based on issues found"""
    recommendations = []
    
    if any('not installed' in issue for issue in issues):
        recommendations.append("Install missing Python packages using pip")
        
    if any('Port' in issue and 'in use' in issue for issue in issues):
        recommendations.append("Check for other processes using port 10089 (use 'lsof -i :10089' on Linux/Mac)")
        
    if any('syntax error' in issue for issue in issues):
        recommendations.append("Review and fix syntax errors in the affected scripts")
        
    if not recommendations:
        recommendations.append("System appears to be working correctly")
        
    return recommendations

def main():
    """Run all troubleshooting checks"""
    print("=" * 60)
    print("TROUBLESHOOTING ALL SYSTEMS")
    print("=" * 60)
    
    report = TroubleshootingReport()
    all_issues = []
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Required Modules", check_required_modules),
        ("Script Files", check_script_files),
        ("Script Execution", test_script_execution),
        ("Port Availability", check_port_availability),
        ("Connector Socket", test_connector_socket)
    ]
    
    for check_name, check_func in checks:
        try:
            success, message = check_func()
            report.report['system_info'][check_name] = {
                'success': success,
                'message': message
            }
            if not success:
                all_issues.append(message)
                report.add_issue(check_name, message)
        except Exception as e:
            error_msg = f"Check failed with error: {str(e)}"
            print(f"\n✗ {check_name}: {error_msg}")
            report.report['system_info'][check_name] = {
                'success': False,
                'message': error_msg
            }
            all_issues.append(error_msg)
            report.add_issue(check_name, error_msg)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING SUMMARY")
    print("=" * 60)
    
    if not all_issues:
        print("\n✓ All systems operational!")
        print("  - All scripts are valid and executable")
        print("  - Connectors can communicate properly")
        print("  - Required dependencies are installed")
    else:
        print("\n✗ Issues found:")
        for issue in all_issues:
            print(f"  - {issue}")
            
        # Add recommendations
        recommendations = generate_recommendations(all_issues)
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
            report.add_recommendation(rec)
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"troubleshooting_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("TROUBLESHOOTING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        if not all_issues:
            f.write("STATUS: ALL SYSTEMS OPERATIONAL\n\n")
        else:
            f.write("STATUS: ISSUES DETECTED\n\n")
            f.write("ISSUES:\n")
            for issue in all_issues:
                f.write(f"- {issue}\n")
            f.write("\nRECOMMENDATIONS:\n")
            for rec in recommendations:
                f.write(f"- {rec}\n")
                
    print(f"\nDetailed report saved to: {report_filename}")

if __name__ == "__main__":
    main()