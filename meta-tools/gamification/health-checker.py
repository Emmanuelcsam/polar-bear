#!/usr/bin/env python3
"""
Project Health Checker - Check for common issues and best practices
Identifies problems like empty files, missing docs, large files, etc.
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import json
import mimetypes

class HealthChecker:
    def __init__(self, path="."):
        self.root = Path(path).resolve()
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.recommendations = []
        self.score = 100  # Start with perfect score
        
    def run_checks(self):
        """Run all health checks"""
        print(f"🏥 Running health check on: {self.root}")
        print("-" * 50)
        
        checks = [
            self._check_readme,
            self._check_gitignore,
            self._check_license,
            self._check_empty_files,
            self._check_large_files,
            self._check_binary_files,
            self._check_temp_files,
            self._check_naming_conventions,
            self._check_file_permissions,
            self._check_documentation,
            self._check_dependencies,
            self._check_security,
            self._check_structure
        ]
        
        for i, check in enumerate(checks, 1):
            print(f"\rRunning check {i}/{len(checks)}...", end='')
            try:
                check()
            except Exception as e:
                self.issues['warning'].append(f"Check failed: {check.__name__} - {str(e)}")
        
        print("\n✓ All checks completed")
        
    def _check_readme(self):
        """Check for README file"""
        readme_files = ['README.md', 'README.txt', 'README.rst', 'README']
        found = False
        
        for readme in readme_files:
            if (self.root / readme).exists():
                found = True
                # Check if README is not empty
                size = (self.root / readme).stat().st_size
                if size < 100:
                    self.issues['warning'].append("README file is very small (< 100 bytes)")
                    self.score -= 5
                break
        
        if not found:
            self.issues['critical'].append("No README file found")
            self.recommendations.append("Add a README.md file with project description")
            self.score -= 10
    
    def _check_gitignore(self):
        """Check for .gitignore file"""
        gitignore = self.root / '.gitignore'
        
        if not gitignore.exists():
            if (self.root / '.git').exists():
                self.issues['warning'].append("Git repository without .gitignore file")
                self.recommendations.append("Add a .gitignore file to exclude unnecessary files")
                self.score -= 5
        else:
            # Check for common patterns
            with open(gitignore, 'r') as f:
                content = f.read()
            
            important_patterns = ['__pycache__', '.env', 'node_modules', '.DS_Store', '*.log']
            missing = [p for p in important_patterns if p not in content]
            
            if missing:
                self.issues['info'].append(f"Consider adding to .gitignore: {', '.join(missing)}")
    
    def _check_license(self):
        """Check for license file"""
        license_files = ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING']
        found = False
        
        for license_file in license_files:
            if (self.root / license_file).exists():
                found = True
                break
        
        if not found:
            self.issues['warning'].append("No license file found")
            self.recommendations.append("Add a LICENSE file to specify usage terms")
            self.score -= 5
    
    def _check_empty_files(self):
        """Check for empty files"""
        empty_files = []
        
        for root, dirs, files in os.walk(self.root):
            # Skip hidden and system directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                path = Path(root) / file
                try:
                    if path.stat().st_size == 0:
                        empty_files.append(path.relative_to(self.root))
                except:
                    pass
        
        if empty_files:
            self.issues['warning'].append(f"Found {len(empty_files)} empty files")
            if len(empty_files) <= 5:
                for f in empty_files:
                    self.issues['info'].append(f"Empty file: {f}")
            self.score -= min(10, len(empty_files))
    
    def _check_large_files(self):
        """Check for unusually large files"""
        large_files = []
        size_limit = 50 * 1024 * 1024  # 50MB
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                path = Path(root) / file
                try:
                    size = path.stat().st_size
                    if size > size_limit:
                        large_files.append((path.relative_to(self.root), size))
                except:
                    pass
        
        if large_files:
            large_files.sort(key=lambda x: x[1], reverse=True)
            self.issues['warning'].append(f"Found {len(large_files)} large files (>50MB)")
            
            for path, size in large_files[:3]:
                self.issues['info'].append(f"Large file: {path} ({self._format_size(size)})")
            
            self.recommendations.append("Consider using Git LFS for large files")
            self.score -= 5
    
    def _check_binary_files(self):
        """Check for binary files in repository"""
        binary_extensions = {'.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite'}
        binary_files = []
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv']]
            
            for file in files:
                path = Path(root) / file
                if path.suffix.lower() in binary_extensions:
                    binary_files.append(path.relative_to(self.root))
        
        if binary_files:
            self.issues['info'].append(f"Found {len(binary_files)} binary files")
            if len(binary_files) > 10:
                self.recommendations.append("Consider if all binary files are necessary")
                self.score -= 3
    
    def _check_temp_files(self):
        """Check for temporary files"""
        temp_patterns = ['~', '.tmp', '.temp', '.swp', '.swo', '.bak', '.orig']
        temp_files = []
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(pattern) for pattern in temp_patterns):
                    temp_files.append(Path(root) / file)
        
        if temp_files:
            self.issues['warning'].append(f"Found {len(temp_files)} temporary files")
            self.recommendations.append("Clean up temporary files")
            self.score -= 5
    
    def _check_naming_conventions(self):
        """Check file naming conventions"""
        issues = []
        
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Check for spaces in filenames
                if ' ' in file:
                    issues.append(f"Space in filename: {file}")
                
                # Check for special characters
                if re.search(r'[<>:"|?*]', file):
                    issues.append(f"Special character in filename: {file}")
        
        if issues:
            self.issues['info'].append(f"Found {len(issues)} naming convention issues")
            for issue in issues[:5]:
                self.issues['info'].append(issue)
            self.score -= min(5, len(issues))
    
    def _check_file_permissions(self):
        """Check for executable files that shouldn't be"""
        if os.name == 'posix':  # Unix-like systems only
            executable_issues = []
            
            for root, dirs, files in os.walk(self.root):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    path = Path(root) / file
                    try:
                        # Check if file is executable
                        if os.access(path, os.X_OK):
                            # Check if it's a script or binary
                            ext = path.suffix.lower()
                            if ext not in ['.sh', '.py', '.exe', '.bat', '.command'] and not file.startswith('run'):
                                executable_issues.append(path.relative_to(self.root))
                    except:
                        pass
            
            if executable_issues:
                self.issues['info'].append(f"Found {len(executable_issues)} files with unexpected execute permissions")
    
    def _check_documentation(self):
        """Check for documentation"""
        doc_dirs = ['docs', 'doc', 'documentation']
        doc_found = False
        
        for doc_dir in doc_dirs:
            if (self.root / doc_dir).is_dir():
                doc_found = True
                # Check if docs directory has content
                doc_files = list((self.root / doc_dir).rglob('*'))
                if len(doc_files) < 2:
                    self.issues['info'].append(f"Documentation directory '{doc_dir}' has very few files")
                break
        
        # Check for inline documentation in code
        code_files = list(self.root.rglob('*.py')) + list(self.root.rglob('*.js'))
        if code_files:
            undocumented = 0
            for code_file in code_files[:20]:  # Sample first 20 files
                try:
                    with open(code_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Simple check for docstrings/comments
                    if '"""' not in content and "'''" not in content and '/**' not in content:
                        undocumented += 1
                except:
                    pass
            
            if undocumented > len(code_files) * 0.5:
                self.issues['info'].append("Many code files lack documentation")
                self.recommendations.append("Add docstrings/comments to your code")
    
    def _check_dependencies(self):
        """Check for dependency files"""
        dep_files = {
            'requirements.txt': 'Python',
            'package.json': 'Node.js',
            'Gemfile': 'Ruby',
            'pom.xml': 'Java/Maven',
            'build.gradle': 'Java/Gradle',
            'Cargo.toml': 'Rust',
            'go.mod': 'Go'
        }
        
        found_deps = []
        for dep_file, lang in dep_files.items():
            if (self.root / dep_file).exists():
                found_deps.append((dep_file, lang))
                
                # Check if dependency file is recent
                mtime = (self.root / dep_file).stat().st_mtime
                if datetime.fromtimestamp(mtime) < datetime.now() - timedelta(days=180):
                    self.issues['info'].append(f"{dep_file} hasn't been updated in 6+ months")
        
        # Check for lock files
        if any(f for f, _ in found_deps if f in ['requirements.txt', 'package.json']):
            lock_files = ['requirements-lock.txt', 'package-lock.json', 'yarn.lock']
            if not any((self.root / lf).exists() for lf in lock_files):
                self.issues['info'].append("Consider using dependency lock files for reproducible builds")
    
    def _check_security(self):
        """Check for security issues"""
        sensitive_patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*=\s*["\']?[a-zA-Z0-9]{20,}', 'API key'),
            (r'(?i)(secret|password|passwd|pwd)\s*=\s*["\']?[^\s"\']+', 'Password/Secret'),
            (r'[a-zA-Z0-9+/]{40,}={0,2}', 'Possible Base64 encoded secret'),
            (r'(?i)aws[_-]?access[_-]?key', 'AWS credentials'),
            (r'(?i)github[_-]?token', 'GitHub token')
        ]
        
        security_issues = []
        
        # Check common config files
        config_files = list(self.root.glob('*.env')) + \
                      list(self.root.glob('config.*')) + \
                      list(self.root.glob('*.conf'))
        
        for config_file in config_files:
            if config_file.name == '.env' and (self.root / '.gitignore').exists():
                # Check if .env is in gitignore
                with open(self.root / '.gitignore', 'r') as f:
                    if '.env' not in f.read():
                        self.issues['critical'].append(".env file not in .gitignore!")
                        self.score -= 10
            
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern, desc in sensitive_patterns:
                    if re.search(pattern, content):
                        security_issues.append(f"Possible {desc} in {config_file.name}")
                        break
            except:
                pass
        
        if security_issues:
            self.issues['critical'].append(f"Found {len(security_issues)} potential security issues")
            for issue in security_issues[:3]:
                self.issues['critical'].append(issue)
            self.recommendations.append("Review and remove hardcoded secrets")
            self.score -= 15
    
    def _check_structure(self):
        """Check project structure"""
        # Check for common structure patterns
        common_dirs = ['src', 'lib', 'test', 'tests', 'bin', 'scripts']
        structure_score = 0
        
        for common_dir in common_dirs:
            if (self.root / common_dir).is_dir():
                structure_score += 1
        
        if structure_score < 2:
            self.issues['info'].append("Project structure could be more organized")
            self.recommendations.append("Consider organizing code into src/, tests/, etc.")
        
        # Check for deeply nested directories
        max_depth = 0
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            depth = len(Path(root).relative_to(self.root).parts)
            max_depth = max(max_depth, depth)
        
        if max_depth > 6:
            self.issues['info'].append(f"Very deep directory nesting detected (max depth: {max_depth})")
            self.recommendations.append("Consider flattening directory structure")
    
    def display_results(self):
        """Display health check results"""
        print("\n" + "="*80)
        print("🏥 PROJECT HEALTH REPORT")
        print("="*80)
        
        # Calculate final score
        self.score = max(0, self.score)  # Don't go below 0
        
        # Determine health status
        if self.score >= 90:
            status = "🟢 Excellent"
            color = '\033[92m'
        elif self.score >= 70:
            status = "🟡 Good"
            color = '\033[93m'
        elif self.score >= 50:
            status = "🟠 Fair"
            color = '\033[93m'
        else:
            status = "🔴 Needs Attention"
            color = '\033[91m'
        
        print(f"\nHealth Score: {color}{self.score}/100{color} - {status}\033[0m")
        
        # Display issues by severity
        if self.issues['critical']:
            print("\n🚨 CRITICAL ISSUES:")
            for issue in self.issues['critical']:
                print(f"  ❌ {issue}")
        
        if self.issues['warning']:
            print("\n⚠️  WARNINGS:")
            for issue in self.issues['warning'][:10]:
                print(f"  ⚠️  {issue}")
            if len(self.issues['warning']) > 10:
                print(f"  ... and {len(self.issues['warning']) - 10} more warnings")
        
        if self.issues['info']:
            print("\nℹ️  INFORMATION:")
            for issue in self.issues['info'][:5]:
                print(f"  ℹ️  {issue}")
            if len(self.issues['info']) > 5:
                print(f"  ... and {len(self.issues['info']) - 5} more info items")
        
        # Display recommendations
        if self.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Save report
        self._save_report()
    
    def _save_report(self):
        """Save health report"""
        stats_dir = Path('.project-stats')
        stats_dir.mkdir(exist_ok=True)
        
        report_file = stats_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'path': str(self.root),
            'score': self.score,
            'issues': self.issues,
            'recommendations': self.recommendations,
            'summary': {
                'critical': len(self.issues['critical']),
                'warning': len(self.issues['warning']),
                'info': len(self.issues['info'])
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Health report saved to: {report_file}")
        
        # Track health over time
        self._track_health_history()
    
    def _track_health_history(self):
        """Track health score over time"""
        stats_dir = Path('.project-stats')
        history_file = stats_dir / 'health_history.json'
        
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'date': datetime.now().isoformat(),
            'score': self.score
        })
        
        # Keep last 30 entries
        history = history[-30:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Show trend
        if len(history) >= 2:
            prev_score = history[-2]['score']
            if self.score > prev_score:
                print(f"📈 Health improved by {self.score - prev_score} points!")
            elif self.score < prev_score:
                print(f"📉 Health decreased by {prev_score - self.score} points")
    
    def _format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check project health and best practices')
    parser.add_argument('path', nargs='?', default='.', help='Directory to check')
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.path)
    checker.run_checks()
    checker.display_results()

if __name__ == "__main__":
    main()
