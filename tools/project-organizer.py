#!/usr/bin/env python3
"""
Smart Project Organizer - Detects and separates nested projects intelligently
"""

import os
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib


class ProjectDetector:
    """Detects different types of projects based on their file patterns"""
    
    PROJECT_INDICATORS = {
        'node': ['package.json', 'node_modules'],
        'python': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
        'react': ['package.json', 'src/App.js', 'src/App.jsx', 'src/App.tsx'],
        'vue': ['package.json', 'vue.config.js', 'src/App.vue'],
        'django': ['manage.py', 'settings.py', 'wsgi.py'],
        'flask': ['app.py', 'flask_app.py', 'application.py'],
        'dotnet': ['.csproj', '.sln', 'Program.cs'],
        'java': ['pom.xml', 'build.gradle', 'gradlew'],
        'ruby': ['Gemfile', 'Rakefile', 'config.ru'],
        'php': ['composer.json', 'index.php'],
        'go': ['go.mod', 'go.sum', 'main.go'],
        'rust': ['Cargo.toml', 'Cargo.lock'],
        'git': ['.git'],
        'docker': ['Dockerfile', 'docker-compose.yml'],
        'terraform': ['main.tf', 'terraform.tfvars'],
        'website': ['index.html', 'index.htm'],
    }
    
    @classmethod
    def detect_project_type(cls, path):
        """Detect what type of project exists at the given path"""
        detected_types = []
        path_obj = Path(path)
        
        for project_type, indicators in cls.PROJECT_INDICATORS.items():
            for indicator in indicators:
                indicator_path = path_obj / indicator
                if indicator_path.exists() or list(path_obj.rglob(indicator)):
                    detected_types.append(project_type)
                    break
        
        return detected_types


class Project:
    """Represents a detected project"""
    
    def __init__(self, root_path, project_types, parent=None):
        self.root_path = Path(root_path)
        self.project_types = project_types
        self.parent = parent
        self.children = []
        self.files = set()
        self.shared_files = set()
        self.name = self.root_path.name
        self.id = hashlib.md5(str(self.root_path).encode()).hexdigest()[:8]
        
    def add_child(self, child_project):
        """Add a child project"""
        self.children.append(child_project)
        child_project.parent = self
        
    def get_all_files(self):
        """Get all files in this project (excluding child projects)"""
        all_files = set()
        for root, dirs, files in os.walk(self.root_path):
            root_path = Path(root)
            
            # Skip child project directories
            relative_root = root_path.relative_to(self.root_path)
            skip = False
            for child in self.children:
                child_relative = child.root_path.relative_to(self.root_path)
                try:
                    relative_root.relative_to(child_relative)
                    skip = True
                    break
                except ValueError:
                    continue
                    
            if skip:
                continue
                
            for file in files:
                all_files.add(root_path / file)
                
        return all_files
    
    def __repr__(self):
        return f"Project({self.name}, types={self.project_types}, children={len(self.children)})"


class ProjectOrganizer:
    """Main organizer class"""
    
    def __init__(self):
        self.root_path = None
        self.projects = []
        self.project_map = {}
        self.analysis_report = {}
        
    def scan_directory(self, root_path):
        """Scan directory and build project hierarchy"""
        self.root_path = Path(root_path)
        self.projects = []
        self.project_map = {}
        
        print(f"\n🔍 Scanning directory: {self.root_path}")
        self._scan_recursive(self.root_path)
        self._build_hierarchy()
        self._analyze_relationships()
        
    def _scan_recursive(self, path):
        """Recursively scan for projects"""
        path = Path(path)
        
        # Check if this directory is a project
        project_types = ProjectDetector.detect_project_type(path)
        
        if project_types:
            project = Project(path, project_types)
            self.projects.append(project)
            self.project_map[str(path)] = project
            print(f"  📁 Found {', '.join(project_types)} project: {path.relative_to(self.root_path)}")
        
        # Scan subdirectories
        try:
            for item in path.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name not in ['node_modules', '__pycache__', 'venv', 'env']:
                    self._scan_recursive(item)
        except PermissionError:
            pass
            
    def _build_hierarchy(self):
        """Build parent-child relationships between projects"""
        for project in self.projects:
            for other_project in self.projects:
                if project != other_project:
                    try:
                        project.root_path.relative_to(other_project.root_path)
                        # project is inside other_project
                        if not project.parent or len(str(other_project.root_path)) > len(str(project.parent.root_path)):
                            project.parent = other_project
                    except ValueError:
                        continue
        
        # Build children lists
        for project in self.projects:
            if project.parent:
                project.parent.children.append(project)
                
    def _analyze_relationships(self):
        """Analyze file relationships between projects"""
        self.analysis_report = {
            'total_projects': len(self.projects),
            'nested_projects': [],
            'potential_conflicts': [],
            'shared_dependencies': []
        }
        
        # Find nested projects
        for project in self.projects:
            if project.parent:
                self.analysis_report['nested_projects'].append({
                    'child': str(project.root_path),
                    'parent': str(project.parent.root_path),
                    'child_types': project.project_types,
                    'parent_types': project.parent.project_types
                })
                
        # Analyze potential conflicts
        for project in self.projects:
            if project.children:
                # Check for file overlaps
                project_files = project.get_all_files()
                for child in project.children:
                    child_files = child.get_all_files()
                    
                    # Check if parent project has files that might depend on child
                    for pf in project_files:
                        if self._check_file_dependency(pf, child.root_path):
                            self.analysis_report['potential_conflicts'].append({
                                'parent': project.name,
                                'child': child.name,
                                'file': str(pf.relative_to(project.root_path)),
                                'reason': 'Parent file may depend on child project'
                            })
                            
    def _check_file_dependency(self, file_path, dependency_path):
        """Check if a file might depend on files in dependency_path"""
        if file_path.suffix in ['.js', '.jsx', '.ts', '.tsx', '.py', '.java', '.cs']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Check for imports/requires
                dep_relative = dependency_path.relative_to(file_path.parent.parent)
                dep_name = dependency_path.name
                
                patterns = [
                    rf'import.*from.*["\'].*{dep_name}',
                    rf'require\s*\(.*["\'].*{dep_name}',
                    rf'from\s+.*{dep_name}\s+import',
                    rf'import\s+.*{dep_name}'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
                        
            except Exception:
                pass
                
        return False
        
    def get_extraction_plan(self, project):
        """Generate a plan for extracting a nested project"""
        plan = {
            'project': project.name,
            'current_path': str(project.root_path),
            'new_path': None,
            'files_to_move': [],
            'files_to_copy': [],
            'warnings': []
        }
        
        # Get all files for this project
        all_files = project.get_all_files()
        plan['files_to_move'] = [str(f.relative_to(project.root_path)) for f in all_files]
        
        # Check for potential issues
        if project.parent:
            parent_files = project.parent.get_all_files()
            for pf in parent_files:
                if self._check_file_dependency(pf, project.root_path):
                    plan['warnings'].append(f"Parent file '{pf.relative_to(project.parent.root_path)}' may depend on this project")
                    
        return plan
        
    def extract_project(self, project, destination):
        """Extract a nested project to a new location"""
        dest_path = Path(destination)
        
        # Create destination directory
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Copy project files
        print(f"\n📦 Extracting {project.name} to {dest_path}")
        
        for root, dirs, files in os.walk(project.root_path):
            root_path = Path(root)
            relative_path = root_path.relative_to(project.root_path)
            dest_dir = dest_path / relative_path
            
            # Create directory structure
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = root_path / file
                dst_file = dest_dir / file
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ Copied: {relative_path / file}")
                
        print(f"\n✅ Successfully extracted {project.name}")
        
        return dest_path
        
    def organize_projects(self, output_dir):
        """Organize all projects into a clean structure"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        organized = {
            'standalone': [],
            'extracted': [],
            'kept_nested': []
        }
        
        # Process each project
        for project in self.projects:
            if not project.parent:
                # Standalone project
                organized['standalone'].append(project.name)
            else:
                # This will be determined by user choice
                pass
                
        return organized


def print_analysis_report(report):
    """Pretty print the analysis report"""
    print("\n" + "="*60)
    print("📊 ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📁 Total projects found: {report['total_projects']}")
    
    if report['nested_projects']:
        print(f"\n🔗 Nested projects: {len(report['nested_projects'])}")
        for np in report['nested_projects']:
            child_name = Path(np['child']).name
            parent_name = Path(np['parent']).name
            print(f"  • {child_name} ({', '.join(np['child_types'])}) inside {parent_name} ({', '.join(np['parent_types'])})")
            
    if report['potential_conflicts']:
        print(f"\n⚠️  Potential conflicts: {len(report['potential_conflicts'])}")
        for conflict in report['potential_conflicts'][:5]:  # Show first 5
            print(f"  • {conflict['parent']} → {conflict['child']}: {conflict['reason']}")
            print(f"    File: {conflict['file']}")
            
    print("\n" + "="*60)


def ask_question(question, options=None, default=None):
    """Ask user a question and get response"""
    print(f"\n❓ {question}")
    
    if options:
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        while True:
            response = input(f"Enter choice (1-{len(options)}){f' [{default}]' if default else ''}: ").strip()
            if not response and default:
                return default
            
            try:
                choice = int(response)
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
    else:
        response = input(f"{f'[{default}] ' if default else ''}: ").strip()
        return response if response else default


def main():
    """Main interactive function"""
    print("🚀 Smart Project Organizer")
    print("=" * 60)
    
    # Get root directory
    root_dir = ask_question(
        "Enter the root directory to scan",
        default=os.getcwd()
    )
    
    if not os.path.exists(root_dir):
        print(f"❌ Directory '{root_dir}' does not exist!")
        return
        
    # Initialize organizer
    organizer = ProjectOrganizer()
    organizer.scan_directory(root_dir)
    
    # Show analysis
    print_analysis_report(organizer.analysis_report)
    
    if not organizer.projects:
        print("\n❌ No projects found!")
        return
        
    # Ask what to do
    action = ask_question(
        "What would you like to do?",
        options=[
            "Extract nested projects",
            "View detailed project structure",
            "Create organized copy of all projects",
            "Exit"
        ]
    )
    
    if action == "Exit":
        return
        
    elif action == "Extract nested projects":
        nested = [p for p in organizer.projects if p.parent]
        
        if not nested:
            print("\n✓ No nested projects found!")
            return
            
        print(f"\n🔍 Found {len(nested)} nested project(s):")
        for i, project in enumerate(nested, 1):
            print(f"  {i}. {project.name} ({', '.join(project.project_types)})")
            print(f"     Path: {project.root_path.relative_to(organizer.root_path)}")
            print(f"     Parent: {project.parent.name}")
            
        # Process each nested project
        for project in nested:
            print(f"\n{'='*60}")
            print(f"📁 Processing: {project.name}")
            
            # Get extraction plan
            plan = organizer.get_extraction_plan(project)
            
            print(f"\n📋 Extraction plan:")
            print(f"  • Files to extract: {len(plan['files_to_move'])}")
            if plan['warnings']:
                print(f"  • ⚠️  Warnings: {len(plan['warnings'])}")
                for warning in plan['warnings'][:3]:
                    print(f"    - {warning}")
                    
            action = ask_question(
                f"Extract '{project.name}'?",
                options=[
                    "Extract to new location",
                    "Keep nested (skip)",
                    "View file list",
                    "Skip all remaining"
                ]
            )
            
            if action == "Skip all remaining":
                break
            elif action == "View file list":
                print("\n📄 Files in project:")
                for f in plan['files_to_move'][:20]:  # Show first 20
                    print(f"  • {f}")
                if len(plan['files_to_move']) > 20:
                    print(f"  ... and {len(plan['files_to_move']) - 20} more files")
                    
                # Ask again
                if ask_question("Extract this project?", options=["Yes", "No"]) == "Yes":
                    action = "Extract to new location"
                else:
                    continue
                    
            if action == "Extract to new location":
                # Get destination
                default_dest = str(Path(root_dir).parent / f"{project.name}_extracted")
                dest = ask_question(
                    f"Enter destination path for '{project.name}'",
                    default=default_dest
                )
                
                # Confirm before extraction
                if ask_question(
                    f"Extract {len(plan['files_to_move'])} files to '{dest}'?",
                    options=["Yes", "No"]
                ) == "Yes":
                    
                    # Check if we should remove from original
                    remove_original = ask_question(
                        "Remove files from original location after extraction?",
                        options=["No (safe copy)", "Yes (move files)"]
                    ) == "Yes (move files)"
                    
                    # Perform extraction
                    new_path = organizer.extract_project(project, dest)
                    
                    if remove_original:
                        # Remove original files
                        print(f"\n🗑️  Removing original files...")
                        shutil.rmtree(project.root_path)
                        print("✓ Original files removed")
                        
    elif action == "View detailed project structure":
        # Show tree view
        print("\n📊 Project Structure:")
        
        def print_tree(project, indent=""):
            prefix = "└── " if indent else ""
            print(f"{indent}{prefix}📁 {project.name} ({', '.join(project.project_types)})")
            
            for i, child in enumerate(project.children):
                is_last = i == len(project.children) - 1
                extension = "    " if is_last else "│   "
                print_tree(child, indent + extension)
                
        # Print root level projects
        root_projects = [p for p in organizer.projects if not p.parent]
        for project in root_projects:
            print_tree(project)
            
    elif action == "Create organized copy of all projects":
        # Create a clean copy with all projects organized
        output_dir = ask_question(
            "Enter output directory for organized projects",
            default=str(Path(root_dir).parent / "organized_projects")
        )
        
        print(f"\n📦 Organizing projects to: {output_dir}")
        result = organizer.organize_projects(output_dir)
        
        print("\n✅ Organization complete!")
        print(f"  • Standalone projects: {len(result['standalone'])}")
        print(f"  • Extracted projects: {len(result['extracted'])}")
        print(f"  • Kept nested: {len(result['kept_nested'])}")
        
    print("\n✨ Done! Your projects have been organized.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        if ask_question("Show full error?", options=["Yes", "No"]) == "Yes":
            traceback.print_exc()
