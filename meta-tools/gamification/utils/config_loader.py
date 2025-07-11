#!/usr/bin/env python3
"""
Configuration loader module for gamification tools
Handles loading and managing project directory configurations
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

class ConfigLoader:
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration loader"""
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.current_project = None
        
    def _load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if not self.config_file.exists():
            # Create default config if it doesn't exist
            default_config = {
                "project_directories": {
                    "default": ".",
                    "projects": [
                        {
                            "name": "current_directory",
                            "path": ".",
                            "description": "Analyze current directory",
                            "active": True
                        }
                    ]
                },
                "analysis_settings": {
                    "include_subdirectories": True,
                    "max_depth": 10,
                    "ignore_patterns": ["__pycache__", "*.pyc", ".git", "node_modules", "venv", "env"],
                    "stats_directory": ".project-stats",
                    "create_stats_in_target": False
                },
                "script_settings": {
                    "verbose": True,
                    "auto_save_reports": True,
                    "report_format": "json"
                }
            }
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def get_active_project(self) -> Optional[Dict]:
        """Get the currently active project configuration"""
        projects = self.config.get('project_directories', {}).get('projects', [])
        for project in projects:
            if project.get('active', False):
                return project
        return None
    
    def get_project_by_name(self, name: str) -> Optional[Dict]:
        """Get project configuration by name"""
        projects = self.config.get('project_directories', {}).get('projects', [])
        for project in projects:
            if project.get('name') == name:
                return project
        return None
    
    def get_project_path(self, project_name: Optional[str] = None) -> Path:
        """Get the path for a specific project or the active one"""
        if project_name:
            project = self.get_project_by_name(project_name)
            if project:
                return Path(project['path']).expanduser().resolve()
        
        # Try to get active project
        active_project = self.get_active_project()
        if active_project:
            return Path(active_project['path']).expanduser().resolve()
        
        # Fall back to default
        default_path = self.config.get('project_directories', {}).get('default', '.')
        return Path(default_path).expanduser().resolve()
    
    def list_projects(self) -> List[Dict]:
        """List all configured projects"""
        return self.config.get('project_directories', {}).get('projects', [])
    
    def set_active_project(self, name: str) -> bool:
        """Set a project as active"""
        projects = self.config.get('project_directories', {}).get('projects', [])
        found = False
        
        for project in projects:
            if project.get('name') == name:
                project['active'] = True
                found = True
            else:
                project['active'] = False
        
        if found:
            self._save_config()
        return found
    
    def add_project(self, name: str, path: str, description: str = "") -> bool:
        """Add a new project to configuration"""
        projects = self.config.get('project_directories', {}).get('projects', [])
        
        # Check if project already exists
        for project in projects:
            if project.get('name') == name:
                return False
        
        # Add new project
        projects.append({
            "name": name,
            "path": path,
            "description": description,
            "active": False
        })
        
        self.config['project_directories']['projects'] = projects
        self._save_config()
        return True
    
    def remove_project(self, name: str) -> bool:
        """Remove a project from configuration"""
        projects = self.config.get('project_directories', {}).get('projects', [])
        original_length = len(projects)
        
        projects = [p for p in projects if p.get('name') != name]
        
        if len(projects) < original_length:
            self.config['project_directories']['projects'] = projects
            self._save_config()
            return True
        return False
    
    def get_stats_directory(self, project_path: Path) -> Path:
        """Get the stats directory for a project"""
        stats_dir_name = self.config.get('analysis_settings', {}).get('stats_directory', '.project-stats')
        create_in_target = self.config.get('analysis_settings', {}).get('create_stats_in_target', False)
        
        if create_in_target:
            return project_path / stats_dir_name
        else:
            # Create stats in the script directory
            return Path.cwd() / stats_dir_name
    
    def get_ignore_patterns(self) -> List[str]:
        """Get ignore patterns for file analysis"""
        return self.config.get('analysis_settings', {}).get('ignore_patterns', [])
    
    def should_include_subdirectories(self) -> bool:
        """Check if subdirectories should be included"""
        return self.config.get('analysis_settings', {}).get('include_subdirectories', True)
    
    def get_max_depth(self) -> int:
        """Get maximum directory depth for analysis"""
        return self.config.get('analysis_settings', {}).get('max_depth', 10)
    
    def _save_config(self):
        """Save configuration back to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_common_argparser(self, description: str) -> argparse.ArgumentParser:
        """Get a common argument parser for all scripts"""
        parser = argparse.ArgumentParser(description=description)
        
        # Project selection arguments
        parser.add_argument('path', nargs='?', default=None,
                          help='Direct path to analyze (overrides config)')
        parser.add_argument('--project', '-p', type=str,
                          help='Project name from config to analyze')
        parser.add_argument('--list-projects', action='store_true',
                          help='List all configured projects')
        parser.add_argument('--set-active', type=str,
                          help='Set a project as active')
        parser.add_argument('--add-project', nargs=3, metavar=('NAME', 'PATH', 'DESCRIPTION'),
                          help='Add a new project to config')
        parser.add_argument('--remove-project', type=str,
                          help='Remove a project from config')
        
        # Analysis options
        parser.add_argument('--no-subdirs', action='store_true',
                          help='Do not include subdirectories')
        parser.add_argument('--max-depth', type=int,
                          help='Maximum directory depth')
        parser.add_argument('--stats-dir', type=str,
                          help='Custom stats directory')
        
        return parser
    
    def handle_common_args(self, args) -> Optional[Path]:
        """Handle common command line arguments and return project path"""
        # Handle project management commands
        if args.list_projects:
            print("\n📋 Configured Projects:")
            print("="*50)
            for project in self.list_projects():
                active = "✓" if project.get('active', False) else " "
                print(f"[{active}] {project['name']:20} - {project['path']}")
                if project.get('description'):
                    print(f"    {project['description']}")
            return None
        
        if args.set_active:
            if self.set_active_project(args.set_active):
                print(f"✅ Set '{args.set_active}' as active project")
            else:
                print(f"❌ Project '{args.set_active}' not found")
            return None
        
        if args.add_project:
            name, path, desc = args.add_project
            if self.add_project(name, path, desc):
                print(f"✅ Added project '{name}'")
            else:
                print(f"❌ Project '{name}' already exists")
            return None
        
        if args.remove_project:
            if self.remove_project(args.remove_project):
                print(f"✅ Removed project '{args.remove_project}'")
            else:
                print(f"❌ Project '{args.remove_project}' not found")
            return None
        
        # Determine project path
        if args.path:
            # Direct path provided
            project_path = Path(args.path).expanduser().resolve()
        elif args.project:
            # Project name provided
            project_path = self.get_project_path(args.project)
            if not project_path:
                print(f"❌ Project '{args.project}' not found")
                return None
        else:
            # Use active project or default
            project_path = self.get_project_path()
        
        # Verify path exists
        if not project_path.exists():
            print(f"❌ Path does not exist: {project_path}")
            return None
        
        if not project_path.is_dir():
            print(f"❌ Path is not a directory: {project_path}")
            return None
        
        # Override settings if provided
        if args.no_subdirs:
            self.config['analysis_settings']['include_subdirectories'] = False
        
        if args.max_depth:
            self.config['analysis_settings']['max_depth'] = args.max_depth
            
        return project_path


# Utility function for scripts to use
def get_project_config(description: str = "Project analysis tool") -> Tuple[ConfigLoader, Path, argparse.Namespace]:
    """
    Standard configuration setup for all scripts
    Returns: (config_loader, project_path, args)
    """
    config = ConfigLoader()
    parser = config.get_common_argparser(description)
    args = parser.parse_args()
    
    project_path = config.handle_common_args(args)
    
    return config, project_path, args