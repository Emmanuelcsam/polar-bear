#!/usr/bin/env python3
"""
Interactive Configuration Helper
Provides interactive prompts for all scripts instead of command-line arguments
"""

import os
from pathlib import Path
from typing import Optional, Tuple

try:
    # When imported as part of utils package
    from utils.config_loader import ConfigLoader
except ImportError:
    # When run directly or in same directory
    from config_loader import ConfigLoader

class InteractiveConfig:
    def __init__(self):
        self.config = ConfigLoader()
        
    def get_project_choice(self, script_name: str = "Script") -> Tuple[Optional[Path], ConfigLoader]:
        """Interactive project selection"""
        print(f"\n🎯 {script_name}")
        print("="*50)
        print("\nHow would you like to select a project to analyze?")
        print("\n1. Use current directory")
        print("2. Enter a custom path")
        print("3. Select from saved projects")
        print("4. Manage projects (add/remove/list)")
        print("5. Exit")
        
        while True:
            choice = input("\nYour choice (1-5): ").strip()
            
            if choice == '1':
                # Current directory
                project_path = Path.cwd()
                print(f"\n✓ Using current directory: {project_path}")
                return project_path, self.config
                
            elif choice == '2':
                # Custom path
                path_str = input("\nEnter the path to analyze: ").strip()
                if not path_str:
                    print("❌ No path entered, please try again.")
                    continue
                    
                project_path = Path(path_str).expanduser().resolve()
                if not project_path.exists():
                    print(f"❌ Path does not exist: {project_path}")
                    continue
                if not project_path.is_dir():
                    print(f"❌ Path is not a directory: {project_path}")
                    continue
                    
                print(f"\n✓ Using path: {project_path}")
                return project_path, self.config
                
            elif choice == '3':
                # Select from projects
                project_path = self._select_project()
                if project_path:
                    return project_path, self.config
                    
            elif choice == '4':
                # Manage projects
                self._manage_projects()
                
            elif choice == '5':
                # Exit
                print("\n👋 Goodbye!")
                return None, self.config
                
            else:
                print("❌ Invalid choice, please enter 1-5")
    
    def _select_project(self) -> Optional[Path]:
        """Select from saved projects"""
        projects = self.config.list_projects()
        
        if not projects:
            print("\n❌ No projects configured yet!")
            add_now = input("Would you like to add a project now? (y/n): ").strip().lower()
            if add_now == 'y':
                self._add_project()
            return None
        
        print("\n📋 Saved Projects:")
        print("-"*50)
        for i, project in enumerate(projects, 1):
            active = "✓" if project.get('active', False) else " "
            print(f"{i}. [{active}] {project['name']}")
            print(f"   Path: {project['path']}")
            if project.get('description'):
                print(f"   Description: {project['description']}")
        
        print(f"\n0. Go back")
        
        while True:
            choice = input("\nSelect project number: ").strip()
            
            if choice == '0':
                return None
                
            if choice.isdigit() and 1 <= int(choice) <= len(projects):
                project = projects[int(choice) - 1]
                project_path = Path(project['path']).expanduser().resolve()
                
                if not project_path.exists():
                    print(f"❌ Project path no longer exists: {project_path}")
                    remove = input("Remove this project? (y/n): ").strip().lower()
                    if remove == 'y':
                        self.config.remove_project(project['name'])
                        print("✓ Project removed")
                    return None
                
                # Set as active
                self.config.set_active_project(project['name'])
                print(f"\n✓ Selected: {project['name']} ({project_path})")
                return project_path
            else:
                print("❌ Invalid selection")
    
    def _manage_projects(self):
        """Project management menu"""
        while True:
            print("\n📁 Project Management")
            print("-"*50)
            print("1. Add a new project")
            print("2. Remove a project")
            print("3. List all projects")
            print("4. Set active project")
            print("0. Go back")
            
            choice = input("\nYour choice: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self._add_project()
            elif choice == '2':
                self._remove_project()
            elif choice == '3':
                self._list_projects()
            elif choice == '4':
                self._set_active_project()
            else:
                print("❌ Invalid choice")
    
    def _add_project(self):
        """Add a new project interactively"""
        print("\n➕ Add New Project")
        print("-"*30)
        
        name = input("Project name (short identifier): ").strip()
        if not name:
            print("❌ Name cannot be empty")
            return
        
        path = input("Project path: ").strip()
        if not path:
            print("❌ Path cannot be empty")
            return
        
        # Validate path
        project_path = Path(path).expanduser().resolve()
        if not project_path.exists():
            print(f"❌ Path does not exist: {project_path}")
            create = input("Path doesn't exist. Create it? (y/n): ").strip().lower()
            if create == 'y':
                try:
                    project_path.mkdir(parents=True, exist_ok=True)
                    print("✓ Directory created")
                except Exception as e:
                    print(f"❌ Failed to create directory: {e}")
                    return
            else:
                return
        
        description = input("Description (optional): ").strip()
        
        if self.config.add_project(name, str(project_path), description):
            print(f"✅ Added project '{name}'")
            set_active = input("Set as active project? (y/n): ").strip().lower()
            if set_active == 'y':
                self.config.set_active_project(name)
                print("✓ Set as active project")
        else:
            print(f"❌ Project '{name}' already exists")
    
    def _remove_project(self):
        """Remove a project interactively"""
        projects = self.config.list_projects()
        if not projects:
            print("\n❌ No projects to remove")
            return
        
        print("\n➖ Remove Project")
        print("-"*30)
        
        for i, project in enumerate(projects, 1):
            print(f"{i}. {project['name']} - {project['path']}")
        
        print("0. Cancel")
        
        choice = input("\nSelect project to remove: ").strip()
        
        if choice == '0':
            return
            
        if choice.isdigit() and 1 <= int(choice) <= len(projects):
            project = projects[int(choice) - 1]
            confirm = input(f"Remove '{project['name']}'? (y/n): ").strip().lower()
            if confirm == 'y':
                if self.config.remove_project(project['name']):
                    print("✅ Project removed")
                else:
                    print("❌ Failed to remove project")
        else:
            print("❌ Invalid selection")
    
    def _list_projects(self):
        """List all projects"""
        projects = self.config.list_projects()
        
        if not projects:
            print("\n❌ No projects configured")
            return
        
        print("\n📋 All Projects")
        print("-"*50)
        
        for project in projects:
            active = "✓" if project.get('active', False) else " "
            print(f"[{active}] {project['name']}")
            print(f"    Path: {project['path']}")
            if project.get('description'):
                print(f"    Description: {project['description']}")
            print()
        
        input("Press Enter to continue...")
    
    def _set_active_project(self):
        """Set the active project"""
        projects = self.config.list_projects()
        if not projects:
            print("\n❌ No projects configured")
            return
        
        print("\n🎯 Set Active Project")
        print("-"*30)
        
        for i, project in enumerate(projects, 1):
            active = "✓" if project.get('active', False) else " "
            print(f"{i}. [{active}] {project['name']}")
        
        print("0. Cancel")
        
        choice = input("\nSelect project to set as active: ").strip()
        
        if choice == '0':
            return
            
        if choice.isdigit() and 1 <= int(choice) <= len(projects):
            project = projects[int(choice) - 1]
            if self.config.set_active_project(project['name']):
                print(f"✅ '{project['name']}' is now the active project")
            else:
                print("❌ Failed to set active project")
        else:
            print("❌ Invalid selection")
    
    def get_analysis_options(self) -> dict:
        """Get additional analysis options"""
        print("\n⚙️  Analysis Options")
        print("-"*30)
        
        options = {}
        
        # Subdirectories
        include_subdirs = input("Include subdirectories? (Y/n): ").strip().lower()
        options['include_subdirectories'] = include_subdirs != 'n'
        
        if options['include_subdirectories']:
            # Max depth
            depth_input = input("Maximum depth (default: 10, enter for default): ").strip()
            if depth_input.isdigit():
                options['max_depth'] = int(depth_input)
            else:
                options['max_depth'] = 10
        
        # Custom stats directory
        custom_stats = input("Use custom stats directory? (y/N): ").strip().lower()
        if custom_stats == 'y':
            stats_dir = input("Stats directory name (default: .project-stats): ").strip()
            if stats_dir:
                options['stats_directory'] = stats_dir
        
        return options
    
    def confirm_action(self, message: str) -> bool:
        """Simple yes/no confirmation"""
        response = input(f"\n{message} (y/n): ").strip().lower()
        return response == 'y'


# Helper function for scripts
def get_interactive_project_config(script_name: str = "Script") -> Tuple[Optional[Path], ConfigLoader]:
    """Get project configuration interactively"""
    interactive = InteractiveConfig()
    return interactive.get_project_choice(script_name)