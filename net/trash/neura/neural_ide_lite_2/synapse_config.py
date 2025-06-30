"""
Synapse Config: Manages project and script configurations for Synapse IDE.
- Uses clear, dataclass-based structures for projects and scripts.
- Saves configurations in human-readable YAML format.
- Automatically creates default templates to help users get started.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import networkx as nx

# --- Data Structures for Projects and Scripts ---

@dataclass
class ScriptConfig:
    """Configuration for a single script within a project."""
    id: str
    name: str
    description: str = "A Python script."
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # --- NEW: Configuration for autonomous operation ---
    # When true, automatically analyzes the script's syntax on every save.
    auto_check_on_save: bool = False
    # When true, runs the script in a self-healing loop until it succeeds.
    autonomous_mode: bool = False
    
@dataclass
class ProjectConfig:
    """Configuration for an entire Synapse IDE project."""
    name: str
    description: str = "A Synapse IDE project."
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    scripts: List[ScriptConfig] = field(default_factory=list)
    global_parameters: Dict[str, Any] = field(default_factory=dict)

# --- Main Configuration Manager Class ---

class ConfigManager:
    """
    Handles loading, saving, and managing all project configurations and templates.
    Provides a clean API for the main application to interact with the file system.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initializes the config manager and ensures necessary directories exist."""
        self.config_dir = base_dir or Path.home() / ".synapse_ide"
        self.projects_dir = self.config_dir / "projects"
        self.templates_dir = self.config_dir / "templates"
        
        self.config_dir.mkdir(exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        
        self._create_default_templates_if_needed()

    def _create_default_templates_if_needed(self):
        """Checks if default templates exist and creates them if not."""
        if any(self.templates_dir.iterdir()):
            return
        
        print("Creating default project templates...")
        
        ping_pong_template = ProjectConfig(
            name="Ping Pong Communicator",
            description="A simple project demonstrating two scripts sending messages to each other.",
            scripts=[
                ScriptConfig(id="script_1", name="Ping", file_path="ping.py"),
                ScriptConfig(id="script_2", name="Pong", dependencies=["script_1"], file_path="pong.py")
            ]
        )
        self.save_template("ping_pong", ping_pong_template)

    def save_project(self, project_config: ProjectConfig) -> Path:
        """Saves a project configuration to a YAML file."""
        project_config.modified_at = datetime.now().isoformat()
        safe_name = project_config.name.replace(' ', '_').lower()
        file_path = self.projects_dir / f"{safe_name}.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(project_config), f, default_flow_style=False, sort_keys=False)
        return file_path

    def load_project(self, project_name: str) -> ProjectConfig:
        """Loads a project configuration from a YAML file."""
        safe_name = project_name.replace(' ', '_').lower()
        file_path = self.projects_dir / f"{safe_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Project '{project_name}' not found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        data['scripts'] = [ScriptConfig(**s) for s in data.get('scripts', [])]
        return ProjectConfig(**data)

    def list_projects(self) -> List[str]:
        """Returns a list of all available project names."""
        return [p.stem.replace('_', ' ').title() for p in self.projects_dir.glob("*.yaml")]

    def save_template(self, template_name: str, project_config: ProjectConfig):
        """Saves a project configuration as a reusable template."""
        safe_name = template_name.replace(' ', '_').lower()
        file_path = self.templates_dir / f"{safe_name}.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(project_config), f, default_flow_style=False, sort_keys=False)

    def load_template(self, template_name: str) -> ProjectConfig:
        """Loads a template and returns it as a new ProjectConfig object."""
        safe_name = template_name.replace(' ', '_').lower()
        file_path = self.templates_dir / f"{safe_name}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found.")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        data['scripts'] = [ScriptConfig(**s) for s in data.get('scripts', [])]
        project_config = ProjectConfig(**data)
        project_config.name = f"New Project from {template_name.replace('_', ' ').title()}"
        return project_config

    def list_templates(self) -> List[str]:
        """Returns a list of all available template names."""
        return [p.stem.replace('_', ' ').title() for p in self.templates_dir.glob("*.yaml")]

    @staticmethod
    def validate_project(project_config: ProjectConfig) -> List[str]:
        """
        Validates a project for common issues like duplicate IDs or circular dependencies.
        Returns a list of error messages.
        """
        errors = []
        script_ids = [script.id for script in project_config.scripts]

        if len(script_ids) != len(set(script_ids)):
            errors.append("Error: Duplicate Script IDs found. Each script must have a unique ID.")

        graph = nx.DiGraph()
        for script in project_config.scripts:
            graph.add_node(script.id)
            for dep_id in script.dependencies:
                if dep_id not in script_ids:
                    errors.append(f"Error: Script '{script.name}' has a broken dependency on non-existent script ID '{dep_id}'.")
                else:
                    graph.add_edge(dep_id, script.id)
        
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                cycle_path = " -> ".join(cycles[0]) + f" -> {cycles[0][0]}"
                errors.append(f"Error: A circular dependency was detected: {cycle_path}.")
        except Exception as e:
            errors.append(f"An unexpected error occurred during validation: {e}")
            
        return errors

    @staticmethod
    def get_execution_order(project_config: ProjectConfig) -> List[str]:
        """Performs a topological sort to get the correct script execution order."""
        graph = nx.DiGraph()
        for script in project_config.scripts:
            graph.add_node(script.id)
            for dep_id in script.dependencies:
                graph.add_edge(dep_id, script.id)
        
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Cannot determine execution order due to circular dependencies.")
        
        return list(nx.topological_sort(graph))
