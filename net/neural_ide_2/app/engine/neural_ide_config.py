# === config_manager.py ===
"""
Configuration Manager for Neural Script IDE
Handles project configurations, templates, and settings
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Some features may not work.")
    yaml = None

@dataclass
class ScriptConfig:
    """Configuration for a single script"""
    id: str
    name: str
    file_path: Optional[str] = None
    dependencies: List[str] = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    parameters: Dict[str, Any] = None
    auto_restart: bool = False
    max_retries: int = 3
    timeout: Optional[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.inputs is None:
            self.inputs = {}
        if self.outputs is None:
            self.outputs = {}
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ProjectConfig:
    """Configuration for an entire project"""
    name: str
    version: str
    description: str = ""
    created: str = ""
    modified: str = ""
    scripts: List[ScriptConfig] = None
    global_parameters: Dict[str, Any] = None
    message_broker_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()
        if self.scripts is None:
            self.scripts = []
        if self.global_parameters is None:
            self.global_parameters = {}
        if self.message_broker_config is None:
            self.message_broker_config = {
                'buffer_size': 1000,
                'message_timeout': 30,
                'heartbeat_interval': 5
            }

class ConfigurationManager:
    """Manages IDE configurations and projects"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".neural_script_ide"
        self.projects_dir = self.config_dir / "projects"
        self.templates_dir = self.config_dir / "templates"
        self.current_project = None
        
        # Create directories
        self.config_dir.mkdir(exist_ok=True)
        self.projects_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Load default templates
        self._create_default_templates()
        
    def _create_default_templates(self):
        """Create default project templates"""
        templates = {
            'simple_pipeline': self._create_pipeline_template(),
            'neural_network': self._create_neural_network_template(),
            'data_processing': self._create_data_processing_template(),
            'microservices': self._create_microservices_template()
        }
        
        for name, config in templates.items():
            self.save_template(name, config)
            
    def _create_pipeline_template(self) -> ProjectConfig:
        """Create a simple pipeline template"""
        return ProjectConfig(
            name="Simple Pipeline",
            version="1.0",
            description="A simple data processing pipeline",
            scripts=[
                ScriptConfig(
                    id="data_reader",
                    name="Data Reader",
                    outputs={"data": "raw_data"},
                    parameters={"source": "file.csv"}
                ),
                ScriptConfig(
                    id="processor",
                    name="Data Processor",
                    dependencies=["data_reader"],
                    inputs={"data": "raw_data"},
                    outputs={"data": "processed_data"}
                ),
                ScriptConfig(
                    id="writer",
                    name="Data Writer",
                    dependencies=["processor"],
                    inputs={"data": "processed_data"},
                    parameters={"destination": "output.csv"}
                )
            ]
        )
        
    def _create_neural_network_template(self) -> ProjectConfig:
        """Create a neural network template"""
        return ProjectConfig(
            name="Neural Network",
            version="1.0",
            description="Multi-layer neural network with parallel processing",
            scripts=[
                ScriptConfig(
                    id="input_layer",
                    name="Input Layer",
                    outputs={"activations": "input_features"},
                    parameters={
                        "input_size": 784,
                        "batch_size": 32
                    }
                ),
                ScriptConfig(
                    id="hidden_layer_1",
                    name="Hidden Layer 1",
                    dependencies=["input_layer"],
                    inputs={"features": "input_features"},
                    outputs={"activations": "hidden_1_output"},
                    parameters={
                        "neurons": 128,
                        "activation": "relu",
                        "dropout": 0.2
                    }
                ),
                ScriptConfig(
                    id="hidden_layer_2",
                    name="Hidden Layer 2",
                    dependencies=["input_layer"],
                    inputs={"features": "input_features"},
                    outputs={"activations": "hidden_2_output"},
                    parameters={
                        "neurons": 64,
                        "activation": "tanh"
                    }
                ),
                ScriptConfig(
                    id="attention_layer",
                    name="Attention Mechanism",
                    dependencies=["hidden_layer_1", "hidden_layer_2"],
                    inputs={
                        "hidden_1": "hidden_1_output",
                        "hidden_2": "hidden_2_output"
                    },
                    outputs={"attention_weights": "attention_output"}
                ),
                ScriptConfig(
                    id="output_layer",
                    name="Output Layer",
                    dependencies=["attention_layer"],
                    inputs={"features": "attention_output"},
                    outputs={"predictions": "final_output"},
                    parameters={
                        "num_classes": 10,
                        "activation": "softmax"
                    }
                ),
                ScriptConfig(
                    id="loss_calculator",
                    name="Loss Calculator",
                    dependencies=["output_layer"],
                    inputs={"predictions": "final_output"},
                    outputs={"loss": "training_loss"},
                    parameters={
                        "loss_function": "cross_entropy"
                    }
                )
            ],
            global_parameters={
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 100
            }
        )
        
    def _create_data_processing_template(self) -> ProjectConfig:
        """Create a data processing template"""
        return ProjectConfig(
            name="Data Processing Pipeline",
            version="1.0",
            description="ETL pipeline with validation and monitoring",
            scripts=[
                ScriptConfig(
                    id="extractor",
                    name="Data Extractor",
                    outputs={"raw_data": "extracted_data"},
                    parameters={
                        "sources": ["database", "api", "files"],
                        "batch_size": 1000
                    }
                ),
                ScriptConfig(
                    id="validator",
                    name="Data Validator",
                    dependencies=["extractor"],
                    inputs={"data": "extracted_data"},
                    outputs={
                        "valid_data": "validated_data",
                        "errors": "validation_errors"
                    },
                    parameters={
                        "rules": "validation_rules.json"
                    }
                ),
                ScriptConfig(
                    id="transformer",
                    name="Data Transformer",
                    dependencies=["validator"],
                    inputs={"data": "validated_data"},
                    outputs={"transformed": "transformed_data"},
                    parameters={
                        "transformations": ["normalize", "aggregate", "enrich"]
                    }
                ),
                ScriptConfig(
                    id="loader",
                    name="Data Loader",
                    dependencies=["transformer"],
                    inputs={"data": "transformed_data"},
                    parameters={
                        "destination": "data_warehouse",
                        "mode": "append"
                    }
                ),
                ScriptConfig(
                    id="monitor",
                    name="Pipeline Monitor",
                    dependencies=["extractor", "validator", "transformer", "loader"],
                    inputs={
                        "errors": "validation_errors"
                    },
                    parameters={
                        "alert_threshold": 0.05,
                        "metrics": ["throughput", "error_rate", "latency"]
                    }
                )
            ]
        )
        
    def _create_microservices_template(self) -> ProjectConfig:
        """Create a microservices template"""
        return ProjectConfig(
            name="Microservices Architecture",
            version="1.0",
            description="Distributed microservices with load balancing",
            scripts=[
                ScriptConfig(
                    id="api_gateway",
                    name="API Gateway",
                    outputs={"requests": "incoming_requests"},
                    parameters={
                        "port": 8080,
                        "rate_limit": 100,
                        "auth_enabled": True
                    }
                ),
                ScriptConfig(
                    id="load_balancer",
                    name="Load Balancer",
                    dependencies=["api_gateway"],
                    inputs={"requests": "incoming_requests"},
                    outputs={"distributed": "balanced_requests"},
                    parameters={
                        "algorithm": "round_robin",
                        "health_check_interval": 30
                    }
                ),
                ScriptConfig(
                    id="auth_service",
                    name="Authentication Service",
                    dependencies=["load_balancer"],
                    inputs={"requests": "balanced_requests"},
                    outputs={"authenticated": "auth_requests"},
                    parameters={
                        "token_expiry": 3600,
                        "max_attempts": 3
                    }
                ),
                ScriptConfig(
                    id="user_service",
                    name="User Service",
                    dependencies=["auth_service"],
                    inputs={"requests": "auth_requests"},
                    outputs={"responses": "user_responses"},
                    parameters={
                        "cache_enabled": True,
                        "database": "users_db"
                    },
                    auto_restart=True
                ),
                ScriptConfig(
                    id="order_service",
                    name="Order Service",
                    dependencies=["auth_service"],
                    inputs={"requests": "auth_requests"},
                    outputs={"responses": "order_responses"},
                    parameters={
                        "database": "orders_db",
                        "transaction_timeout": 30
                    },
                    auto_restart=True
                ),
                ScriptConfig(
                    id="notification_service",
                    name="Notification Service",
                    dependencies=["user_service", "order_service"],
                    inputs={
                        "user_events": "user_responses",
                        "order_events": "order_responses"
                    },
                    parameters={
                        "channels": ["email", "sms", "push"],
                        "queue_size": 1000
                    }
                )
            ],
            message_broker_config={
                "type": "rabbitmq",
                "host": "localhost",
                "port": 5672,
                "prefetch_count": 10
            }
        )
        
    def save_project(self, project: ProjectConfig, name: str) -> Path:
        """Save a project configuration"""
        project_file = self.projects_dir / f"{name}.yaml"
        
        with open(project_file, 'w') as f:
            yaml.dump(asdict(project), f, default_flow_style=False)
            
        return project_file
        
    def load_project(self, name: str) -> ProjectConfig:
        """Load a project configuration"""
        project_file = self.projects_dir / f"{name}.yaml"
        
        if not project_file.exists():
            raise FileNotFoundError(f"Project '{name}' not found")
            
        with open(project_file, 'r') as f:
            data = yaml.safe_load(f)
            
        # Convert to ProjectConfig
        scripts = [ScriptConfig(**s) for s in data.get('scripts', [])]
        data['scripts'] = scripts
        
        return ProjectConfig(**data)
        
    def save_template(self, name: str, config: ProjectConfig):
        """Save a project template"""
        template_file = self.templates_dir / f"{name}.yaml"
        
        with open(template_file, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False)
            
    def list_projects(self) -> List[str]:
        """List all saved projects"""
        return [f.stem for f in self.projects_dir.glob("*.yaml")]
        
    def list_templates(self) -> List[str]:
        """List all available templates"""
        return [f.stem for f in self.templates_dir.glob("*.yaml")]
        
    def export_project(self, project: ProjectConfig, output_path: str, 
                      include_scripts: bool = True):
        """Export project with all scripts"""
        export_dir = Path(output_path) / project.name.replace(" ", "_").lower()
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save project config
        with open(export_dir / "project.yaml", 'w') as f:
            yaml.dump(asdict(project), f, default_flow_style=False)
            
        # Create scripts directory
        if include_scripts:
            scripts_dir = export_dir / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            
            # Copy script files
            for script in project.scripts:
                if script.file_path and Path(script.file_path).exists():
                    import shutil
                    shutil.copy2(script.file_path, 
                               scripts_dir / f"{script.id}.py")
                               
        # Create README
        readme_content = f"""# {project.name}

{project.description}

## Version: {project.version}

## Scripts:
"""
        for script in project.scripts:
            readme_content += f"\n### {script.name} ({script.id})"
            if script.dependencies:
                readme_content += f"\n- Dependencies: {', '.join(script.dependencies)}"
            if script.parameters:
                readme_content += f"\n- Parameters: {json.dumps(script.parameters, indent=2)}"
            readme_content += "\n"
            
        with open(export_dir / "README.md", 'w') as f:
            f.write(readme_content)
            
        print(f"Project exported to: {export_dir}")
        
    def validate_project(self, project: ProjectConfig) -> List[str]:
        """Validate project configuration"""
        errors = []
        
        # Check for duplicate script IDs
        script_ids = [s.id for s in project.scripts]
        if len(script_ids) != len(set(script_ids)):
            errors.append("Duplicate script IDs found")
            
        # Check dependencies
        for script in project.scripts:
            for dep in script.dependencies:
                if dep not in script_ids:
                    errors.append(f"Script '{script.id}' depends on unknown script '{dep}'")
                    
        # Check for circular dependencies
        try:
            self._check_circular_dependencies(project)
        except ValueError as e:
            errors.append(str(e))
            
        return errors
        
    def _check_circular_dependencies(self, project: ProjectConfig):
        """Check for circular dependencies using DFS"""
        # Build adjacency list
        graph = {s.id: s.dependencies for s in project.scripts}
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for script_id in graph:
            if script_id not in visited:
                if has_cycle(script_id):
                    raise ValueError(f"Circular dependency detected involving '{script_id}'")


# === project_wizard.py ===
"""
Project Creation Wizard for Neural Script IDE
Interactive project setup with guided configuration
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Any, Optional
import json

class ProjectWizard:
    """Interactive wizard for creating projects"""
    
    def __init__(self, parent, config_manager):
        self.parent = parent
        self.config_manager = config_manager
        self.result = None
        
        self.window = tk.Toplevel(parent)
        self.window.title("New Project Wizard")
        self.window.geometry("800x600")
        
        # Current page
        self.current_page = 0
        self.pages = []
        
        # Project data
        self.project_data = {
            'name': '',
            'description': '',
            'template': None,
            'scripts': [],
            'parameters': {}
        }
        
        self.setup_ui()
        self.create_pages()
        self.show_page(0)
        
    def setup_ui(self):
        """Setup wizard UI"""
        # Header
        header_frame = ttk.Frame(self.window)
        header_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.title_label = ttk.Label(header_frame, text="New Project Wizard",
                                    font=("Arial", 16, "bold"))
        self.title_label.pack()
        
        self.subtitle_label = ttk.Label(header_frame, text="Step 1 of 5")
        self.subtitle_label.pack()
        
        # Content area
        self.content_frame = ttk.Frame(self.window)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Navigation
        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.back_button = ttk.Button(nav_frame, text="< Back", 
                                     command=self.previous_page)
        self.back_button.pack(side=tk.LEFT)
        
        self.next_button = ttk.Button(nav_frame, text="Next >", 
                                     command=self.next_page)
        self.next_button.pack(side=tk.RIGHT)
        
        self.cancel_button = ttk.Button(nav_frame, text="Cancel", 
                                       command=self.window.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
    def create_pages(self):
        """Create wizard pages"""
        self.pages = [
            self.create_welcome_page,
            self.create_template_page,
            self.create_scripts_page,
            self.create_connections_page,
            self.create_summary_page
        ]
        
    def show_page(self, index):
        """Show specific page"""
        # Clear content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
            
        # Update navigation
        self.current_page = index
        self.subtitle_label.config(text=f"Step {index + 1} of {len(self.pages)}")
        
        self.back_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
        self.next_button.config(text="Finish" if index == len(self.pages) - 1 else "Next >")
        
        # Create page
        self.pages[index]()
        
    def next_page(self):
        """Go to next page"""
        if self.current_page == len(self.pages) - 1:
            self.finish()
        else:
            self.show_page(self.current_page + 1)
            
    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.show_page(self.current_page - 1)
            
    def create_welcome_page(self):
        """Create welcome page"""
        self.title_label.config(text="Welcome to Project Wizard")
        
        # Project name
        ttk.Label(self.content_frame, text="Project Name:",
                 font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=10)
                 
        self.name_var = tk.StringVar(value=self.project_data.get('name', ''))
        name_entry = ttk.Entry(self.content_frame, textvariable=self.name_var, width=40)
        name_entry.grid(row=0, column=1, sticky=tk.W, pady=10)
        
        # Description
        ttk.Label(self.content_frame, text="Description:",
                 font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.NW, pady=10)
                 
        self.desc_text = tk.Text(self.content_frame, width=50, height=5)
        self.desc_text.grid(row=1, column=1, sticky=tk.W, pady=10)
        self.desc_text.insert(1.0, self.project_data.get('description', ''))
        
        # Project type
        ttk.Label(self.content_frame, text="Project Type:",
                 font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=10)
                 
        self.type_var = tk.StringVar(value="custom")
        types = [
            ("Start from template", "template"),
            ("Create custom project", "custom"),
            ("Import existing project", "import")
        ]
        
        for i, (text, value) in enumerate(types):
            ttk.Radiobutton(self.content_frame, text=text, variable=self.type_var,
                           value=value).grid(row=2+i, column=1, sticky=tk.W, pady=2)
                           
    def create_template_page(self):
        """Create template selection page"""
        self.title_label.config(text="Select Template")
        
        if self.type_var.get() != "template":
            # Skip to next page
            self.next_page()
            return
            
        # Template list
        ttk.Label(self.content_frame, text="Available Templates:",
                 font=("Arial", 10, "bold")).pack(pady=10)
                 
        # Template listbox with descriptions
        template_frame = ttk.Frame(self.content_frame)
        template_frame.pack(fill=tk.BOTH, expand=True)
        
        # Listbox
        self.template_listbox = tk.Listbox(template_frame, height=10)
        self.template_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Description
        desc_frame = ttk.LabelFrame(template_frame, text="Description", padding=10)
        desc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.template_desc = tk.Text(desc_frame, width=40, height=10, wrap=tk.WORD)
        self.template_desc.pack(fill=tk.BOTH, expand=True)
        
        # Load templates
        templates = self.config_manager.list_templates()
        for template in templates:
            self.template_listbox.insert(tk.END, template)
            
        # Bind selection
        self.template_listbox.bind('<<ListboxSelect>>', self.on_template_select)
        
    def on_template_select(self, event):
        """Handle template selection"""
        selection = self.template_listbox.curselection()
        if selection:
            template_name = self.template_listbox.get(selection[0])
            
            # Load template
            try:
                # Load template from templates directory
                template_file = self.config_manager.templates_dir / f"{template_name}.yaml"
                if template_file.exists():
                    with open(template_file, 'r') as f:
                        template_data = yaml.safe_load(f)
                    
                    # Show description
                    self.template_desc.delete(1.0, tk.END)
                    self.template_desc.insert(1.0, template_data.get('description', 'No description available'))
                    
                    # Store selection
                    self.project_data['template'] = template_name
                    self.project_data['template_data'] = template_data
                else:
                    raise FileNotFoundError(f"Template '{template_name}' not found")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load template: {e}")
                
    def create_scripts_page(self):
        """Create scripts configuration page"""
        self.title_label.config(text="Configure Scripts")
        
        # Script list
        list_frame = ttk.Frame(self.content_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(list_frame)
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="Add Script", command=self.add_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Remove", command=self.remove_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Edit", command=self.edit_script).pack(side=tk.LEFT, padx=2)
        
        # Script tree
        columns = ('name', 'type', 'dependencies')
        self.script_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.script_tree.heading(col, text=col.title())
            self.script_tree.column(col, width=150)
            
        self.script_tree.pack(fill=tk.BOTH, expand=True)
        
        # Load scripts if template selected
        if self.project_data.get('template'):
            self.load_template_scripts()
            
    def load_template_scripts(self):
        """Load scripts from selected template"""
        # Implementation to load template scripts
        pass
        
    def add_script(self):
        """Add new script"""
        dialog = ScriptDialog(self.window)
        if dialog.result:
            self.project_data['scripts'].append(dialog.result)
            self.update_script_tree()
            
    def remove_script(self):
        """Remove selected script"""
        selection = self.script_tree.selection()
        if selection:
            # Remove script
            pass
            
    def edit_script(self):
        """Edit selected script"""
        selection = self.script_tree.selection()
        if selection:
            # Edit script
            pass
            
    def update_script_tree(self):
        """Update script tree display"""
        # Clear tree
        for item in self.script_tree.get_children():
            self.script_tree.delete(item)
            
        # Add scripts
        for script in self.project_data['scripts']:
            self.script_tree.insert('', tk.END, values=(
                script.get('name', ''),
                script.get('type', 'python'),
                ', '.join(script.get('dependencies', []))
            ))
            
    def create_connections_page(self):
        """Create connections configuration page"""
        self.title_label.config(text="Configure Connections")
        
        # Visual connection editor
        ttk.Label(self.content_frame, text="Drag to create connections between scripts").pack()
        
        # Canvas for visual editing
        canvas = tk.Canvas(self.content_frame, bg='white', height=400)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        # Draw script nodes
        self.draw_script_nodes(canvas)
        
    def draw_script_nodes(self, canvas):
        """Draw script nodes on canvas"""
        # Implementation for visual node editor
        pass
        
    def create_summary_page(self):
        """Create summary page"""
        self.title_label.config(text="Project Summary")
        
        # Summary text
        summary = tk.Text(self.content_frame, width=60, height=20)
        summary.pack(fill=tk.BOTH, expand=True)
        
        # Generate summary
        summary_text = f"""Project: {self.name_var.get()}
Description: {self.desc_text.get(1.0, tk.END).strip()}

Scripts: {len(self.project_data['scripts'])}
"""
        
        for script in self.project_data['scripts']:
            summary_text += f"\n- {script.get('name', 'Unnamed')}"
            
        summary.insert(1.0, summary_text)
        summary.config(state='disabled')
        
    def finish(self):
        """Complete wizard"""
        # Create project configuration
        project = ProjectConfig(
            name=self.name_var.get(),
            version="1.0",
            description=self.desc_text.get(1.0, tk.END).strip(),
            scripts=self.project_data['scripts']
        )
        
        # Validate
        errors = self.config_manager.validate_project(project)
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return
            
        # Save project
        try:
            self.config_manager.save_project(project, project.name)
            self.result = project
            messagebox.showinfo("Success", f"Project '{project.name}' created successfully!")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project: {e}")


class ScriptDialog:
    """Dialog for adding/editing scripts"""
    
    def __init__(self, parent, script_data=None):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Script Configuration")
        self.dialog.geometry("500x400")
        
        # Script data
        self.script_data = script_data or {}
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI"""
        # Basic info
        info_frame = ttk.LabelFrame(self.dialog, text="Basic Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(info_frame, text="Script ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.id_var = tk.StringVar(value=self.script_data.get('id', ''))
        ttk.Entry(info_frame, textvariable=self.id_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Name:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.name_var = tk.StringVar(value=self.script_data.get('name', ''))
        ttk.Entry(info_frame, textvariable=self.name_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(self.dialog, text="Parameters", padding=10)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Parameter editor
        self.param_text = tk.Text(param_frame, height=10)
        self.param_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        
    def ok(self):
        """Save script configuration"""
        self.result = {
            'id': self.id_var.get(),
            'name': self.name_var.get(),
            'parameters': {}  # Parse from text
        }
        self.dialog.destroy()


# Example usage
if __name__ == "__main__":
    # Test configuration manager
    config_mgr = ConfigurationManager()
    
    # List templates
    print("Available templates:")
    for template in config_mgr.list_templates():
        print(f"  - {template}")
        
    # Load a template
    neural_net = config_mgr.load_project("neural_network")
    print(f"\nLoaded template: {neural_net.name}")
    print(f"Scripts: {len(neural_net.scripts)}")
    
    # Validate project
    errors = config_mgr.validate_project(neural_net)
    if errors:
        print("Validation errors:", errors)
    else:
        print("Project is valid!")
