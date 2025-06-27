"""
Weaver Config: Configuration Manager for Neural Weaver
Handles the loading, saving, and validation of "Flows" (formerly projects)
and "Blocks" (formerly scripts). It uses human-readable YAML files.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
import networkx as nx

# --- Data Structures for Flows and Blocks ---

@dataclass
class BlockConfig:
    """
    Configuration for a single 'Block' on the canvas.
    This is the new, more user-friendly version of 'ScriptConfig'.
    """
    id: str
    name: str
    description: str = "A configurable processing block."
    block_type: str = "custom"  # e.g., 'data_input', 'text_processor', 'image_classifier'
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    # User-friendly settings, replacing 'parameters'
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # For UI positioning on the canvas
    position: Dict[str, int] = field(default_factory=lambda: {"x": 100, "y": 100})
    
    # Advanced options
    auto_restart: bool = False
    max_retries: int = 3
    
@dataclass
class FlowConfig:
    """
    Configuration for an entire 'Flow' or 'Weave'.
    This is the new version of 'ProjectConfig'.
    """
    name: str
    version: str = "1.0"
    description: str = "A Neural Weaver data flow."
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    blocks: List[BlockConfig] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)

# --- Main Configuration Manager Class ---

class ConfigManager:
    """
    Manages all configurations for Neural Weaver, including Flows and Templates.
    Provides a clean API for the main application to interact with the file system.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initializes the configuration manager, creating necessary directories.
        """
        if base_dir:
            self.config_dir = base_dir
        else:
            self.config_dir = Path.home() / ".neural_weaver"
            
        self.flows_dir = self.config_dir / "flows"
        self.templates_dir = self.config_dir / "templates"
        
        # Ensure all necessary directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.flows_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        
        # Populate with default templates if the directory is empty
        self._create_default_templates_if_needed()

    def _create_default_templates_if_needed(self):
        """Checks if default templates exist and creates them if not."""
        if any(self.templates_dir.iterdir()):
            return # Templates already exist
        
        print("Creating default flow templates...")
        
        # --- Template Definitions ---
        
        # 1. Simple Data Processing Pipeline
        pipeline_template = FlowConfig(
            name="Simple Data Pipeline",
            description="A basic ETL (Extract, Transform, Load) pipeline.",
            blocks=[
                BlockConfig(id="block_1", name="Load CSV Data", block_type="data_input", position={"x": 50, "y": 100},
                            settings={"file_path": "path/to/your/data.csv"}),
                BlockConfig(id="block_2", name="Filter Rows", block_type="data_transform", position={"x": 300, "y": 100},
                            dependencies=["block_1"], settings={"filter_column": "age", "operator": ">", "value": 30}),
                BlockConfig(id="block_3", name="Save Results", block_type="data_output", position={"x": 550, "y": 100},
                            dependencies=["block_2"], settings={"output_file": "path/to/filtered_data.csv"})
            ]
        )
        self.save_template("simple_data_pipeline", pipeline_template)
        
        # 2. Basic Neural Network
        nn_template = FlowConfig(
            name="Basic Neural Network",
            description="A simple feed-forward neural network for classification.",
            blocks=[
                BlockConfig(id="input", name="Input Layer", block_type="nn_layer", position={"x": 50, "y": 150},
                            settings={"neurons": 784, "activation": "none"}),
                BlockConfig(id="hidden1", name="Hidden Layer", block_type="nn_layer", position={"x": 300, "y": 50},
                            dependencies=["input"], settings={"neurons": 128, "activation": "relu"}),
                BlockConfig(id="hidden2", name="Hidden Layer", block_type="nn_layer", position={"x": 300, "y": 250},
                            dependencies=["input"], settings={"neurons": 128, "activation": "relu"}),
                BlockConfig(id="output", name="Output Layer", block_type="nn_layer", position={"x": 550, "y": 150},
                            dependencies=["hidden1", "hidden2"], settings={"neurons": 10, "activation": "softmax"})
            ],
            global_settings={"learning_rate": 0.001, "optimizer": "Adam", "epochs": 10}
        )
        self.save_template("basic_neural_network", nn_template)

    def save_flow(self, flow_config: FlowConfig) -> Path:
        """
        Saves a Flow configuration to a YAML file.
        Updates the 'modified_at' timestamp automatically.
        """
        flow_config.modified_at = datetime.now().isoformat()
        file_path = self.flows_dir / f"{flow_config.name.replace(' ', '_').lower()}.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            # Use asdict to convert the dataclass to a dictionary for dumping
            yaml.dump(asdict(flow_config), f, default_flow_style=False, sort_keys=False)
        return file_path

    def load_flow(self, flow_name: str) -> FlowConfig:
        """Loads a Flow configuration from a YAML file."""
        file_path = self.flows_dir / f"{flow_name.replace(' ', '_').lower()}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Flow '{flow_name}' not found at {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        # Reconstruct the dataclass from the loaded dictionary
        data['blocks'] = [BlockConfig(**b) for b in data.get('blocks', [])]
        return FlowConfig(**data)

    def delete_flow(self, flow_name: str):
        """Deletes a flow file."""
        file_path = self.flows_dir / f"{flow_name.replace(' ', '_').lower()}.yaml"
        if file_path.exists():
            file_path.unlink()
        else:
            raise FileNotFoundError(f"Flow '{flow_name}' not found for deletion.")

    def list_flows(self) -> List[str]:
        """Returns a list of all available Flow names."""
        return [p.stem.replace('_', ' ').title() for p in self.flows_dir.glob("*.yaml")]

    def save_template(self, template_name: str, flow_config: FlowConfig):
        """Saves a Flow configuration as a reusable template."""
        file_path = self.templates_dir / f"{template_name.replace(' ', '_').lower()}.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(flow_config), f, default_flow_style=False, sort_keys=False)

    def load_template(self, template_name: str) -> FlowConfig:
        """Loads a template and returns it as a new FlowConfig object."""
        file_path = self.templates_dir / f"{template_name.replace(' ', '_').lower()}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found at {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        data['blocks'] = [BlockConfig(**b) for b in data.get('blocks', [])]
        # Reset name so the user can define a new one
        data['name'] = f"New Flow from {template_name.replace('_', ' ').title()}"
        return FlowConfig(**data)

    def list_templates(self) -> List[str]:
        """Returns a list of all available template names."""
        return [p.stem.replace('_', ' ').title() for p in self.templates_dir.glob("*.yaml")]

    @staticmethod
    def validate_flow(flow_config: FlowConfig) -> List[str]:
        """
        Validates a flow for common issues like duplicate IDs or cyclic dependencies.
        Returns a list of error messages. An empty list means the flow is valid.
        """
        errors = []
        block_ids = [block.id for block in flow_config.blocks]

        # 1. Check for duplicate block IDs
        if len(block_ids) != len(set(block_ids)):
            errors.append("Error: Duplicate Block IDs found. Each block must have a unique ID.")

        # 2. Check for dependency issues using NetworkX for robust cycle detection
        graph = nx.DiGraph()
        for block in flow_config.blocks:
            graph.add_node(block.id)
            for dep_id in block.dependencies:
                if dep_id not in block_ids:
                    errors.append(f"Error: Block '{block.name}' has a broken connection to non-existent block '{dep_id}'.")
                else:
                    graph.add_edge(dep_id, block.id)
        
        # 3. Check for cycles (e.g., A -> B -> A)
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                # Provide a more user-friendly cycle description
                cycle_paths = " -> ".join(cycles[0]) + f" -> {cycles[0][0]}"
                errors.append(f"Error: A circular connection was detected: {cycle_paths}. Blocks cannot depend on each other in a loop.")
        except Exception as e:
            errors.append(f"An unexpected error occurred during validation: {e}")
            
        return errors

# --- Example Usage ---
if __name__ == "__main__":
    # This demonstrates how the ConfigManager works.
    print("--- Testing Weaver ConfigManager ---")
    
    # Create a temporary directory for testing
    test_dir = Path("./temp_weaver_config")
    if not test_dir.exists():
        test_dir.mkdir()
        
    config_mgr = ConfigManager(base_dir=test_dir)
    
    # List initial templates
    print("\nAvailable Templates:")
    templates = config_mgr.list_templates()
    for t in templates:
        print(f"- {t}")
        
    # Create a new flow from a template
    new_flow = config_mgr.load_template(templates[0])
    new_flow.name = "My Test Data Flow"
    new_flow.description = "A test flow created from a template."
    
    # Add a new block
    new_block = BlockConfig(id="block_4", name="Log Output", dependencies=["block_3"], position={"x": 800, "y": 100})
    new_flow.blocks.append(new_block)
    
    # Save the new flow
    config_mgr.save_flow(new_flow)
    print(f"\nSaved new flow: '{new_flow.name}'")
    
    # List all flows
    print("\nAvailable Flows:")
    for f in config_mgr.list_flows():
        print(f"- {f}")
        
    # Load and validate the flow
    loaded_flow = config_mgr.load_flow(new_flow.name)
    print(f"\nLoaded flow '{loaded_flow.name}' with {len(loaded_flow.blocks)} blocks.")
    
    validation_errors = ConfigManager.validate_flow(loaded_flow)
    if not validation_errors:
        print("✅ Flow validation successful.")
    else:
        print("❌ Flow validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
            
    # Clean up the temporary directory
    import shutil
    shutil.rmtree(test_dir)
    print("\nCleaned up temporary test directory.")
