

"""
integration_api.py
------------------

This script is the primary entry point for integrating the 'artificial-intelligence'
module with external systems, such as the 'iteration6-lab-framework'.

It initializes the necessary components (logging, config, dependencies) and
exposes the core functionalities of the Synapse orchestrator through a
simple, function-based API.

To use this module from another part of the project:
1. Add the 'artificial-intelligence' directory to your Python path.
   e.g., sys.path.append('/path/to/your/project/modules/artificial-intelligence')
2. Import the functions from this script.
   e.g., from integration_api import get_synapse, analyze_image
"""

import sys
from pathlib import Path
import logging
import pandas as pd

# Ensure the 'core' and 'modules' directories are in the Python path
# This makes the script runnable from anywhere in the project
sys.path.append(str(Path(__file__).parent))

from core.logging_manager import setup_logging, get_logger
from core.dependency_manager import check_dependencies
from core.config_manager import ConfigManager
from core.module_loader import ModuleLoader
from core.synapse import Synapse, RunnableNode

__all__ = [
    'initialize_ai_system', 
    'get_synapse', 
    'get_node',
    'list_nodes',
    'get_parameters', 
    'set_parameters'
]

_synapse_instance = None
_initialized = False

def initialize_ai_system(interactive: bool = True) -> Synapse:
    """
    Initializes the entire AI system stack.
    """
    global _synapse_instance, _initialized
    if _initialized:
        get_logger("API").info("AI system already initialized.")
        return _synapse_instance

    setup_logging()
    logger = get_logger("API")
    logger.info("Initializing AI System via API...")

    if interactive:
        check_dependencies()

    config_manager = ConfigManager()
    config_manager.load_config()

    loader = ModuleLoader()
    _synapse_instance = Synapse(config_manager, loader)
    _initialized = True
    
    logger.info("AI System initialization complete.")
    return _synapse_instance

def get_synapse() -> Synapse:
    """
    Returns the singleton Synapse instance, initializing if necessary.
    """
    if not _initialized:
        initialize_ai_system(interactive=False)
    return _synapse_instance

def list_nodes(search_term: str = None) -> list:
    """
    Lists all available nodes in the Synapse network.
    """
    synapse = get_synapse()
    return synapse.list_nodes(search_term)

def get_node(node_name: str) -> RunnableNode:
    """
    Retrieves a specific node from the network.
    """
    synapse = get_synapse()
    return synapse.get_node(node_name)

def get_parameters() -> dict:
    """
    Gets all tunable parameters from all nodes in the network.
    """
    synapse = get_synapse()
    return synapse.get_all_tunable_parameters()

def set_parameters(params: dict):
    """
    Sets tunable parameters for one or more nodes.
    """
    synapse = get_synapse()
    synapse.set_all_tunable_parameters(params)

if __name__ == '__main__':
    print("--- Running AI Module API Demo ---")
    initialize_ai_system(interactive=False)
    
    print("\n1. Listing all available nodes (first 20)...")
    all_nodes = list_nodes()
    print(f"Total nodes found: {len(all_nodes)}")
    for node_name in all_nodes[:20]:
        print(f"  - {node_name}")
        
    print("\n2. Getting a specific node...")
    try:
        node = get_node('system_analyzer.main')
        print(f"Successfully retrieved node: {node.name}")
        params = node.get_tunable_parameters()
        print(f"  - Parameters: {params}")
    except KeyError as e:
        print(e)
    
    print("\n--- API Demo Complete ---")

