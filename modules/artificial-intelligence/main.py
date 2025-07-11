
import sys
import logging

# Setup sys.path to include the 'core' directory
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.logging_manager import setup_logging, get_logger
from core.logging_manager import setup_logging, get_logger
from core.dependency_manager import check_dependencies
from core.config_manager import ConfigManager
from core.module_loader import ModuleLoader
from core.synapse import Synapse
import json

def main_menu(synapse: Synapse):
    """
    The main user interface for interacting with the Synapse network.
    """
    logger = get_logger("main")
    logger.info("Synapse network operational. Welcome to the command interface.")
    
    print("\n--- Comprehensive Neural Network Integration System ---")
    
    while True:
        print("\n[1] List all available nodes")
        print("[2] Inspect a node")
        print("[3] Execute a node")
        print("[4] Re-scan project to update registry")
        print("[5] Exit")
        
        try:
            choice = input("Please select an option: ").strip()
            
            if choice == '1':
                search = input("Enter search term (or leave blank for all): ").strip()
                nodes = synapse.list_nodes(search)
                print(f"\n--- Found {len(nodes)} Nodes ---")
                for node_name in nodes:
                    print(f"  - {node_name}")

            elif choice == '2':
                node_name = input("Enter the full name of the node to inspect: ").strip()
                try:
                    node = synapse.get_node(node_name)
                    print(f"\n--- Node Inspection: {node.name} ---")
                    print(f"  File Path: {node.file_path}")
                    print(f"  Tunable Parameters: {json.dumps(node.get_tunable_parameters(), indent=4)}")
                except KeyError as e:
                    logger.warning(e)
                    print(f"Error: {e}")

            elif choice == '3':
                node_name = input("Enter the full name of the node to execute: ").strip()
                try:
                    node = synapse.get_node(node_name)
                    params = node.get_tunable_parameters()
                    print(f"Node '{node.name}' requires the following parameters:")
                    print(json.dumps(params, indent=4))
                    print("Enter parameters as key=value pairs, separated by commas.")
                    print("Example: arg1=value1, arg2=123")
                    user_input = input("Parameters: ").strip()
                    
                    kwargs = {}
                    if user_input:
                        for pair in user_input.split(','):
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                # Basic type casting
                                if value.isdigit(): value = int(value)
                                elif value.lower() in ['true', 'false']: value = value.lower() == 'true'
                                kwargs[key.strip()] = value
                    
                    print("\n--- Executing Node ---")
                    result = node.execute(**kwargs)
                    print("--- Execution Result ---")
                    print(result)

                except KeyError as e:
                    logger.warning(e)
                    print(f"Error: {e}")
                except Exception as e:
                    logger.error(f"An error occurred during node execution: {e}", exc_info=True)
                    print(f"An error occurred: {e}")

            elif choice == '4':
                logger.info("User requested project re-scan.")
                import system_analyzer
                system_analyzer.main()
                print("Registry has been updated. Please restart the application to load the new registry.")

            elif choice == '5':
                logger.info("Exiting application.")
                break
            else:
                logger.warning(f"Invalid option selected: {choice}")
                print("Invalid choice, please try again.")

        except (KeyboardInterrupt, EOFError):
            logger.info("Exiting application due to user interruption.")
            break
            
    print("\nSystem shut down.")

if __name__ == "__main__":
    setup_logging()
    check_dependencies()
    config_manager = ConfigManager()
    config_manager.load_config()
    
    try:
        loader = ModuleLoader()
        synapse = Synapse(config_manager, loader)
        main_menu(synapse)
    except FileNotFoundError as e:
        get_logger("main").error(f"A critical error occurred: {e}")
        print(f"\nFATAL ERROR: {e}")
        print("Please run 'python3 system_analyzer.py' to build the registry first.")
    except Exception as e:
        get_logger("main").critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"\nAn unexpected critical error occurred: {e}")
