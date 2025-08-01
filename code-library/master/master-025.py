
import logging
import inspect
from .config_manager import ConfigManager
from .module_loader import ModuleLoader

logger = logging.getLogger(__name__)

class RunnableNode:
    """
    A wrapper that makes any discovered function or class method in the registry
    executable and tunable, turning it into a 'node' in our network.
    """
    def __init__(self, module_info: dict, callable_info: dict, loader: ModuleLoader):
        self.name = f"{Path(module_info['file_path']).stem}.{callable_info['name']}"
        self.file_path = module_info['file_path']
        self.callable_name = callable_info['name']
        self.callable_type = 'function' if 'args' in callable_info else 'class'
        self.loader = loader
        self._callable = None
        self._tunable_parameters = self._introspect_parameters(callable_info)
        
        logger.debug(f"Initialized node '{self.name}' from {self.file_path}")

    def _introspect_parameters(self, callable_info: dict) -> dict:
        """
        Introspects the arguments of a function/method to make them tunable.
        In a real system, we'd parse default values from the AST.
        For now, we'll make all args tunable with a default of None.
        """
        params = {}
        args = callable_info.get('args', [])
        # For methods, 'self' is usually the first arg and shouldn't be tunable
        if 'self' in args:
            args.remove('self')
        for arg in args:
            params[arg] = None # Default value
        return params

    def _load_callable(self):
        """Lazy-loads the actual Python callable object."""
        if self._callable is None:
            # This is a simplification. For a class method, we'd need to find the class name.
            self._callable = self.loader.load_callable(self.file_path, function_name=self.callable_name)
            if self._callable is None:
                # Try loading as a class if it's not a top-level function
                # This part needs more robust logic to find the parent class.
                logger.warning(f"Could not load '{self.callable_name}' as a function. A more advanced loader would try to find its parent class.")

    def execute(self, **kwargs):
        """
        Executes the node with the given parameters.
        """
        self._load_callable()
        if not self._callable:
            raise RuntimeError(f"Node '{self.name}' could not be loaded for execution.")

        # Merge runtime args with tunable defaults
        exec_params = self._tunable_parameters.copy()
        exec_params.update(kwargs)
        
        # Filter only the arguments that the function actually accepts
        valid_args = inspect.signature(self._callable).parameters
        final_args = {k: v for k, v in exec_params.items() if k in valid_args}

        logger.info(f"Executing node '{self.name}' with parameters: {final_args}")
        try:
            return self._callable(**final_args)
        except Exception as e:
            logger.error(f"Exception during execution of node '{self.name}': {e}", exc_info=True)
            raise

    def get_tunable_parameters(self) -> dict:
        return self._tunable_parameters

    def set_tunable_parameters(self, params: dict):
        for key, value in params.items():
            if key in self._tunable_parameters:
                self._tunable_parameters[key] = value
            else:
                logger.warning(f"Node '{self.name}' has no tunable parameter '{key}'")


class Synapse:
    """
    The central orchestrator. It uses the ModuleLoader to discover and
    create a network of RunnableNodes from the entire codebase.
    """
    def __init__(self, config_manager: ConfigManager, loader: ModuleLoader):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config_manager
        self.loader = loader
        self.nodes: Dict[str, RunnableNode] = {}
        self._build_network()

    def _build_network(self):
        """
        Scans the module registry and builds the network of runnable nodes.
        """
        self.logger.info("Building neural network from module registry...")
        all_modules = self.loader.get_all_modules()
        
        for module_info in all_modules:
            # Create nodes for top-level functions
            for func_info in module_info.get('functions', []):
                node = RunnableNode(module_info, func_info, self.loader)
                if node.name in self.nodes:
                    logger.warning(f"Duplicate node name '{node.name}'. Overwriting.")
                self.nodes[node.name] = node
            
            # Create nodes for class methods
            for class_name, class_info in module_info.get('classes', {}).items():
                for method_info in class_info.get('methods', []):
                    # A more advanced system would handle class instantiation.
                    # For now, we treat methods like static functions for simplicity.
                    method_node_info = method_info.copy()
                    method_node_info['name'] = f"{class_name}.{method_info['name']}"
                    node = RunnableNode(module_info, method_node_info, self.loader)
                    if node.name in self.nodes:
                        logger.warning(f"Duplicate node name '{node.name}'. Overwriting.")
                    self.nodes[node.name] = node

        self.logger.info(f"Network build complete. {len(self.nodes)} runnable nodes available.")

    def list_nodes(self, search_term: str = None) -> List[str]:
        """
        Lists the names of all available nodes, with an optional search filter.
        """
        node_names = sorted(self.nodes.keys())
        if search_term:
            return [name for name in node_names if search_term in name]
        return node_names

    def get_node(self, node_name: str) -> RunnableNode:
        """
        Retrieves a specific node by its unique name.
        """
        node = self.nodes.get(node_name)
        if not node:
            raise KeyError(f"Node '{node_name}' not found in the network.")
        return node

    def get_all_tunable_parameters(self) -> dict:
        """
        Gathers and returns all tunable parameters from all nodes.
        """
        return {name: node.get_tunable_parameters() for name, node in self.nodes.items()}

    def set_all_tunable_parameters(self, params: dict):
        """
        Sets tunable parameters for multiple nodes from a given dictionary.
        """
        for node_name, node_params in params.items():
            try:
                node = self.get_node(node_name)
                node.set_tunable_parameters(node_params)
            except KeyError:
                self.logger.warning(f"Attempted to set params for non-existent node '{node_name}'")
