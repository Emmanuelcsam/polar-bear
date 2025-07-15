
from abc import ABC, abstractmethod
import logging

class BaseModule(ABC):
    """
    Abstract Base Class for all modules in the system.
    It defines a standard interface for initialization, execution,
    and parameter management.
    """
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self._tunable_parameters = {}
        self.logger.info(f"Module '{self.__class__.__name__}' initialized.")

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        The main execution method for the module.
        This method must be implemented by all subclasses.
        """
        pass

    def get_tunable_parameters(self) -> dict:
        """
        Returns a dictionary of parameters that can be tuned by the
        external neural network.
        """
        self.logger.debug(f"Fetching tunable parameters: {self._tunable_parameters}")
        return self._tunable_parameters

    def set_tunable_parameters(self, params: dict):
        """
        Sets the tunable parameters for the module.
        """
        self.logger.info(f"Updating tunable parameters with: {params}")
        for key, value in params.items():
            if key in self._tunable_parameters:
                self._tunable_parameters[key] = value
                self.logger.debug(f"Set parameter '{key}' to '{value}'")
            else:
                self.logger.warning(f"Attempted to set unknown parameter '{key}'")

    def _register_tunable_parameter(self, name: str, default_value):
        """
        Registers a parameter as tunable.
        """
        self._tunable_parameters[name] = default_value
        self.logger.debug(f"Registered tunable parameter '{name}' with default value '{default_value}'")
