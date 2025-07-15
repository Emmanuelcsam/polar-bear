
import threading

class SharedState:
    """
    A thread-safe class to hold the shared state of a script, 
    including parameters and control flags.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "running": True,
            "parameters": {}
        }

    def get(self, key, default=None):
        """Get a value from the state."""
        with self._lock:
            return self._state.get(key, default)

    def set(self, key, value):
        """Set a value in the state."""
        with self._lock:
            self._state[key] = value

    def get_all_parameters(self):
        """Get a copy of the parameters."""
        with self._lock:
            return self._state["parameters"].copy()

    def get_parameter(self, param_key, default=None):
        """Get a specific parameter."""
        with self._lock:
            return self._state["parameters"].get(param_key, default)

    def set_parameter(self, param_key, value):
        """Set a specific parameter."""
        with self._lock:
            self._state["parameters"][param_key] = value
