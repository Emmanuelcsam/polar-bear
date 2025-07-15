#!/usr/bin/env python3
"""
Integration Wrapper - Enables scripts to work independently and with connectors
Provides seamless integration without modifying original script functionality
"""

import sys
import functools
import threading
from typing import Any, Callable, Dict, Optional

# Global integration state
_integration_context = threading.local()

def get_integration():
    """Get the current integration context if available"""
    return getattr(_integration_context, 'integration', None)

def set_integration(integration):
    """Set the integration context for the current thread"""
    _integration_context.integration = integration

def integrated(func: Callable) -> Callable:
    """Decorator to make functions integration-aware"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        integration = get_integration()
        
        # Report function call if integrated
        if integration and hasattr(integration, 'publish_event'):
            integration.publish_event('function_call', {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            })
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Report success if integrated
            if integration and hasattr(integration, 'publish_event'):
                integration.publish_event('result', {
                    'function': func.__name__,
                    'status': 'success'
                })
            
            return result
            
        except Exception as e:
            # Report error if integrated
            if integration and hasattr(integration, 'publish_event'):
                integration.publish_event('error', {
                    'function': func.__name__,
                    'error': str(e)
                })
            raise
    
    return wrapper

def get_parameter(name: str, default: Any = None) -> Any:
    """Get a parameter value, checking integration first"""
    integration = get_integration()
    
    if integration and hasattr(integration, 'get_parameter'):
        value = integration.get_parameter(name, default)
        if value != default:
            return value
    
    # Fall back to module globals
    frame = sys._getframe(1)
    return frame.f_globals.get(name, default)

def set_parameter(name: str, value: Any):
    """Set a parameter value, updating integration if available"""
    integration = get_integration()
    
    # Update integration
    if integration and hasattr(integration, 'set_parameter'):
        integration.set_parameter(name, value)
    
    # Update module globals
    frame = sys._getframe(1)
    frame.f_globals[name] = value

def update_state(key: str, value: Any):
    """Update shared state"""
    integration = get_integration()
    
    if integration and hasattr(integration, 'update_state'):
        integration.update_state(key, value)

def get_state(key: str, default: Any = None) -> Any:
    """Get shared state value"""
    integration = get_integration()
    
    if integration and hasattr(integration, 'get_state'):
        return integration.get_state(key, default)
    
    return default

def report_progress(progress: float, message: str = ""):
    """Report progress (0.0 to 1.0)"""
    integration = get_integration()
    
    if integration and hasattr(integration, 'report_progress'):
        integration.report_progress(progress, message)
    else:
        # Fallback to console output
        print(f"Progress: {progress * 100:.1f}% - {message}")

def publish_event(event_type: str, data: Dict[str, Any], target: Optional[str] = None):
    """Publish an event"""
    integration = get_integration()
    
    if integration and hasattr(integration, 'publish_event'):
        integration.publish_event(event_type, data, target)

class IntegrationContext:
    """Context manager for integration"""
    
    def __init__(self, integration=None):
        self.integration = integration
        self.previous = None
    
    def __enter__(self):
        self.previous = get_integration()
        set_integration(self.integration)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_integration(self.previous)

# Auto-inject integration functions into main namespace when imported
def inject_integration():
    """Inject integration functions into the calling module's namespace"""
    frame = sys._getframe(1)
    namespace = frame.f_globals
    
    # Only inject if not already present
    if '_integration' not in namespace:
        namespace['_integration'] = get_integration()
        namespace['get_parameter'] = get_parameter
        namespace['set_parameter'] = set_parameter
        namespace['update_state'] = update_state
        namespace['get_state'] = get_state
        namespace['report_progress'] = report_progress
        namespace['publish_event'] = publish_event
        namespace['integrated'] = integrated
        namespace['IntegrationContext'] = IntegrationContext

# Example usage patterns
if __name__ == "__main__":
    print("Integration Wrapper")
    print("==================")
    print("\nThis module provides integration capabilities for scripts.")
    print("\nUsage in scripts:")
    print("1. Import at the beginning:")
    print("   try:")
    print("       from integration_wrapper import *")
    print("       inject_integration()")
    print("   except ImportError:")
    print("       # Define stubs for independent running")
    print("       def report_progress(p, m=''): print(f'Progress: {p*100:.1f}% - {m}')")
    print("       def get_parameter(n, d=None): return globals().get(n, d)")
    print("       # ... etc")
    print("\n2. Use integration functions:")
    print("   report_progress(0.5, 'Halfway done')")
    print("   learning_rate = get_parameter('LEARNING_RATE', 0.01)")
    print("   update_state('model', model)")
    print("\n3. Decorate functions for tracking:")
    print("   @integrated")
    print("   def train_model():")
    print("       # Your code here")
    print("       pass")