# orchestrator.py
import pkgutil, importlib, sys, os
from script_interface import ScriptManager

# Control parameters
_params = {'auto_load': True, 'exclude': ['__pycache__', '.git', 'venv']}
_loaded_modules = {}
_script_manager = None

def load_modules():
    global _loaded_modules, _script_manager
    
    # Initialize script manager
    _script_manager = ScriptManager(os.getcwd())
    
    # Discover and load scripts
    scripts = _script_manager.discover_scripts()
    load_results = _script_manager.load_all_scripts()
    
    # Also load modules the traditional way for backward compatibility
    mods = [m.name for m in pkgutil.iter_modules(['.']) if m.name not in _params['exclude']]
    
    for m in mods:
        if m.endswith('.py'): m = m[:-3]
        try:
            if m not in ['orchestrator', 'connector', 'hivemind_connector', 'script_interface']:
                _loaded_modules[m] = importlib.import_module(m)
        except Exception as e:
            print(f"Failed to load {m}: {e}")
    
    return {'modules': list(_loaded_modules.keys()), 'scripts': scripts}

def orchestrate_workflow(workflow):
    """Execute a workflow across multiple modules"""
    if _script_manager:
        return _script_manager.orchestrate_collaboration(workflow)
    return {'error': 'Script manager not initialized'}

def call_module_function(module_name, function_name, *args, **kwargs):
    """Call a function in a loaded module"""
    if module_name in _loaded_modules:
        module = _loaded_modules[module_name]
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            return func(*args, **kwargs)
    return {'error': f'Module {module_name} or function {function_name} not found'}

def get_module_info(module_name=None):
    """Get information about loaded modules"""
    if module_name:
        if module_name in _loaded_modules:
            module = _loaded_modules[module_name]
            return {
                'name': module_name,
                'functions': [f for f in dir(module) if not f.startswith('_') and callable(getattr(module, f))],
                'params': getattr(module, '_params', {})
            }
    return {'loaded': list(_loaded_modules.keys())}

# Connector interface
def set_param(k,v): _params[k] = v; return True
def get_param(k): return _params.get(k)
def get_info(): return {'params': _params, 'loaded_modules': list(_loaded_modules.keys())}

if __name__=='__main__': 
    if _params['auto_load']:
        result = load_modules()
        print(f"Loaded modules: {result['modules']}")

