#!/usr/bin/env python3
"""
Integration Template for Hivemind Connector System
This template shows how to integrate any script with the hivemind connector.
Copy this template and modify for your specific script.
"""

from connector_interface import setup_connector, get_hivemind_parameter, send_hivemind_status

# Import your existing modules here
# import your_module

def your_main_function(param1, param2, connector=None):
    """
    Your main processing function.
    Add connector parameter to enable hivemind integration.
    """
    # Get parameters from hivemind if available
    if connector:
        param1 = connector.get_parameter('param1', param1)
        param2 = connector.get_parameter('param2', param2)
    
    # Your existing logic here
    result = {}  # Replace with actual processing
    
    # Send status updates to hivemind
    if connector:
        connector.send_status({
            'action': 'processing',
            'param1': param1,
            'param2': param2
        })
    
    # Do your actual work here
    # ...
    
    # Send completion status
    if connector:
        connector.send_status({
            'action': 'complete',
            'result': result
        })
    
    return result


def callback_function_example(arg):
    """Example callback that can be triggered by hivemind"""
    return f"Callback executed with arg: {arg}"


def main():
    """Main function with hivemind integration"""
    # Setup connector
    connector = setup_connector("your_script_name.py")
    
    if connector.is_connected:
        print("Connected to hivemind system")
        
        # Register parameters that can be controlled by hivemind
        connector.register_parameter("param1", "default_value1", "Description of param1")
        connector.register_parameter("param2", 42, "Description of param2")
        connector.register_parameter("listen_mode", False, "Whether to listen for commands")
        
        # Register callbacks that hivemind can trigger
        connector.register_callback("main_function", lambda: your_main_function(
            connector.get_parameter('param1'),
            connector.get_parameter('param2'),
            connector
        ))
        connector.register_callback("callback_example", callback_function_example)
        connector.register_callback("get_status", lambda: {"status": "ready"})
    else:
        print("Running in standalone mode")
    
    # Run your main logic
    try:
        result = your_main_function("value1", 42, connector)
        print(f"Result: {result}")
        
        # If connected and in listen mode, start listening for commands
        if connector.is_connected and connector.get_parameter('listen_mode', False):
            print("Entering listen mode for hivemind commands...")
            connector.listen_for_commands()
            
    except Exception as e:
        print(f"Error: {e}")
        if connector.is_connected:
            connector.send_status({'error': str(e)})


if __name__ == "__main__":
    main()