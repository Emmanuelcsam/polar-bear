#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced connector system
"""

import json
import time
from connector import ConnectorSystem

def main():
    """Demonstrate the connector system capabilities"""
    
    print("=== Connector System Demo ===\n")
    
    # Initialize the connector system
    connector = ConnectorSystem()
    connector.scan_scripts()
    connector.enable_collaboration()
    
    # 1. Show system status
    print("1. System Status:")
    status = connector.get_system_status()
    print(f"   Scripts loaded: {status['scripts_loaded']}")
    print(f"   Available scripts: {', '.join(status['scripts'].keys())}")
    print()
    
    # 2. Execute correlation finder with test data
    print("2. Running Correlation Analysis:")
    test_data = "1 10%\n2 15%\n3 25%\n4 30%\n5 45%\n6 50%\n7 55%\n8 60%\n9 65%\n10 70%"
    result = connector.control_script('correlation-finder', 'execute', {'input_data': test_data})
    
    if result.get('success'):
        print(f"   {result['stdout'].strip()}")
        print(f"   Variables after execution: {result.get('variables', {})}")
    print()
    
    # 3. Test shared memory
    print("3. Shared Memory Test:")
    shared_data = connector.shared_memory.get('data', {})
    if 'correlation_results' in shared_data:
        corr_results = shared_data['correlation_results']
        print(f"   R-squared: {corr_results.get('r_squared', 'N/A'):.4f}")
        print(f"   Peak count: {corr_results.get('peak_count', 0)}")
    print()
    
    # 4. Test collaborative task
    print("4. Collaborative Task Example:")
    task = {
        'name': 'multi_step_analysis',
        'steps': [
            {
                'script': 'correlation-finder',
                'action': 'set_variable',
                'params': {'variable': 'analysis_name', 'value': 'Demo Analysis'}
            },
            {
                'script': 'correlation-finder',
                'action': 'execute',
                'params': {'input_data': '1 5%\n2 10%\n3 20%\n4 35%\n5 50%'}
            }
        ]
    }
    
    task_result = connector.execute_collaborative_task(task)
    print(f"   Task completed with {len(task_result)} steps")
    print()
    
    # 5. Test intensity matcher (if image available)
    print("5. Intensity Matcher Test:")
    # Create a simple test image
    try:
        from PIL import Image
        import numpy as np
        
        # Create a test image with known intensity distribution
        test_img = np.zeros((100, 100), dtype=np.uint8)
        test_img[25:75, 25:75] = 128  # 25% of pixels at intensity 128
        Image.fromarray(test_img, 'L').save('demo_test.jpg')
        
        # Set shared data for intensity matcher
        connector.control_script('intensity-matcher', 'set_variable', 
                               {'variable': 'target_value', 'value': 128})
        
        # Execute intensity matcher
        result = connector.control_script('intensity-matcher', 'execute', 
                                        {'input_data': None})  # Will use demo_test.jpg
        
        if result.get('success'):
            print("   Intensity matcher executed successfully")
            
        # Cleanup
        import os
        if os.path.exists('demo_test.jpg'):
            os.remove('demo_test.jpg')
            
    except Exception as e:
        print(f"   Could not test intensity matcher: {e}")
    
    print("\nDemo completed successfully!")
    
    # Show how to use with hivemind connector
    print("\n" + "=" * 50)
    print("To use with Hivemind Connector:")
    print("1. Run in terminal 1: python hivemind_connector.py")
    print("2. Run in terminal 2: python demo_hivemind_client.py")
    print("=" * 50)

if __name__ == "__main__":
    main()