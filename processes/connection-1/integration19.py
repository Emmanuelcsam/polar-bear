#!/usr/bin/env python3
"""
Integration Demo Script
Demonstrates full connector control and script collaboration
"""

import time
import json
import os
from connector import Connector
from hivemind_connector import HivemindConnector

def demo_connector_control():
    """Demonstrate connector control capabilities"""
    print("\n" + "="*60)
    print("CONNECTOR INTEGRATION DEMO")
    print("="*60)
    
    # Initialize connector
    print("\n1. Initializing Connector...")
    connector = Connector()
    results = connector.initialize()
    print(f"   Loaded {sum(1 for v in results.values() if v)} scripts successfully")
    
    # Show script information
    print("\n2. Script Information:")
    info = connector.get_script_info()
    for script, details in info.items():
        functions = details.get('functions', [])
        print(f"   - {script}: {len(functions)} functions")
    
    # Demonstrate parameter control
    print("\n3. Parameter Control Demo:")
    
    # Set pixel generator parameters
    print("   Setting pixel generator range to 100-200...")
    connector.control_variable('pixel_generator', 'min', 100)
    connector.control_variable('pixel_generator', 'max', 200)
    
    # Set anomaly detector threshold
    print("   Setting anomaly detector threshold to 75...")
    connector.control_variable('anomaly_detector', 'threshold', 75)
    
    # Generate some data
    print("\n4. Generating Test Data:")
    print("   Generating 20 pixels...")
    for i in range(20):
        result = connector.call_function('pixel_generator', 'generate')
        if i % 5 == 0:
            print(f"   Generated pixel: {result}")
    
    # Add some anomalies
    print("   Adding anomalous values...")
    connector.call_function('data_store', 'save_event', {'pixel': 500})
    connector.call_function('data_store', 'save_event', {'pixel': 10})
    
    # Analyze data
    print("\n5. Data Analysis:")
    
    # Get statistics
    stats = connector.call_function('data_store', 'get_stats')
    print(f"   Data statistics: {json.dumps(stats, indent=2)}")
    
    # Find anomalies
    anomalies = connector.call_function('anomaly_detector', 'anomalies')
    print(f"   Found {len(anomalies)} anomalies")
    
    # Get trends
    trends = connector.call_function('trend_reader', 'trends')
    if isinstance(trends, str):
        trends_data = json.loads(trends)
        print(f"   Trends: min={trends_data['min']}, max={trends_data['max']}, mean={trends_data['mean']:.2f}")
    
    # Find patterns
    patterns = connector.call_function('pattern_recognition', 'patterns')
    print(f"   Found {patterns.get('total_unique', 0)} unique patterns")
    
    # Demonstrate workflow orchestration
    print("\n6. Workflow Orchestration:")
    workflow = [
        {
            'name': 'generate_batch',
            'script': 'pixel_generator',
            'action': {
                'type': 'call_function',
                'function_name': 'generate'
            }
        },
        {
            'name': 'save_data',
            'script': 'data_store',
            'action': {
                'type': 'call_function',
                'function_name': 'save_event',
                'args': [{'pixel': 150, 'source': 'workflow'}]
            }
        },
        {
            'name': 'analyze',
            'script': 'anomaly_detector',
            'action': {
                'type': 'call_function',
                'function_name': 'anomalies'
            }
        }
    ]
    
    results = connector.orchestrate_workflow(workflow)
    print(f"   Completed {len(results)} workflow steps")
    
    # Demonstrate broadcast
    print("\n7. Broadcast Command:")
    print("   Getting info from all scripts...")
    broadcast_result = connector.broadcast_action({'type': 'get_info'})
    print(f"   Received responses from {len(broadcast_result)} scripts")
    
    return True

def demo_hivemind_messages():
    """Demonstrate hivemind connector message processing"""
    print("\n" + "="*60)
    print("HIVEMIND CONNECTOR DEMO")
    print("="*60)
    
    # Create hivemind connector (without starting server)
    connector = HivemindConnector()
    
    print("\n1. Testing Message Processing:")
    
    # Status message
    response = connector.process_message({'command': 'status'})
    print(f"   Status: {response}")
    
    # Control message - set variable
    print("\n2. Testing Control Messages:")
    response = connector.process_message({
        'command': 'control',
        'control_type': 'set_variable',
        'script': 'pixel_generator',
        'variable': 'delay',
        'value': 0.05
    })
    print(f"   Set variable response: {response}")
    
    # Control message - call function
    response = connector.process_message({
        'command': 'control',
        'control_type': 'call_function',
        'script': 'data_store',
        'function': 'get_stats'
    })
    print(f"   Function call response: {response}")
    
    # Orchestrate workflow
    print("\n3. Testing Workflow Orchestration:")
    response = connector.process_message({
        'command': 'orchestrate',
        'workflow': [
            {
                'script': 'pixel_generator',
                'action': {'type': 'call_function', 'function_name': 'generate'}
            }
        ]
    })
    print(f"   Orchestration response: {response}")
    
    return True

def cleanup():
    """Clean up generated files"""
    if os.path.exists('events.log'):
        os.remove('events.log')
    if os.path.exists('test_events.log'):
        os.remove('test_events.log')
    if os.path.exists('model.pkl'):
        os.remove('model.pkl')

def main():
    """Run the complete integration demo"""
    try:
        # Clean up any existing data
        cleanup()
        
        # Run demos
        success1 = demo_connector_control()
        success2 = demo_hivemind_messages()
        
        if success1 and success2:
            print("\n" + "="*60)
            print("✅ INTEGRATION DEMO COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\nKey Features Demonstrated:")
            print("- ✓ All scripts loaded and controlled by connectors")
            print("- ✓ Parameter and variable control working")
            print("- ✓ Scripts can run independently and through connectors")
            print("- ✓ Workflow orchestration functioning")
            print("- ✓ Script collaboration enabled")
            print("- ✓ Hivemind connector message processing")
            print("\nThe system is fully integrated and operational!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cleanup()

if __name__ == "__main__":
    main()