#!/usr/bin/env python3
"""
Quick demo of the connector system without timeouts
"""

from connector import ConnectorSystem
import json

# Initialize
print("Initializing connector system...")
connector = ConnectorSystem()
connector.scan_scripts()
connector.enable_collaboration()

# Show loaded scripts
print(f"\nLoaded {len(connector.script_controllers)} scripts:")
for name in connector.script_controllers:
    print(f"  - {name}")

# Test correlation finder
print("\nTesting correlation-finder with sample data...")
result = connector.control_script(
    'correlation-finder', 
    'execute',
    {'input_data': '1 10%\n2 20%\n3 30%\n4 40%\n5 50%'}
)

if result.get('success'):
    print("Success! Output:")
    print(result['stdout'])
    
# Check shared memory
print("\nShared memory contents:")
shared = connector.shared_memory.get('data', {})
for key, value in shared.items():
    print(f"  {key}: {type(value).__name__}")
    
print("\nDemo complete!")