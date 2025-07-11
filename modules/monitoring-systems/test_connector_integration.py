#!/usr/bin/env python3
"""
Test script for connector integration
Demonstrates bidirectional control and monitoring
"""

import time
import json
import socket
import logging
import random
from connector_interface import ConnectorInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize connector
connector = ConnectorInterface('test_connector_integration.py')

# Register controllable parameters
connector.register_parameter(
    'processing_rate',
    10,
    'int',
    'Processing rate (items/second)',
    min_value=1,
    max_value=100,
    callback=lambda old, new: logger.info(f"Processing rate changed from {old} to {new}")
)

connector.register_parameter(
    'threshold',
    0.5,
    'float', 
    'Detection threshold',
    min_value=0.0,
    max_value=1.0
)

connector.register_parameter(
    'mode',
    'normal',
    'str',
    'Operating mode',
    choices=['normal', 'aggressive', 'conservative']
)

connector.register_parameter(
    'enabled',
    True,
    'bool',
    'Enable/disable processing'
)

def simulate_processing():
    """Simulate data processing with metrics"""
    logger.info("Starting simulated processing...")
    connector.set_status('running')
    
    items_processed = 0
    errors = 0
    
    try:
        while True:
            # Check if enabled
            if not connector.get_parameter('enabled'):
                connector.set_status('paused')
                time.sleep(1)
                continue
                
            connector.set_status('running')
            
            # Get current parameters
            rate = connector.get_parameter('processing_rate')
            threshold = connector.get_parameter('threshold')
            mode = connector.get_parameter('mode')
            
            # Simulate processing
            for _ in range(rate):
                # Simulate work
                value = random.random()
                
                if value > threshold:
                    items_processed += 1
                    
                    # Occasionally simulate an error
                    if random.random() < 0.01:
                        errors += 1
                        
            # Update metrics
            connector.update_metric('items_processed', items_processed)
            connector.update_metric('errors', errors)
            connector.update_metric('error_rate', errors / max(items_processed, 1))
            connector.update_metric('current_threshold', threshold)
            connector.update_metric('operating_mode', mode)
            
            # Log status
            logger.info(f"Processed: {items_processed}, Errors: {errors}, Mode: {mode}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted")
        connector.set_status('stopped')
    except Exception as e:
        logger.error(f"Processing error: {e}")
        connector.set_status('error', str(e))
    finally:
        connector.cleanup()


def test_connector_control():
    """Test controlling the script through connector"""
    logger.info("Testing connector control...")
    
    try:
        # Connect to connector
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 10130))
        
        # Test 1: Get script state
        logger.info("Test 1: Getting script state")
        command = {
            'command': 'get_script_state',
            'script': 'test_connector_integration.py'
        }
        sock.send(json.dumps(command).encode())
        response = sock.recv(4096)
        logger.info(f"State response: {response.decode()}")
        
        # Test 2: Set parameter
        logger.info("Test 2: Setting parameter")
        command = {
            'command': 'set_parameter',
            'script': 'test_connector_integration.py',
            'parameter': 'processing_rate',
            'value': 50
        }
        sock.send(json.dumps(command).encode())
        response = sock.recv(4096)
        logger.info(f"Set parameter response: {response.decode()}")
        
        # Test 3: Pause script
        logger.info("Test 3: Pausing script")
        command = {
            'command': 'control_script',
            'script': 'test_connector_integration.py',
            'action': 'pause'
        }
        sock.send(json.dumps(command).encode())
        response = sock.recv(4096)
        logger.info(f"Pause response: {response.decode()}")
        
        sock.close()
        logger.info("Connector control tests completed")
        
    except Exception as e:
        logger.error(f"Control test failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--control-test':
        # Run control test
        test_connector_control()
    else:
        # Run processing simulation
        simulate_processing()