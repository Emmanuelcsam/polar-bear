#!/usr/bin/env python3
"""
Simple test script to verify connector integration without external dependencies
"""

import sys
import time
import random
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from script_interface import ScriptInterface, ConnectorClient

class SimpleTestScript(ScriptInterface):
    """Simple test script that doesn't require external packages"""
    
    def __init__(self):
        super().__init__("simple_test", "Simple Connector Integration Test")
        
        # Register parameters
        self.register_parameter("iterations", 10, range(1, 101))
        self.register_parameter("delay", 1.0, [0.5, 1.0, 2.0, 5.0])
        self.register_parameter("operation", "sine", ["sine", "cosine", "random", "linear"])
        
        # Register variables
        self.register_variable("current_iteration", 0)
        self.register_variable("current_value", 0.0)
        self.register_variable("min_value", float('inf'))
        self.register_variable("max_value", float('-inf'))
        self.register_variable("completion_percentage", 0.0)
        
        # Initialize client for collaboration
        self.client = ConnectorClient(self)
        
    def run(self):
        """Main execution"""
        print(f"\n{'='*50}")
        print(f"Simple Test Script - Connector Integration")
        print(f"{'='*50}")
        
        # Register with connector if available
        if hasattr(self, 'client'):
            self.client.register_script()
            print("✓ Registered with connector")
        
        iterations = self.get_parameter("iterations")
        delay = self.get_parameter("delay")
        operation = self.get_parameter("operation")
        
        print(f"\nConfiguration:")
        print(f"- Iterations: {iterations}")
        print(f"- Delay: {delay}s")
        print(f"- Operation: {operation}")
        
        print(f"\nStarting execution...")
        
        for i in range(iterations):
            # Update iteration
            self.set_variable("current_iteration", i + 1)
            
            # Calculate value based on operation
            if operation == "sine":
                value = math.sin(i * 0.1) * 100
            elif operation == "cosine":
                value = math.cos(i * 0.1) * 100
            elif operation == "random":
                value = random.uniform(-100, 100)
            else:  # linear
                value = i * 10
            
            # Update variables
            self.set_variable("current_value", value)
            
            # Track min/max
            current_min = self.get_variable("min_value")
            current_max = self.get_variable("max_value")
            
            if value < current_min:
                self.set_variable("min_value", value)
            if value > current_max:
                self.set_variable("max_value", value)
            
            # Update completion percentage
            completion = ((i + 1) / iterations) * 100
            self.set_variable("completion_percentage", completion)
            
            # Print progress
            print(f"Iteration {i+1}/{iterations}: value={value:.2f}, completion={completion:.1f}%")
            
            # Update results
            self.update_results("latest_value", value)
            self.update_results("iterations_completed", i + 1)
            
            # Broadcast progress if connected
            if hasattr(self, 'client') and i % 5 == 0:
                self.client.broadcast_data({
                    "type": "progress_update",
                    "iteration": i + 1,
                    "value": value,
                    "completion": completion
                })
            
            # Check if parameters changed
            if self.get_parameter("operation") != operation:
                print(f"Operation changed to: {self.get_parameter('operation')}")
                operation = self.get_parameter("operation")
            
            # Sleep
            time.sleep(self.get_parameter("delay"))
        
        # Final results
        print(f"\n{'='*50}")
        print("Execution Complete!")
        print(f"{'='*50}")
        print(f"Total iterations: {iterations}")
        print(f"Min value: {self.get_variable('min_value'):.2f}")
        print(f"Max value: {self.get_variable('max_value'):.2f}")
        
        # Update final results
        self.update_results("final_stats", {
            "total_iterations": iterations,
            "min_value": self.get_variable("min_value"),
            "max_value": self.get_variable("max_value"),
            "final_value": self.get_variable("current_value")
        })
        
        # Broadcast completion
        if hasattr(self, 'client'):
            self.client.broadcast_data({
                "type": "execution_complete",
                "results": self.results
            })

def main():
    """Main function"""
    script = SimpleTestScript()
    
    if "--with-connector" in sys.argv:
        print("Running with connector integration")
        script.run_with_connector()
    else:
        print("Running in standalone mode")
        script.run()

if __name__ == "__main__":
    main()