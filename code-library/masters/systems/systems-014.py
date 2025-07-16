#!/usr/bin/env python3
"""
Main Neural Network Integration System
Orchestrates all components into a unified neural network
"""

import os
import sys
import time
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import signal
import atexit

# Add core modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logger import logger, LogChannel, LogLevel
from core.dependency_manager import DependencyManager
from core.config_manager import config_manager, ConfigLevel
from core.node_base import BaseNode, NodeState, AtomicNode, CompositeNode, NodeMetadata, NodeInput, NodeOutput
from core.synapse import SynchronousSynapse, AsynchronousSynapse, StreamingSynapse, BroadcastSynapse, SynapticRouter
from core.module_analyzer import ModuleAnalyzer, NodeFactory
from core.parameter_tuner import ParameterTuner, TuningStrategy, create_performance_objective


class NeuralNetwork:
    """Main neural network orchestration system"""
    
    def __init__(self, name: str = "PolarBearNeuralNetwork"):
        """Initialize the neural network"""
        self.name = name
        self.nodes: Dict[str, BaseNode] = {}
        self.synapses: Dict[str, Any] = {}
        self.router = SynapticRouter()
        self.module_analyzer = ModuleAnalyzer()
        self.node_factory = None
        self.parameter_tuner = ParameterTuner()
        self.running = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)
        
        logger.info(LogChannel.NEURAL, f"Neural Network '{name}' initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(LogChannel.SYSTEM, f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self, module_paths: List[str]):
        """Initialize the neural network with module paths"""
        logger.info(LogChannel.NEURAL, "Initializing neural network components...")
        
        # 1. Check and install dependencies
        self._check_dependencies(module_paths)
        
        # 2. Load configuration
        if config_manager.interactive_setup(ConfigLevel.BEGINNER):
            config_manager.auto_detect_optimal_settings()
        
        # 3. Analyze modules
        self._analyze_modules(module_paths)
        
        # 4. Create nodes
        self._create_nodes()
        
        # 5. Establish connections
        self._establish_connections()
        
        # 6. Initialize all nodes
        self._initialize_nodes()
        
        # 7. Register nodes with parameter tuner
        self._register_tunable_nodes()
        
        logger.success(LogChannel.NEURAL, "Neural network initialization complete",
                      nodes=len(self.nodes), connections=len(self.synapses))
    
    def _check_dependencies(self, module_paths: List[str]):
        """Check and install required dependencies"""
        logger.info(LogChannel.SYSTEM, "Checking dependencies...")
        
        dm = DependencyManager()
        
        # Scan all module paths
        all_imports = set()
        for path in module_paths:
            if os.path.isdir(path):
                imports = dm.scan_directory(path)
                all_imports.update(imports)
        
        # Auto-install missing dependencies
        if all_imports:
            logger.info(LogChannel.SYSTEM, f"Found {len(all_imports)} unique imports")
            results = dm.auto_install_missing(os.path.dirname(module_paths[0]))
            
            # Create requirements file
            dm.create_requirements_file("requirements_neural.txt")
    
    def _analyze_modules(self, module_paths: List[str]):
        """Analyze all modules to understand their structure"""
        logger.info(LogChannel.MODULE, "Analyzing modules...")
        
        total_modules = 0
        for path in module_paths:
            if os.path.isdir(path):
                modules = self.module_analyzer.analyze_directory(path, recursive=True)
                total_modules += len(modules)
            elif os.path.isfile(path) and path.endswith('.py'):
                self.module_analyzer.analyze_file(path)
                total_modules += 1
        
        logger.info(LogChannel.MODULE, f"Analyzed {total_modules} modules",
                   functions=len(self.module_analyzer.functions),
                   classes=len(self.module_analyzer.classes))
        
        # Create node factory
        self.node_factory = NodeFactory(self.module_analyzer)
    
    def _create_nodes(self):
        """Create neural network nodes from analyzed modules"""
        logger.info(LogChannel.MODULE, "Creating neural network nodes...")
        
        # Create function nodes
        func_nodes = self.node_factory.create_all_nodes()
        for name, node in func_nodes.items():
            self.add_node(node, name)
        
        # Create module composite nodes
        module_nodes = self.node_factory.create_module_nodes()
        for name, node in module_nodes.items():
            self.add_node(node, f"module_{name}")
        
        # Create special nodes for specific functionality
        self._create_special_nodes()
    
    def _create_special_nodes(self):
        """Create special-purpose nodes"""
        # Image processing pipeline node
        image_pipeline = CompositeNode(NodeMetadata(
            name="image_processing_pipeline",
            description="Complete image processing pipeline"
        ))
        
        # Add relevant nodes to pipeline
        image_nodes = [
            name for name in self.nodes 
            if any(keyword in name.lower() for keyword in ['image', 'cv', 'pixel', 'intensity'])
        ]
        
        for i, node_name in enumerate(image_nodes[:5]):  # Limit to 5 for demo
            if node_name in self.nodes:
                image_pipeline.add_node(self.nodes[node_name], f"stage_{i}")
        
        if image_pipeline.sub_nodes:
            self.add_node(image_pipeline, "image_pipeline")
        
        # Anomaly detection ensemble node
        anomaly_ensemble = CompositeNode(NodeMetadata(
            name="anomaly_detection_ensemble",
            description="Ensemble of anomaly detection methods"
        ))
        
        anomaly_nodes = [
            name for name in self.nodes 
            if 'anomaly' in name.lower() or 'outlier' in name.lower()
        ]
        
        for node_name in anomaly_nodes:
            if node_name in self.nodes:
                anomaly_ensemble.add_node(self.nodes[node_name], node_name)
        
        if anomaly_ensemble.sub_nodes:
            self.add_node(anomaly_ensemble, "anomaly_ensemble")
    
    def _establish_connections(self):
        """Establish synaptic connections between nodes"""
        logger.info(LogChannel.SYNAPSE, "Establishing synaptic connections...")
        
        # Auto-connect based on dependencies
        self.node_factory.connect_nodes_by_dependencies()
        
        # Create broadcast synapse for system events
        if "image_pipeline" in self.nodes:
            broadcast = BroadcastSynapse(self.nodes["image_pipeline"])
            
            # Connect to all anomaly detectors
            anomaly_nodes = [
                self.nodes[name] for name in self.nodes 
                if 'anomaly' in name.lower()
            ]
            
            if anomaly_nodes:
                broadcast.connect(anomaly_nodes)
                self.synapses["image_to_anomaly_broadcast"] = broadcast
        
        # Create streaming connections for real-time processing
        realtime_nodes = [
            name for name in self.nodes 
            if 'realtime' in name.lower() or 'stream' in name.lower()
        ]
        
        for node_name in realtime_nodes:
            if node_name in self.nodes:
                stream = StreamingSynapse(self.nodes[node_name])
                self.synapses[f"{node_name}_stream"] = stream
        
        logger.info(LogChannel.SYNAPSE, f"Created {len(self.synapses)} synaptic connections")
    
    def _initialize_nodes(self):
        """Initialize all registered nodes"""
        logger.info(LogChannel.MODULE, "Initializing all nodes...")
        
        initialized = 0
        failed = 0
        
        for name, node in self.nodes.items():
            try:
                if node.initialize():
                    initialized += 1
                else:
                    failed += 1
                    logger.warning(LogChannel.MODULE, f"Failed to initialize node: {name}")
            except Exception as e:
                failed += 1
                logger.error(LogChannel.MODULE, f"Error initializing node {name}: {str(e)}")
        
        logger.info(LogChannel.MODULE, f"Node initialization complete",
                   initialized=initialized, failed=failed)
    
    def _register_tunable_nodes(self):
        """Register nodes with tunable parameters"""
        logger.info(LogChannel.NEURAL, "Registering tunable parameters...")
        
        registered = 0
        for node in self.nodes.values():
            if node.tunable_params:
                self.parameter_tuner.register_node(node)
                registered += 1
        
        logger.info(LogChannel.NEURAL, f"Registered {registered} nodes with tunable parameters")
    
    def add_node(self, node: BaseNode, name: Optional[str] = None):
        """Add a node to the network"""
        with self._lock:
            node_name = name or node.metadata.name
            self.nodes[node_name] = node
            logger.debug(LogChannel.MODULE, f"Added node: {node_name}")
    
    def remove_node(self, name: str):
        """Remove a node from the network"""
        with self._lock:
            if name in self.nodes:
                node = self.nodes[name]
                node.shutdown()
                del self.nodes[name]
                logger.info(LogChannel.MODULE, f"Removed node: {name}")
    
    def get_node(self, name: str) -> Optional[BaseNode]:
        """Get a node by name"""
        return self.nodes.get(name)
    
    def process(self, data: Any, node_name: str, **kwargs) -> Optional[NodeOutput]:
        """Process data through a specific node"""
        node = self.get_node(node_name)
        if not node:
            logger.error(LogChannel.NEURAL, f"Node not found: {node_name}")
            return None
        
        input_data = NodeInput(data=data, metadata=kwargs)
        return node.execute(input_data)
    
    async def process_async(self, data: Any, node_name: str, **kwargs) -> Optional[NodeOutput]:
        """Process data asynchronously"""
        node = self.get_node(node_name)
        if not node:
            logger.error(LogChannel.NEURAL, f"Node not found: {node_name}")
            return None
        
        input_data = NodeInput(data=data, metadata=kwargs)
        return await node.execute_async(input_data)
    
    def broadcast(self, data: Any, synapse_name: str, **kwargs) -> int:
        """Broadcast data through a broadcast synapse"""
        if synapse_name not in self.synapses:
            logger.error(LogChannel.SYNAPSE, f"Synapse not found: {synapse_name}")
            return 0
        
        synapse = self.synapses[synapse_name]
        if hasattr(synapse, 'send'):
            return synapse.send(data, **kwargs)
        return 0
    
    def tune_parameters(self, strategy: TuningStrategy = TuningStrategy.ADAPTIVE,
                       max_iterations: int = 100, test_data: Any = None):
        """Run parameter tuning on the network"""
        logger.info(LogChannel.NEURAL, "Starting parameter tuning...")
        
        # Add performance objective
        self.parameter_tuner.add_objective(create_performance_objective(test_data))
        
        # Run tuning
        best_result = self.parameter_tuner.tune(strategy, max_iterations)
        
        # Apply best parameters
        self.parameter_tuner.apply_best_parameters()
        
        # Save results
        self.parameter_tuner.save_results("neural_tuning_results.json")
        
        return best_result
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = {
            'name': self.name,
            'nodes': {
                'total': len(self.nodes),
                'by_state': {},
                'by_type': {}
            },
            'connections': len(self.synapses),
            'performance': {
                'total_calls': 0,
                'total_errors': 0,
                'average_latency': 0
            }
        }
        
        # Aggregate node statistics
        total_latency = 0
        latency_count = 0
        
        for node in self.nodes.values():
            # Count by state
            state = node.state.value
            stats['nodes']['by_state'][state] = stats['nodes']['by_state'].get(state, 0) + 1
            
            # Count by type
            node_type = node.metadata.type.value
            stats['nodes']['by_type'][node_type] = stats['nodes']['by_type'].get(node_type, 0) + 1
            
            # Aggregate performance
            metrics = node.get_metrics()
            stats['performance']['total_calls'] += metrics['metrics']['total_calls']
            stats['performance']['total_errors'] += metrics['metrics']['failed_calls']
            
            if metrics['metrics']['average_processing_time'] > 0:
                total_latency += metrics['metrics']['average_processing_time']
                latency_count += 1
        
        if latency_count > 0:
            stats['performance']['average_latency'] = total_latency / latency_count
        
        return stats
    
    def save_state(self, filepath: str):
        """Save network state to file"""
        state = {
            'name': self.name,
            'timestamp': time.time(),
            'nodes': {
                name: {
                    'metadata': node.metadata.to_dict(),
                    'state': node.state.value,
                    'parameters': node.parameters,
                    'metrics': node.get_metrics()
                }
                for name, node in self.nodes.items()
            },
            'configuration': config_manager.values,
            'network_stats': self.get_network_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(LogChannel.NEURAL, f"Network state saved to {filepath}")
    
    def visualize_network(self, output_file: str = "network_graph.json"):
        """Generate visualization data for the network"""
        graph = self.module_analyzer.generate_node_graph()
        
        # Add runtime information
        for node_data in graph['nodes']:
            node_id = node_data['id']
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node_data['state'] = node.state.value
                node_data['metrics'] = node.get_metrics()
        
        with open(output_file, 'w') as f:
            json.dump(graph, f, indent=2)
        
        logger.info(LogChannel.NEURAL, f"Network visualization saved to {output_file}")
    
    def run(self):
        """Run the neural network in interactive mode"""
        self.running = True
        
        print("\n🧠 Neural Network Running 🧠")
        print("=" * 50)
        print("Commands:")
        print("  stats    - Show network statistics")
        print("  nodes    - List all nodes")
        print("  process  - Process data through a node")
        print("  tune     - Run parameter tuning")
        print("  save     - Save network state")
        print("  quit     - Shutdown network")
        print()
        
        while self.running and not self._shutdown_event.is_set():
            try:
                command = input("neural> ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "stats":
                    stats = self.get_network_stats()
                    print(json.dumps(stats, indent=2))
                elif command == "nodes":
                    for name, node in sorted(self.nodes.items()):
                        print(f"  {name}: {node.state.value}")
                elif command == "process":
                    node_name = input("Node name: ").strip()
                    data = input("Data (JSON): ").strip()
                    try:
                        data = json.loads(data)
                        result = self.process(data, node_name)
                        if result:
                            print(f"Result: {result.data}")
                            print(f"Success: {result.success}")
                            print(f"Processing time: {result.processing_time:.4f}s")
                        else:
                            print("Processing failed")
                    except json.JSONDecodeError:
                        print("Invalid JSON data")
                elif command == "tune":
                    print("Starting parameter tuning...")
                    self.tune_parameters(max_iterations=10)
                    print("Tuning complete!")
                elif command == "save":
                    self.save_state("network_state.json")
                    print("State saved!")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\nUse 'quit' to shutdown properly")
            except Exception as e:
                logger.error(LogChannel.NEURAL, f"Command error: {str(e)}")
    
    def shutdown(self):
        """Shutdown the neural network gracefully"""
        if self._shutdown_event.is_set():
            return
        
        self._shutdown_event.set()
        self.running = False
        
        logger.info(LogChannel.NEURAL, "Shutting down neural network...")
        
        # Shutdown all nodes
        with self._lock:
            for name, node in self.nodes.items():
                try:
                    node.shutdown()
                except Exception as e:
                    logger.error(LogChannel.MODULE, f"Error shutting down node {name}: {str(e)}")
        
        # Close all synapses
        for name, synapse in self.synapses.items():
            try:
                if hasattr(synapse, 'disconnect'):
                    synapse.disconnect()
            except Exception as e:
                logger.error(LogChannel.SYNAPSE, f"Error closing synapse {name}: {str(e)}")
        
        # Shutdown parameter tuner
        self.parameter_tuner.stop()
        
        # Save final state
        try:
            self.save_state("network_final_state.json")
        except:
            pass
        
        logger.info(LogChannel.NEURAL, "Neural network shutdown complete")


def main():
    """Main entry point"""
    print("\n🚀 Polar Bear Neural Network Integration System 🚀")
    print("=" * 60)
    
    # Define module paths to analyze
    module_paths = [
        "/home/jarvis/Documents/GitHub/polar-bear/modules/iteration6-lab-framework",
        "/home/jarvis/Documents/GitHub/polar-bear/modules/iteration4-modular-start",
        "/home/jarvis/Documents/GitHub/polar-bear/modules/experimental-features",
        "/home/jarvis/Documents/GitHub/polar-bear/modules/artificial-intelligence",
        "/home/jarvis/Documents/GitHub/polar-bear/modules/computer-vision",
        "/home/jarvis/Documents/GitHub/polar-bear/modules/real-time-monitoring"
    ]
    
    # Filter to existing paths
    existing_paths = [p for p in module_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("❌ No module paths found!")
        return
    
    print(f"\nFound {len(existing_paths)} module directories")
    
    # Create neural network
    network = NeuralNetwork()
    
    try:
        # Initialize network
        network.initialize(existing_paths)
        
        # Show stats
        stats = network.get_network_stats()
        print(f"\n📊 Network Statistics:")
        print(f"  Total Nodes: {stats['nodes']['total']}")
        print(f"  Connections: {stats['connections']}")
        print(f"  Node Types: {stats['nodes']['by_type']}")
        
        # Save visualization
        network.visualize_network()
        
        # Run interactive mode
        network.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        logger.error(LogChannel.NEURAL, f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        network.shutdown()
        print("\n✅ Neural network terminated")


if __name__ == "__main__":
    main()