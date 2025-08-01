#!/usr/bin/env python3
"""
Collaboration Manager for Polar Bear System
Facilitates inter-script communication and coordinated processing
"""

import socket
import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
from queue import Queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CollaborationManager')

# Shared state for collaboration
COLLABORATION_STATE = {
    'active_pipelines': {},
    'task_queue': Queue(),
    'results_cache': {},
    'script_capabilities': {},
    'processing_stats': {
        'total_tasks': 0,
        'completed_tasks': 0,
        'failed_tasks': 0,
        'average_processing_time': 0
    }
}

class CollaborationManager:
    """Manages collaboration between different scripts through the connector"""
    
    def __init__(self, connector_host='localhost', connector_port=10001):
        self.connector_host = connector_host
        self.connector_port = connector_port
        self.running = False
        self.worker_threads = []
        
    def send_connector_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the hivemind connector"""
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10)
            client.connect((self.connector_host, self.connector_port))
            
            client.send(json.dumps(command).encode('utf-8'))
            response = json.loads(client.recv(8192).decode('utf-8'))
            client.close()
            
            return response
        except Exception as e:
            logger.error(f"Error sending command to connector: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def discover_capabilities(self) -> Dict[str, Any]:
        """Discover capabilities of all available scripts"""
        response = self.send_connector_command({'command': 'get_scripts'})
        
        if response.get('status') == 'ok':
            scripts = response.get('scripts', {})
            COLLABORATION_STATE['script_capabilities'] = scripts
            
            logger.info(f"Discovered {len(scripts)} scripts with capabilities")
            for script_name, info in scripts.items():
                logger.info(f"  - {script_name}: {len(info.get('functions', []))} functions, "
                          f"{len(info.get('parameters', []))} parameters")
            
            return scripts
        return {}
    
    def coordinate_image_processing(self, image_path: str) -> Dict[str, Any]:
        """Coordinate image processing across multiple scripts"""
        logger.info(f"Starting coordinated processing of {image_path}")
        
        results = {
            'image_path': image_path,
            'timestamp': time.time(),
            'stages': {}
        }
        
        # Stage 1: Initialize speed-test pipeline if needed
        logger.info("Stage 1: Initializing processing pipeline")
        init_response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'initialize_pipeline',
            'args': [],
            'kwargs': {}
        })
        
        if init_response.get('status') != 'ok':
            logger.error(f"Failed to initialize pipeline: {init_response}")
            results['stages']['initialization'] = {'status': 'failed', 'error': init_response}
            return results
        
        results['stages']['initialization'] = {'status': 'success'}
        
        # Stage 2: Process image with speed-test
        logger.info("Stage 2: Processing image with speed-test pipeline")
        process_response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'process_single_image',
            'args': [image_path],
            'kwargs': {}
        })
        
        if process_response.get('status') == 'ok':
            results['stages']['processing'] = {
                'status': 'success',
                'result': process_response.get('result')
            }
        else:
            results['stages']['processing'] = {
                'status': 'failed',
                'error': process_response
            }
        
        # Stage 3: Get processing status
        logger.info("Stage 3: Getting pipeline status")
        status_response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'get_pipeline_status',
            'args': [],
            'kwargs': {}
        })
        
        if status_response.get('status') == 'ok':
            results['stages']['status'] = {
                'status': 'success',
                'pipeline_status': status_response.get('result')
            }
        
        # Update statistics
        COLLABORATION_STATE['processing_stats']['total_tasks'] += 1
        COLLABORATION_STATE['processing_stats']['completed_tasks'] += 1
        
        # Cache results
        COLLABORATION_STATE['results_cache'][image_path] = results
        
        logger.info(f"Completed coordinated processing of {image_path}")
        return results
    
    def batch_coordinate_processing(self, image_directory: str, max_images: int = None) -> Dict[str, Any]:
        """Coordinate batch processing of multiple images"""
        image_dir = Path(image_directory)
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        if max_images:
            image_files = image_files[:max_images]
        
        logger.info(f"Starting batch coordination for {len(image_files)} images")
        
        batch_results = {
            'directory': image_directory,
            'total_images': len(image_files),
            'results': []
        }
        
        for img_path in image_files:
            result = self.coordinate_image_processing(str(img_path))
            batch_results['results'].append(result)
        
        return batch_results
    
    def update_shared_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """Update configuration across all scripts"""
        logger.info(f"Updating shared configuration: {config_updates}")
        
        # Update speed-test configuration
        response = self.send_connector_command({
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'update_configuration',
            'args': [config_updates],
            'kwargs': {}
        })
        
        if response.get('status') == 'ok':
            logger.info("Successfully updated configuration")
            return True
        else:
            logger.error(f"Failed to update configuration: {response}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all components"""
        health = {
            'timestamp': time.time(),
            'components': {}
        }
        
        # Check connector status
        connector_status = self.send_connector_command({'command': 'status'})
        health['components']['hivemind_connector'] = {
            'status': connector_status.get('status', 'unknown'),
            'details': connector_status
        }
        
        # Check speed-test status
        pipeline_status = self.send_connector_command({
            'command': 'execute_function',
            'script': 'speed-test',
            'function': 'get_pipeline_status',
            'args': [],
            'kwargs': {}
        })
        health['components']['speed_test_pipeline'] = {
            'status': 'ok' if pipeline_status.get('status') == 'ok' else 'error',
            'details': pipeline_status.get('result', {})
        }
        
        # Add processing statistics
        health['processing_stats'] = COLLABORATION_STATE['processing_stats']
        
        return health
    
    def start_task_worker(self):
        """Start a worker thread to process tasks from the queue"""
        def worker():
            while self.running:
                try:
                    task = COLLABORATION_STATE['task_queue'].get(timeout=1)
                    logger.info(f"Processing task: {task['type']}")
                    
                    if task['type'] == 'process_image':
                        self.coordinate_image_processing(task['image_path'])
                    elif task['type'] == 'update_config':
                        self.update_shared_configuration(task['config'])
                    
                    COLLABORATION_STATE['task_queue'].task_done()
                except:
                    continue
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.worker_threads.append(thread)
    
    def add_task(self, task_type: str, **kwargs) -> bool:
        """Add a task to the processing queue"""
        task = {'type': task_type, 'timestamp': time.time()}
        task.update(kwargs)
        
        COLLABORATION_STATE['task_queue'].put(task)
        logger.info(f"Added task to queue: {task_type}")
        return True
    
    def start(self, num_workers: int = 2):
        """Start the collaboration manager"""
        self.running = True
        
        # Discover capabilities
        self.discover_capabilities()
        
        # Start worker threads
        for i in range(num_workers):
            self.start_task_worker()
        
        logger.info(f"Collaboration Manager started with {num_workers} workers")
    
    def stop(self):
        """Stop the collaboration manager"""
        self.running = False
        
        # Wait for workers to finish
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        logger.info("Collaboration Manager stopped")

# Exposed functions for connector integration
def initialize_collaboration_manager():
    """Initialize the collaboration manager"""
    global collaboration_manager
    collaboration_manager = CollaborationManager()
    collaboration_manager.start()
    return True

def coordinate_processing(image_path: str) -> Dict[str, Any]:
    """Coordinate processing of a single image"""
    if 'collaboration_manager' not in globals():
        initialize_collaboration_manager()
    
    return collaboration_manager.coordinate_image_processing(image_path)

def batch_coordinate(directory: str, max_images: Optional[int] = None) -> Dict[str, Any]:
    """Coordinate batch processing"""
    if 'collaboration_manager' not in globals():
        initialize_collaboration_manager()
    
    return collaboration_manager.batch_coordinate_processing(directory, max_images)

def update_all_configurations(config: Dict[str, Any]) -> bool:
    """Update configuration across all scripts"""
    if 'collaboration_manager' not in globals():
        initialize_collaboration_manager()
    
    return collaboration_manager.update_shared_configuration(config)

def get_collaboration_status() -> Dict[str, Any]:
    """Get status of the collaboration system"""
    return {
        'manager_initialized': 'collaboration_manager' in globals(),
        'collaboration_state': COLLABORATION_STATE,
        'system_health': collaboration_manager.get_system_health() if 'collaboration_manager' in globals() else None
    }

def queue_processing_task(image_path: str) -> bool:
    """Queue an image for processing"""
    if 'collaboration_manager' not in globals():
        initialize_collaboration_manager()
    
    return collaboration_manager.add_task('process_image', image_path=image_path)

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test mode
            manager = CollaborationManager()
            manager.discover_capabilities()
            health = manager.get_system_health()
            print(f"System Health: {json.dumps(health, indent=2)}")
        elif sys.argv[1] == "--coordinate" and len(sys.argv) > 2:
            # Coordinate processing of a single image
            manager = CollaborationManager()
            result = manager.coordinate_image_processing(sys.argv[2])
            print(f"Coordination Result: {json.dumps(result, indent=2)}")
        else:
            print("Usage: python collaboration_manager.py [--test | --coordinate <image_path>]")
    else:
        # Interactive mode
        print("Collaboration Manager - Facilitating Inter-Script Communication")
        print("\nInitializing collaboration manager...")
        
        manager = CollaborationManager()
        manager.start()
        
        print("\nCollaboration Manager is running. Available functions:")
        print("  - coordinate_processing(image_path)")
        print("  - batch_coordinate(directory)")
        print("  - update_all_configurations(config)")
        print("  - get_collaboration_status()")
        print("  - queue_processing_task(image_path)")
        
        print("\nPress Ctrl+C to exit")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            manager.stop()