#!/usr/bin/env python3
"""
Neural Network Hivemind Web Interface
A modern web-based UI for the Neural Hivemind system
"""

import os
import sys
import json
import time
import pickle
import logging
import hashlib
import inspect
import importlib
import subprocess
import ast
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback
import re
import base64
import io

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet
eventlet.monkey_patch()

# Import the core hivemind functionality
from neural_hivemind import NeuralHivemind

class WebNeuralHivemind(NeuralHivemind):
    """Extended Neural Hivemind with web interface capabilities"""
    
    def __init__(self, socketio):
        super().__init__()
        self.socketio = socketio
        self.analysis_progress = 0
        self.total_scripts_to_analyze = 0
        self.active_connections = 0
        
    def log(self, message, level="INFO"):
        """Override log to also emit to web interface"""
        super().log(message, level)
        
        # Emit log to web interface
        if hasattr(self, 'socketio'):
            self.socketio.emit('log_update', {
                'timestamp': datetime.now().isoformat(),
                'level': level,
                'message': message
            })
            
    def emit_progress(self, task, progress, total=100):
        """Emit progress updates to web interface"""
        self.socketio.emit('progress_update', {
            'task': task,
            'progress': progress,
            'total': total,
            'percentage': (progress / total * 100) if total > 0 else 0
        })
        
    def analyze_script(self, script_path: str) -> Dict[str, Any]:
        """Analyze script with progress updates"""
        result = super().analyze_script(script_path)
        
        self.analysis_progress += 1
        self.emit_progress('script_analysis', 
                         self.analysis_progress, 
                         self.total_scripts_to_analyze)
        
        return result
        
    def get_network_graph_data(self):
        """Generate network graph data for visualization"""
        nodes = []
        edges = []
        
        # Create nodes for each script
        for script in self.scripts:
            metadata = self.script_metadata.get(script, {})
            nodes.append({
                'id': script,
                'label': Path(script).name,
                'group': Path(script).parent.name,
                'size': len(metadata.get('functions', [])) + len(metadata.get('classes', [])),
                'metadata': {
                    'functions': len(metadata.get('functions', [])),
                    'classes': len(metadata.get('classes', [])),
                    'imports': len(metadata.get('imports', [])),
                    'parameters': len(metadata.get('parameters', {}))
                }
            })
            
        # Create edges based on connections
        edge_id = 0
        for source, targets in self.script_connections.items():
            for target in targets:
                edges.append({
                    'id': edge_id,
                    'source': source,
                    'target': target,
                    'weight': 1
                })
                edge_id += 1
                
        return {'nodes': nodes, 'edges': edges}
        
    def get_performance_metrics(self):
        """Get performance metrics for dashboard"""
        metrics = {
            'total_scripts': len(self.scripts),
            'total_images': len(self.image_files),
            'total_data_files': len(self.data_files),
            'analyzed_scripts': len(self.script_metadata),
            'total_connections': sum(len(v) for v in self.script_connections.values()),
            'total_parameters': sum(len(m.get('parameters', {})) 
                                  for m in self.script_metadata.values()),
            'optimization_runs': len(self.script_combinations),
            'best_performance': max([c['score'] for c in self.script_combinations], 
                                  default=0)
        }
        
        # Get top performing combinations
        if self.script_combinations:
            top_combos = sorted(self.script_combinations, 
                              key=lambda x: x['score'], 
                              reverse=True)[:5]
            metrics['top_combinations'] = [
                {
                    'scripts': [Path(s).name for s in combo['combination']],
                    'score': combo['score']
                }
                for combo in top_combos
            ]
        else:
            metrics['top_combinations'] = []
            
        return metrics

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'neural-hivemind-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app)

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global hivemind instance
hivemind = None

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    global hivemind
    
    if hivemind is None:
        return jsonify({
            'status': 'not_initialized',
            'message': 'Hivemind not yet initialized'
        })
        
    return jsonify({
        'status': 'ready',
        'metrics': hivemind.get_performance_metrics()
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_hivemind():
    """Initialize the hivemind system"""
    global hivemind
    
    data = request.json
    focus_area = data.get('focus_area', 'fiberoptic')
    optimization = data.get('optimization', 'balanced')
    max_threads = data.get('max_threads', 10)
    
    # Create new hivemind instance
    hivemind = WebNeuralHivemind(socketio)
    hivemind.focus_area = focus_area
    hivemind.optimization_preference = optimization
    hivemind.max_threads = max_threads
    
    # Run initialization in background
    def init_task():
        try:
            hivemind.log("Starting Neural Hivemind initialization...")
            hivemind.check_and_install_dependencies()
            hivemind.crawl_directories()
            
            hivemind.total_scripts_to_analyze = len(hivemind.scripts)
            hivemind.analysis_progress = 0
            
            hivemind.analyze_all_scripts()
            hivemind.find_script_connections()
            hivemind.save_state()
            
            socketio.emit('initialization_complete', {
                'status': 'success',
                'metrics': hivemind.get_performance_metrics()
            })
            
        except Exception as e:
            hivemind.log(f"Initialization error: {str(e)}", "ERROR")
            socketio.emit('initialization_complete', {
                'status': 'error',
                'error': str(e)
            })
            
    threading.Thread(target=init_task).start()
    
    return jsonify({'status': 'initializing'})

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze an uploaded image"""
    global hivemind
    
    if hivemind is None:
        return jsonify({'error': 'Hivemind not initialized'}), 400
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save uploaded file
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Create execution plan
    task = request.form.get('task', 'defect detection')
    execution_plan = hivemind.create_execution_plan(task)
    
    # Execute analysis
    results = hivemind.execute_script_combination(execution_plan, filepath)
    
    # Clean up uploaded file
    os.remove(filepath)
    
    return jsonify({
        'execution_plan': [Path(s).name for s in execution_plan],
        'results': {
            'execution_time': results['execution_time'],
            'outputs': len(results['outputs']),
            'errors': len(results['errors'])
        }
    })

@app.route('/api/network-graph')
def get_network_graph():
    """Get network graph visualization data"""
    global hivemind
    
    if hivemind is None:
        return jsonify({'nodes': [], 'edges': []})
        
    return jsonify(hivemind.get_network_graph_data())

@app.route('/api/optimize', methods=['POST'])
def start_optimization():
    """Start network optimization"""
    global hivemind
    
    if hivemind is None:
        return jsonify({'error': 'Hivemind not initialized'}), 400
        
    data = request.json
    epochs = data.get('epochs', 5)
    
    # Run optimization in background
    def optimize_task():
        try:
            # Create synthetic training data for demo
            training_data = [f"sample_{i}.png" for i in range(10)]
            hivemind.optimize_network(training_data, epochs=epochs)
            
            socketio.emit('optimization_complete', {
                'status': 'success',
                'metrics': hivemind.get_performance_metrics()
            })
            
        except Exception as e:
            socketio.emit('optimization_complete', {
                'status': 'error',
                'error': str(e)
            })
            
    threading.Thread(target=optimize_task).start()
    
    return jsonify({'status': 'optimizing'})

@app.route('/api/scripts')
def get_scripts():
    """Get list of all scripts with metadata"""
    global hivemind
    
    if hivemind is None:
        return jsonify({'scripts': []})
        
    scripts = []
    for script_path, metadata in hivemind.script_metadata.items():
        scripts.append({
            'path': script_path,
            'name': Path(script_path).name,
            'directory': str(Path(script_path).parent),
            'functions': len(metadata.get('functions', [])),
            'classes': len(metadata.get('classes', [])),
            'parameters': len(metadata.get('parameters', {})),
            'imports': len(metadata.get('imports', [])),
            'has_error': metadata.get('error') is not None
        })
        
    return jsonify({'scripts': scripts})

@app.route('/api/script/<path:script_path>')
def get_script_details(script_path):
    """Get detailed information about a specific script"""
    global hivemind
    
    if hivemind is None:
        return jsonify({'error': 'Hivemind not initialized'}), 400
        
    script_path = '/' + script_path  # Restore leading slash
    metadata = hivemind.script_metadata.get(script_path)
    
    if not metadata:
        return jsonify({'error': 'Script not found'}), 404
        
    return jsonify({
        'path': script_path,
        'name': Path(script_path).name,
        'metadata': metadata,
        'connections': {
            'imports_from': hivemind.script_connections.get(script_path, []),
            'imported_by': [s for s, conns in hivemind.script_connections.items() 
                          if script_path in conns]
        }
    })

@app.route('/api/logs')
def get_logs():
    """Get recent log entries"""
    log_dir = Path("/home/jarvis/Documents/GitHub/polar-bear/hivemind_logs")
    
    if not log_dir.exists():
        return jsonify({'logs': []})
        
    # Get most recent log file
    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_files:
        return jsonify({'logs': []})
        
    # Read last 100 lines
    with open(log_files[0], 'r') as f:
        lines = f.readlines()[-100:]
        
    return jsonify({'logs': lines})

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    if hivemind:
        hivemind.active_connections += 1
        emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if hivemind:
        hivemind.active_connections -= 1

if __name__ == '__main__':
    print("\n=== Neural Hivemind Web Interface ===")
    print("Starting server on http://localhost:5000")
    print("Open your browser to access the interface\n")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)