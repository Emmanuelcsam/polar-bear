#!/usr/bin/env python3
"""
Neural Script IDE - Production Version
A sophisticated IDE for building neural networks from interconnected Python scripts
Features:
- Advanced inter-script communication with message passing
- Real-time dependency visualization
- Advanced debugging and profiling
- Automatic logic and data flow checking
- Script orchestration and scheduling
- Performance monitoring
- Visual script relationship mapping
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import queue
import platform
import os
import sys
import json
import re
import ast
import tempfile
import time
from pathlib import Path
from datetime import datetime
import pickle
import socket
import multiprocessing
from multiprocessing import Queue, Process, Manager
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import psutil
import inspect
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import websockets
import yaml
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np

# Try to import optional dependencies
try:
    import pylint.lint
    import pylint.reporters.text
    from io import StringIO
    HAS_PYLINT = True
except ImportError:
    HAS_PYLINT = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of inter-script messages"""
    DATA = "data"
    CONTROL = "control"
    STATUS = "status"
    ERROR = "error"
    LOG = "log"
    METRIC = "metric"
    HEARTBEAT = "heartbeat"

@dataclass
class ScriptMessage:
    """Message structure for inter-script communication"""
    sender_id: str
    receiver_id: str
    message_type: MessageType
    payload: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self):
        return {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            message_type=MessageType(data['message_type']),
            payload=data['payload'],
            timestamp=data.get('timestamp', time.time())
        )

class ScriptNode:
    """Represents a script in the neural network"""
    def __init__(self, script_id: str, file_path: str = None):
        self.id = script_id
        self.file_path = file_path
        self.dependencies = set()  # Scripts this node depends on
        self.dependents = set()    # Scripts that depend on this node
        self.inputs = {}           # Expected inputs
        self.outputs = {}          # Expected outputs
        self.status = "idle"       # idle, running, completed, error
        self.metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'message_count': 0,
            'error_count': 0,
            'execution_time': 0
        }
        self.process = None
        self.last_heartbeat = time.time()

class MessageBroker:
    """Central message broker for inter-script communication"""
    def __init__(self):
        self.manager = Manager()
        self.message_queue = self.manager.Queue()
        self.subscribers = self.manager.dict()
        self.message_history = deque(maxlen=1000)
        self.running = False
        self.broker_thread = None
        
    def start(self):
        """Start the message broker"""
        self.running = True
        self.broker_thread = threading.Thread(target=self._broker_loop, daemon=True)
        self.broker_thread.start()
        
    def stop(self):
        """Stop the message broker"""
        self.running = False
        if self.broker_thread:
            self.broker_thread.join(timeout=1)
            
    def _broker_loop(self):
        """Main broker loop for routing messages"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    message = self.message_queue.get(timeout=0.1)
                    self._route_message(message)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Broker error: {e}")
            time.sleep(0.01)
            
    def _route_message(self, message: ScriptMessage):
        """Route message to appropriate receiver"""
        self.message_history.append(message)
        
        if message.receiver_id == "BROADCAST":
            # Broadcast to all scripts
            for script_id, queue_obj in self.subscribers.items():
                if script_id != message.sender_id:
                    try:
                        queue_obj.put(message.to_dict())
                    except:
                        pass
        elif message.receiver_id in self.subscribers:
            # Direct message
            try:
                self.subscribers[message.receiver_id].put(message.to_dict())
            except:
                pass
                
    def subscribe(self, script_id: str) -> Queue:
        """Subscribe a script to receive messages"""
        script_queue = self.manager.Queue()
        self.subscribers[script_id] = script_queue
        return script_queue
        
    def unsubscribe(self, script_id: str):
        """Unsubscribe a script"""
        if script_id in self.subscribers:
            del self.subscribers[script_id]
            
    def send_message(self, message: ScriptMessage):
        """Send a message through the broker"""
        self.message_queue.put(message)

class ScriptDebugger:
    """Advanced debugger for script analysis"""
    def __init__(self):
        self.breakpoints = defaultdict(set)
        self.watch_variables = defaultdict(list)
        self.profiling_data = defaultdict(dict)
        
    def analyze_script(self, script_content: str, script_id: str) -> Dict:
        """Analyze script for potential issues and patterns"""
        analysis = {
            'syntax_errors': [],
            'logic_warnings': [],
            'dependencies': [],
            'input_patterns': [],
            'output_patterns': [],
            'complexity_score': 0,
            'suggested_optimizations': []
        }
        
        try:
            # Parse AST
            tree = ast.parse(script_content)
            
            # Find imports and dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['dependencies'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis['dependencies'].append(node.module)
                    
            # Analyze function calls for inter-script communication
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ['send_message', 'receive_message', 'broadcast']:
                            # Found inter-script communication
                            analysis['output_patterns'].append({
                                'type': node.func.attr,
                                'line': node.lineno
                            })
                            
            # Calculate complexity score (simplified McCabe complexity)
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                    complexity += 1
            analysis['complexity_score'] = complexity
            
            # Check for common issues
            self._check_logic_issues(tree, analysis)
            
        except SyntaxError as e:
            analysis['syntax_errors'].append({
                'line': e.lineno,
                'message': str(e.msg),
                'offset': e.offset
            })
            
        return analysis
        
    def _check_logic_issues(self, tree: ast.AST, analysis: Dict):
        """Check for common logic issues"""
        # Check for infinite loops
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    analysis['logic_warnings'].append({
                        'type': 'infinite_loop',
                        'line': node.lineno,
                        'message': 'Potential infinite loop detected'
                    })
                    
        # Check for unreachable code
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return) and i < len(node.body) - 1:
                        analysis['logic_warnings'].append({
                            'type': 'unreachable_code',
                            'line': node.body[i + 1].lineno,
                            'message': 'Unreachable code after return statement'
                        })

class DependencyVisualizer:
    """Visualize script dependencies and data flow"""
    def __init__(self, parent):
        self.parent = parent
        self.graph = nx.DiGraph()
        self.positions = {}
        self.figure = None
        self.canvas = None
        
    def update_graph(self, scripts: Dict[str, ScriptNode]):
        """Update the dependency graph"""
        self.graph.clear()
        
        # Add nodes
        for script_id, node in scripts.items():
            self.graph.add_node(script_id, 
                              status=node.status,
                              metrics=node.metrics)
            
        # Add edges
        for script_id, node in scripts.items():
            for dep in node.dependencies:
                if dep in scripts:
                    self.graph.add_edge(dep, script_id)
                    
    def draw(self, frame: tk.Frame):
        """Draw the dependency graph"""
        if self.figure:
            plt.close(self.figure)
            
        self.figure = plt.Figure(figsize=(8, 6), dpi=100)
        ax = self.figure.add_subplot(111)
        
        if len(self.graph.nodes) > 0:
            # Calculate positions
            if not self.positions or set(self.positions.keys()) != set(self.graph.nodes):
                self.positions = nx.spring_layout(self.graph, k=2, iterations=50)
                
            # Color nodes based on status
            node_colors = []
            for node in self.graph.nodes:
                status = self.graph.nodes[node].get('status', 'idle')
                if status == 'running':
                    node_colors.append('#FFA500')
                elif status == 'completed':
                    node_colors.append('#00FF00')
                elif status == 'error':
                    node_colors.append('#FF0000')
                else:
                    node_colors.append('#87CEEB')
                    
            # Draw graph
            nx.draw(self.graph, self.positions, ax=ax,
                   node_color=node_colors,
                   node_size=2000,
                   font_size=10,
                   font_weight='bold',
                   arrows=True,
                   edge_color='gray',
                   width=2)
            
            # Add labels with metrics
            labels = {}
            for node in self.graph.nodes:
                metrics = self.graph.nodes[node].get('metrics', {})
                labels[node] = f"{node}\nCPU: {metrics.get('cpu_usage', 0):.1f}%"
                
            nx.draw_networkx_labels(self.graph, self.positions, labels, 
                                  font_size=8, ax=ax)
                                  
        ax.set_title("Script Dependency Graph")
        ax.axis('off')
        
        # Embed in tkinter
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        self.canvas = FigureCanvasTkAgg(self.figure, frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class ScriptOrchestrator:
    """Orchestrate script execution based on dependencies"""
    def __init__(self, message_broker: MessageBroker):
        self.message_broker = message_broker
        self.scripts = {}
        self.execution_order = []
        self.running = False
        
    def add_script(self, script_node: ScriptNode):
        """Add a script to the orchestrator"""
        self.scripts[script_node.id] = script_node
        self._update_execution_order()
        
    def _update_execution_order(self):
        """Calculate execution order based on dependencies"""
        # Build dependency graph
        graph = nx.DiGraph()
        for script_id, node in self.scripts.items():
            graph.add_node(script_id)
            for dep in node.dependencies:
                if dep in self.scripts:
                    graph.add_edge(dep, script_id)
                    
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Circular dependencies detected!")
            
        # Topological sort
        self.execution_order = list(nx.topological_sort(graph))
        
    def execute_all(self, parallel: bool = True):
        """Execute all scripts respecting dependencies"""
        if parallel:
            self._execute_parallel()
        else:
            self._execute_sequential()
            
    def _execute_sequential(self):
        """Execute scripts sequentially"""
        for script_id in self.execution_order:
            if script_id in self.scripts:
                node = self.scripts[script_id]
                logger.info(f"Executing {script_id}")
                # Execute script and wait for completion
                # Implementation depends on script execution mechanism
                
    def _execute_parallel(self):
        """Execute scripts in parallel respecting dependencies"""
        completed = set()
        running = set()
        
        while len(completed) < len(self.scripts):
            # Find scripts ready to run
            ready = []
            for script_id in self.execution_order:
                if script_id not in completed and script_id not in running:
                    node = self.scripts[script_id]
                    # Check if all dependencies are completed
                    if all(dep in completed for dep in node.dependencies):
                        ready.append(script_id)
                        
            # Start ready scripts
            for script_id in ready:
                logger.info(f"Starting {script_id}")
                running.add(script_id)
                # Start script execution
                # Monitor for completion
                
            time.sleep(0.1)

class EnhancedScriptTab:
    """Enhanced script tab with advanced features"""
    def __init__(self, parent, tab_id, runner):
        self.parent = parent
        self.tab_id = tab_id
        self.runner = runner
        self.script_node = ScriptNode(f"script_{tab_id}")
        self.process = None
        self.output_queue = queue.Queue()
        self.message_queue = None
        self.file_path = None
        self.last_saved_content = ""
        self.syntax_errors = []
        self.logic_analysis = {}
        self.performance_monitor = None
        
        self.setup_ui()
        self._setup_message_handler()
        
    def setup_ui(self):
        """Setup enhanced UI for this script tab"""
        # Main container with notebook for multiple views
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.view_notebook = ttk.Notebook(self.frame)
        self.view_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Editor view
        self.editor_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.editor_frame, text="Editor")
        self._setup_editor_view()
        
        # Debug view
        self.debug_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.debug_frame, text="Debug")
        self._setup_debug_view()
        
        # Messages view
        self.messages_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.messages_frame, text="Messages")
        self._setup_messages_view()
        
        # Metrics view
        self.metrics_frame = ttk.Frame(self.view_notebook)
        self.view_notebook.add(self.metrics_frame, text="Metrics")
        self._setup_metrics_view()
        
    def _setup_editor_view(self):
        """Setup the main editor view"""
        # Paned window for editor and output
        paned = ttk.PanedWindow(self.editor_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Top frame for editor
        editor_container = ttk.LabelFrame(paned, text="Script Editor", padding=5)
        paned.add(editor_container, weight=3)
        
        # Enhanced toolbar
        toolbar = ttk.Frame(editor_container)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        
        # File operations
        file_frame = ttk.Frame(toolbar)
        file_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="📁 Load", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="💾 Save", command=self.save_file).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        # Execution controls
        exec_frame = ttk.Frame(toolbar)
        exec_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(exec_frame, text="▶️ Run", command=self.run_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(exec_frame, text="⏸️ Debug", command=self.debug_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(exec_frame, text="⏹️ Stop", command=self.stop_script).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        # Analysis controls
        analysis_frame = ttk.Frame(toolbar)
        analysis_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(analysis_frame, text="✓ Check", command=self.analyze_script).pack(side=tk.LEFT, padx=2)
        ttk.Button(analysis_frame, text="🔍 Profile", command=self.profile_script).pack(side=tk.LEFT, padx=2)
        
        # Status
        self.status_label = ttk.Label(toolbar, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Editor with enhanced features
        editor_container_inner = ttk.Frame(editor_container)
        editor_container_inner.pack(fill=tk.BOTH, expand=True)
        
        # Line numbers
        self.line_numbers = tk.Text(editor_container_inner, width=5, padx=3, takefocus=0,
                                   wrap=tk.NONE, state='disabled',
                                   font=("Consolas", 10))
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # Main editor
        self.editor = scrolledtext.ScrolledText(editor_container_inner, wrap=tk.NONE,
                                               font=("Consolas", 10), undo=True,
                                               maxundo=-1)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Syntax highlighting setup
        self._setup_syntax_highlighting()
        
        # Analysis panel
        self.analysis_text = scrolledtext.ScrolledText(editor_container, height=5, 
                                                      wrap=tk.WORD,
                                                      font=("Consolas", 9))
        # Initially hidden
        
        # Bind events
        self.editor.bind('<KeyRelease>', self.on_editor_change)
        self.editor.bind('<Control-s>', lambda e: self.save_file())
        self.editor.bind('<F5>', lambda e: self.run_script())
        
        # Output frame
        output_frame = ttk.LabelFrame(paned, text="Output", padding=5)
        paned.add(output_frame, weight=2)
        
        # Output controls
        output_toolbar = ttk.Frame(output_frame)
        output_toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(output_toolbar, text="Clear", command=self.clear_output).pack(side=tk.LEFT, padx=2)
        ttk.Button(output_toolbar, text="Copy", command=self.copy_output).pack(side=tk.LEFT, padx=2)
        
        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_toolbar, text="Auto-scroll", 
                       variable=self.auto_scroll).pack(side=tk.LEFT, padx=10)
        
        # Filter options
        ttk.Label(output_toolbar, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.output_filter = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(output_toolbar, textvariable=self.output_filter,
                                   values=["all", "stdout", "stderr", "messages", "metrics"],
                                   width=10, state="readonly")
        filter_combo.pack(side=tk.LEFT)
        
        # Output text with tabs
        output_notebook = ttk.Notebook(output_frame)
        output_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Console output
        self.output = scrolledtext.ScrolledText(output_notebook, wrap=tk.WORD,
                                               font=("Consolas", 9),
                                               state='disabled')
        output_notebook.add(self.output, text="Console")
        
        # Message log
        self.message_log = scrolledtext.ScrolledText(output_notebook, wrap=tk.WORD,
                                                    font=("Consolas", 9),
                                                    state='disabled')
        output_notebook.add(self.message_log, text="Messages")
        
        # Configure output tags
        self.output.tag_config("stdout", foreground="black")
        self.output.tag_config("stderr", foreground="red")
        self.output.tag_config("system", foreground="blue", font=("Consolas", 9, "italic"))
        self.output.tag_config("debug", foreground="purple")
        self.output.tag_config("message_in", foreground="green")
        self.output.tag_config("message_out", foreground="orange")
        
        self.update_line_numbers()
        
    def _setup_debug_view(self):
        """Setup debug view with advanced debugging features"""
        # Debug toolbar
        toolbar = ttk.Frame(self.debug_frame)
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="Set Breakpoint", command=self.set_breakpoint).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Step Over", command=self.step_over).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Step Into", command=self.step_into).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Continue", command=self.continue_execution).pack(side=tk.LEFT, padx=2)
        
        # Debug panels
        debug_paned = ttk.PanedWindow(self.debug_frame, orient=tk.HORIZONTAL)
        debug_paned.pack(fill=tk.BOTH, expand=True)
        
        # Variables panel
        var_frame = ttk.LabelFrame(debug_paned, text="Variables", padding=5)
        debug_paned.add(var_frame, weight=1)
        
        self.variables_tree = ttk.Treeview(var_frame, columns=('value', 'type'), show='tree headings')
        self.variables_tree.heading('#0', text='Name')
        self.variables_tree.heading('value', text='Value')
        self.variables_tree.heading('type', text='Type')
        self.variables_tree.pack(fill=tk.BOTH, expand=True)
        
        # Call stack panel
        stack_frame = ttk.LabelFrame(debug_paned, text="Call Stack", padding=5)
        debug_paned.add(stack_frame, weight=1)
        
        self.call_stack = tk.Listbox(stack_frame, font=("Consolas", 9))
        self.call_stack.pack(fill=tk.BOTH, expand=True)
        
        # Breakpoints panel
        break_frame = ttk.LabelFrame(debug_paned, text="Breakpoints", padding=5)
        debug_paned.add(break_frame, weight=1)
        
        self.breakpoints_list = tk.Listbox(break_frame, font=("Consolas", 9))
        self.breakpoints_list.pack(fill=tk.BOTH, expand=True)
        
    def _setup_messages_view(self):
        """Setup messages view for inter-script communication"""
        # Message controls
        controls = ttk.Frame(self.messages_frame)
        controls.pack(fill=tk.X, pady=5)
        
        ttk.Label(controls, text="Send to:").pack(side=tk.LEFT, padx=5)
        self.message_target = tk.StringVar(value="BROADCAST")
        self.target_combo = ttk.Combobox(controls, textvariable=self.message_target, width=15)
        self.target_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(controls, text="Message:").pack(side=tk.LEFT, padx=5)
        self.message_entry = ttk.Entry(controls, width=40)
        self.message_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls, text="Send", command=self.send_message).pack(side=tk.LEFT, padx=5)
        
        # Message history
        history_frame = ttk.LabelFrame(self.messages_frame, text="Message History", padding=5)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for messages
        columns = ('time', 'from', 'to', 'type', 'message')
        self.message_tree = ttk.Treeview(history_frame, columns=columns, show='headings')
        
        for col in columns:
            self.message_tree.heading(col, text=col.title())
            self.message_tree.column(col, width=100)
            
        self.message_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.message_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.message_tree.configure(yscrollcommand=scrollbar.set)
        
    def _setup_metrics_view(self):
        """Setup metrics view for performance monitoring"""
        # Metrics display
        metrics_container = ttk.Frame(self.metrics_frame)
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current metrics
        current_frame = ttk.LabelFrame(metrics_container, text="Current Metrics", padding=10)
        current_frame.pack(fill=tk.X, pady=5)
        
        self.metric_labels = {}
        metrics = ['CPU Usage', 'Memory Usage', 'Messages/sec', 'Execution Time', 'Error Count']
        
        for i, metric in enumerate(metrics):
            label_frame = ttk.Frame(current_frame)
            label_frame.grid(row=i//3, column=(i%3)*2, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(label_frame, text=f"{metric}:").pack(side=tk.LEFT)
            value_label = ttk.Label(label_frame, text="0", font=("Consolas", 10, "bold"))
            value_label.pack(side=tk.LEFT, padx=5)
            self.metric_labels[metric] = value_label
            
        # Performance graph placeholder
        graph_frame = ttk.LabelFrame(metrics_container, text="Performance Graph", padding=5)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Would add matplotlib graph here
        ttk.Label(graph_frame, text="Performance visualization would go here").pack(expand=True)
        
    def _setup_syntax_highlighting(self):
        """Setup syntax highlighting for Python"""
        # Python keywords
        self.editor.tag_config("keyword", foreground="#0000FF", font=("Consolas", 10, "bold"))
        self.editor.tag_config("builtin", foreground="#8B008B")
        self.editor.tag_config("string", foreground="#008000")
        self.editor.tag_config("comment", foreground="#808080", font=("Consolas", 10, "italic"))
        self.editor.tag_config("function", foreground="#FF1493")
        self.editor.tag_config("class", foreground="#FF8C00", font=("Consolas", 10, "bold"))
        self.editor.tag_config("number", foreground="#FF0000")
        
    def _setup_message_handler(self):
        """Setup message handling for inter-script communication"""
        if self.runner.message_broker:
            self.message_queue = self.runner.message_broker.subscribe(self.script_node.id)
            # Start message monitoring thread
            threading.Thread(target=self._monitor_messages, daemon=True).start()
            
    def _monitor_messages(self):
        """Monitor incoming messages"""
        while True:
            try:
                if self.message_queue and not self.message_queue.empty():
                    message_dict = self.message_queue.get(timeout=0.1)
                    message = ScriptMessage.from_dict(message_dict)
                    self.frame.after(0, lambda m=message: self._handle_message(m))
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Message monitoring error: {e}")
            time.sleep(0.01)
            
    def _handle_message(self, message: ScriptMessage):
        """Handle incoming message"""
        # Update message tree
        timestamp = datetime.fromtimestamp(message.timestamp).strftime("%H:%M:%S")
        self.message_tree.insert('', 0, values=(
            timestamp,
            message.sender_id,
            message.receiver_id,
            message.message_type.value,
            str(message.payload)[:100]
        ))
        
        # Log to message output
        self.add_output(f"[MESSAGE IN] From {message.sender_id}: {message.payload}\n", "message_in")
        
        # Handle special message types
        if message.message_type == MessageType.CONTROL:
            self._handle_control_message(message)
            
    def _handle_control_message(self, message: ScriptMessage):
        """Handle control messages"""
        payload = message.payload
        if isinstance(payload, dict):
            action = payload.get('action')
            if action == 'stop':
                self.stop_script()
            elif action == 'pause':
                # Implement pause functionality
                pass
                
    def send_message(self):
        """Send a message to another script"""
        target = self.message_target.get()
        message_text = self.message_entry.get()
        
        if not message_text:
            return
            
        message = ScriptMessage(
            sender_id=self.script_node.id,
            receiver_id=target,
            message_type=MessageType.DATA,
            payload=message_text
        )
        
        self.runner.message_broker.send_message(message)
        
        # Log outgoing message
        self.add_output(f"[MESSAGE OUT] To {target}: {message_text}\n", "message_out")
        
        # Clear entry
        self.message_entry.delete(0, tk.END)
        
    def analyze_script(self):
        """Perform comprehensive script analysis"""
        content = self.editor.get(1.0, tk.END)
        
        # Show analysis panel
        if not self.analysis_text.winfo_viewable():
            self.analysis_text.pack(fill=tk.X, pady=(5, 0))
            
        self.analysis_text.config(state='normal')
        self.analysis_text.delete(1.0, tk.END)
        
        # Perform analysis
        analysis = self.runner.debugger.analyze_script(content, self.script_node.id)
        self.logic_analysis = analysis
        
        # Display results
        self.analysis_text.insert(tk.END, "=== Script Analysis ===\n\n", "header")
        
        # Syntax errors
        if analysis['syntax_errors']:
            self.analysis_text.insert(tk.END, "❌ Syntax Errors:\n", "error")
            for error in analysis['syntax_errors']:
                self.analysis_text.insert(tk.END, f"  Line {error['line']}: {error['message']}\n")
        else:
            self.analysis_text.insert(tk.END, "✅ No syntax errors\n", "success")
            
        # Logic warnings
        if analysis['logic_warnings']:
            self.analysis_text.insert(tk.END, "\n⚠️ Logic Warnings:\n", "warning")
            for warning in analysis['logic_warnings']:
                self.analysis_text.insert(tk.END, f"  Line {warning['line']}: {warning['message']}\n")
                
        # Complexity
        self.analysis_text.insert(tk.END, f"\n📊 Complexity Score: {analysis['complexity_score']}\n")
        
        # Dependencies
        if analysis['dependencies']:
            self.analysis_text.insert(tk.END, "\n📦 Dependencies:\n")
            for dep in analysis['dependencies']:
                self.analysis_text.insert(tk.END, f"  - {dep}\n")
                
        # Communication patterns
        if analysis['output_patterns']:
            self.analysis_text.insert(tk.END, "\n💬 Communication Patterns:\n")
            for pattern in analysis['output_patterns']:
                self.analysis_text.insert(tk.END, f"  - {pattern['type']} at line {pattern['line']}\n")
                
        self.analysis_text.config(state='disabled')
        
        # Update script node dependencies
        self._update_dependencies(analysis['dependencies'])
        
    def _update_dependencies(self, dependencies):
        """Update script node dependencies"""
        # Parse dependencies to find other scripts
        for dep in dependencies:
            if dep.startswith('script_'):
                self.script_node.dependencies.add(dep)
                
        # Update orchestrator
        self.runner.orchestrator.add_script(self.script_node)
        
    def profile_script(self):
        """Profile script performance"""
        messagebox.showinfo("Profiling", "Script profiling will be implemented here")
        
    def debug_script(self):
        """Start script in debug mode"""
        messagebox.showinfo("Debug", "Debug mode will be implemented here")
        
    def set_breakpoint(self):
        """Set a breakpoint at current line"""
        insert = self.editor.index(tk.INSERT)
        line = int(insert.split('.')[0])
        
        self.runner.debugger.breakpoints[self.script_node.id].add(line)
        self.breakpoints_list.insert(tk.END, f"Line {line}")
        
        # Visual indicator
        self.editor.tag_add("breakpoint", f"{line}.0", f"{line}.end")
        self.editor.tag_config("breakpoint", background="#FFCCCC")
        
    def step_over(self):
        """Step over in debug mode"""
        # Implementation for debugging
        pass
        
    def step_into(self):
        """Step into in debug mode"""
        # Implementation for debugging
        pass
        
    def continue_execution(self):
        """Continue execution in debug mode"""
        # Implementation for debugging
        pass
        
    def on_editor_change(self, event=None):
        """Handle editor content changes"""
        self.update_line_numbers()
        self._apply_syntax_highlighting()
        
        # Schedule analysis (debounced)
        if hasattr(self, '_analysis_timer'):
            self.frame.after_cancel(self._analysis_timer)
        self._analysis_timer = self.frame.after(1500, self.analyze_script)
        
    def _apply_syntax_highlighting(self):
        """Apply syntax highlighting to editor content"""
        # Simple Python syntax highlighting
        content = self.editor.get(1.0, tk.END)
        
        # Remove existing tags
        for tag in ["keyword", "string", "comment", "function", "class", "number"]:
            self.editor.tag_remove(tag, 1.0, tk.END)
            
        # Keywords
        import keyword
        for kw in keyword.kwlist:
            start = 1.0
            while True:
                pos = self.editor.search(rf'\b{kw}\b', start, tk.END, regexp=True)
                if not pos:
                    break
                end = f"{pos}+{len(kw)}c"
                self.editor.tag_add("keyword", pos, end)
                start = end
                
        # Strings (simple version)
        for quote in ['"', "'"]:
            start = 1.0
            while True:
                pos = self.editor.search(quote, start, tk.END)
                if not pos:
                    break
                end = self.editor.search(quote, f"{pos}+1c", tk.END)
                if not end:
                    break
                self.editor.tag_add("string", pos, f"{end}+1c")
                start = f"{end}+1c"
                
        # Comments
        start = 1.0
        while True:
            pos = self.editor.search('#', start, tk.END)
            if not pos:
                break
            end = self.editor.search('\n', pos, tk.END)
            if not end:
                end = tk.END
            self.editor.tag_add("comment", pos, end)
            start = end
            
    def update_line_numbers(self):
        """Update line numbers in the editor"""
        self.line_numbers.config(state='normal')
        self.line_numbers.delete(1.0, tk.END)
        
        line_count = self.editor.get(1.0, tk.END).count('\n')
        line_numbers_string = "\n".join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.insert(1.0, line_numbers_string)
        self.line_numbers.config(state='disabled')
        
    def run_script(self):
        """Run the script with enhanced monitoring"""
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Warning", "Script is already running!")
            return
            
        content = self.editor.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No script to run!")
            return
            
        # Perform analysis first
        self.analyze_script()
        
        if self.logic_analysis.get('syntax_errors'):
            if not messagebox.askyesno("Syntax Errors", 
                                     "Syntax errors found. Run anyway?"):
                return
                
        # Inject communication framework
        enhanced_content = self._inject_communication_framework(content)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(enhanced_content)
            self.temp_script = f.name
            
        # Start process with monitoring
        self._start_process_with_monitoring()
        
    def _inject_communication_framework(self, content):
        """Inject inter-script communication framework"""
        framework = '''
import sys
import json
import time
import multiprocessing
from multiprocessing import Queue

# Script communication framework
class ScriptCommunicator:
    def __init__(self, script_id):
        self.script_id = script_id
        self.message_queue = None
        
    def send_message(self, target, payload):
        message = {
            'sender_id': self.script_id,
            'receiver_id': target,
            'message_type': 'data',
            'payload': payload,
            'timestamp': time.time()
        }
        print(f"[SCRIPT_MESSAGE_OUT]{json.dumps(message)}")
        sys.stdout.flush()
        
    def broadcast(self, payload):
        self.send_message('BROADCAST', payload)
        
    def log_metric(self, name, value):
        message = {
            'sender_id': self.script_id,
            'receiver_id': 'SYSTEM',
            'message_type': 'metric',
            'payload': {'name': name, 'value': value},
            'timestamp': time.time()
        }
        print(f"[SCRIPT_METRIC]{json.dumps(message)}")
        sys.stdout.flush()

# Initialize communicator
comm = ScriptCommunicator('{}')

# User script below
{}
'''.format(self.script_node.id, content)
        
        return framework
        
    def _start_process_with_monitoring(self):
        """Start process with enhanced monitoring"""
        try:
            # Create process
            self.process = subprocess.Popen(
                [sys.executable, '-u', self.temp_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Update status
            self.script_node.status = "running"
            self.status_label.config(text="Running...", foreground="orange")
            
            # Clear output
            self.clear_output()
            self.add_output(f"Starting script {self.script_node.id}...\n", "system")
            
            # Start monitoring
            self.performance_monitor = threading.Thread(
                target=self._monitor_performance, daemon=True)
            self.performance_monitor.start()
            
            # Start output monitoring
            threading.Thread(target=self._monitor_output, 
                           args=(self.process.stdout, "stdout"), daemon=True).start()
            threading.Thread(target=self._monitor_output, 
                           args=(self.process.stderr, "stderr"), daemon=True).start()
            threading.Thread(target=self._monitor_process, daemon=True).start()
            
            # Update visualization
            self.runner.update_visualization()
            
        except Exception as e:
            self.add_output(f"Failed to start script: {str(e)}\n", "stderr")
            self.script_node.status = "error"
            self.status_label.config(text="Error", foreground="red")
            
    def _monitor_output(self, pipe, tag):
        """Monitor process output with message detection"""
        try:
            for line in pipe:
                # Check for special messages
                if line.startswith("[SCRIPT_MESSAGE_OUT]"):
                    # Parse and route message
                    try:
                        message_data = json.loads(line[20:])
                        message = ScriptMessage.from_dict(message_data)
                        self.runner.message_broker.send_message(message)
                        self.frame.after(0, lambda: self.add_output(
                            f"→ Sent message to {message.receiver_id}\n", "message_out"))
                    except:
                        pass
                elif line.startswith("[SCRIPT_METRIC]"):
                    # Parse metric
                    try:
                        metric_data = json.loads(line[15:])
                        self._update_metric(metric_data['payload'])
                    except:
                        pass
                else:
                    # Regular output
                    self.output_queue.put((line, tag))
                    self.frame.after(0, self.process_output_queue)
        except:
            pass
            
    def _monitor_performance(self):
        """Monitor script performance metrics"""
        if not self.process:
            return
            
        try:
            proc = psutil.Process(self.process.pid)
            
            while self.process and self.process.poll() is None:
                # Get metrics
                cpu_percent = proc.cpu_percent(interval=1)
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Update metrics
                self.script_node.metrics['cpu_usage'] = cpu_percent
                self.script_node.metrics['memory_usage'] = memory_mb
                
                # Update UI
                self.frame.after(0, lambda: self._update_metrics_display())
                
                time.sleep(1)
        except:
            pass
            
    def _update_metrics_display(self):
        """Update metrics display"""
        metrics = self.script_node.metrics
        
        self.metric_labels['CPU Usage'].config(
            text=f"{metrics['cpu_usage']:.1f}%")
        self.metric_labels['Memory Usage'].config(
            text=f"{metrics['memory_usage']:.1f} MB")
        self.metric_labels['Messages/sec'].config(
            text=f"{metrics.get('message_rate', 0):.1f}")
        self.metric_labels['Execution Time'].config(
            text=f"{metrics.get('execution_time', 0):.1f}s")
        self.metric_labels['Error Count'].config(
            text=str(metrics['error_count']))
            
    def _update_metric(self, metric_data):
        """Update custom metric"""
        name = metric_data.get('name')
        value = metric_data.get('value')
        
        if name and value is not None:
            self.script_node.metrics[name] = value
            
    def _monitor_process(self):
        """Monitor process completion"""
        start_time = time.time()
        self.process.wait()
        execution_time = time.time() - start_time
        
        return_code = self.process.returncode
        self.script_node.metrics['execution_time'] = execution_time
        
        self.frame.after(0, lambda: self._on_process_complete(return_code))
        
    def _on_process_complete(self, return_code):
        """Handle process completion"""
        self.add_output("-" * 50 + "\n", "system")
        self.add_output(f"Process exited with code: {return_code}\n", "system")
        self.add_output(f"Execution time: {self.script_node.metrics['execution_time']:.2f}s\n", "system")
        
        if return_code == 0:
            self.script_node.status = "completed"
            self.status_label.config(text="Completed", foreground="green")
        else:
            self.script_node.status = "error"
            self.script_node.metrics['error_count'] += 1
            self.status_label.config(text=f"Error (code {return_code})", foreground="red")
            
        # Clean up
        try:
            if hasattr(self, 'temp_script') and os.path.exists(self.temp_script):
                os.unlink(self.temp_script)
        except:
            pass
            
        self.process = None
        
        # Update visualization
        self.runner.update_visualization()
        
    def stop_script(self):
        """Stop the running script"""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.add_output("\nScript terminated by user\n", "system")
            self.script_node.status = "idle"
            self.status_label.config(text="Terminated", foreground="red")
            
    def process_output_queue(self):
        """Process queued output"""
        try:
            while True:
                line, tag = self.output_queue.get_nowait()
                self.add_output(line, tag)
        except queue.Empty:
            pass
            
    def add_output(self, text, tag="stdout"):
        """Add text to output with filtering"""
        # Apply filter
        filter_value = self.output_filter.get()
        if filter_value != "all" and tag != filter_value:
            return
            
        self.output.config(state='normal')
        self.output.insert(tk.END, text, tag)
        if self.auto_scroll.get():
            self.output.see(tk.END)
        self.output.config(state='disabled')
        
        # Also add to message log if it's a message
        if tag in ["message_in", "message_out"]:
            self.message_log.config(state='normal')
            self.message_log.insert(tk.END, text, tag)
            if self.auto_scroll.get():
                self.message_log.see(tk.END)
            self.message_log.config(state='disabled')
            
    def clear_output(self):
        """Clear all output"""
        for widget in [self.output, self.message_log]:
            widget.config(state='normal')
            widget.delete(1.0, tk.END)
            widget.config(state='disabled')
            
    def copy_output(self):
        """Copy output to clipboard"""
        content = self.output.get(1.0, tk.END)
        self.frame.clipboard_clear()
        self.frame.clipboard_append(content)
        
    def load_file(self):
        """Load a script file"""
        filename = filedialog.askopenfilename(
            title="Open Script",
            filetypes=[
                ("Python files", "*.py"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.editor.delete(1.0, tk.END)
                self.editor.insert(1.0, content)
                self.file_path = filename
                self.last_saved_content = content
                self.script_node.file_path = filename
                
                # Update tab title
                self.runner.notebook.tab(self.frame, text=os.path.basename(filename))
                
                # Analyze immediately
                self.analyze_script()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def save_file(self):
        """Save the script to file"""
        if not self.file_path:
            self.file_path = filedialog.asksaveasfilename(
                title="Save Script",
                defaultextension=".py",
                filetypes=[
                    ("Python files", "*.py"),
                    ("All files", "*.*")
                ]
            )
            
        if self.file_path:
            try:
                content = self.editor.get(1.0, tk.END)
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.last_saved_content = content
                self.script_node.file_path = self.file_path
                
                # Update tab title
                self.runner.notebook.tab(self.frame, text=os.path.basename(self.file_path))
                
                self.add_output(f"Saved to: {self.file_path}\n", "system")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")


class NeuralScriptIDE:
    """Main application for the Neural Script IDE"""
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Script IDE - Production Version")
        self.root.geometry("1400x900")
        
        # Core components
        self.message_broker = MessageBroker()
        self.debugger = ScriptDebugger()
        self.orchestrator = ScriptOrchestrator(self.message_broker)
        self.dependency_visualizer = None
        
        self.tabs = {}
        self.tab_counter = 0
        
        # Configuration
        self.config_file = Path.home() / ".neural_script_ide_config.json"
        self.load_config()
        
        self.setup_ui()
        self.setup_menu()
        
        # Start services
        self.message_broker.start()
        
        # Create initial tab
        self.create_new_tab()
        
        # Bind shortcuts
        self.setup_keyboard_shortcuts()
        
    def load_config(self):
        """Load configuration"""
        default_config = {
            "theme": "dark",
            "auto_save": True,
            "auto_analyze": True,
            "debug_mode": False,
            "performance_monitoring": True,
            "message_history_size": 1000
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = {**default_config, **json.load(f)}
            except:
                self.config = default_config
        else:
            self.config = default_config
            
    def save_config(self):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def setup_ui(self):
        """Setup main UI"""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Script tabs
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)
        
        # Tab bar
        tab_toolbar = ttk.Frame(left_frame)
        tab_toolbar.pack(fill=tk.X)
        
        ttk.Button(tab_toolbar, text="➕ New Tab", command=self.create_new_tab).pack(side=tk.LEFT, padx=2)
        ttk.Button(tab_toolbar, text="📊 Show Dependencies", command=self.show_dependencies).pack(side=tk.LEFT, padx=2)
        ttk.Button(tab_toolbar, text="▶️ Run All", command=self.run_all_scripts).pack(side=tk.LEFT, padx=2)
        ttk.Button(tab_toolbar, text="⏹️ Stop All", command=self.stop_all_scripts).pack(side=tk.LEFT, padx=2)
        
        # Script tabs notebook
        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Visualization and monitoring
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # Right panel notebook
        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Dependency graph tab
        dep_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(dep_frame, text="Dependencies")
        self.dependency_visualizer = DependencyVisualizer(self)
        
        # System monitor tab
        monitor_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(monitor_frame, text="System Monitor")
        self.setup_system_monitor(monitor_frame)
        
        # Message flow tab
        message_frame = ttk.Frame(self.right_notebook)
        self.right_notebook.add(message_frame, text="Message Flow")
        self.setup_message_flow(message_frame)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_system_monitor(self, parent):
        """Setup system monitoring view"""
        # System metrics
        metrics_frame = ttk.LabelFrame(parent, text="System Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.system_labels = {}
        metrics = ['Total CPU', 'Total Memory', 'Active Scripts', 'Messages/sec']
        
        for i, metric in enumerate(metrics):
            label_frame = ttk.Frame(metrics_frame)
            label_frame.grid(row=i//2, column=(i%2)*2, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(label_frame, text=f"{metric}:").pack(side=tk.LEFT)
            value_label = ttk.Label(label_frame, text="0", font=("Arial", 10, "bold"))
            value_label.pack(side=tk.LEFT, padx=5)
            self.system_labels[metric] = value_label
            
        # Script status list
        status_frame = ttk.LabelFrame(parent, text="Script Status", padding=5)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ('script', 'status', 'cpu', 'memory', 'messages')
        self.status_tree = ttk.Treeview(status_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.status_tree.heading(col, text=col.title())
            self.status_tree.column(col, width=80)
            
        self.status_tree.pack(fill=tk.BOTH, expand=True)
        
        # Start monitoring
        self.monitor_system()
        
    def setup_message_flow(self, parent):
        """Setup message flow visualization"""
        # Message statistics
        stats_frame = ttk.LabelFrame(parent, text="Message Statistics", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(stats_frame, text="Total Messages: ").grid(row=0, column=0, sticky=tk.W)
        self.total_messages_label = ttk.Label(stats_frame, text="0")
        self.total_messages_label.grid(row=0, column=1, sticky=tk.W)
        
        # Recent messages
        recent_frame = ttk.LabelFrame(parent, text="Recent Messages", padding=5)
        recent_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.message_flow_text = scrolledtext.ScrolledText(recent_frame, height=15, 
                                                          wrap=tk.WORD, 
                                                          font=("Consolas", 9))
        self.message_flow_text.pack(fill=tk.BOTH, expand=True)
        
        # Update periodically
        self.update_message_flow()
        
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(self.status_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        self.tab_count_label = ttk.Label(self.status_frame, text="Scripts: 0")
        self.tab_count_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(self.status_frame, orient='vertical').pack(side=tk.LEFT, fill='y', padx=5)
        
        self.message_count_label = ttk.Label(self.status_frame, text="Messages: 0")
        self.message_count_label.pack(side=tk.LEFT, padx=5)
        
    def setup_menu(self):
        """Setup application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Script", command=self.create_new_tab, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Script", command=self.open_script, accelerator="Ctrl+O")
        file_menu.add_command(label="Save All", command=self.save_all_scripts, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Import Project", command=self.import_project)
        file_menu.add_command(label="Export Project", command=self.export_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Find", accelerator="Ctrl+F")
        edit_menu.add_command(label="Replace", accelerator="Ctrl+H")
        
        # Scripts menu
        scripts_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Scripts", menu=scripts_menu)
        scripts_menu.add_command(label="Run All", command=self.run_all_scripts, accelerator="F5")
        scripts_menu.add_command(label="Stop All", command=self.stop_all_scripts, accelerator="Shift+F5")
        scripts_menu.add_separator()
        scripts_menu.add_command(label="Analyze All", command=self.analyze_all_scripts)
        scripts_menu.add_command(label="Check Dependencies", command=self.check_dependencies)
        
        # Debug menu
        debug_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Debug", menu=debug_menu)
        debug_menu.add_command(label="Start Debugging", accelerator="F10")
        debug_menu.add_command(label="Step Over", accelerator="F11")
        debug_menu.add_command(label="Toggle Breakpoint", accelerator="F9")
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Message Inspector", command=self.open_message_inspector)
        tools_menu.add_command(label="Performance Profiler", command=self.open_profiler)
        tools_menu.add_command(label="Script Templates", command=self.open_templates)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings", command=self.open_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.open_documentation)
        help_menu.add_command(label="Examples", command=self.open_examples)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-n>', lambda e: self.create_new_tab())
        self.root.bind('<Control-o>', lambda e: self.open_script())
        self.root.bind('<Control-w>', lambda e: self.close_current_tab())
        self.root.bind('<Control-Tab>', lambda e: self.next_tab())
        self.root.bind('<Control-Shift-Tab>', lambda e: self.prev_tab())
        self.root.bind('<F5>', lambda e: self.run_current_script())
        self.root.bind('<Shift-F5>', lambda e: self.stop_all_scripts())
        self.root.bind('<Control-Shift-s>', lambda e: self.save_all_scripts())
        
    def create_new_tab(self):
        """Create a new script tab"""
        self.tab_counter += 1
        tab_id = self.tab_counter
        
        tab = EnhancedScriptTab(self.notebook, tab_id, self)
        self.tabs[tab_id] = tab
        
        self.notebook.add(tab.frame, text=f"Script {tab_id}")
        self.notebook.select(tab.frame)
        
        # Update target options for messages
        self.update_message_targets()
        
        self.update_tab_count()
        
    def open_script(self):
        """Open an existing script"""
        tab = self.create_new_tab()
        current_tab = self.get_current_tab()
        if current_tab:
            current_tab.load_file()
            
    def close_current_tab(self):
        """Close the current tab"""
        if len(self.tabs) <= 1:
            messagebox.showwarning("Warning", "Cannot close the last tab!")
            return
            
        current_tab_widget = self.notebook.select()
        for tab_id, tab in self.tabs.items():
            if str(tab.frame) == current_tab_widget:
                # Check if script is running
                if tab.process and tab.process.poll() is None:
                    if not messagebox.askyesno("Script Running", 
                                             "Script is still running. Close anyway?"):
                        return
                    tab.stop_script()
                    
                # Unsubscribe from messages
                if self.message_broker:
                    self.message_broker.unsubscribe(tab.script_node.id)
                    
                self.notebook.forget(current_tab_widget)
                del self.tabs[tab_id]
                self.update_tab_count()
                self.update_message_targets()
                break
                
    def get_current_tab(self):
        """Get the current active tab"""
        current_tab_widget = self.notebook.select()
        for tab_id, tab in self.tabs.items():
            if str(tab.frame) == current_tab_widget:
                return tab
        return None
        
    def next_tab(self):
        """Switch to next tab"""
        tabs = self.notebook.tabs()
        if tabs:
            current = self.notebook.select()
            current_index = tabs.index(current)
            next_index = (current_index + 1) % len(tabs)
            self.notebook.select(tabs[next_index])
            
    def prev_tab(self):
        """Switch to previous tab"""
        tabs = self.notebook.tabs()
        if tabs:
            current = self.notebook.select()
            current_index = tabs.index(current)
            prev_index = (current_index - 1) % len(tabs)
            self.notebook.select(tabs[prev_index])
            
    def update_tab_count(self):
        """Update tab count in status bar"""
        self.tab_count_label.config(text=f"Scripts: {len(self.tabs)}")
        

    def update_message_targets(self):
        """Update message target options for all tabs"""
        targets = ["BROADCAST"]
        for tab_id, tab in self.tabs.items():
            targets.append(tab.script_node.id)
            
        for tab in self.tabs.values():
            tab.message_target.set("BROADCAST")
            if hasattr(tab, 'target_combo'):
                tab.target_combo['values'] = targets
                tab.target_combo.set("BROADCAST")
                

            
    def run_current_script(self):
        """Run the current script"""
        tab = self.get_current_tab()
        if tab:
            tab.run_script()
            
    def run_all_scripts(self):
        """Run all scripts with orchestration"""
        try:
            # Update orchestrator with all scripts
            for tab in self.tabs.values():
                self.orchestrator.add_script(tab.script_node)
                
            # Check for circular dependencies
            self.orchestrator._update_execution_order()
            
            # Run scripts
            for tab_id in self.orchestrator.execution_order:
                if tab_id in [tab.script_node.id for tab in self.tabs.values()]:
                    tab = next(t for t in self.tabs.values() if t.script_node.id == tab_id)
                    tab.run_script()
                    time.sleep(0.1)  # Small delay between starts
                    
        except ValueError as e:
            messagebox.showerror("Dependency Error", str(e))
            
    def stop_all_scripts(self):
        """Stop all running scripts"""
        for tab in self.tabs.values():
            tab.stop_script()
            
    def save_all_scripts(self):
        """Save all scripts"""
        for tab in self.tabs.values():
            if tab.editor.get(1.0, tk.END).strip():
                tab.save_file()
                
    def analyze_all_scripts(self):
        """Analyze all scripts"""
        for tab in self.tabs.values():
            tab.analyze_script()
            
    def check_dependencies(self):
        """Check and visualize dependencies"""
        self.show_dependencies()
        
        # Check for issues
        issues = []
        
        # Check for circular dependencies
        try:
            self.orchestrator._update_execution_order()
        except ValueError as e:
            issues.append(str(e))
            
        # Check for missing dependencies
        all_script_ids = {tab.script_node.id for tab in self.tabs.values()}
        for tab in self.tabs.values():
            for dep in tab.script_node.dependencies:
                if dep not in all_script_ids:
                    issues.append(f"Script {tab.script_node.id} depends on missing script {dep}")
                    
        if issues:
            messagebox.showwarning("Dependency Issues", "\n".join(issues))
        else:
            messagebox.showinfo("Dependencies", "All dependencies are valid!")
            
    def show_dependencies(self):
        """Show dependency visualization"""
        # Update graph
        scripts = {tab.script_node.id: tab.script_node for tab in self.tabs.values()}
        self.dependency_visualizer.update_graph(scripts)
        
        # Draw graph
        dep_frame = self.right_notebook.tabs()[0]
        dep_widget = self.right_notebook.nametowidget(dep_frame)
        
        # Clear existing content
        for child in dep_widget.winfo_children():
            child.destroy()
            
        self.dependency_visualizer.draw(dep_widget)
        
        # Switch to dependencies tab
        self.right_notebook.select(0)
        
    def update_visualization(self):
        """Update all visualizations"""
        self.show_dependencies()
        self.update_status_tree()
        
    def monitor_system(self):
        """Monitor system metrics"""
        def update():
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            active_scripts = sum(1 for tab in self.tabs.values() 
                               if tab.script_node.status == "running")
            
            # Calculate message rate
            message_count = len(self.message_broker.message_history)
            
            # Update labels
            self.system_labels['Total CPU'].config(text=f"{cpu_percent:.1f}%")
            self.system_labels['Total Memory'].config(text=f"{memory.percent:.1f}%")
            self.system_labels['Active Scripts'].config(text=str(active_scripts))
            self.system_labels['Messages/sec'].config(text="0")  # Would need proper calculation
            
            # Schedule next update
            self.root.after(1000, update)
            
        update()
        
    def update_status_tree(self):
        """Update script status tree"""
        # Clear existing
        for item in self.status_tree.get_children():
            self.status_tree.delete(item)
            
        # Add current scripts
        for tab in self.tabs.values():
            node = tab.script_node
            self.status_tree.insert('', tk.END, values=(
                node.id,
                node.status,
                f"{node.metrics['cpu_usage']:.1f}%",
                f"{node.metrics['memory_usage']:.1f} MB",
                node.metrics['message_count']
            ))
            
        # Schedule next update
        self.root.after(1000, self.update_status_tree)
        
    def update_message_flow(self):
        """Update message flow display"""
        # Get recent messages
        recent_messages = list(self.message_broker.message_history)[-20:]
        
        self.message_flow_text.config(state='normal')
        self.message_flow_text.delete(1.0, tk.END)
        
        for msg in recent_messages:
            timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S.%f")[:-3]
            self.message_flow_text.insert(tk.END, 
                f"[{timestamp}] {msg.sender_id} → {msg.receiver_id}: {msg.payload}\n")
            
        self.message_flow_text.config(state='disabled')
        self.message_flow_text.see(tk.END)
        
        # Update count
        self.total_messages_label.config(text=str(len(self.message_broker.message_history)))
        self.message_count_label.config(text=f"Messages: {len(self.message_broker.message_history)}")
        
        # Schedule next update
        self.root.after(500, self.update_message_flow)
        
    def import_project(self):
        """Import a project configuration"""
        filename = filedialog.askopenfilename(
            title="Import Project",
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    if filename.endswith('.yaml'):
                        project = yaml.safe_load(f)
                    else:
                        project = json.load(f)
                        
                # Load scripts
                for script_config in project.get('scripts', []):
                    tab = self.create_new_tab()
                    # Load script content and configuration
                    # Implementation would load file paths, dependencies, etc.
                    
                messagebox.showinfo("Success", "Project imported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import project: {str(e)}")
                
    def export_project(self):
        """Export current project configuration"""
        filename = filedialog.asksaveasfilename(
            title="Export Project",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("YAML files", "*.yaml")]
        )
        
        if filename:
            try:
                project = {
                    'name': 'Neural Script Project',
                    'version': '1.0',
                    'scripts': []
                }
                
                for tab in self.tabs.values():
                    script_config = {
                        'id': tab.script_node.id,
                        'file_path': tab.file_path,
                        'dependencies': list(tab.script_node.dependencies),
                        'status': tab.script_node.status
                    }
                    project['scripts'].append(script_config)
                    
                with open(filename, 'w') as f:
                    if filename.endswith('.yaml'):
                        yaml.dump(project, f)
                    else:
                        json.dump(project, f, indent=2)
                        
                messagebox.showinfo("Success", "Project exported successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export project: {str(e)}")
                
    def open_message_inspector(self):
        """Open message inspector tool"""
        inspector = tk.Toplevel(self.root)
        inspector.title("Message Inspector")
        inspector.geometry("800x600")
        
        # Message history viewer with filtering
        ttk.Label(inspector, text="Message Inspector - Coming Soon").pack(expand=True)
        
    def open_profiler(self):
        """Open performance profiler"""
        profiler = tk.Toplevel(self.root)
        profiler.title("Performance Profiler")
        profiler.geometry("900x700")
        
        ttk.Label(profiler, text="Performance Profiler - Coming Soon").pack(expand=True)
        
    def open_templates(self):
        """Open script templates"""
        templates = tk.Toplevel(self.root)
        templates.title("Script Templates")
        templates.geometry("600x400")
        
        # Template categories
        notebook = ttk.Notebook(templates)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic templates
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic")
        
        templates_list = [
            ("Data Processor", self._get_data_processor_template),
            ("Message Handler", self._get_message_handler_template),
            ("Neural Node", self._get_neural_node_template),
        ]
        
        for i, (name, template_func) in enumerate(templates_list):
            ttk.Button(basic_frame, text=name, 
                      command=lambda f=template_func: self._apply_template(f())).grid(
                          row=i//3, column=i%3, padx=10, pady=10)
                          
    def _get_data_processor_template(self):
        """Get data processor template"""
        return '''# Data Processor Script
import time
import json

def process_data(data):
    """Process incoming data"""
    # Your processing logic here
    processed = data.upper() if isinstance(data, str) else str(data)
    return processed

def main():
    # Listen for messages
    while True:
        # In real implementation, this would receive from message queue
        data = input("Enter data: ")
        
        if data.lower() == 'quit':
            break
            
        # Process data
        result = process_data(data)
        
        # Send result
        comm.send_message('BROADCAST', result)
        comm.log_metric('processed_items', 1)
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
'''

    def _get_message_handler_template(self):
        """Get message handler template"""
        return '''# Message Handler Script
import json
import time
from collections import defaultdict

class MessageHandler:
    def __init__(self):
        self.message_count = defaultdict(int)
        
    def handle_message(self, message):
        """Handle incoming message"""
        sender = message.get('sender_id', 'unknown')
        payload = message.get('payload', '')
        
        # Count messages
        self.message_count[sender] += 1
        
        # Process based on message type
        if isinstance(payload, dict):
            action = payload.get('action')
            if action == 'process':
                return self.process_action(payload)
        
        return None
        
    def process_action(self, payload):
        """Process specific action"""
        # Implement your logic here
        return {'status': 'processed', 'data': payload}

def main():
    handler = MessageHandler()
    
    # Main message loop
    while True:
        # In real implementation, receive from message queue
        # For now, simulate with input
        try:
            message_str = input("Message JSON: ")
            if message_str.lower() == 'quit':
                break
                
            message = json.loads(message_str)
            result = handler.handle_message(message)
            
            if result:
                comm.send_message('BROADCAST', result)
                
        except json.JSONDecodeError:
            print("Invalid JSON")
        except Exception as e:
            comm.send_message('SYSTEM', {'error': str(e)})
            
        # Log metrics
        total_messages = sum(handler.message_count.values())
        comm.log_metric('total_messages', total_messages)

if __name__ == "__main__":
    main()
'''

    def _get_neural_node_template(self):
        """Get neural network node template"""
        return '''# Neural Network Node Script
import numpy as np
import time

class NeuralNode:
    def __init__(self, node_id, input_size=10, output_size=5):
        self.node_id = node_id
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
        self.activation_count = 0
        
    def forward(self, inputs):
        """Forward pass through the node"""
        # Linear transformation
        output = np.dot(inputs, self.weights) + self.bias
        
        # Activation (ReLU)
        output = np.maximum(0, output)
        
        self.activation_count += 1
        return output
        
    def receive_input(self, sender, data):
        """Receive input from another node"""
        if isinstance(data, list):
            inputs = np.array(data)
        else:
            inputs = data
            
        # Process through node
        output = self.forward(inputs)
        
        # Send to next nodes
        self.send_output(output)
        
    def send_output(self, output):
        """Send output to connected nodes"""
        # Convert to list for JSON serialization
        output_list = output.tolist()
        
        # Send to specific nodes or broadcast
        comm.send_message('BROADCAST', {
            'node_id': self.node_id,
            'output': output_list,
            'activation': self.activation_count
        })

def main():
    # Initialize node
    node = NeuralNode('node_1')
    
    # Main processing loop
    while True:
        # Simulate receiving input
        # In real implementation, would receive from message queue
        try:
            input_data = input("Input data (comma-separated numbers): ")
            if input_data.lower() == 'quit':
                break
                
            # Parse input
            inputs = [float(x) for x in input_data.split(',')]
            
            # Process
            node.receive_input('user', inputs)
            
            # Log metrics
            comm.log_metric('activations', node.activation_count)
            
        except ValueError:
            print("Invalid input format")
        except Exception as e:
            comm.send_message('SYSTEM', {'error': str(e)})
            
        time.sleep(0.1)

if __name__ == "__main__":
    main()
'''

    def _apply_template(self, template_code):
        """Apply template to current tab"""
        tab = self.get_current_tab()
        if tab:
            tab.editor.delete(1.0, tk.END)
            tab.editor.insert(1.0, template_code)
            
    def open_settings(self):
        """Open settings dialog"""
        settings = tk.Toplevel(self.root)
        settings.title("Settings")
        settings.geometry("600x500")
        
        notebook = ttk.Notebook(settings)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # General settings
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="General")
        
        settings_vars = {}
        
        # Auto-save
        auto_save_var = tk.BooleanVar(value=self.config.get('auto_save', True))
        ttk.Checkbutton(general_frame, text="Auto-save scripts", 
                       variable=auto_save_var).pack(anchor=tk.W, padx=10, pady=5)
        settings_vars['auto_save'] = auto_save_var
        
        # Auto-analyze
        auto_analyze_var = tk.BooleanVar(value=self.config.get('auto_analyze', True))
        ttk.Checkbutton(general_frame, text="Auto-analyze scripts", 
                       variable=auto_analyze_var).pack(anchor=tk.W, padx=10, pady=5)
        settings_vars['auto_analyze'] = auto_analyze_var
        
        # Performance monitoring
        perf_var = tk.BooleanVar(value=self.config.get('performance_monitoring', True))
        ttk.Checkbutton(general_frame, text="Enable performance monitoring", 
                       variable=perf_var).pack(anchor=tk.W, padx=10, pady=5)
        settings_vars['performance_monitoring'] = perf_var
        
        # Advanced settings
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        
        ttk.Label(advanced_frame, text="Message History Size:").pack(anchor=tk.W, padx=10, pady=5)
        history_var = tk.IntVar(value=self.config.get('message_history_size', 1000))
        ttk.Spinbox(advanced_frame, from_=100, to=10000, textvariable=history_var, 
                   width=10).pack(anchor=tk.W, padx=20)
        settings_vars['message_history_size'] = history_var
        
        # Save button
        def save_settings():
            for key, var in settings_vars.items():
                self.config[key] = var.get()
            self.save_config()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            settings.destroy()
            
        ttk.Button(settings, text="Save", command=save_settings).pack(pady=10)
        
    def open_documentation(self):
        """Open documentation"""
        messagebox.showinfo("Documentation", 
                          "Documentation is available at:\n"
                          "https://neural-script-ide.readthedocs.io")
                          
    def open_examples(self):
        """Open examples"""
        examples = tk.Toplevel(self.root)
        examples.title("Examples")
        examples.geometry("800x600")
        
        text = scrolledtext.ScrolledText(examples, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True)
        
        example_text = """Neural Script IDE Examples
=======================

1. Simple Message Passing
------------------------
Create two scripts that communicate:

Script 1 (Producer):
```python
import time
for i in range(10):
    comm.send_message('script_2', f'Data item {i}')
    time.sleep(1)
```

Script 2 (Consumer):
```python
# This script will automatically receive messages
# from Script 1 through the message broker
```

2. Data Processing Pipeline
--------------------------
Create a pipeline of scripts that process data sequentially.

3. Neural Network Simulation
---------------------------
Build a simple neural network with scripts acting as nodes.

For more examples, visit our GitHub repository.
"""
        
        text.insert(1.0, example_text)
        text.config(state='disabled')
        
    def show_about(self):
        """Show about dialog"""
        about_text = """Neural Script IDE
Version 1.0.0

A sophisticated IDE for building neural networks 
from interconnected Python scripts.

Features:
• Advanced inter-script communication
• Real-time dependency visualization  
• Comprehensive debugging tools
• Performance monitoring
• Script orchestration

© 2024 Neural Script IDE Team"""
        
        messagebox.showinfo("About Neural Script IDE", about_text)
        
    def quit_app(self):
        """Quit application"""
        # Check for running scripts
        running_scripts = [tab for tab in self.tabs.values() 
                          if tab.process and tab.process.poll() is None]
        
        if running_scripts:
            if not messagebox.askyesno("Scripts Running",
                                     f"{len(running_scripts)} scripts are still running. Quit anyway?"):
                return
                
        # Stop services
        self.message_broker.stop()
        
        # Save config
        self.save_config()
        
        self.root.quit()


def main():
    """Main entry point"""
    # Check dependencies
    missing_deps = []
    
    try:
        import networkx
    except ImportError:
        missing_deps.append("networkx")
        
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
        
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
        
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
            
    # Create and run application
    root = tk.Tk()
    app = NeuralScriptIDE(root)
    
    # Set icon if available
    try:
        if platform.system() == "Windows":
            root.iconbitmap(default='neural_ide.ico')
    except:
        pass
        
    root.mainloop()


if __name__ == "__main__":
    main()
