# === performance_profiler.py ===
"""
Advanced Performance Profiler for Neural Script IDE
Real-time performance analysis and optimization suggestions
"""

import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    message_rate: float
    latency_ms: float
    error_rate: float
    custom_metrics: Dict[str, float] = None

@dataclass
class ScriptProfile:
    """Performance profile for a script"""
    script_id: str
    metrics_history: deque
    bottlenecks: List[str]
    optimization_suggestions: List[str]
    resource_usage_pattern: str  # "cpu_intensive", "memory_intensive", "io_bound", "balanced"
    
class PerformanceProfiler:
    """Advanced performance profiling system"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.script_profiles = {}
        self.global_metrics = deque(maxlen=max_history)
        self.monitoring = False
        self.monitor_thread = None
        
        # Analysis thresholds
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 80.0,
            'latency_high': 100.0,  # ms
            'error_rate_high': 0.05,
            'message_rate_low': 1.0
        }
        
    def start_monitoring(self, scripts: Dict[str, any]):
        """Start performance monitoring"""
        self.monitoring = True
        self.scripts = scripts
        
        # Initialize profiles
        for script_id in scripts:
            if script_id not in self.script_profiles:
                self.script_profiles[script_id] = ScriptProfile(
                    script_id=script_id,
                    metrics_history=deque(maxlen=self.max_history),
                    bottlenecks=[],
                    optimization_suggestions=[],
                    resource_usage_pattern="unknown"
                )
                
        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics for each script
                for script_id, script_obj in self.scripts.items():
                    if hasattr(script_obj, 'process') and script_obj.process:
                        metrics = self._collect_metrics(script_obj)
                        if metrics:
                            self.script_profiles[script_id].metrics_history.append(metrics)
                            
                # Analyze performance
                self._analyze_performance()
                
                # Global system metrics
                global_metric = PerformanceMetric(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(interval=0.1),
                    memory_mb=psutil.virtual_memory().used / 1024 / 1024,
                    message_rate=self._calculate_global_message_rate(),
                    latency_ms=self._calculate_global_latency(),
                    error_rate=self._calculate_global_error_rate()
                )
                self.global_metrics.append(global_metric)
                
                time.sleep(1)  # Monitor interval
                
            except Exception as e:
                print(f"Monitor error: {e}")
                
    def _collect_metrics(self, script_obj) -> Optional[PerformanceMetric]:
        """Collect metrics for a single script"""
        try:
            proc = psutil.Process(script_obj.process.pid)
            
            # Basic metrics
            cpu_percent = proc.cpu_percent(interval=0.1)
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Get custom metrics from script
            custom_metrics = {}
            if hasattr(script_obj, 'script_node'):
                custom_metrics = script_obj.script_node.metrics.copy()
                
            # Calculate rates
            message_rate = custom_metrics.get('message_rate', 0)
            latency_ms = custom_metrics.get('latency_ms', 0)
            error_rate = custom_metrics.get('error_rate', 0)
            
            return PerformanceMetric(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                message_rate=message_rate,
                latency_ms=latency_ms,
                error_rate=error_rate,
                custom_metrics=custom_metrics
            )
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
            
    def _analyze_performance(self):
        """Analyze performance and identify issues"""
        for script_id, profile in self.script_profiles.items():
            if len(profile.metrics_history) < 10:
                continue
                
            # Get recent metrics
            recent_metrics = list(profile.metrics_history)[-100:]
            
            # Reset analysis
            profile.bottlenecks.clear()
            profile.optimization_suggestions.clear()
            
            # Analyze CPU usage
            avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
            max_cpu = np.max([m.cpu_percent for m in recent_metrics])
            
            if avg_cpu > self.thresholds['cpu_high']:
                profile.bottlenecks.append("High CPU usage")
                profile.optimization_suggestions.append(
                    "Consider optimizing compute-intensive operations or using multiprocessing"
                )
                
            # Analyze memory usage
            avg_memory = np.mean([m.memory_mb for m in recent_metrics])
            memory_trend = np.polyfit(range(len(recent_metrics)), 
                                    [m.memory_mb for m in recent_metrics], 1)[0]
                                    
            if memory_trend > 1.0:  # Growing by >1MB per measurement
                profile.bottlenecks.append("Memory leak detected")
                profile.optimization_suggestions.append(
                    "Check for growing data structures or unclosed resources"
                )
                
            # Analyze message latency
            if recent_metrics[-1].latency_ms > self.thresholds['latency_high']:
                profile.bottlenecks.append("High message latency")
                profile.optimization_suggestions.append(
                    "Consider batching messages or optimizing message processing"
                )
                
            # Determine resource pattern
            if avg_cpu > 70:
                profile.resource_usage_pattern = "cpu_intensive"
            elif avg_memory > 1000:  # >1GB
                profile.resource_usage_pattern = "memory_intensive"
            elif recent_metrics[-1].message_rate > 100:
                profile.resource_usage_pattern = "io_bound"
            else:
                profile.resource_usage_pattern = "balanced"
                
    def get_optimization_report(self, script_id: str) -> Dict:
        """Get detailed optimization report for a script"""
        if script_id not in self.script_profiles:
            return {"error": "Script not found"}
            
        profile = self.script_profiles[script_id]
        
        if len(profile.metrics_history) < 10:
            return {"error": "Insufficient data"}
            
        recent_metrics = list(profile.metrics_history)[-100:]
        
        report = {
            "script_id": script_id,
            "resource_pattern": profile.resource_usage_pattern,
            "bottlenecks": profile.bottlenecks,
            "suggestions": profile.optimization_suggestions,
            "metrics_summary": {
                "avg_cpu": np.mean([m.cpu_percent for m in recent_metrics]),
                "max_cpu": np.max([m.cpu_percent for m in recent_metrics]),
                "avg_memory_mb": np.mean([m.memory_mb for m in recent_metrics]),
                "max_memory_mb": np.max([m.memory_mb for m in recent_metrics]),
                "avg_latency_ms": np.mean([m.latency_ms for m in recent_metrics]),
                "total_errors": sum([m.error_rate for m in recent_metrics])
            },
            "performance_score": self._calculate_performance_score(recent_metrics)
        }
        
        return report
        
    def _calculate_performance_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics:
            return 0.0
            
        # Weighted scoring
        cpu_score = 100 - min(np.mean([m.cpu_percent for m in metrics]), 100)
        memory_score = 100 - min(np.mean([m.memory_mb for m in metrics]) / 10, 100)
        latency_score = 100 - min(np.mean([m.latency_ms for m in metrics]) / 10, 100)
        error_score = 100 - min(np.mean([m.error_rate for m in metrics]) * 1000, 100)
        
        # Weighted average
        score = (cpu_score * 0.3 + 
                memory_score * 0.3 + 
                latency_score * 0.3 + 
                error_score * 0.1)
                
        return round(score, 1)
        
    def _calculate_global_message_rate(self) -> float:
        """Calculate global message rate"""
        # Implementation would sum across all scripts
        return sum([
            profile.metrics_history[-1].message_rate 
            if profile.metrics_history else 0
            for profile in self.script_profiles.values()
        ])
        
    def _calculate_global_latency(self) -> float:
        """Calculate average global latency"""
        latencies = [
            profile.metrics_history[-1].latency_ms 
            if profile.metrics_history else 0
            for profile in self.script_profiles.values()
        ]
        return np.mean(latencies) if latencies else 0
        
    def _calculate_global_error_rate(self) -> float:
        """Calculate global error rate"""
        error_rates = [
            profile.metrics_history[-1].error_rate 
            if profile.metrics_history else 0
            for profile in self.script_profiles.values()
        ]
        return np.mean(error_rates) if error_rates else 0


class PerformanceViewer(ttk.Frame):
    """Performance visualization widget"""
    
    def __init__(self, parent, profiler: PerformanceProfiler):
        super().__init__(parent)
        self.profiler = profiler
        self.selected_script = None
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Setup performance viewer UI"""
        # Control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Script:").pack(side=tk.LEFT, padx=5)
        
        self.script_var = tk.StringVar()
        self.script_combo = ttk.Combobox(control_frame, textvariable=self.script_var,
                                        state="readonly", width=20)
        self.script_combo.pack(side=tk.LEFT, padx=5)
        self.script_combo.bind('<<ComboboxSelected>>', self.on_script_select)
        
        ttk.Button(control_frame, text="Refresh", 
                  command=self.update_display).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(control_frame, text="Export Report", 
                  command=self.export_report).pack(side=tk.LEFT, padx=5)
                  
        # Main display area
        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Real-time metrics tab
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="Real-time Metrics")
        self.setup_metrics_view(metrics_frame)
        
        # Analysis tab
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="Analysis")
        self.setup_analysis_view(analysis_frame)
        
        # Optimization tab
        optimization_frame = ttk.Frame(self.notebook)
        self.notebook.add(optimization_frame, text="Optimization")
        self.setup_optimization_view(optimization_frame)
        
    def setup_metrics_view(self, parent):
        """Setup real-time metrics view"""
        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=80)
        
        # Create subplots
        self.cpu_ax = self.figure.add_subplot(221)
        self.memory_ax = self.figure.add_subplot(222)
        self.latency_ax = self.figure.add_subplot(223)
        self.rate_ax = self.figure.add_subplot(224)
        
        # Configure axes
        for ax, title in [
            (self.cpu_ax, "CPU Usage (%)"),
            (self.memory_ax, "Memory (MB)"),
            (self.latency_ax, "Latency (ms)"),
            (self.rate_ax, "Message Rate")
        ]:
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def setup_analysis_view(self, parent):
        """Setup analysis view"""
        # Summary frame
        summary_frame = ttk.LabelFrame(parent, text="Performance Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=60)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Bottlenecks frame
        bottleneck_frame = ttk.LabelFrame(parent, text="Identified Bottlenecks", padding=10)
        bottleneck_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bottleneck_list = tk.Listbox(bottleneck_frame, height=10)
        self.bottleneck_list.pack(fill=tk.BOTH, expand=True)
        
    def setup_optimization_view(self, parent):
        """Setup optimization suggestions view"""
        # Suggestions text
        self.suggestions_text = tk.Text(parent, wrap=tk.WORD, padx=10, pady=10)
        self.suggestions_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure tags
        self.suggestions_text.tag_config("header", font=("Arial", 12, "bold"))
        self.suggestions_text.tag_config("suggestion", font=("Arial", 10), 
                                       foreground="blue")
                                       
    def update_display(self):
        """Update all displays"""
        # Update script list
        scripts = list(self.profiler.script_profiles.keys())
        self.script_combo['values'] = scripts
        
        if self.selected_script:
            self.update_metrics()
            self.update_analysis()
            self.update_optimization()
            
        # Schedule next update
        self.after(1000, self.update_display)
        
    def on_script_select(self, event):
        """Handle script selection"""
        self.selected_script = self.script_var.get()
        self.update_metrics()
        self.update_analysis()
        self.update_optimization()
        
    def update_metrics(self):
        """Update metrics graphs"""
        if not self.selected_script:
            return
            
        profile = self.profiler.script_profiles.get(self.selected_script)
        if not profile or len(profile.metrics_history) < 2:
            return
            
        # Get data
        metrics = list(profile.metrics_history)
        times = [(m.timestamp - metrics[0].timestamp) for m in metrics]
        
        # Clear and update plots
        for ax in [self.cpu_ax, self.memory_ax, self.latency_ax, self.rate_ax]:
            ax.clear()
            
        # CPU plot
        self.cpu_ax.plot(times, [m.cpu_percent for m in metrics], 'b-')
        self.cpu_ax.set_title("CPU Usage (%)")
        self.cpu_ax.set_ylim(0, 100)
        
        # Memory plot
        self.memory_ax.plot(times, [m.memory_mb for m in metrics], 'g-')
        self.memory_ax.set_title("Memory (MB)")
        
        # Latency plot
        self.latency_ax.plot(times, [m.latency_ms for m in metrics], 'r-')
        self.latency_ax.set_title("Latency (ms)")
        
        # Message rate plot
        self.rate_ax.plot(times, [m.message_rate for m in metrics], 'm-')
        self.rate_ax.set_title("Message Rate")
        
        # Add grid
        for ax in [self.cpu_ax, self.memory_ax, self.latency_ax, self.rate_ax]:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (s)")
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_analysis(self):
        """Update analysis display"""
        if not self.selected_script:
            return
            
        report = self.profiler.get_optimization_report(self.selected_script)
        
        if "error" in report:
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(1.0, report["error"])
            return
            
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        summary = f"""Script: {report['script_id']}
Resource Pattern: {report['resource_pattern']}
Performance Score: {report['performance_score']}/100

Metrics Summary:
- Average CPU: {report['metrics_summary']['avg_cpu']:.1f}%
- Maximum CPU: {report['metrics_summary']['max_cpu']:.1f}%
- Average Memory: {report['metrics_summary']['avg_memory_mb']:.1f} MB
- Maximum Memory: {report['metrics_summary']['max_memory_mb']:.1f} MB
- Average Latency: {report['metrics_summary']['avg_latency_ms']:.1f} ms
- Total Errors: {report['metrics_summary']['total_errors']}
"""
        self.summary_text.insert(1.0, summary)
        
        # Update bottlenecks
        self.bottleneck_list.delete(0, tk.END)
        for bottleneck in report['bottlenecks']:
            self.bottleneck_list.insert(tk.END, f"⚠️ {bottleneck}")
            
    def update_optimization(self):
        """Update optimization suggestions"""
        if not self.selected_script:
            return
            
        report = self.profiler.get_optimization_report(self.selected_script)
        
        if "error" in report:
            return
            
        # Clear and update suggestions
        self.suggestions_text.delete(1.0, tk.END)
        
        self.suggestions_text.insert(tk.END, "Optimization Suggestions\n\n", "header")
        
        for i, suggestion in enumerate(report['suggestions'], 1):
            self.suggestions_text.insert(tk.END, f"{i}. {suggestion}\n\n", "suggestion")
            
        # Add resource-specific suggestions
        if report['resource_pattern'] == "cpu_intensive":
            self.suggestions_text.insert(tk.END, "\nCPU Optimization Tips:\n", "header")
            self.suggestions_text.insert(tk.END, 
                "• Use NumPy for numerical computations\n"
                "• Consider Cython for performance-critical code\n"
                "• Implement caching for repeated calculations\n"
                "• Use multiprocessing for parallel tasks\n", "suggestion")
                
        elif report['resource_pattern'] == "memory_intensive":
            self.suggestions_text.insert(tk.END, "\nMemory Optimization Tips:\n", "header")
            self.suggestions_text.insert(tk.END,
                "• Use generators instead of lists for large datasets\n"
                "• Implement data streaming instead of loading all at once\n"
                "• Clear unused variables explicitly\n"
                "• Consider using memory-mapped files\n", "suggestion")
                
    def export_report(self):
        """Export performance report"""
        if not self.selected_script:
            return
            
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            report = self.profiler.get_optimization_report(self.selected_script)
            
            # Add timestamp
            report['exported_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            tk.messagebox.showinfo("Export Complete", f"Report saved to {filename}")


# === message_debugger.py ===
"""
Message Debugger for Neural Script IDE
Advanced message inspection and debugging tools
"""

import json
import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple
import tkinter as tk
from tkinter import ttk, messagebox
import re

class MessageFilter:
    """Filter for messages"""
    def __init__(self):
        self.sender_pattern = None
        self.receiver_pattern = None
        self.message_type = None
        self.payload_pattern = None
        self.time_range = None
        
    def matches(self, message) -> bool:
        """Check if message matches filter criteria"""
        # Sender filter
        if self.sender_pattern:
            if not re.match(self.sender_pattern, message.sender_id):
                return False
                
        # Receiver filter
        if self.receiver_pattern:
            if not re.match(self.receiver_pattern, message.receiver_id):
                return False
                
        # Type filter
        if self.message_type:
            if message.message_type != self.message_type:
                return False
                
        # Payload filter
        if self.payload_pattern:
            payload_str = str(message.payload)
            if not re.search(self.payload_pattern, payload_str):
                return False
                
        # Time filter
        if self.time_range:
            start, end = self.time_range
            if not (start <= message.timestamp <= end):
                return False
                
        return True

class MessageDebugger:
    """Advanced message debugging system"""
    
    def __init__(self, message_broker):
        self.message_broker = message_broker
        self.breakpoints = []
        self.message_log = deque(maxlen=10000)
        self.statistics = defaultdict(lambda: defaultdict(int))
        self.paused_messages = []
        self.debug_mode = False
        
    def add_breakpoint(self, filter_: MessageFilter):
        """Add message breakpoint"""
        self.breakpoints.append(filter_)
        
    def remove_breakpoint(self, index: int):
        """Remove breakpoint by index"""
        if 0 <= index < len(self.breakpoints):
            del self.breakpoints[index]
            
    def intercept_message(self, message) -> bool:
        """Intercept message for debugging"""
        # Log message
        self.message_log.append(message)
        
        # Update statistics
        self.statistics[message.sender_id]['sent'] += 1
        self.statistics[message.receiver_id]['received'] += 1
        self.statistics['_global']['total'] += 1
        
        # Check breakpoints
        if self.debug_mode:
            for bp in self.breakpoints:
                if bp.matches(message):
                    self.paused_messages.append(message)
                    return True  # Pause message delivery
                    
        return False  # Continue normal delivery
        
    def get_message_flow(self, start_time: float = None, 
                        end_time: float = None) -> List[Tuple[str, str, float]]:
        """Get message flow for visualization"""
        flow = []
        
        for msg in self.message_log:
            if start_time and msg.timestamp < start_time:
                continue
            if end_time and msg.timestamp > end_time:
                continue
                
            flow.append((msg.sender_id, msg.receiver_id, msg.timestamp))
            
        return flow
        
    def get_statistics(self) -> Dict:
        """Get message statistics"""
        return {
            'total_messages': self.statistics['_global']['total'],
            'by_sender': dict(self.statistics),
            'message_rate': self._calculate_message_rate(),
            'average_size': self._calculate_average_size(),
            'peak_rate': self._calculate_peak_rate()
        }
        
    def _calculate_message_rate(self) -> float:
        """Calculate current message rate"""
        if len(self.message_log) < 2:
            return 0.0
            
        recent = list(self.message_log)[-100:]
        time_span = recent[-1].timestamp - recent[0].timestamp
        
        if time_span > 0:
            return len(recent) / time_span
        return 0.0
        
    def _calculate_average_size(self) -> float:
        """Calculate average message payload size"""
        if not self.message_log:
            return 0.0
            
        sizes = []
        for msg in list(self.message_log)[-100:]:
            sizes.append(len(json.dumps(msg.payload)))
            
        return sum(sizes) / len(sizes) if sizes else 0.0
        
    def _calculate_peak_rate(self) -> float:
        """Calculate peak message rate"""
        # Implementation for peak rate calculation
        return 0.0

class MessageDebuggerUI(ttk.Frame):
    """Message debugger UI"""
    
    def __init__(self, parent, debugger: MessageDebugger):
        super().__init__(parent)
        self.debugger = debugger
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Setup debugger UI"""
        # Control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.debug_var = tk.BooleanVar(value=self.debugger.debug_mode)
        ttk.Checkbutton(control_frame, text="Debug Mode", 
                       variable=self.debug_var,
                       command=self.toggle_debug).pack(side=tk.LEFT, padx=5)
                       
        ttk.Button(control_frame, text="Clear Log",
                  command=self.clear_log).pack(side=tk.LEFT, padx=5)
                  
        ttk.Button(control_frame, text="Add Breakpoint",
                  command=self.add_breakpoint).pack(side=tk.LEFT, padx=5)
                  
        # Filter controls
        filter_frame = ttk.LabelFrame(control_frame, text="Filter", padding=5)
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.filter_sender = tk.StringVar()
        ttk.Label(filter_frame, text="Sender:").pack(side=tk.LEFT)
        ttk.Entry(filter_frame, textvariable=self.filter_sender, 
                 width=15).pack(side=tk.LEFT, padx=2)
                 
        self.filter_type = tk.StringVar()
        ttk.Label(filter_frame, text="Type:").pack(side=tk.LEFT)
        ttk.Combobox(filter_frame, textvariable=self.filter_type,
                    values=["all", "data", "control", "status", "error"],
                    width=10).pack(side=tk.LEFT, padx=2)
                    
        ttk.Button(filter_frame, text="Apply",
                  command=self.apply_filter).pack(side=tk.LEFT, padx=5)
                  
        # Main display
        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create paned window
        paned = ttk.PanedWindow(display_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Message log
        log_frame = ttk.LabelFrame(paned, text="Message Log", padding=5)
        paned.add(log_frame, weight=2)
        
        # Create treeview for messages
        columns = ('time', 'sender', 'receiver', 'type', 'size', 'preview')
        self.message_tree = ttk.Treeview(log_frame, columns=columns, show='headings')
        
        # Configure columns
        self.message_tree.heading('time', text='Time')
        self.message_tree.heading('sender', text='Sender')
        self.message_tree.heading('receiver', text='Receiver')
        self.message_tree.heading('type', text='Type')
        self.message_tree.heading('size', text='Size')
        self.message_tree.heading('preview', text='Preview')
        
        self.message_tree.column('time', width=80)
        self.message_tree.column('sender', width=100)
        self.message_tree.column('receiver', width=100)
        self.message_tree.column('type', width=60)
        self.message_tree.column('size', width=60)
        self.message_tree.column('preview', width=200)
        
        self.message_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL,
                                 command=self.message_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.message_tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind selection
        self.message_tree.bind('<<TreeviewSelect>>', self.on_message_select)
        
        # Details panel
        details_frame = ttk.LabelFrame(paned, text="Message Details", padding=5)
        paned.add(details_frame, weight=1)
        
        self.details_text = tk.Text(details_frame, wrap=tk.WORD)
        self.details_text.pack(fill=tk.BOTH, expand=True)
        
        # Breakpoints panel
        bp_frame = ttk.LabelFrame(display_frame, text="Breakpoints", padding=5)
        bp_frame.pack(fill=tk.X, pady=5)
        
        self.bp_listbox = tk.Listbox(bp_frame, height=3)
        self.bp_listbox.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        ttk.Button(bp_frame, text="Remove",
                  command=self.remove_breakpoint).pack(side=tk.LEFT, padx=5)
                  
    def toggle_debug(self):
        """Toggle debug mode"""
        self.debugger.debug_mode = self.debug_var.get()
        
    def clear_log(self):
        """Clear message log"""
        self.debugger.message_log.clear()
        self.update_display()
        
    def add_breakpoint(self):
        """Add new breakpoint"""
        dialog = BreakpointDialog(self)
        if dialog.result:
            self.debugger.add_breakpoint(dialog.result)
            self.update_breakpoints()
            
    def remove_breakpoint(self):
        """Remove selected breakpoint"""
        selection = self.bp_listbox.curselection()
        if selection:
            self.debugger.remove_breakpoint(selection[0])
            self.update_breakpoints()
            
    def apply_filter(self):
        """Apply display filter"""
        self.update_display()
        
    def update_display(self):
        """Update message display"""
        # Clear tree
        for item in self.message_tree.get_children():
            self.message_tree.delete(item)
            
        # Get filter criteria
        sender_filter = self.filter_sender.get()
        type_filter = self.filter_type.get()
        
        # Add messages
        for msg in reversed(list(self.debugger.message_log)):
            # Apply filters
            if sender_filter and sender_filter not in msg.sender_id:
                continue
            if type_filter != "all" and msg.message_type.value != type_filter:
                continue
                
            # Format message
            time_str = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S.%f")[:-3]
            size = len(json.dumps(msg.payload))
            preview = str(msg.payload)[:100]
            
            self.message_tree.insert('', 0, values=(
                time_str,
                msg.sender_id,
                msg.receiver_id,
                msg.message_type.value,
                size,
                preview
            ))
            
        # Update statistics
        self.update_statistics()
        
        # Schedule next update
        self.after(500, self.update_display)
        
    def update_breakpoints(self):
        """Update breakpoints display"""
        self.bp_listbox.delete(0, tk.END)
        
        for i, bp in enumerate(self.debugger.breakpoints):
            desc = f"BP{i}: "
            if bp.sender_pattern:
                desc += f"sender={bp.sender_pattern} "
            if bp.receiver_pattern:
                desc += f"receiver={bp.receiver_pattern} "
            if bp.message_type:
                desc += f"type={bp.message_type} "
                
            self.bp_listbox.insert(tk.END, desc)
            
    def update_statistics(self):
        """Update statistics display"""
        stats = self.debugger.get_statistics()
        
        # Update status bar or statistics panel
        # Implementation depends on UI layout
        
    def on_message_select(self, event):
        """Handle message selection"""
        selection = self.message_tree.selection()
        if not selection:
            return
            
        # Get selected message index
        item = self.message_tree.item(selection[0])
        values = item['values']
        
        # Find corresponding message
        # This is simplified - in real implementation would use message ID
        time_str = values[0]
        
        # Display full message details
        self.details_text.delete(1.0, tk.END)
        
        # Find message by matching fields
        for msg in self.debugger.message_log:
            msg_time = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S.%f")[:-3]
            if msg_time == time_str:
                # Format message details
                details = f"""Timestamp: {msg.timestamp}
Time: {datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")}
Sender: {msg.sender_id}
Receiver: {msg.receiver_id}
Type: {msg.message_type.value}

Payload:
{json.dumps(msg.payload, indent=2)}
"""
                self.details_text.insert(1.0, details)
                break

class BreakpointDialog:
    """Dialog for creating message breakpoints"""
    
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Add Breakpoint")
        self.dialog.geometry("400x300")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup dialog UI"""
        # Sender pattern
        ttk.Label(self.dialog, text="Sender Pattern (regex):").grid(
            row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.sender_entry = ttk.Entry(self.dialog, width=30)
        self.sender_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Receiver pattern
        ttk.Label(self.dialog, text="Receiver Pattern (regex):").grid(
            row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.receiver_entry = ttk.Entry(self.dialog, width=30)
        self.receiver_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Message type
        ttk.Label(self.dialog, text="Message Type:").grid(
            row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.type_var = tk.StringVar(value="any")
        type_combo = ttk.Combobox(self.dialog, textvariable=self.type_var,
                                 values=["any", "data", "control", "status", "error"],
                                 width=28)
        type_combo.grid(row=2, column=1, padx=10, pady=5)
        
        # Payload pattern
        ttk.Label(self.dialog, text="Payload Pattern (regex):").grid(
            row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.payload_entry = ttk.Entry(self.dialog, width=30)
        self.payload_entry.grid(row=3, column=1, padx=10, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="OK", command=self.ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
                  
    def ok(self):
        """Create breakpoint from inputs"""
        filter_ = MessageFilter()
        
        # Set patterns
        if self.sender_entry.get():
            filter_.sender_pattern = self.sender_entry.get()
        if self.receiver_entry.get():
            filter_.receiver_pattern = self.receiver_entry.get()
        if self.type_var.get() != "any":
            filter_.message_type = self.type_var.get()
        if self.payload_entry.get():
            filter_.payload_pattern = self.payload_entry.get()
            
        self.result = filter_
        self.dialog.destroy()


# Example usage
if __name__ == "__main__":
    # Test performance profiler
    profiler = PerformanceProfiler()
    
    # Simulate some metrics
    test_metric = PerformanceMetric(
        timestamp=time.time(),
        cpu_percent=45.2,
        memory_mb=256.8,
        message_rate=150.5,
        latency_ms=23.4,
        error_rate=0.01
    )
    
    print("Performance metric created:", test_metric)
