"""
Weaver Tools: Advanced analysis tools for Neural Weaver.
This module contains the backend logic for performance profiling and message
debugging, adapted from the original 'neural_ide_tools.py'. The UI components
have been removed, as they will be built directly into the main application's
inspector panel for a more integrated experience.
"""

import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import json
import re

# --- Performance Profiler ---

@dataclass
class PerformanceMetrics:
    """A single snapshot of performance measurements for a Block."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    # Custom metrics can be logged from the Block's code
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class BlockProfile:
    """Holds the complete performance profile for a single Block."""
    block_id: str
    # Use a deque for efficient, fixed-size history
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=300)) # 5 minutes of data
    bottlenecks: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    # Describes how the block uses resources
    usage_pattern: str = "Balanced" # e.g., "CPU Intensive", "Memory Heavy"

class PerformanceProfiler:
    """
    The backend engine for monitoring the performance of all active blocks.
    It runs in a separate thread to avoid blocking the UI.
    """
    
    def __init__(self):
        self.block_profiles: Dict[str, BlockProfile] = {}
        self.monitored_processes: Dict[str, psutil.Process] = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Configurable thresholds for analysis
        self.thresholds = {
            'cpu_high': 80.0,      # percent
            'memory_high': 1024.0, # MB
            'memory_leak_slope': 0.5 # MB increase per second
        }

    def start_monitoring(self):
        """Starts the performance monitoring thread."""
        if self.is_monitoring:
            return
        print("Starting performance profiler...")
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stops the performance monitoring thread."""
        print("Stopping performance profiler...")
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

    def add_block(self, block_id: str, process: psutil.Process):
        """Registers a new block and its process to be monitored."""
        if block_id not in self.block_profiles:
            self.block_profiles[block_id] = BlockProfile(block_id=block_id)
        self.monitored_processes[block_id] = process
        print(f"Profiler now tracking block: {block_id} (PID: {process.pid})")

    def remove_block(self, block_id: str):
        """Stops monitoring a block, typically when its process ends."""
        if block_id in self.monitored_processes:
            del self.monitored_processes[block_id]
        # We keep the profile data for later analysis
        print(f"Profiler stopped tracking block: {block_id}")

    def _monitor_loop(self):
        """The main loop that periodically collects and analyzes metrics."""
        while self.is_monitoring:
            try:
                # Iterate over a copy of keys to allow modification during loop
                for block_id in list(self.monitored_processes.keys()):
                    process = self.monitored_processes.get(block_id)
                    if process and process.is_running():
                        metrics = self._collect_metrics(process)
                        profile = self.block_profiles[block_id]
                        profile.metrics_history.append(metrics)
                        # Analyze performance every few seconds
                        if len(profile.metrics_history) % 5 == 0:
                            self._analyze_profile(profile)
                    else:
                        # Process has stopped, remove it from monitoring
                        self.remove_block(block_id)
                time.sleep(1) # Collection interval
            except Exception as e:
                print(f"Error in profiler loop: {e}")

    def _collect_metrics(self, process: psutil.Process) -> PerformanceMetrics:
        """Collects CPU and Memory metrics for a single process."""
        try:
            with process.oneshot():
                cpu = process.cpu_percent(interval=0.1)
                mem = process.memory_info().rss / (1024 * 1024) # to MB
                return PerformanceMetrics(cpu_percent=cpu, memory_mb=mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # This can happen if the process ends between checks
            return PerformanceMetrics()

    def _analyze_profile(self, profile: BlockProfile):
        """Analyzes a block's recent performance history to find issues."""
        if len(profile.metrics_history) < 10:
            return # Not enough data to analyze

        history = list(profile.metrics_history)
        recent_metrics = history[-60:] # Analyze last minute
        
        # Reset analysis results
        profile.bottlenecks.clear()
        profile.suggestions.clear()
        
        # --- Analysis Logic ---
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_mem = np.mean([m.memory_mb for m in recent_metrics])

        # High CPU usage
        if avg_cpu > self.thresholds['cpu_high']:
            profile.bottlenecks.append(f"High CPU Usage (avg {avg_cpu:.1f}%)")
            profile.suggestions.append("The code is very compute-intensive. Consider optimizing algorithms or using more efficient libraries like NumPy.")
        
        # High Memory usage
        if avg_mem > self.thresholds['memory_high']:
            profile.bottlenecks.append(f"High Memory Usage (avg {avg_mem:.1f} MB)")
            profile.suggestions.append("The block is using a lot of memory. Try processing data in smaller chunks or streams.")
            
        # Memory Leak Detection
        if len(recent_metrics) > 30:
            timestamps = np.array([m.timestamp for m in recent_metrics])
            memory_points = np.array([m.memory_mb for m in recent_metrics])
            # Fit a line to the memory usage over time
            slope, _ = np.polyfit(timestamps, memory_points, 1)
            if slope > self.thresholds['memory_leak_slope']:
                profile.bottlenecks.append(f"Potential Memory Leak (growing at {slope:.2f} MB/s)")
                profile.suggestions.append("Memory usage is consistently increasing. Check for data being appended to lists in a loop without being cleared.")

        # Determine usage pattern
        if avg_cpu > 60:
            profile.usage_pattern = "CPU Intensive"
        elif avg_mem > 500:
            profile.usage_pattern = "Memory Heavy"
        else:
            profile.usage_pattern = "Balanced"

    def get_profile(self, block_id: str) -> Optional[BlockProfile]:
        """Returns the full performance profile for a given block."""
        return self.block_profiles.get(block_id)

# --- Message Debugger ---

@dataclass
class Message:
    """Represents a single message passed between Blocks."""
    sender_id: str
    receiver_id: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self):
        """Serializes the message for transport or logging."""
        return json.dumps({
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "payload": self.payload,
            "timestamp": self.timestamp
        })

class MessageDebugger:
    """
    A backend service to log, inspect, and manage the flow of messages
    between all blocks in a flow.
    """
    def __init__(self, max_log_size: int = 5000):
        self.message_log: deque = deque(maxlen=max_log_size)
        self.stats = defaultdict(lambda: {"sent": 0, "received": 0})
    
    def log_message(self, message: Message):
        """Logs a message and updates statistics."""
        self.message_log.append(message)
        self.stats[message.sender_id]["sent"] += 1
        if message.receiver_id != "BROADCAST":
            self.stats[message.receiver_id]["received"] += 1
        else:
            # In a broadcast, we can't easily track all receivers here,
            # but the UI could show this.
            pass

    def get_messages(self, filter_text: str = "") -> List[Message]:
        """
        Returns a list of logged messages, optionally filtered by a string
        that can match sender, receiver, or payload content.
        """
        if not filter_text:
            return list(self.message_log)

        filtered_messages = []
        try:
            # Support simple text search and basic regex
            filter_regex = re.compile(filter_text, re.IGNORECASE)
            for msg in self.message_log:
                # Search in sender, receiver, and string representation of payload
                if filter_regex.search(msg.sender_id) or \
                   filter_regex.search(msg.receiver_id) or \
                   filter_regex.search(str(msg.payload)):
                    filtered_messages.append(msg)
        except re.error:
            # If regex is invalid, fall back to simple string contains
            for msg in self.message_log:
                 if filter_text.lower() in msg.sender_id.lower() or \
                    filter_text.lower() in msg.receiver_id.lower() or \
                    filter_text.lower() in str(msg.payload).lower():
                     filtered_messages.append(msg)
                     
        return filtered_messages

    def get_stats(self) -> Dict:
        """Returns statistics about message traffic."""
        return {
            "total_messages": len(self.message_log),
            "block_stats": dict(self.stats)
        }

    def clear(self):
        """Clears the message log and statistics."""
        self.message_log.clear()
        self.stats.clear()

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Testing Weaver Tools ---")

    # --- Profiler Test ---
    print("\n--- Performance Profiler Test ---")
    profiler = PerformanceProfiler()
    profiler.start_monitoring()
    
    # Simulate a running process (in a real scenario, this would be a subprocess)
    # For this test, we monitor the current python process
    try:
        current_process = psutil.Process(os.getpid())
        profiler.add_block("test_block_1", current_process)
        
        print("Monitoring current process for 5 seconds...")
        time.sleep(5)
        
        profile = profiler.get_profile("test_block_1")
        if profile and profile.metrics_history:
            print(f"Collected {len(profile.metrics_history)} metric snapshots.")
            last_metric = profile.metrics_history[-1]
            print(f"Last CPU: {last_metric.cpu_percent:.2f}%, Last Memory: {last_metric.memory_mb:.2f} MB")
            print(f"Identified usage pattern: {profile.usage_pattern}")
        else:
            print("Failed to collect metrics.")
            
    except Exception as e:
        print(f"Could not run profiler test: {e}")
    finally:
        profiler.stop_monitoring()


    # --- Message Debugger Test ---
    print("\n--- Message Debugger Test ---")
    debugger = MessageDebugger()
    
    # Simulate some message traffic
    debugger.log_message(Message("block_A", "block_B", {"data": "hello world"}))
    debugger.log_message(Message("block_B", "block_C", {"value": 123}))
    debugger.log_message(Message("block_A", "BROADCAST", {"alert": "system update"}))
    
    all_msgs = debugger.get_messages()
    print(f"Logged {len(all_msgs)} messages.")
    
    filtered_msgs = debugger.get_messages(filter_text="block_A")
    print(f"Found {len(filtered_msgs)} messages from 'block_A'.")

    stats = debugger.get_stats()
    print("Message Stats:")
    import json
    print(json.dumps(stats, indent=2))
