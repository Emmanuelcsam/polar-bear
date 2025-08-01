#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-time Monitoring System for Fiber CNN Training
Provides live metrics, GPU monitoring, and visualization during training
"""

import os
import sys
import time
import json
import threading
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import psutil
import GPUtil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import queue
import signal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Data class for storing training metrics"""
    epoch: int = 0
    batch: int = 0
    total_batches: int = 0
    loss: float = 0.0
    zone_loss: float = 0.0
    defect_loss: float = 0.0
    quality_loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    batch_time: float = 0.0
    data_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'epoch': self.epoch,
            'batch': self.batch,
            'total_batches': self.total_batches,
            'loss': self.loss,
            'zone_loss': self.zone_loss,
            'defect_loss': self.defect_loss,
            'quality_loss': self.quality_loss,
            'learning_rate': self.learning_rate,
            'gpu_memory_used': self.gpu_memory_used,
            'gpu_memory_total': self.gpu_memory_total,
            'gpu_utilization': self.gpu_utilization,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'batch_time': self.batch_time,
            'data_time': self.data_time,
            'timestamp': self.timestamp.isoformat()
        }

class SystemMonitor:
    """Monitor system resources in real-time"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        metrics = {}
        
        # CPU and Memory
        metrics['cpu_utilization'] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        metrics['memory_utilization'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_total_gb'] = memory.total / (1024**3)
        
        # GPU metrics
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics['gpu_utilization'] = gpu.load * 100
                    metrics['gpu_memory_used'] = gpu.memoryUsed
                    metrics['gpu_memory_total'] = gpu.memoryTotal
                    metrics['gpu_temperature'] = gpu.temperature
                else:
                    # Fallback to torch.cuda
                    if torch.cuda.is_available():
                        metrics['gpu_utilization'] = 0.0  # Not easily accessible via torch
                        metrics['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
                        metrics['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        metrics['gpu_temperature'] = 0.0
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
                metrics.update({
                    'gpu_utilization': 0.0,
                    'gpu_memory_used': 0.0,
                    'gpu_memory_total': 0.0,
                    'gpu_temperature': 0.0
                })
        else:
            metrics.update({
                'gpu_utilization': 0.0,
                'gpu_memory_used': 0.0,
                'gpu_memory_total': 0.0,
                'gpu_temperature': 0.0
            })
        
        return metrics

class MetricsCollector:
    """Collect and store training metrics"""
    
    def __init__(self, log_dir: str = "logs/monitoring"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_queue = queue.Queue()
        self.metrics_history: List[TrainingMetrics] = []
        self.system_monitor = SystemMonitor()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def add_metrics(self, metrics: TrainingMetrics):
        """Add new metrics to the queue"""
        self.metrics_queue.put(metrics)
        
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Get system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                
                # Process queued metrics
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    
                    # Add system metrics
                    metrics.cpu_utilization = system_metrics['cpu_utilization']
                    metrics.memory_utilization = system_metrics['memory_utilization']
                    metrics.gpu_utilization = system_metrics['gpu_utilization']
                    metrics.gpu_memory_used = system_metrics['gpu_memory_used']
                    metrics.gpu_memory_total = system_metrics['gpu_memory_total']
                    
                    # Store in history
                    self.metrics_history.append(metrics)
                    
                    # Write to TensorBoard
                    self._write_to_tensorboard(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _write_to_tensorboard(self, metrics: TrainingMetrics):
        """Write metrics to TensorBoard"""
        step = metrics.epoch * metrics.total_batches + metrics.batch
        
        # Losses
        self.writer.add_scalar('Loss/Total', metrics.loss, step)
        self.writer.add_scalar('Loss/Zone', metrics.zone_loss, step)
        self.writer.add_scalar('Loss/Defect', metrics.defect_loss, step)
        self.writer.add_scalar('Loss/Quality', metrics.quality_loss, step)
        
        # Learning rate
        self.writer.add_scalar('Training/Learning_Rate', metrics.learning_rate, step)
        
        # System metrics
        self.writer.add_scalar('System/CPU_Utilization', metrics.cpu_utilization, step)
        self.writer.add_scalar('System/Memory_Utilization', metrics.memory_utilization, step)
        self.writer.add_scalar('System/GPU_Utilization', metrics.gpu_utilization, step)
        self.writer.add_scalar('System/GPU_Memory_Used_GB', metrics.gpu_memory_used, step)
        self.writer.add_scalar('System/GPU_Memory_Total_GB', metrics.gpu_memory_total, step)
        
        # Timing
        self.writer.add_scalar('Timing/Batch_Time', metrics.batch_time, step)
        self.writer.add_scalar('Timing/Data_Time', metrics.data_time, step)
        
        self.writer.flush()
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get the latest metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, window_size: int = 100) -> List[TrainingMetrics]:
        """Get recent metrics history"""
        return self.metrics_history[-window_size:] if self.metrics_history else []
    
    def save_metrics(self, filename: str = None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        metrics_data = [metrics.to_dict() for metrics in self.metrics_history]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {filepath}")
        return filepath

class RealTimeVisualizer:
    """Real-time visualization of training metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.root = tk.Tk()
        self.root.title("Fiber CNN Training Monitor")
        self.root.geometry("1200x800")
        
        # Create figure and canvas
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self._setup_plots()
        
        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, interval=1000, blit=False
        )
        
        # Add control panel
        self._setup_control_panel()
        
    def _setup_plots(self):
        """Setup the plots"""
        self.axes[0, 0].set_title("Training Loss")
        self.axes[0, 0].set_ylabel("Loss")
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title("System Resources")
        self.axes[0, 1].set_ylabel("Utilization (%)")
        self.axes[0, 1].grid(True)
        
        self.axes[0, 2].set_title("GPU Memory")
        self.axes[0, 2].set_ylabel("Memory (GB)")
        self.axes[0, 2].grid(True)
        
        self.axes[1, 0].set_title("Learning Rate")
        self.axes[1, 0].set_ylabel("LR")
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title("Batch Timing")
        self.axes[1, 1].set_ylabel("Time (s)")
        self.axes[1, 1].grid(True)
        
        self.axes[1, 2].set_title("Progress")
        self.axes[1, 2].set_ylabel("Progress (%)")
        self.axes[1, 2].grid(True)
        
        plt.tight_layout()
    
    def _update_plots(self, frame):
        """Update all plots with latest data"""
        metrics_history = self.metrics_collector.get_metrics_history(100)
        if not metrics_history:
            return
        
        # Clear all plots
        for ax in self.axes.flat:
            ax.clear()
            ax.grid(True)
        
        # Extract data
        steps = [m.epoch * m.total_batches + m.batch for m in metrics_history]
        losses = [m.loss for m in metrics_history]
        zone_losses = [m.zone_loss for m in metrics_history]
        defect_losses = [m.defect_loss for m in metrics_history]
        quality_losses = [m.quality_loss for m in metrics_history]
        
        cpu_utils = [m.cpu_utilization for m in metrics_history]
        gpu_utils = [m.gpu_utilization for m in metrics_history]
        memory_utils = [m.memory_utilization for m in metrics_history]
        
        gpu_memory_used = [m.gpu_memory_used for m in metrics_history]
        gpu_memory_total = [m.gpu_memory_total for m in metrics_history]
        
        lrs = [m.learning_rate for m in metrics_history]
        batch_times = [m.batch_time for m in metrics_history]
        data_times = [m.data_time for m in metrics_history]
        
        # Plot 1: Training Loss
        self.axes[0, 0].plot(steps, losses, 'b-', label='Total Loss', linewidth=2)
        self.axes[0, 0].plot(steps, zone_losses, 'r-', label='Zone Loss', alpha=0.7)
        self.axes[0, 0].plot(steps, defect_losses, 'g-', label='Defect Loss', alpha=0.7)
        self.axes[0, 0].plot(steps, quality_losses, 'y-', label='Quality Loss', alpha=0.7)
        self.axes[0, 0].set_title("Training Loss")
        self.axes[0, 0].legend()
        
        # Plot 2: System Resources
        self.axes[0, 1].plot(steps, cpu_utils, 'b-', label='CPU', linewidth=2)
        self.axes[0, 1].plot(steps, gpu_utils, 'r-', label='GPU', linewidth=2)
        self.axes[0, 1].plot(steps, memory_utils, 'g-', label='Memory', linewidth=2)
        self.axes[0, 1].set_title("System Resources")
        self.axes[0, 1].legend()
        
        # Plot 3: GPU Memory
        self.axes[0, 2].plot(steps, gpu_memory_used, 'r-', label='Used', linewidth=2)
        self.axes[0, 2].plot(steps, gpu_memory_total, 'b--', label='Total', linewidth=2)
        self.axes[0, 2].set_title("GPU Memory")
        self.axes[0, 2].legend()
        
        # Plot 4: Learning Rate
        self.axes[1, 0].plot(steps, lrs, 'b-', linewidth=2)
        self.axes[1, 0].set_title("Learning Rate")
        
        # Plot 5: Batch Timing
        self.axes[1, 1].plot(steps, batch_times, 'b-', label='Batch Time', linewidth=2)
        self.axes[1, 1].plot(steps, data_times, 'r-', label='Data Time', linewidth=2)
        self.axes[1, 1].set_title("Batch Timing")
        self.axes[1, 1].legend()
        
        # Plot 6: Progress
        if metrics_history:
            latest = metrics_history[-1]
            progress = (latest.epoch * latest.total_batches + latest.batch) / (latest.total_batches * 50) * 100  # Assuming 50 epochs
            self.axes[1, 2].bar(['Progress'], [progress], color='green')
            self.axes[1, 2].set_ylim(0, 100)
            self.axes[1, 2].set_title("Training Progress")
        
        plt.tight_layout()
    
    def _setup_control_panel(self):
        """Setup control panel"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Save metrics button
        save_btn = ttk.Button(control_frame, text="Save Metrics", 
                             command=self._save_metrics)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Monitoring...")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Update status
        self._update_status()
    
    def _save_metrics(self):
        """Save current metrics"""
        try:
            filepath = self.metrics_collector.save_metrics()
            self.status_label.config(text=f"Saved: {filepath.name}")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
    
    def _update_status(self):
        """Update status display"""
        latest = self.metrics_collector.get_latest_metrics()
        if latest:
            status = f"Epoch {latest.epoch}, Batch {latest.batch}, Loss: {latest.loss:.4f}"
            self.status_label.config(text=status)
        
        self.root.after(1000, self._update_status)
    
    def run(self):
        """Start the visualization"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("Visualization stopped by user")
        finally:
            self.metrics_collector.writer.close()

def create_monitoring_hook(metrics_collector: MetricsCollector):
    """Create a monitoring hook for training loops"""
    
    def monitoring_hook(epoch: int, batch: int, total_batches: int, 
                       losses: Dict[str, float], lr: float, 
                       batch_time: float, data_time: float):
        """Hook function to be called during training"""
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            total_batches=total_batches,
            loss=losses.get('total', 0.0),
            zone_loss=losses.get('zone', 0.0),
            defect_loss=losses.get('defect', 0.0),
            quality_loss=losses.get('quality', 0.0),
            learning_rate=lr,
            batch_time=batch_time,
            data_time=data_time
        )
        metrics_collector.add_metrics(metrics)
    
    return monitoring_hook

def main():
    """Main function for standalone monitoring"""
    parser = argparse.ArgumentParser(description='Real-time Fiber CNN Training Monitor')
    parser.add_argument('--log-dir', type=str, default='logs/monitoring',
                       help='Directory for storing monitoring logs')
    parser.add_argument('--visualize', action='store_true',
                       help='Start real-time visualization')
    parser.add_argument('--port', type=int, default=6006,
                       help='TensorBoard port')
    
    args = parser.parse_args()
    
    # Create metrics collector
    metrics_collector = MetricsCollector(args.log_dir)
    
    if args.visualize:
        # Start visualization
        visualizer = RealTimeVisualizer(metrics_collector)
        logger.info("Starting real-time visualization...")
        visualizer.run()
    else:
        # Just run the monitoring loop
        logger.info("Starting monitoring loop...")
        logger.info(f"TensorBoard logs available at: {args.log_dir}/tensorboard")
        logger.info(f"Run: tensorboard --logdir={args.log_dir}/tensorboard --port={args.port}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            metrics_collector.save_metrics()

if __name__ == "__main__":
    main() 