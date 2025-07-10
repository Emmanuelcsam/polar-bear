#!/usr/bin/env python3
"""
Example Application: Real-Time Shape Analysis Dashboard
======================================================

This example demonstrates how to use the Integrated Geometry Detection System
to build a custom application with:
- Real-time shape statistics
- Shape filtering and tracking
- Data export and visualization
- Custom alerts and notifications

Usage: python shape_analysis_dashboard.py
"""

import cv2
import numpy as np
import time
import json
from collections import defaultdict, deque
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
import queue

# Import from the integrated system
from src.core.integrated_geometry_system import (
    GeometryDetectionSystem, 
    GeometryDetector,
    ShapeType,
    Config,
    setup_logging
)

class ShapeAnalysisDashboard:
    """Custom application for shape analysis with dashboard"""
    
    def __init__(self):
        self.logger = setup_logging("ShapeAnalysisDashboard")
        
        # Initialize detection system
        self.detector = GeometryDetector(use_gpu=True)
        
        # Data storage
        self.shape_history = defaultdict(lambda: deque(maxlen=300))  # 10 seconds at 30fps
        self.shape_counts = defaultdict(int)
        self.total_shapes_detected = 0
        self.start_time = time.time()
        
        # Filtering settings
        self.min_area_filter = 500
        self.max_area_filter = 50000
        self.shape_filter = set(ShapeType)  # All shapes by default
        
        # Alert settings
        self.alert_thresholds = {
            ShapeType.CIRCLE: 5,    # Alert if more than 5 circles
            ShapeType.TRIANGLE: 3,  # Alert if more than 3 triangles
        }
        self.alerts = deque(maxlen=10)
        
        # Visualization
        self.show_dashboard = True
        self.dashboard_update_interval = 1.0  # seconds
        self.last_dashboard_update = 0
        
        # Export settings
        self.export_queue = queue.Queue()
        self.export_thread = threading.Thread(target=self._export_worker)
        self.export_thread.daemon = True
        self.export_thread.start()
        
        self.logger.info("Shape Analysis Dashboard initialized")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame and update analytics"""
        # Detect shapes
        shapes = self.detector.detect_shapes(frame)
        
        # Filter shapes
        filtered_shapes = self._filter_shapes(shapes)
        
        # Update statistics
        self._update_statistics(filtered_shapes)
        
        # Check alerts
        self._check_alerts(filtered_shapes)
        
        # Draw visualization
        output = self._draw_visualization(frame, filtered_shapes)
        
        # Update dashboard
        if self.show_dashboard and time.time() - self.last_dashboard_update > self.dashboard_update_interval:
            self._update_dashboard(output)
            self.last_dashboard_update = time.time()
        
        return output
    
    def _filter_shapes(self, shapes):
        """Apply shape filters"""
        filtered = []
        
        for shape in shapes:
            # Type filter
            if shape.shape_type not in self.shape_filter:
                continue
            
            # Area filter
            if not (self.min_area_filter <= shape.area <= self.max_area_filter):
                continue
            
            # Confidence filter
            if shape.confidence < 0.7:
                continue
            
            filtered.append(shape)
        
        return filtered
    
    def _update_statistics(self, shapes):
        """Update shape statistics"""
        current_time = time.time()
        
        # Count shapes by type
        frame_counts = defaultdict(int)
        for shape in shapes:
            frame_counts[shape.shape_type] += 1
            self.shape_counts[shape.shape_type] += 1
            self.total_shapes_detected += 1
        
        # Update history
        for shape_type in ShapeType:
            count = frame_counts.get(shape_type, 0)
            self.shape_history[shape_type].append((current_time, count))
    
    def _check_alerts(self, shapes):
        """Check for alert conditions"""
        # Count current shapes
        current_counts = defaultdict(int)
        for shape in shapes:
            current_counts[shape.shape_type] += 1
        
        # Check thresholds
        for shape_type, threshold in self.alert_thresholds.items():
            if current_counts[shape_type] > threshold:
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'threshold_exceeded',
                    'shape': shape_type.value,
                    'count': current_counts[shape_type],
                    'threshold': threshold
                }
                self.alerts.append(alert)
                self.logger.warning(f"Alert: {shape_type.value} count ({current_counts[shape_type]}) exceeded threshold ({threshold})")
    
    def _draw_visualization(self, frame, shapes):
        """Draw shapes and statistics on frame"""
        output = frame.copy()
        
        # Draw shapes
        for shape in shapes:
            # Draw contour
            cv2.drawContours(output, [shape.contour], -1, shape.color, 2)
            
            # Draw center with label
            cv2.circle(output, shape.center, 5, (255, 0, 0), -1)
            
            # Add detailed info
            info_text = f"{shape.shape_type.value}"
            cv2.putText(output, info_text, 
                       (shape.center[0] - 30, shape.center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape.color, 2)
            
            # Add area
            area_text = f"A:{int(shape.area)}"
            cv2.putText(output, area_text,
                       (shape.center[0] - 30, shape.center[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw statistics panel
        self._draw_stats_panel(output)
        
        # Draw alerts
        self._draw_alerts(output)
        
        return output
    
    def _draw_stats_panel(self, frame):
        """Draw statistics panel"""
        # Create semi-transparent panel
        panel_height = 200
        panel = np.zeros((panel_height, 300, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(panel, "Shape Statistics", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Runtime
        runtime = time.time() - self.start_time
        cv2.putText(panel, f"Runtime: {runtime:.1f}s", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Total shapes
        cv2.putText(panel, f"Total: {self.total_shapes_detected}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Shape counts
        y = 100
        for shape_type, count in sorted(self.shape_counts.items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
            if count > 0:
                color = self.detector.colors[shape_type]
                text = f"{shape_type.value}: {count}"
                cv2.putText(panel, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw bar
                bar_width = int(200 * count / max(self.shape_counts.values()))
                cv2.rectangle(panel, (100, y-15), (100 + bar_width, y-2), color, -1)
                
                y += 25
        
        # Overlay panel on frame
        if frame.shape[0] >= panel_height:
            frame[10:10+panel_height, 10:310] = cv2.addWeighted(
                frame[10:10+panel_height, 10:310], 0.3, panel, 0.7, 0)
    
    def _draw_alerts(self, frame):
        """Draw recent alerts"""
        if not self.alerts:
            return
        
        y = frame.shape[0] - 100
        for alert in list(self.alerts)[-3:]:  # Show last 3 alerts
            alert_text = f"⚠ {alert['shape']} count: {alert['count']} > {alert['threshold']}"
            
            # Draw background
            cv2.rectangle(frame, (10, y-20), (400, y+5), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, y-20), (400, y+5), (0, 0, 255), 2)
            
            # Draw text
            cv2.putText(frame, alert_text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            y += 30
    
    def _update_dashboard(self, frame):
        """Update analytics dashboard"""
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Shape Detection Analytics Dashboard', fontsize=16)
        
        # 1. Shape counts pie chart
        ax = axes[0, 0]
        if self.shape_counts:
            labels = []
            sizes = []
            colors = []
            
            for shape_type, count in self.shape_counts.items():
                if count > 0:
                    labels.append(shape_type.value)
                    sizes.append(count)
                    color = self.detector.colors[shape_type]
                    colors.append([c/255 for c in color[::-1]])  # BGR to RGB
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Total Shape Distribution')
        else:
            ax.text(0.5, 0.5, 'No shapes detected', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # 2. Shape timeline
        ax = axes[0, 1]
        current_time = time.time()
        
        for shape_type in [ShapeType.CIRCLE, ShapeType.RECTANGLE, ShapeType.TRIANGLE]:
            if shape_type in self.shape_history:
                history = self.shape_history[shape_type]
                if history:
                    times = [t - current_time for t, _ in history]
                    counts = [c for _, c in history]
                    color = self.detector.colors[shape_type]
                    rgb_color = [c/255 for c in color[::-1]]
                    ax.plot(times, counts, label=shape_type.value, 
                           color=rgb_color, linewidth=2)
        
        ax.set_xlabel('Time (seconds ago)')
        ax.set_ylabel('Count')
        ax.set_title('Shape Detection Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 0)
        
        # 3. Detection rate
        ax = axes[1, 0]
        if self.total_shapes_detected > 0:
            runtime = time.time() - self.start_time
            rate = self.total_shapes_detected / runtime
            
            ax.bar(['Detection Rate'], [rate], color='green')
            ax.set_ylabel('Shapes per second')
            ax.set_title(f'Average Detection Rate: {rate:.1f} shapes/sec')
            ax.set_ylim(0, max(rate * 1.2, 10))
        
        # 4. Current frame preview
        ax = axes[1, 1]
        # Convert BGR to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize for display
        preview = cv2.resize(frame_rgb, (320, 240))
        ax.imshow(preview)
        ax.set_title('Current Frame')
        ax.axis('off')
        
        # Convert to image
        plt.tight_layout()
        
        # Show in window (non-blocking)
        plt.show(block=False)
        plt.pause(0.001)
        plt.close()
    
    def _export_worker(self):
        """Background thread for data export"""
        while True:
            try:
                # Export every 30 seconds
                time.sleep(30)
                
                # Prepare export data
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'runtime_seconds': time.time() - self.start_time,
                    'total_shapes': self.total_shapes_detected,
                    'shape_counts': dict(self.shape_counts),
                    'alerts': list(self.alerts),
                    'settings': {
                        'min_area': self.min_area_filter,
                        'max_area': self.max_area_filter,
                        'filtered_shapes': [s.value for s in self.shape_filter]
                    }
                }
                
                # Save to file
                filename = f"shape_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Exported analytics to {filename}")
                
            except Exception as e:
                self.logger.error(f"Export error: {e}")
    
    def handle_keyboard(self, key):
        """Handle keyboard input"""
        if key == ord('d'):
            self.show_dashboard = not self.show_dashboard
            self.logger.info(f"Dashboard: {'ON' if self.show_dashboard else 'OFF'}")
        
        elif key == ord('f'):
            # Cycle through shape filters
            if len(self.shape_filter) == len(ShapeType):
                self.shape_filter = {ShapeType.CIRCLE}
            elif ShapeType.CIRCLE in self.shape_filter:
                self.shape_filter = {ShapeType.RECTANGLE, ShapeType.SQUARE}
            elif ShapeType.RECTANGLE in self.shape_filter:
                self.shape_filter = {ShapeType.TRIANGLE}
            else:
                self.shape_filter = set(ShapeType)
            
            self.logger.info(f"Filter: {[s.value for s in self.shape_filter]}")
        
        elif key == ord('r'):
            # Reset statistics
            self.shape_counts.clear()
            self.shape_history.clear()
            self.total_shapes_detected = 0
            self.alerts.clear()
            self.start_time = time.time()
            self.logger.info("Statistics reset")
        
        elif key == ord('['):
            self.min_area_filter = max(100, self.min_area_filter - 100)
            self.logger.info(f"Min area filter: {self.min_area_filter}")
        
        elif key == ord(']'):
            self.min_area_filter = min(5000, self.min_area_filter + 100)
            self.logger.info(f"Min area filter: {self.min_area_filter}")
    
    def run(self, camera_source=0):
        """Run the dashboard application"""
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            self.logger.error("Failed to open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\nShape Analysis Dashboard")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'd' - Toggle dashboard")
        print("  'f' - Cycle shape filters")
        print("  'r' - Reset statistics")
        print("  '[/]' - Adjust minimum area filter")
        print("  's' - Save screenshot")
        print("="*50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                output = self.process_frame(frame)
                
                # Display
                cv2.imshow('Shape Analysis Dashboard', output)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, output)
                    self.logger.info(f"Screenshot saved: {filename}")
                else:
                    self.handle_keyboard(key)
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Final export
            self.logger.info("Performing final export...")
            self._export_worker()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shape Analysis Dashboard Example')
    parser.add_argument('-s', '--source', default=0,
                       help='Camera source (index or path)')
    
    args = parser.parse_args()
    
    # Convert source to int if numeric
    source = args.source
    if str(source).isdigit():
        source = int(source)
    
    # Create and run dashboard
    dashboard = ShapeAnalysisDashboard()
    dashboard.run(source)

if __name__ == "__main__":
    main()
