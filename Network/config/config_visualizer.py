#!/usr/bin/env python3
"""
Interactive Configuration Editor GUI for Fiber Optics Neural Network System

This is a standalone PyQt5 application that provides:
- Interactive configuration editing with real-time validation
- Signal visualization based on configuration parameters
- System architecture diagram
- Performance prediction based on settings
- Export/import configuration functionality

This is a STANDALONE application - run directly with: python config_visualizer.py
Not imported by other modules.
"""

import sys
import yaml
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import math

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SystemArchitectureVisualizer:
    """
    Visualizes the entire Fiber Optics Neural Network system architecture
    Shows data flow, component relationships, and processing pipeline
    """

    def __init__(self):
        self.components = {
            'data_loader': {'inputs': [], 'outputs': ['tensor_processor']},
            'tensor_processor': {'inputs': ['data_loader'], 'outputs': ['feature_extractor']},
            'feature_extractor': {'inputs': ['tensor_processor'], 'outputs': ['integrated_network']},
            'integrated_network': {
                'inputs': ['feature_extractor', 'reference_comparator'],
                'outputs': ['segmentation', 'anomaly_detector', 'similarity_calculator']
            },
            'reference_comparator': {'inputs': ['reference_loader'], 'outputs': ['integrated_network']},
            'reference_loader': {'inputs': [], 'outputs': ['reference_comparator']},
            'segmentation': {'inputs': ['integrated_network'], 'outputs': ['results']},
            'anomaly_detector': {'inputs': ['integrated_network'], 'outputs': ['results']},
            'similarity_calculator': {'inputs': ['integrated_network'], 'outputs': ['results']},
            'results': {'inputs': ['segmentation', 'anomaly_detector', 'similarity_calculator'], 'outputs': []}
        }

    def generate_architecture_diagram(self) -> Figure:
        """Generate system architecture diagram"""
        fig = Figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

        # Position components
        positions = {
            'data_loader': (1, 5),
            'tensor_processor': (3, 5),
            'feature_extractor': (5, 5),
            'integrated_network': (7, 5),
            'reference_loader': (1, 3),
            'reference_comparator': (3, 3),
            'segmentation': (9, 7),
            'anomaly_detector': (9, 5),
            'similarity_calculator': (9, 3),
            'results': (11, 5)
        }

        # Draw components
        for comp, pos in positions.items():
            rect = plt.Rectangle((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6,
                               fill=True, facecolor='lightblue',
                               edgecolor='darkblue', linewidth=2)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], comp.replace('_', '\n'),
                   ha='center', va='center', fontsize=9, weight='bold')

        # Draw connections
        for comp, connections in self.components.items():
            if comp in positions:
                for output in connections['outputs']:
                    if output in positions:
                        start = positions[comp]
                        end = positions[output]
                        ax.arrow(start[0]+0.4, start[1],
                               end[0]-start[0]-0.8, end[1]-start[1],
                               head_width=0.1, head_length=0.1,
                               fc='green', ec='green', alpha=0.7)

        ax.set_xlim(0, 12)
        ax.set_ylim(2, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Fiber Optics Neural Network System Architecture', fontsize=16, weight='bold')

        return fig


class ConfigSignalGenerator:
    """
    Generates oscillating signals based on configuration parameters
    Maps config values to signal characteristics
    """

    def __init__(self):
        self.time_points = np.linspace(0, 10, 1000)
        self.system_visualizer = SystemArchitectureVisualizer()

    def generate_config_signal(self, config: Dict) -> np.ndarray:
        """
        Generate signal based on configuration parameters
        Signal characteristics represent predicted performance
        """
        # Base frequency from learning rate
        base_freq = 1.0 / (config['optimizer']['learning_rate'] * 100)

        # Amplitude from loss weights balance
        loss_weights = config['loss']['weights']
        weight_variance = np.var(list(loss_weights.values()))
        amplitude = 1.0 / (1.0 + weight_variance * 10)

        # Phase shifts from thresholds
        sim_threshold = config['similarity']['threshold']
        anomaly_threshold = config['anomaly']['threshold']
        # Phase shift calculated from deviation of thresholds from baseline values
        # Using 0.7 as baseline for similarity and 0.3 as baseline for anomaly
        phase_shift = (sim_threshold - 0.7) * np.pi + (0.3 - anomaly_threshold) * np.pi

        # Noise level from regularization
        weight_decay = config['optimizer']['weight_decay']
        noise_level = weight_decay * 5

        # Damping from momentum parameters
        betas = config['optimizer']['betas']
        damping = (1 - betas[0]) * 0.5

        # Generate base signal
        signal = amplitude * np.exp(-damping * self.time_points) * \
                np.sin(2 * np.pi * base_freq * self.time_points + phase_shift)

        # Add harmonics based on architecture complexity
        if config['model']['use_se_blocks']:
            signal += 0.1 * amplitude * np.sin(4 * np.pi * base_freq * self.time_points)

        if config['model']['use_deformable_conv']:
            signal += 0.15 * amplitude * np.sin(6 * np.pi * base_freq * self.time_points)

        # Add noise
        signal += np.random.normal(0, noise_level, len(self.time_points))

        # Apply stability factor from SAM/Lookahead
        if config['optimizer']['type'] == 'sam_lookahead':
            stability = 1.0 - config['optimizer']['sam_rho'] * 2
            signal *= stability

        return signal

    def generate_ideal_signal(self) -> np.ndarray:
        """Generate ideal reference signal (perfect configuration)"""
        # Perfect sine wave with optimal characteristics
        frequency = 2.0  # Optimal frequency
        amplitude = 0.9  # Near maximum amplitude

        signal = amplitude * np.sin(2 * np.pi * frequency * self.time_points)

        # Add slight exponential envelope for realism
        envelope = 0.9 + 0.1 * np.exp(-0.1 * self.time_points)
        signal *= envelope

        return signal

    def calculate_signal_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics from signal"""
        # RMS (stability)
        rms = np.sqrt(np.mean(signal**2))

        # Peak-to-peak (dynamic range)
        peak_to_peak = np.max(signal) - np.min(signal)

        # Frequency stability (via FFT)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
        dominant_freq = abs(freqs[dominant_freq_idx])

        # Signal-to-noise ratio
        signal_power = np.mean(signal**2)
        noise = signal - np.mean(signal)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

        # Convergence rate (decay factor)
        envelope = np.abs(signal)
        if len(envelope) > 100:
            decay_rate = -np.polyfit(self.time_points[:100], np.log(envelope[:100] + 1e-10), 1)[0]
        else:
            decay_rate = 0

        return {
            'rms': rms,
            'peak_to_peak': peak_to_peak,
            'dominant_frequency': dominant_freq,
            'snr': snr,
            'convergence_rate': decay_rate,
            'stability_score': rms * (1 - abs(decay_rate)) * min(snr / 20, 1.0)
        }


class ConfigParameterWidget(QWidget):
    """Base widget for configuration parameters"""

    valueChanged = pyqtSignal(str, object)  # key, value

    def __init__(self, key: str, value: Any, parent=None):
        super().__init__(parent)
        self.key = key
        self.value = value
        self.setup_ui()

    def setup_ui(self):
        """Override in subclasses"""
        pass


class SliderWidget(ConfigParameterWidget):
    """Draggable slider for numeric parameters"""
    
    def __init__(self, key: str, value: float, min_val: float, max_val: float,
                 step: float = 0.01, is_int: bool = False, parent=None):
        # Added is_int flag to distinguish integer vs float parameters (e.g., batch_size vs learning_rate),
        # as original assumed all numeric are float with step=0.01, leading to fractional values for ints.
        self.is_int = is_int
        self.min_val = min_val
        self.max_val = max_val
        self.step = step if not is_int else 1.0  # Force step=1 for ints
        super().__init__(key, value, parent)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header row with label and current value
        header_layout = QHBoxLayout()

        # Label with better formatting
        label = QLabel(self.key.replace('_', ' ').title())
        label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: 600;
                color: #ffffff;
                padding: 2px 0px;
            }
        """)
        header_layout.addWidget(label)

        header_layout.addStretch()

        # Current value display (larger and more prominent)
        # Updated format to show int without decimals if is_int, fixing display for integer parameters.
        value_text = f"{int(self.value)}" if self.is_int else f"{self.value:.4f}"
        self.value_label = QLabel(value_text)
        self.value_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #14b8a6;
                background-color: #2a2a2a;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 70px;
            }
        """)
        self.value_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.value_label)

        layout.addLayout(header_layout)

        # Slider row with range indicators
        slider_layout = QHBoxLayout()

        # Min value indicator (format based on type)
        min_text = f"{int(self.min_val)}" if self.is_int else f"{self.min_val:.3f}"
        min_label = QLabel(min_text)
        min_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #999999;
                font-weight: 500;
            }
        """)
        min_label.setFixedWidth(50)
        slider_layout.addWidget(min_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(self.min_val / self.step))
        self.slider.setMaximum(int(self.max_val / self.step))
        self.slider.setValue(int(self.value / self.step))
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setMinimumHeight(30)
        slider_layout.addWidget(self.slider, 1)

        # Max value indicator
        max_text = f"{int(self.max_val)}" if self.is_int else f"{self.max_val:.3f}"
        max_label = QLabel(max_text)
        max_label.setStyleSheet("""
            QLabel {
                font-size: 10px;
                color: #999999;
                font-weight: 500;
            }
        """)
        max_label.setFixedWidth(50)
        max_label.setAlignment(Qt.AlignRight)
        slider_layout.addWidget(max_label)

        layout.addLayout(slider_layout)

        # Bottom row with precise input
        input_layout = QHBoxLayout()

        # Description label
        desc_label = QLabel("Precise value:")
        desc_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #cccccc;
                font-weight: 500;
            }
        """)
        input_layout.addWidget(desc_label)

        input_layout.addStretch()

        # Input box for precise values (use QSpinBox for ints, QDoubleSpinBox for floats)
        # Added conditional for spin_box type to handle integer inputs properly, preventing decimal entry for int params like batch_size.
        if self.is_int:
            self.spin_box = QSpinBox()
            self.spin_box.setMinimum(int(self.min_val))
            self.spin_box.setMaximum(int(self.max_val))
            self.spin_box.setSingleStep(int(self.step))
        else:
            self.spin_box = QDoubleSpinBox()
            self.spin_box.setMinimum(self.min_val)
            self.spin_box.setMaximum(self.max_val)
            self.spin_box.setSingleStep(self.step)
            self.spin_box.setDecimals(4)
        
        self.spin_box.setValue(self.value)
        self.spin_box.valueChanged.connect(self.on_spin_changed)
        self.spin_box.setMinimumWidth(100)
        input_layout.addWidget(self.spin_box)

        layout.addLayout(input_layout)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("""
            QFrame {
                color: #404040;
                background-color: #404040;
                border: none;
                height: 1px;
                margin: 4px 0px;
            }
        """)
        layout.addWidget(separator)

    def on_slider_changed(self, value):
        self.value = value * self.step
        # Updated value_label format based on is_int, to show whole numbers for integers.
        value_text = f"{int(self.value)}" if self.is_int else f"{self.value:.4f}"
        self.value_label.setText(value_text)
        self.spin_box.blockSignals(True)
        self.spin_box.setValue(self.value)
        self.spin_box.blockSignals(False)
        self.valueChanged.emit(self.key, self.value)

    def on_spin_changed(self, value):
        self.value = value
        # Updated value_label format based on is_int, consistent with slider change.
        value_text = f"{int(self.value)}" if self.is_int else f"{self.value:.4f}"
        self.value_label.setText(value_text)
        self.slider.blockSignals(True)
        self.slider.setValue(int(value / self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.key, self.value)


class SwitchWidget(ConfigParameterWidget):
    """Toggle switch for boolean parameters"""

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header row with label
        header_layout = QHBoxLayout()

        # Label with better formatting
        label = QLabel(self.key.replace('_', ' ').title())
        label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: 600;
                color: #ffffff;
                padding: 2px 0px;
            }
        """)
        header_layout.addWidget(label)

        header_layout.addStretch()

        # Status indicator (larger and more prominent)
        self.status_label = QLabel("ENABLED" if self.value else "DISABLED")
        status_color = "#20c997" if self.value else "#dc3545"
        bg_color = "#1a4c42" if self.value else "#4c1a1a"
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                font-weight: bold;
                color: {status_color};
                background-color: {bg_color};
                border: 1px solid {status_color};
                border-radius: 4px;
                padding: 4px 12px;
                min-width: 80px;
            }}
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.status_label)

        layout.addLayout(header_layout)

        # Switch control row
        switch_layout = QHBoxLayout()

        # Description
        desc_label = QLabel("Toggle setting:")
        desc_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #cccccc;
                font-weight: 500;
            }
        """)
        switch_layout.addWidget(desc_label)

        switch_layout.addStretch()

        # Enhanced switch
        self.switch = QCheckBox()
        self.switch.setChecked(self.value)
        self.switch.stateChanged.connect(self.on_switch_changed)
        self.switch.setStyleSheet("""
            QCheckBox {
                font-size: 12px;
                font-weight: 600;
                spacing: 12px;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
                border: 2px solid #606060;
                border-radius: 6px;
                background-color: #404040;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border-color: #14b8a6;
            }
            QCheckBox::indicator:hover {
                border-color: #808080;
                background-color: #4a4a4a;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #14b8a6;
                border-color: #20c997;
            }
        """)
        switch_layout.addWidget(self.switch)

        layout.addLayout(switch_layout)

        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("""
            QFrame {
                color: #404040;
                background-color: #404040;
                border: none;
                height: 1px;
                margin: 4px 0px;
            }
        """)
        layout.addWidget(separator)

    def on_switch_changed(self, state):
        self.value = bool(state)

        # Update status label with new styling
        status_text = "ENABLED" if self.value else "DISABLED"
        status_color = "#20c997" if self.value else "#dc3545"
        bg_color = "#1a4c42" if self.value else "#4c1a1a"

        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 12px;
                font-weight: bold;
                color: {status_color};
                background-color: {bg_color};
                border: 1px solid {status_color};
                border-radius: 4px;
                padding: 4px 12px;
                min-width: 80px;
            }}
        """)

        self.valueChanged.emit(self.key, self.value)


class SystemArchitectureWidget(QWidget):
    """Widget for displaying system architecture"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = SystemArchitectureVisualizer()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = self.visualizer.generate_architecture_diagram()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add description
        description = QTextEdit()
        description.setReadOnly(True)
        description.setMaximumHeight(150)
        description.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 10px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        description.setHtml("""
        <h3>Fiber Optics Neural Network Architecture</h3>
        <p>This diagram shows the complete data flow through the system:</p>
        <ul>
            <li><b>Data Loader</b>: Loads fiber optic images from dataset</li>
            <li><b>Tensor Processor</b>: Converts images to tensorized format</li>
            <li><b>Feature Extractor</b>: Extracts multi-scale features</li>
            <li><b>Integrated Network</b>: Core neural network processing</li>
            <li><b>Reference Comparator</b>: Compares with reference database</li>
            <li><b>Segmentation</b>: Identifies core, cladding, and ferrule regions</li>
            <li><b>Anomaly Detector</b>: Detects defects and anomalies</li>
            <li><b>Similarity Calculator</b>: Computes similarity score (must achieve > {self.config['similarity']['threshold']})</li>
        </ul>
        <p>The equation <b>I=Ax1+Bx2+Cx3...=S(R)</b> governs the entire process.</p>
        """)
        layout.addWidget(description)


class EquationVisualizationWidget(QWidget):
    """Widget for visualizing the main equation I=Ax1+Bx2+Cx3...=S(R)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Initialize visualization
        self.update_equation_visualization()

    def update_equation_visualization(self, coefficients=None):
        """Update equation visualization with current coefficients"""
        self.figure.clear()

        if coefficients is None:
            coefficients = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 1.0}

        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_eq = self.figure.add_subplot(gs[0, :])
        ax_weights = self.figure.add_subplot(gs[1, 0])
        ax_contrib = self.figure.add_subplot(gs[1, 1])

        # Display equation
        ax_eq.text(0.5, 0.7, r'$I = A \cdot x_1 + B \cdot x_2 + C \cdot x_3 + D \cdot x_4 + E \cdot x_5 = S(R)$',
                fontsize=20, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))

        # Show current coefficients
        coeff_text = '\n'.join([f'{k} = {v:.3f}' for k, v in coefficients.items()])
        ax_eq.text(0.5, 0.3, coeff_text, fontsize=14, ha='center', va='center',
                family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))

        ax_eq.set_xlim(0, 1)
        ax_eq.set_ylim(0, 1)
        ax_eq.axis('off')
        ax_eq.set_title('Fiber Optics Analysis Equation', fontsize=16, weight='bold')

        # Visualize coefficient weights
        labels = list(coefficients.keys())
        values = list(coefficients.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

        bars = ax_weights.bar(labels, values, color=colors, edgecolor='black', linewidth=2)
        ax_weights.set_ylabel('Coefficient Value', fontsize=12)
        ax_weights.set_title('Coefficient Weights', fontsize=14)
        ax_weights.grid(True, alpha=0.3)
        ax_weights.axhline(y=0, color='black', linewidth=0.5)

        # Show component contributions
        components = {
            'x₁': 'Reference Similarity',
            'x₂': 'Trend Adherence',
            'x₃': 'Inverse Anomaly Score',
            'x₄': 'Segmentation Confidence',
            'x₅': 'Reconstruction Quality'
        }

        # Subscript mapping for digits 1-5 (Unicode characters)
        subscript_map = {
            1: '\u2081',
            2: '\u2082',
            3: '\u2083',
            4: '\u2084',
            5: '\u2085',
        }

        contributions = [v * 0.8 for v in values]  # Simulated contributions
        pie_data = [abs(c) for c in contributions]

        wedges, texts, autotexts = ax_contrib.pie(pie_data, labels=[f'{components["x" + subscript_map[i+1]]}\n({labels[i]})'
                                                                    for i in range(len(labels))],
                                                autopct='%1.1f%%', colors=colors)
        ax_contrib.set_title('Component Contributions to Final Score', fontsize=14)

        self.canvas.draw()


class PerformanceMetricsWidget(QWidget):
    """Widget for displaying theoretical performance metrics based on configuration.
    
    Note: This shows PREDICTED performance based on configuration values,
    not actual real-time neural network performance. For actual real-time
    monitoring, use visualization_ui.py instead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = None  # Will be set later
        self.setup_ui()
        self.metric_history = {
            'similarity': [],
            'inference_time': [],
            'memory_usage': [],
            'accuracy': []
        }

    def set_config(self, config):
        """Set configuration for accessing threshold values"""
        self.config = config
        self.update_metrics()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Control buttons
        control_layout = QHBoxLayout()

        self.clear_btn = QPushButton("🗑️ Clear History")
        self.clear_btn.clicked.connect(self.clear_history)
        control_layout.addWidget(self.clear_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Initialize plots
        self.update_metrics()

    def update_metrics(self, new_metrics=None):
        """Update performance metrics visualization"""
        if new_metrics:
            for key in self.metric_history:
                if key in new_metrics:
                    self.metric_history[key].append(new_metrics[key])
                    # Keep only last 100 points
                    if len(self.metric_history[key]) > 100:
                        self.metric_history[key].pop(0)

        self.figure.clear()

        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax_sim = self.figure.add_subplot(gs[0, 0])
        ax_time = self.figure.add_subplot(gs[0, 1])
        ax_mem = self.figure.add_subplot(gs[1, 0])
        ax_acc = self.figure.add_subplot(gs[1, 1])

        axes = [ax_sim, ax_time, ax_mem, ax_acc]
        titles = ['Similarity Score', 'Inference Time (ms)', 'Memory Usage (MB)', 'Accuracy']
        keys = ['similarity', 'inference_time', 'memory_usage', 'accuracy']
        colors = ['blue', 'green', 'red', 'purple']
        # Fixed: Use similarity threshold from config if available, else default to 0.7
        sim_threshold = self.config['similarity']['threshold'] if self.config else 0.7
        thresholds = [sim_threshold, None, None, None]  # Similarity must be > threshold

        for ax, title, key, color, threshold in zip(axes, titles, keys, colors, thresholds):
            data = self.metric_history[key]
            if data:
                ax.plot(data, color=color, linewidth=2, marker='o', markersize=4)
                ax.fill_between(range(len(data)), data, alpha=0.3, color=color)

                # Add threshold line if applicable
                if threshold:
                    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                             label=f'Threshold ({threshold})')
                    ax.legend()

                # Add statistics
                if len(data) > 0:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
                           transform=ax.transAxes, va='top', ha='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')

        self.figure.suptitle('Real-time Performance Metrics', fontsize=16, weight='bold')
        self.canvas.draw()

    def clear_history(self):
        """Clear metric history"""
        for key in self.metric_history:
            self.metric_history[key] = []
        self.update_metrics()


class SignalVisualizationWidget(QWidget):
    """Widget for displaying oscillating signals"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal_generator = ConfigSignalGenerator()
        self.setup_ui()

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.animation_offset = 0

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        self.ax_config = self.figure.add_subplot(gs[0, :])
        self.ax_ideal = self.figure.add_subplot(gs[1, 0])
        self.ax_metrics = self.figure.add_subplot(gs[1, 1])

        # Initialize plots
        self.ax_config.set_title("Configuration Signal (Current Settings)", fontsize=14)
        self.ax_config.set_xlabel("Time")
        self.ax_config.set_ylabel("Amplitude")
        self.ax_config.grid(True, alpha=0.3)

        self.ax_ideal.set_title("Ideal Reference Signal", fontsize=12)
        self.ax_ideal.set_xlabel("Time")
        self.ax_ideal.set_ylabel("Amplitude")
        self.ax_ideal.grid(True, alpha=0.3)

        self.ax_metrics.set_title("Performance Metrics", fontsize=12)
        self.ax_metrics.axis('off')

        # Control buttons with enhanced styling
        control_layout = QHBoxLayout()
        control_layout.setSpacing(12)
        control_layout.setContentsMargins(16, 8, 16, 8)

        self.animate_btn = QPushButton("▶️ Start Animation")
        self.animate_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-height: 25px;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #0d7377;
                border-color: #14b8a6;
                color: #ffffff;
            }
        """)
        self.animate_btn.clicked.connect(self.toggle_animation)
        control_layout.addWidget(self.animate_btn)

        self.reset_btn = QPushButton("🔄 Reset View")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-height: 25px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #6c757d;
                border-color: #868e96;
                color: #ffffff;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

    def update_signals(self, config: Dict):
        """Update signal visualizations based on configuration"""
        # Clear previous plots
        self.ax_config.clear()
        self.ax_ideal.clear()
        self.ax_metrics.clear()

        # Generate signals
        config_signal = self.signal_generator.generate_config_signal(config)
        ideal_signal = self.signal_generator.generate_ideal_signal()

        # Apply animation offset
        time_points = self.signal_generator.time_points + self.animation_offset

        # Plot configuration signal
        self.ax_config.plot(time_points, config_signal, 'b-', linewidth=2, label='Config Signal')
        self.ax_config.plot(time_points, ideal_signal, 'g--', alpha=0.5, linewidth=1, label='Ideal Reference')
        self.ax_config.set_title("Configuration Signal (Current Settings)", fontsize=14)
        self.ax_config.set_xlabel("Time")
        self.ax_config.set_ylabel("Amplitude")
        self.ax_config.grid(True, alpha=0.3)
        self.ax_config.legend()
        self.ax_config.set_ylim(-2, 2)

        # Plot ideal signal
        self.ax_ideal.plot(time_points, ideal_signal, 'g-', linewidth=2)
        self.ax_ideal.set_title("Ideal Reference Signal", fontsize=12)
        self.ax_ideal.set_xlabel("Time")
        self.ax_ideal.set_ylabel("Amplitude")
        self.ax_ideal.grid(True, alpha=0.3)
        self.ax_ideal.set_ylim(-1.5, 1.5)

        # Calculate and display metrics
        config_metrics = self.signal_generator.calculate_signal_metrics(config_signal)
        ideal_metrics = self.signal_generator.calculate_signal_metrics(ideal_signal)

        # Create metrics comparison
        metrics_text = "Performance Metrics Comparison\n" + "="*30 + "\n\n"
        metrics_text += f"{'Metric':<20} {'Current':<10} {'Ideal':<10} {'Ratio':<10}\n"
        metrics_text += "-"*50 + "\n"

        for key in config_metrics:
            current = config_metrics[key]
            ideal = ideal_metrics[key]
            ratio = current / (ideal + 1e-10)

            # Color code based on performance
            if ratio > 0.9:
                status = "✓"
            elif ratio > 0.7:
                status = "~"
            else:
                status = "✗"

            metrics_text += f"{key.replace('_', ' ').title():<20} "
            metrics_text += f"{current:>10.3f} {ideal:>10.3f} {ratio:>10.2f} {status}\n"

        # Overall performance score
        overall_score = config_metrics['stability_score'] / ideal_metrics['stability_score']
        metrics_text += "\n" + "="*50 + "\n"
        metrics_text += f"Overall Performance Score: {overall_score:.2%}\n"

        if overall_score > 0.9:
            metrics_text += "Status: EXCELLENT - Configuration is well optimized"
        elif overall_score > 0.7:
            metrics_text += "Status: GOOD - Minor adjustments recommended"
        elif overall_score > 0.5:
            metrics_text += "Status: FAIR - Significant tuning needed"
        else:
            metrics_text += "Status: POOR - Major configuration issues"

        self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                           fontfamily='monospace', fontsize=10, verticalalignment='top')
        self.ax_metrics.axis('off')

        self.canvas.draw()

    def toggle_animation(self):
        """Toggle signal animation"""
        if self.timer.isActive():
            self.timer.stop()
            self.animate_btn.setText("▶️ Start Animation")
        else:
            self.timer.start(50)  # 20 FPS
            self.animate_btn.setText("⏸️ Stop Animation")

    def update_animation(self):
        """Update animation frame"""
        self.animation_offset += 0.1
        if hasattr(self, 'current_config'):
            self.update_signals(self.current_config)

    def reset_view(self):
        """Reset animation and view"""
        self.animation_offset = 0
        if hasattr(self, 'current_config'):
            self.update_signals(self.current_config)

    def set_config(self, config: Dict):
        """Set current configuration"""
        self.current_config = config
        self.update_signals(config)


class ConfigVisualizerMainWindow(QMainWindow):
    """Main window for configuration visualizer"""

    def __init__(self):
        super().__init__()
        self.config_path = Path(__file__).parent.parent / "config.yaml"
        self.config = {}
        self.parameter_widgets = {}

        self.setWindowTitle("Fiber Optics Configuration Visualizer")
        self.setGeometry(100, 100, 1600, 900)

        # Set enhanced dark theme for better readability
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }

            QLabel {
                color: #ffffff;
                font-size: 12px;
                font-weight: 500;
            }

            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #0d7377;
                border-color: #14b8a6;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0a5a5e;
                border-color: #0d7377;
            }

            QSlider::groove:horizontal {
                height: 12px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #303030);
                border-radius: 6px;
                border: 1px solid #606060;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #14b8a6, stop:1 #0d7377);
                border: 2px solid #ffffff;
                width: 22px;
                height: 22px;
                margin: -6px 0;
                border-radius: 13px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #20c997, stop:1 #14b8a6);
                border-color: #f8f9fa;
            }
            QSlider::handle:horizontal:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0d7377, stop:1 #0a5a5e);
            }

            QGroupBox {
                color: #ffffff;
                border: 2px solid #606060;
                border-radius: 8px;
                margin-top: 16px;
                padding-top: 16px;
                padding-left: 8px;
                padding-right: 8px;
                padding-bottom: 8px;
                font-size: 13px;
                font-weight: 600;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 4px 8px;
                background-color: #0d7377;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
            }

            QTabWidget::pane {
                border: 2px solid #606060;
                background-color: #1e1e1e;
                border-radius: 6px;
                padding: 4px;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 10px 16px;
                margin-right: 3px;
                margin-bottom: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 2px solid #606060;
                border-bottom: none;
                font-size: 12px;
                font-weight: 600;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                border-color: #14b8a6;
                color: #ffffff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #505050;
                border-color: #707070;
            }

            QScrollArea {
                background-color: #1e1e1e;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 4px;
            }

            QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 6px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
                min-width: 60px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #14b8a6;
                background-color: #4a4a4a;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #808080;
                background-color: #4a4a4a;
            }

            QCheckBox {
                color: #ffffff;
                font-size: 12px;
                font-weight: 500;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #606060;
                border-radius: 4px;
                background-color: #404040;
            }
            QCheckBox::indicator:checked {
                background-color: #0d7377;
                border-color: #14b8a6;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEzLjUgNEw2IDExLjUgMi41IDgiIHN0cm9rZT0iI0ZGRkZGRiIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
            }
            QCheckBox::indicator:hover {
                border-color: #808080;
                background-color: #4a4a4a;
            }

            QStatusBar {
                background-color: #2a2a2a;
                color: #ffffff;
                border-top: 1px solid #606060;
                font-size: 11px;
                padding: 4px;
            }

            QMenuBar {
                background-color: #2a2a2a;
                color: #ffffff;
                border-bottom: 1px solid #606060;
                padding: 2px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #0d7377;
            }

            QMenu {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #606060;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 16px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #0d7377;
            }

            QSplitter::handle {
                background-color: #606060;
                width: 3px;
                border-radius: 1px;
            }
            QSplitter::handle:hover {
                background-color: #808080;
            }
        """)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        """Setup main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Left panel - Configuration controls (40% width)
        left_panel = QWidget()
        left_panel.setMaximumWidth(600)
        left_layout = QVBoxLayout(left_panel)

        # Title with enhanced styling
        title = QLabel("⚙️ Configuration Parameters")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: bold;
                padding: 16px;
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d7377, stop:1 #14b8a6);
                border-radius: 8px;
                margin-bottom: 8px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Create tab widget for parameter categories
        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        # Control buttons with enhanced styling
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Load button
        load_btn = QPushButton("📁 Load Config")
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-height: 25px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4a90e2;
                border-color: #6bb6ff;
                color: #ffffff;
            }
        """)
        load_btn.clicked.connect(self.load_config_dialog)
        button_layout.addWidget(load_btn)

        # Save button
        save_btn = QPushButton("💾 Save Config")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-height: 25px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #28a745;
                border-color: #34ce57;
                color: #ffffff;
            }
        """)
        save_btn.clicked.connect(self.save_config)
        button_layout.addWidget(save_btn)

        # Reset button
        reset_btn = QPushButton("🔄 Reset Default")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #ffffff;
                border: 2px solid #606060;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                min-height: 25px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #dc3545;
                border-color: #e74c3c;
                color: #ffffff;
            }
        """)
        reset_btn.clicked.connect(self.reset_config)
        button_layout.addWidget(reset_btn)

        left_layout.addLayout(button_layout)

        # Right panel - Create tabbed view for multiple visualizations
        right_panel = QTabWidget()
        right_panel.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #606060;
                background-color: #1e1e1e;
                border-radius: 6px;
                padding: 4px;
            }
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 10px 16px;
                margin-right: 3px;
                margin-bottom: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 2px solid #606060;
                border-bottom: none;
                font-size: 12px;
                font-weight: 600;
                min-width: 100px;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                border-color: #14b8a6;
                color: #ffffff;
            }
        """)

        # Add signal visualization tab
        self.signal_widget = SignalVisualizationWidget()
        right_panel.addTab(self.signal_widget, "📊 Signal Analysis")

        # Add system architecture tab
        self.architecture_widget = SystemArchitectureWidget()
        right_panel.addTab(self.architecture_widget, "🔧 System Architecture")

        # Add equation visualization tab
        self.equation_widget = EquationVisualizationWidget()
        right_panel.addTab(self.equation_widget, "📐 Equation I=Ax1+Bx2+...")

        # Add performance metrics tab
        self.metrics_widget = PerformanceMetricsWidget()
        right_panel.addTab(self.metrics_widget, "📈 Performance Metrics")

        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2)  # Right panel gets more space

        main_layout.addWidget(splitter)

        # Create menu bar
        self.create_menu_bar()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load Configuration', self)
        load_action.triggered.connect(self.load_config_dialog)
        file_menu.addAction(load_action)

        save_action = QAction('Save Configuration', self)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu('View')

        animate_action = QAction('Toggle Animation', self)
        animate_action.triggered.connect(self.signal_widget.toggle_animation)
        view_menu.addAction(animate_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            self.create_parameter_widgets()
            self.signal_widget.set_config(self.config)
            self.equation_widget.update_equation_visualization(self.config.get('equation', {}).get('coefficients', {}))
            # Fixed: Set config on metrics widget to use correct threshold
            self.metrics_widget.set_config(self.config)
            self.status_bar.showMessage(f"Loaded configuration from {self.config_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configuration: {str(e)}")

    def create_parameter_widgets(self):
        """Create parameter control widgets based on configuration"""
        # Clear existing tabs
        self.tab_widget.clear()
        self.parameter_widgets.clear()

        # Group parameters by category
        categories = {
            'System': ['system'],
            'Model': ['model'],
            'Training': ['training', 'optimizer'],
            'Loss': ['loss'],
            'Similarity': ['similarity', 'equation'],
            'Anomaly': ['anomaly'],
            'Visualization': ['visualization'],
            'Advanced': ['advanced', 'experimental']
        }

        for category, sections in categories.items():
            # Create scroll area for category with enhanced styling
            scroll = QScrollArea()
            scroll.setStyleSheet("""
                QScrollArea {
                    background-color: #1e1e1e;
                    border: 1px solid #404040;
                    border-radius: 6px;
                    padding: 2px;
                }
                QScrollBar:vertical {
                    background-color: #2a2a2a;
                    width: 12px;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical {
                    background-color: #0d7377;
                    border-radius: 6px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #14b8a6;
                }
            """)
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            scroll_layout.setSpacing(12)
            scroll_layout.setContentsMargins(12, 12, 12, 12)

            for section in sections:
                if section in self.config:
                    group_box = QGroupBox(section.title())
                    group_layout = QVBoxLayout(group_box)

                    self.add_section_widgets(self.config[section], section, group_layout)

                    scroll_layout.addWidget(group_box)

            scroll_layout.addStretch()
            scroll.setWidget(scroll_widget)
            scroll.setWidgetResizable(True)

            self.tab_widget.addTab(scroll, category)

    def add_section_widgets(self, section_config: Dict, prefix: str, layout: QVBoxLayout):
        """Recursively add widgets for configuration section"""
        for key, value in section_config.items():
            full_key = f"{prefix}.{key}"

            if isinstance(value, dict):
                # Create sub-group with enhanced styling
                sub_group = QGroupBox(key.replace('_', ' ').title())
                sub_group.setStyleSheet("""
                    QGroupBox {
                        color: #ffffff;
                        border: 1px solid #808080;
                        border-radius: 6px;
                        margin-top: 12px;
                        padding-top: 12px;
                        padding-left: 6px;
                        padding-right: 6px;
                        padding-bottom: 6px;
                        font-size: 12px;
                        font-weight: 600;
                        background-color: #333333;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 2px 6px;
                        background-color: #4a90e2;
                        border-radius: 3px;
                        color: #ffffff;
                        font-weight: bold;
                        font-size: 11px;
                    }
                """)
                sub_layout = QVBoxLayout(sub_group)
                sub_layout.setSpacing(8)
                self.add_section_widgets(value, full_key, sub_layout)
                layout.addWidget(sub_group)

            elif isinstance(value, bool):
                # Create switch widget
                widget = SwitchWidget(key, value)
                widget.valueChanged.connect(self.on_parameter_changed)
                self.parameter_widgets[full_key] = widget
                layout.addWidget(widget)

            elif isinstance(value, (int, float)):
                # Determine appropriate range
                min_val, max_val = self.get_parameter_range(full_key, value)
                
                # Added is_int flag passing to SliderWidget based on original type, to handle int params properly
                # (original treated all as float, leading to fractional sliders for ints like num_epochs).
                is_int = isinstance(value, int)
                widget = SliderWidget(key, float(value), min_val, max_val, is_int=is_int)
                widget.valueChanged.connect(self.on_parameter_changed)
                self.parameter_widgets[full_key] = widget
                layout.addWidget(widget)

            elif isinstance(value, list):
                # Create list editor (simplified for now)
                label = QLabel(f"{key}: {value}")
                layout.addWidget(label)

    def get_parameter_range(self, key: str, value: float) -> Tuple[float, float]:
        """Get appropriate range for parameter"""
        # Define ranges for known parameters
        ranges = {
            'learning_rate': (0.00001, 0.1),
            'weight_decay': (0.0, 0.1),
            'threshold': (0.0, 1.0),
            'alpha': (0.0, 1.0),
            'beta': (0.0, 1.0),
            'gamma': (0.0, 5.0),
            'rho': (0.0, 0.2),
            'temperature': (0.01, 1.0),
            'coefficient': (-2.0, 2.0),
            'sigma': (0.01, 1.0),
            'epsilon': (0.001, 1.0),
            'reduction': (1, 32),
            'channels': (1, 1024),
            'batch_size': (1, 256),
            'num_epochs': (1, 1000),
            'patience': (1, 100),
            'k': (1, 20),
            'population_size': (10, 100),
            'dropout': (0.0, 0.9),
            'momentum': (0.0, 0.999),
        }

        # Check if key contains any of the range keywords
        for keyword, (min_val, max_val) in ranges.items():
            if keyword in key.lower():
                return min_val, max_val

        # Default range based on current value
        if 0 <= value <= 1:
            return 0.0, 1.0
        elif value < 0:
            return value * 2, -value * 2
        else:
            return 0.0, value * 2

    def on_parameter_changed(self, key: str, value: Any):
        """Handle parameter value change"""
        # Find full key
        full_key = None
        for widget_key, widget in self.parameter_widgets.items():
            if widget.key == key:
                full_key = widget_key
                break

        if not full_key:
            return

        # Update configuration
        parts = full_key.split('.')
        config_ref = self.config

        for part in parts[:-1]:
            config_ref = config_ref[part]

        # Added type check and casting: if original was int, round and cast value to int,
        # preventing float values in config for int params (e.g., batch_size=128.3), which could cause errors in training code.
        original_type = type(config_ref[parts[-1]])
        if original_type == int:
            value = int(round(value))
        
        config_ref[parts[-1]] = value

        # Update all visualizations
        self.signal_widget.set_config(self.config)
        self.equation_widget.update_equation_visualization(self.config.get('equation', {}).get('coefficients', {}))

        # Update performance metrics with simulated data
        if 'learning_rate' in full_key:
            self.metrics_widget.update_metrics({
                'similarity': 0.7 + np.random.uniform(-0.1, 0.2),
                'inference_time': 25 + np.random.uniform(-5, 5),
                'memory_usage': 1024 + np.random.uniform(-100, 100),
                'accuracy': 0.9 + np.random.uniform(-0.05, 0.05)
            })

        # Update status
        self.status_bar.showMessage(f"Updated {full_key} = {value}")

    def save_config(self):
        """Save current configuration to YAML file"""
        try:
            # Create backup
            backup_path = self.config_path.with_suffix('.yaml.bak')
            if self.config_path.exists():
                import shutil
                shutil.copy(self.config_path, backup_path)

            # Save configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

            self.status_bar.showMessage(f"Saved configuration to {self.config_path}")
            QMessageBox.information(self, "Success", "Configuration saved successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")

    def load_config_dialog(self):
        """Show dialog to load configuration file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", str(self.config_path.parent),
            "YAML Files (*.yaml *.yml)"
        )

        if file_path:
            self.config_path = Path(file_path)
            self.load_config()

    def reset_config(self):
        """Reset configuration to default values"""
        reply = QMessageBox.question(
            self, "Reset Configuration",
            "Are you sure you want to reset to default configuration?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Reload from file
            self.load_config()

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About",
            "Fiber Optics Configuration Visualizer\n\n"
            "Interactive tool for optimizing neural network configuration\n"
            "by analyzing theoretical performance signals.\n\n"
            "The signal visualization shows how well your configuration\n"
            "is expected to perform compared to an ideal setup."
        )


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = ConfigVisualizerMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
