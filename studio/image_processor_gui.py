#!/usr/bin/env python3

import sys
import os
import re
import subprocess
import shutil
import json
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import cv2

# Qt imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QSplitter, QLineEdit, QMessageBox, QScrollArea, QGroupBox,
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit, QProgressBar, QStatusBar, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor, QCursor


class ImageViewer(QScrollArea):
    """Enhanced image viewer with smooth zoom and pan"""
    
    def __init__(self):
        super().__init__()  # Initialize the parent QScrollArea widget
        self.setWidgetResizable(True)  # Allow the scroll area to resize its contents automatically
        self.image_label = QLabel()  # Create a label widget to hold the displayed image
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image within the label
        self.setWidget(self.image_label)  # Set the label as the scrollable content of this scroll area
        
        self.original_pixmap = None  # Store the unscaled original image for high-quality zooming
        self.scale_factor = 1.0  # Track current zoom level (1.0 = 100%, 2.0 = 200%, etc.)
        self.pan_start = None  # Store mouse position when panning starts (for calculating drag distance)
        self.zoom_point = None  # Store the point around which to zoom (unused in current implementation)
        
        # Enable mouse tracking for zoom
        self.setMouseTracking(True)  # Allow this widget to receive mouse move events even without buttons pressed
        self.image_label.setMouseTracking(True)  # Enable mouse tracking on the image label as well
        
    def set_image(self, image: np.ndarray):
        """Set the image to display"""
        if image is None:  # Check if no image was provided
            self.image_label.clear()  # Remove any existing image from the display
            self.original_pixmap = None  # Clear the stored original image reference
            return  # Exit early since there's nothing to display
        
        # Convert numpy array to QImage
        if len(image.shape) == 2:  # Check if image is grayscale (height, width only)
            height, width = image.shape  # Extract dimensions from 2D array
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)  # Create Qt image from raw data with grayscale format
        else:  # Color (BGR to RGB)
            height, width, channel = image.shape  # Extract dimensions including color channels
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR format to Qt's expected RGB format
            q_image = QImage(rgb_image.data, width, height, channel * width, QImage.Format_RGB888)  # Create Qt image with RGB format and calculated bytes per line
        
        self.original_pixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap for efficient display and store as reference
        self.update_display()  # Refresh the display with the new image at current zoom level
        
    def update_display(self):
        """Update the displayed image with current scale"""
        if self.original_pixmap:  # Only proceed if we have an image to display
            scaled_pixmap = self.original_pixmap.scaled(  # Create a scaled version of the original image
                int(self.original_pixmap.width() * self.scale_factor),  # Calculate new width based on current zoom
                int(self.original_pixmap.height() * self.scale_factor),  # Calculate new height based on current zoom
                Qt.KeepAspectRatio,  # Maintain the original width-to-height ratio during scaling
                Qt.SmoothTransformation  # Use high-quality scaling algorithm for better visual results
            )
            self.image_label.setPixmap(scaled_pixmap)  # Display the scaled image in the label widget
            
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming centered on cursor"""
        if not self.original_pixmap:  # Exit if no image is loaded
            return
        
        # Get cursor position relative to the image
        cursor_pos = event.pos()  # Get mouse cursor position within this widget
        
        # Calculate zoom
        zoom_in_factor = 1.25  # Define zoom increment (25% increase)
        zoom_out_factor = 0.8  # Define zoom decrement (20% decrease)
        old_scale = self.scale_factor  # Store current zoom level before changing it
        
        if event.angleDelta().y() > 0:  # Check if mouse wheel was scrolled up (positive Y delta)
            self.scale_factor *= zoom_in_factor  # Increase zoom level
        else:
            self.scale_factor *= zoom_out_factor  # Decrease zoom level
            
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))  # Clamp zoom between 10% and 1000% to prevent extreme values
        
        # Update display
        self.update_display()  # Refresh the image with new zoom level
        
        # Adjust scrollbars to keep cursor position stable
        if old_scale != self.scale_factor:  # Only adjust scroll position if zoom actually changed
            scale_delta = self.scale_factor / old_scale  # Calculate the ratio of new zoom to old zoom
            
            # Calculate the new scroll positions
            h_bar = self.horizontalScrollBar()  # Get reference to horizontal scrollbar
            v_bar = self.verticalScrollBar()  # Get reference to vertical scrollbar
            
            new_h_value = int((h_bar.value() + cursor_pos.x()) * scale_delta - cursor_pos.x())  # Calculate new horizontal scroll to keep cursor position steady
            new_v_value = int((v_bar.value() + cursor_pos.y()) * scale_delta - cursor_pos.y())  # Calculate new vertical scroll to keep cursor position steady
            
            h_bar.setValue(new_h_value)  # Apply new horizontal scroll position
            v_bar.setValue(new_v_value)  # Apply new vertical scroll position
            
    def mousePressEvent(self, event):
        """Start panning on middle mouse button"""
        if event.button() == Qt.MiddleButton:  # Check if middle mouse button was pressed
            self.pan_start = event.pos()  # Store the starting position for calculating drag distance
            self.setCursor(Qt.ClosedHandCursor)  # Change cursor to closed hand to indicate panning mode
            
    def mouseReleaseEvent(self, event):
        """Stop panning"""
        if event.button() == Qt.MiddleButton:  # Check if middle mouse button was released
            self.pan_start = None  # Clear the panning start position to stop panning
            self.setCursor(Qt.ArrowCursor)  # Reset cursor back to normal arrow
            
    def mouseMoveEvent(self, event):
        """Handle panning"""
        if self.pan_start:  # Only pan if middle mouse button is currently pressed
            delta = event.pos() - self.pan_start  # Calculate how far the mouse has moved since last update
            self.pan_start = event.pos()  # Update start position for next movement calculation
            
            h_bar = self.horizontalScrollBar()  # Get reference to horizontal scrollbar
            v_bar = self.verticalScrollBar()  # Get reference to vertical scrollbar
            h_bar.setValue(h_bar.value() - delta.x())  # Move horizontal scroll opposite to mouse movement
            v_bar.setValue(v_bar.value() - delta.y())  # Move vertical scroll opposite to mouse movement
            
    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.scale_factor = 1.0  # Set zoom back to original size (100%)
        self.update_display()  # Refresh display with new zoom level
        
    def zoom_to_fit(self):
        """Zoom to fit the image in the viewport"""
        if not self.original_pixmap:  # Exit if no image is loaded
            return
            
        viewport_size = self.viewport().size()  # Get the visible area size of the scroll area
        image_size = self.original_pixmap.size()  # Get the original image dimensions
        
        scale_x = viewport_size.width() / image_size.width()  # Calculate horizontal scale needed to fit
        scale_y = viewport_size.height() / image_size.height()  # Calculate vertical scale needed to fit
        
        self.scale_factor = min(scale_x, scale_y) * 0.95  # Use smaller scale to fit both dimensions, with 5% margin
        self.update_display()  # Apply the calculated zoom level


class ScriptCleaner:
    """Cleans scripts with Unicode errors and creates compatible wrappers"""
    
    @staticmethod
    def clean_script_content(content: str) -> str:
        """Remove hardcoded paths and fix common issues"""
        # Remove hardcoded paths
        content = re.sub(r'[a-zA-Z]:\\[^"\'\s\n]+', '', content)  # Remove Windows absolute paths like C:\folder\file
        content = re.sub(r'img_path\s*=\s*["\'][^"\']*["\']', '', content)  # Remove lines that set img_path variable
        content = re.sub(r'image_path\s*=\s*["\'][^"\']*["\']', '', content)  # Remove lines that set image_path variable
        content = re.sub(r'base_path\s*=\s*[^\n]+', '', content)  # Remove lines that set base_path variable
        
        return content  # Return the cleaned script content
        
    @staticmethod
    def create_process_image_wrapper(script_content: str, script_name: str) -> str:
        """Create a process_image function wrapper from script content"""
        # Try to identify the main processing logic
        if 'def process_image' in script_content:  # Check if script already has the required function
            return script_content  # Already has the right format
            
        # Extract the core processing operations
        operations = []  # List to store detected image processing operations
        op_patterns = {  # Dictionary mapping OpenCV function patterns to operation names
            'cv2.GaussianBlur': 'gaussian_blur',
            'cv2.Canny': 'edge_detection',
            'cv2.threshold': 'threshold',
            'cv2.equalizeHist': 'histogram_equalization',
            'cv2.morphologyEx': 'morphology',
            'cv2.HoughCircles': 'circle_detection',
            'cv2.medianBlur': 'median_filter',
            'cv2.Sobel': 'sobel_edge',
            'cv2.Laplacian': 'laplacian_edge',
            'cv2.createCLAHE': 'clahe',
            'cv2.cvtColor.*GRAY': 'grayscale',
        }
        
        detected_ops = []  # List to collect operations found in the script
        for pattern, op_name in op_patterns.items():  # Iterate through each pattern to search for
            if re.search(pattern, script_content):  # Search for the pattern in script content
                detected_ops.append(op_name)  # Add operation name to detected list
                
        # Create a generic wrapper
        wrapper = f'''"""
Auto-generated wrapper for {script_name}
Detected operations: {', '.join(detected_ops) if detected_ops else 'unknown'}
"""
import cv2
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using {script_name} logic"""
    try:
        # Default implementation - modify based on original script
        result = image.copy()
        
        # Add your processing here based on the original script
        {ScriptCleaner._generate_default_processing(detected_ops)}
        
        return result
    except Exception as e:
        print(f"Error in {script_name}: {{e}}")
        return image
'''  # Create a template wrapper function with detected operations
        return wrapper  # Return the generated wrapper code
        
    @staticmethod
    def _generate_default_processing(operations: List[str]) -> str:
        """Generate default processing code based on detected operations"""
        code_lines = []  # List to collect generated code lines
        
        if 'grayscale' in operations:  # Check if grayscale conversion was detected
            code_lines.append("if len(result.shape) == 3:")  # Add condition to check if image is color
            code_lines.append("    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)")  # Convert to grayscale
            
        if 'gaussian_blur' in operations:  # Check if Gaussian blur was detected
            code_lines.append("result = cv2.GaussianBlur(result, (5, 5), 0)")  # Apply default Gaussian blur
            
        if 'edge_detection' in operations:  # Check if edge detection was detected
            code_lines.append("result = cv2.Canny(result, 50, 150)")  # Apply Canny edge detection with default thresholds
            
        if 'threshold' in operations:  # Check if thresholding was detected
            code_lines.append("_, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)")  # Apply binary threshold with default value
            
        return '\n        '.join(code_lines) if code_lines else "# Implement processing logic here"  # Join code lines with proper indentation or return placeholder


class PipelineWorker(QThread):
    """Background thread for executing the image processing pipeline"""
    
    progress = pyqtSignal(int, str)  # Signal to emit progress percentage and status message
    finished = pyqtSignal(np.ndarray)  # Signal to emit the final processed image when complete
    error = pyqtSignal(str)  # Signal to emit error messages if processing fails
    script_executing = pyqtSignal(str)  # Signal to emit the exact filename being executed
    
    def __init__(self):
        super().__init__()  # Initialize the parent QThread
        self.image = None  # Store the input image to be processed
        self.pipeline = []  # Store the list of processing steps to execute
        self.functions = {}  # Store the dictionary of available processing functions
        
    def set_data(self, image: np.ndarray, pipeline: List[Dict], functions: Dict[str, Callable]):
        self.image = image.copy()  # Create a copy of the input image to avoid modifying the original
        self.pipeline = pipeline  # Store the processing pipeline steps
        self.functions = functions  # Store the available processing functions dictionary
        
    def run(self):
        try:
            result = self.image.copy()  # Start with a copy of the original image
            total_steps = len(self.pipeline)  # Count total processing steps for progress calculation
            
            for i, step in enumerate(self.pipeline):  # Iterate through each processing step
                func_name = step['name']  # Extract the function name for this step
                params = step['params']  # Extract the parameters for this step
                
                # Emit the exact script filename being executed
                self.script_executing.emit(f"Executing: {func_name}")  # Notify UI which script is running
                
                # Emit progress with script name
                status_msg = f"Step {i+1}/{total_steps}: Applying {func_name}"  # Create progress status message
                self.progress.emit(int((i / total_steps) * 100), status_msg)  # Emit progress percentage and message
                
                if func_name in self.functions:  # Check if the required function exists
                    func = self.functions[func_name]  # Get the function reference
                    result = func(result, **params)  # Call the function with current image and parameters
                else:
                    raise RuntimeError(f"Function '{func_name}' not found.")  # Raise error if function is missing
                    
            self.progress.emit(100, "Processing complete!")  # Emit 100% completion signal
            self.finished.emit(result)  # Emit the final processed image
            
        except Exception as e:  # Catch any errors during processing
            error_msg = f"Error in '{func_name}': {str(e)}\n\n{traceback.format_exc()}"  # Format detailed error message
            self.error.emit(error_msg)  # Emit the error message to the UI


class FunctionLoader:
    """Enhanced function loader with Unicode error handling"""
    
    def __init__(self, directory: str = "scripts"):
        self.dir = Path(directory)  # Convert directory path to Path object for easier manipulation
        self.functions: Dict[str, Callable] = {}  # Dictionary to store loaded function references
        self.function_info: Dict[str, Dict] = {}  # Dictionary to store metadata about each function
        self.script_cleaner = ScriptCleaner()  # Create instance of script cleaner for handling problematic scripts
        
    def scan(self):
        """Scan directory for functions with enhanced error handling"""
        self.functions.clear()  # Clear previously loaded functions
        self.function_info.clear()  # Clear previously loaded function metadata
        
        if not self.dir.exists():  # Check if scripts directory exists
            self.dir.mkdir(parents=True)  # Create directory and any missing parent directories
            
        # Create a cleaned scripts directory
        cleaned_dir = self.dir / "cleaned"  # Create path for cleaned script cache
        cleaned_dir.mkdir(exist_ok=True)  # Create cleaned directory if it doesn't exist
        
        successful_loads = 0  # Counter for successfully loaded functions
        failed_loads = []  # List to track files that failed to load
        
        for file_path in self.dir.glob("*.py"):  # Iterate through all Python files in directory
            if file_path.name.startswith("_") or file_path.stem == "cleaned":  # Skip private files and cleaned directory
                continue
                
            try:
                # First, try to load directly
                module = self._load_module_direct(file_path)  # Attempt direct module loading
                
                if module and hasattr(module, 'process_image'):  # Check if module has required function
                    self._register_function(file_path, module)  # Register the function if successful
                    successful_loads += 1  # Increment success counter
                else:
                    # Try to clean and create wrapper
                    cleaned_module = self._clean_and_load(file_path, cleaned_dir)  # Attempt to clean and load
                    if cleaned_module:  # Check if cleaning was successful
                        self._register_function(file_path, cleaned_module)  # Register the cleaned function
                        successful_loads += 1  # Increment success counter
                    else:
                        failed_loads.append(file_path.name)  # Add to failed list
                        
            except Exception as e:  # Catch any loading exceptions
                # Try to clean and create wrapper
                try:
                    cleaned_module = self._clean_and_load(file_path, cleaned_dir)  # Attempt cleaning as fallback
                    if cleaned_module:  # Check if fallback cleaning worked
                        self._register_function(file_path, cleaned_module)  # Register the function
                        successful_loads += 1  # Increment success counter
                    else:
                        failed_loads.append(file_path.name)  # Add to failed list
                except:
                    failed_loads.append(file_path.name)  # Add to failed list if all attempts fail
                    print(f"Failed to load {file_path.name}: {e}")  # Print error message
                    
        print(f"Successfully loaded {successful_loads} functions")  # Report success count
        if failed_loads:  # Check if any files failed to load
            print(f"Failed to load: {', '.join(failed_loads)}")  # Report failed files
            
    def _load_module_direct(self, file_path: Path):
        """Try to load module directly"""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)  # Create module specification from file path
            module = importlib.util.module_from_spec(spec)  # Create module object from specification
            spec.loader.exec_module(module)  # Execute the module code to load it into memory
            return module  # Return the loaded module
        except:
            return None  # Return None if loading fails
            
    def _clean_and_load(self, file_path: Path, cleaned_dir: Path):
        """Clean script and create wrapper"""
        try:
            # Read with error handling
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # Open file with Unicode error handling
                content = f.read()  # Read the entire file content
                
            # Clean the content
            cleaned_content = self.script_cleaner.clean_script_content(content)  # Remove problematic patterns from script
            
            # Create wrapper if needed
            if 'def process_image' not in cleaned_content:  # Check if script lacks required function
                cleaned_content = self.script_cleaner.create_process_image_wrapper(  # Generate wrapper function
                    cleaned_content, file_path.stem
                )
                
            # Save cleaned version
            cleaned_path = cleaned_dir / file_path.name  # Create path for cleaned script in cache directory
            with open(cleaned_path, 'w', encoding='utf-8') as f:  # Open cleaned file for writing
                f.write(cleaned_content)  # Write the cleaned content to cache file
                
            # Load the cleaned module
            return self._load_module_direct(cleaned_path)  # Load the cleaned script as a module
            
        except Exception as e:  # Catch any errors during cleaning process
            print(f"Error cleaning {file_path.name}: {e}")  # Print error message
            return None  # Return None if cleaning fails
            
    def _register_function(self, file_path: Path, module):
        """Register a function from a module"""
        func = getattr(module, 'process_image')  # Extract the process_image function from the loaded module
        func_name = file_path.name  # Use full filename as identifier
        
        self.functions[func_name] = func  # Store function reference in functions dictionary
        self.function_info[func_name] = {  # Store metadata about the function
            'name': func_name,  # Store the function identifier name
            'doc': inspect.getdoc(func) or f"Process image using {file_path.stem}",  # Get docstring or create default description
            'params': self._get_params(func),  # Extract parameter information from function signature
            'category': self._categorize_function(file_path.name)  # Determine function category based on filename
        }
        
    def _get_params(self, func: Callable) -> Dict[str, Dict]:
        """Extract parameter information"""
        params = {}  # Dictionary to store parameter metadata
        sig = inspect.signature(func)  # Get function signature using introspection
        
        for name, param in sig.parameters.items():  # Iterate through each parameter in signature
            if name == 'image':  # Skip the main image parameter
                continue
            params[name] = {  # Store parameter metadata
                'type': param.annotation if param.annotation != param.empty else str,  # Use annotation or default to str
                'default': param.default if param.default != param.empty else None  # Use default value or None
            }
        return params  # Return the parameter metadata dictionary
        
    def _categorize_function(self, filename: str) -> str:
        """Categorize function based on filename"""
        filename_lower = filename.lower()  # Convert filename to lowercase for case-insensitive matching
        
        categories = {  # Dictionary mapping categories to keyword lists for classification
            'Filtering': ['blur', 'filter', 'median', 'gaussian'],
            'Edge Detection': ['edge', 'canny', 'sobel', 'laplacian', 'gradient'],
            'Thresholding': ['threshold', 'thresh', 'binary', 'otsu'],
            'Morphology': ['morph', 'erode', 'dilate', 'open', 'close'],
            'Enhancement': ['enhance', 'clahe', 'histogram', 'equalize', 'contrast'],
            'Color': ['color', 'gray', 'grayscale', 'hsv', 'rgb'],
            'Detection': ['detect', 'find', 'circle', 'contour', 'hough'],
            'Transform': ['transform', 'rotate', 'scale', 'resize', 'warp'],
            'Analysis': ['analyze', 'measure', 'profile', 'intensity'],
            'Visualization': ['visualize', 'viz', 'display', 'show', 'heatmap', 'colormap'],
            'Masking': ['mask', 'roi', 'region'],
            'I/O': ['load', 'save', 'read', 'write'],
        }
        
        for category, keywords in categories.items():  # Iterate through each category and its keywords
            if any(keyword in filename_lower for keyword in keywords):  # Check if any keyword matches filename
                return category  # Return the first matching category
                
        return "Other"  # Return default category if no keywords match


class MainWindow(QMainWindow):
    """Main application window with enhanced features"""
    
    def __init__(self):
        super().__init__()  # Initialize the parent QMainWindow

        # ——— State placeholders ———
        self.current_image   = None  # Store the original loaded image
        self.processed_image = None  # Store the result after processing pipeline

        self.current_scripts_dir = "scripts"  # Default directory for loading processing scripts
        # ——— Recent & favorite functions ———
        self.recent_functions   = []      # Store up to 20 most-recently used functions
        self.favorite_functions = set()   # Store user-marked favorite functions (persisted to disk)
        self._favorites_file    = Path("favorites.json")  # File path for saving favorite functions

        if self._favorites_file.exists():
            try:
                with open(self._favorites_file, "r") as f:  # Open favorites file for reading
                    favs = json.load(f)  # Parse JSON data from file
                    # ensure it’s a list before converting
                    if isinstance(favs, list):  # Validate that loaded data is a list
                        self.favorite_functions = set(favs)  # Convert list to set for fast lookup
            except (json.JSONDecodeError, IOError):  # Handle file corruption or access errors
                # corrupted file or IO issues → start fresh
                self.favorite_functions = set()  # Initialize empty favorites set


        self._scripts_dir_file = Path("last_scripts_dir.txt")  # File to remember last used scripts directory
        if self._scripts_dir_file.exists():  # Check if directory preference file exists
            try:
                with open(self._scripts_dir_file, "r") as f:  # Open directory preference file
                    saved_dir = f.read().strip()  # Read and clean directory path
                    if Path(saved_dir).exists():  # Verify the saved directory still exists
                        self.current_scripts_dir = saved_dir  # Use saved directory
            except:
                pass  # Use default if reading fails

        # ——— Core components ———
        self.function_loader = FunctionLoader(self.current_scripts_dir)  # Create function loader for scripts directory
        self.worker          = PipelineWorker()  # Create background worker for processing pipeline

        # ——— Wire up worker signals ———
        self.worker.progress          .connect(self.update_progress)  # Connect progress updates to UI handler
        self.worker.finished          .connect(self.on_processing_finished)  # Connect completion signal to UI handler
        self.worker.error             .connect(self.on_processing_error)  # Connect error signal to UI handler
        self.worker.script_executing  .connect(self.update_executing_script)  # Connect script name updates to UI handler

        # ——— Build UI & load functions ———
        self.init_ui()  # Initialize the user interface components
        self.load_functions()  # Load available processing functions from scripts directory
        
    def init_ui(self):
        self.setWindowTitle("Advanced Image Processing Pipeline Studio")
        self.setGeometry(50, 50, 1600, 900)
        
        # Apply modern stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 8px;
                border-radius: 4px;
                background-color: #2196F3;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QPushButton#ProcessBtn {
                background-color: #4CAF50;
                font-size: 14px;
                padding: 12px;
            }
            QPushButton#ProcessBtn:hover {
                background-color: #45a049;
            }
            QPushButton#ResetBtn {
                background-color: #FF5722;
            }
            QPushButton#ResetBtn:hover {
                background-color: #E64A19;
            }
            QPushButton#UndoBtn {
                background-color: #FF9800;
            }
            QPushButton#UndoBtn:hover {
                background-color: #F57C00;
            }
            QPushButton#ToggleBtn {
                background-color: #9C27B0;
            }
            QPushButton#ToggleBtn:hover {
                background-color: #7B1FA2;
            }
            QPushButton#ToggleBtn:checked {
                background-color: #E91E63;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QLabel#ExecutingScript {
                font-weight: bold;
                color: #D32F2F;
                padding: 8px;
                background-color: #FFEBEE;
                border: 1px solid #FFCDD2;
                border-radius: 4px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panels
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([350, 900, 350])
        
        # Status bar with progress
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Executing script label
        self.executing_label = QLabel("")
        self.executing_label.setObjectName("ExecutingScript")
        self.executing_label.setVisible(False)
        self.status_bar.addPermanentWidget(self.executing_label)
        
    def _create_left_panel(self):
        """Create function library panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Function Library Group
        group = QGroupBox("Function Library")
        group_layout = QVBoxLayout(group)
        filter_layout = QHBoxLayout()
        self.show_recent_cb = QCheckBox("Show Recent")
        self.show_fav_cb    = QCheckBox("Show Favorites")
        self.show_recent_cb.stateChanged.connect(self._apply_filters)
        self.show_fav_cb.stateChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.show_recent_cb)
        filter_layout.addWidget(self.show_fav_cb)
        group_layout.insertLayout(1, filter_layout)  # insert just below the search row

        # Search
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter keywords...")
        self.search_input.textChanged.connect(self.filter_functions)
        search_layout.addWidget(self.search_input)
        group_layout.addLayout(search_layout)
        
        # Category filter
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.currentTextChanged.connect(self.filter_by_category)
        category_layout.addWidget(self.category_combo)
        group_layout.addLayout(category_layout)
        
        # Function table
        self.function_table = QTableWidget()
        self.function_table.setColumnCount(2)
        self.function_table.setHorizontalHeaderLabels(["Script File", "Category"])
        self.function_table.horizontalHeader().setStretchLastSection(True)
        self.function_table.setSelectionBehavior(QTableWidget.SelectRows)
        #self.function_table.itemDoubleClicked.connect(self.add_to_pipeline)
        self.function_table.itemDoubleClicked.connect(self._open_function_file)

        
        self.function_table.itemSelectionChanged.connect(self.show_function_details)
        group_layout.addWidget(self.function_table)
        
        # Function details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        self.details_text.setReadOnly(True)
        group_layout.addWidget(self.details_text)
        
        # Add button
        add_btn = QPushButton("Add to Pipeline →")
        add_btn.clicked.connect(self.add_to_pipeline)
        group_layout.addWidget(add_btn)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Functions")
        refresh_btn.clicked.connect(self.load_functions)
        group_layout.addWidget(refresh_btn)
        
        folder_btn = QPushButton("📁 Choose Scripts Folder")
        folder_btn.clicked.connect(self.choose_scripts_folder)
        folder_btn.setToolTip(f"Current: {self.current_scripts_dir}")
        group_layout.addWidget(folder_btn)

        # Store reference to update tooltip later
        self.folder_btn = folder_btn
        # ——— New: toggle favorite on selected function ———
        fav_btn = QPushButton("★ Toggle Favorite")
        fav_btn.clicked.connect(self._toggle_favorite)
        group_layout.addWidget(fav_btn)

        # ——— New: full‑cache refresh ———
        full_refresh_btn = QPushButton("🔄 Full Refresh Cache")
        full_refresh_btn.clicked.connect(self._full_refresh_cache)
        group_layout.addWidget(full_refresh_btn)
        
        layout.addWidget(group)
        return panel
        
    def _create_center_panel(self):
        """Create image viewer panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        toolbar.addWidget(load_btn)
        
        save_btn = QPushButton("Save Result")
        save_btn.clicked.connect(self.save_image)
        toolbar.addWidget(save_btn)
        
        toolbar.addWidget(QLabel(" | "))
        
        # Reset button
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setObjectName("ResetBtn")
        self.reset_btn.clicked.connect(self.reset_to_original)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setToolTip("Reset to original image (Ctrl+R)")
        self.reset_btn.setShortcut("Ctrl+R")
        toolbar.addWidget(self.reset_btn)
        
        # Toggle original view button
        self.toggle_original_btn = QPushButton("View Original")
        self.toggle_original_btn.setObjectName("ToggleBtn")
        self.toggle_original_btn.setCheckable(True)
        self.toggle_original_btn.toggled.connect(self.toggle_original_view)
        self.toggle_original_btn.setEnabled(False)
        self.toggle_original_btn.setToolTip("Toggle original/processed view (Space)")
        self.toggle_original_btn.setShortcut("Space")
        toolbar.addWidget(self.toggle_original_btn)
        
        toolbar.addStretch()
        
        # Zoom controls
        toolbar.addWidget(QLabel("Zoom:"))
        
        zoom_out_btn = QPushButton("-")
        zoom_out_btn.setMaximumWidth(30)
        zoom_out_btn.clicked.connect(self.zoom_out)
        toolbar.addWidget(zoom_out_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(60)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        toolbar.addWidget(self.zoom_label)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setMaximumWidth(30)
        zoom_in_btn.clicked.connect(self.zoom_in)
        toolbar.addWidget(zoom_in_btn)
        
        zoom_fit_btn = QPushButton("Fit")
        zoom_fit_btn.clicked.connect(self.zoom_fit)
        toolbar.addWidget(zoom_fit_btn)
        
        zoom_reset_btn = QPushButton("100%")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        toolbar.addWidget(zoom_reset_btn)
        
        layout.addLayout(toolbar)
        
        # Image viewer
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer)
        
        # Image info
        self.image_info_label = QLabel("No image loaded")
        self.image_info_label.setStyleSheet(
            "padding: 8px; background-color: #e8e8e8; border-radius: 4px;"
        )
        layout.addWidget(self.image_info_label)
        
        return panel
        
    def _create_right_panel(self):
        """Create pipeline panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Pipeline group
        group = QGroupBox("Processing Pipeline")
        group_layout = QVBoxLayout(group)
        
        # Pipeline list
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QListWidget.InternalMove)
        self.pipeline_list.itemDoubleClicked.connect(self.edit_pipeline_params)
        group_layout.addWidget(self.pipeline_list)
        
        # Pipeline controls
        controls = QHBoxLayout()
        
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_from_pipeline)
        controls.addWidget(remove_btn)
        
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.setObjectName("UndoBtn")
        self.undo_btn.clicked.connect(self.undo_last_step)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setToolTip("Undo last processing step (Ctrl+Z)")
        self.undo_btn.setShortcut("Ctrl+Z")
        controls.addWidget(self.undo_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_pipeline)
        controls.addWidget(clear_btn)
        
        group_layout.addLayout(controls)
        
        # Save/Load pipeline
        io_layout = QHBoxLayout()
        
        save_pipe_btn = QPushButton("Save Pipeline")
        save_pipe_btn.clicked.connect(self.save_pipeline)
        io_layout.addWidget(save_pipe_btn)
        
        load_pipe_btn = QPushButton("Load Pipeline")
        load_pipe_btn.clicked.connect(self.load_pipeline)
        io_layout.addWidget(load_pipe_btn)
        
        group_layout.addLayout(io_layout)
        
        # Process button
        self.process_btn = QPushButton("PROCESS IMAGE")
        self.process_btn.setObjectName("ProcessBtn")
        self.process_btn.clicked.connect(self.process_image)
        group_layout.addWidget(self.process_btn)
        
        layout.addWidget(group)
        
        # Currently executing script display
        exec_group = QGroupBox("Execution Status")
        exec_layout = QVBoxLayout(exec_group)
        
        self.current_script_label = QLabel("Ready")
        self.current_script_label.setWordWrap(True)
        self.current_script_label.setStyleSheet(
            "padding: 10px; background-color: #f0f0f0; border-radius: 4px;"
        )
        exec_layout.addWidget(self.current_script_label)
        
        layout.addWidget(exec_group)
        
        return panel
        
    def load_functions(self):
        """Load all available functions"""
        self.function_loader.scan()  # Scan scripts directory for processing functions
        self.populate_function_table()  # Update the function list in the UI table
        self.update_categories()  # Refresh the category dropdown with available categories
        
        # Add this to update the folder button tooltip
        if hasattr(self, 'folder_btn'):  # Check if folder button exists in UI
            folder_name = os.path.basename(self.current_scripts_dir) or self.current_scripts_dir  # Get display name for directory
            self.folder_btn.setToolTip(f"Current: {self.current_scripts_dir}")  # Update button tooltip with current directory
        
        self.status_bar.showMessage(  # Display loading status in status bar
            f"Loaded {len(self.function_loader.functions)} functions from '{os.path.basename(self.current_scripts_dir) or self.current_scripts_dir}'", 
            3000  # Show message for 3 seconds
        )
        
    def choose_scripts_folder(self):
        """Let user choose a different scripts folder"""
        folder = QFileDialog.getExistingDirectory(  # Open folder selection dialog
            self, 
            "Select Scripts Folder",  # Dialog title
            self.current_scripts_dir,  # Start in current scripts directory
            QFileDialog.ShowDirsOnly  # Only show directories, not files
        )
        
        if folder:  # Check if user selected a folder (not cancelled)
            self.current_scripts_dir = folder  # Update current scripts directory
            
            # Save the selection (add this)
            try:
                with open(self._scripts_dir_file, "w") as f:  # Open preference file for writing
                    f.write(folder)  # Save the selected directory path
            except:
                pass  # Ignore save errors
            
            # Update the function loader with new directory
            self.function_loader = FunctionLoader(self.current_scripts_dir)  # Create new function loader for new directory
            # Reload functions from new directory
            self.load_functions()  # Refresh function list from new directory
            
            # Update status to show current folder
            folder_name = os.path.basename(folder) or folder  # Get display name for status
            self.status_bar.showMessage(  # Show confirmation message
                f"Scripts folder changed to: {folder_name}", 5000  # Display for 5 seconds
            ) 
    # def populate_function_table(self, filter_text="", category="All"):
    #     """Populate the function table"""
    #     self.function_table.setRowCount(0)
    #
    #     for func_name, func_info in sorted(self.function_loader.function_info.items()):
    #         # Apply filters
    #         if filter_text and filter_text.lower() not in func_name.lower():
    #             continue
    #         if category != "All" and func_info['category'] != category:
    #             continue
    #
    #         row = self.function_table.rowCount()
    #         self.function_table.insertRow(row)
    #
    #         # Script name
    #         name_item = QTableWidgetItem(func_name)
    #         name_item.setData(Qt.UserRole, func_name)
    #         self.function_table.setItem(row, 0, name_item)
    #
    #         # Category
    #         category_item = QTableWidgetItem(func_info['category'])
    #         self.function_table.setItem(row, 1, category_item)
    #    self.function_table.resizeColumnsToContents()
    
    
    def populate_function_table(self, filter_text="", category="All"):
        self.function_table.setRowCount(0)  # Clear all existing rows from the table
        for func_name, info in sorted(self.function_loader.function_info.items()):  # Iterate through all loaded functions in alphabetical order
            # text & category filters
            if filter_text and filter_text.lower() not in func_name.lower():  # Skip if search text doesn't match function name
                continue
            if category != "All" and info['category'] != category:  # Skip if category filter doesn't match
                continue
            # recent / favorite filters
            if self.show_recent_cb.isChecked() and func_name not in self.recent_functions:  # Skip if showing recent only and function not recent
                continue
            if self.show_fav_cb.isChecked() and func_name not in self.favorite_functions:  # Skip if showing favorites only and function not favorited
                continue

            row = self.function_table.rowCount()  # Get current row count for inserting new row
            self.function_table.insertRow(row)  # Add a new row to the table
            name_item = QTableWidgetItem(func_name)  # Create table item for function name
            name_item.setData(Qt.UserRole, func_name)  # Store function name as item data for retrieval
            self.function_table.setItem(row, 0, name_item)  # Set function name in first column
            self.function_table.setItem(row, 1, QTableWidgetItem(info['category']))  # Set category in second column
        self.function_table.resizeColumnsToContents()  # Auto-resize columns to fit content

        
    def update_categories(self):
        """Update category combo box"""
        categories = set()
        for func_info in self.function_loader.function_info.values():
            categories.add(func_info['category'])
            
        self.category_combo.clear()
        self.category_combo.addItem("All")
        for category in sorted(categories):
            self.category_combo.addItem(category)
            
    def filter_functions(self, text):
        """Called when the search input changes."""
        self._apply_filters()

    def filter_by_category(self, category):
        """Called when the category combo changes."""
        self._apply_filters()

    def _apply_filters(self):
        """Read current search text and category, then repopulate."""
        current_text = self.search_input.text()
        current_category = self.category_combo.currentText()
        self.populate_function_table(
            filter_text=current_text,
            category=current_category
        )
        
    def show_function_details(self):
        """Show details of selected function"""
        current_row = self.function_table.currentRow()
        if current_row < 0:
            return
            
        name_item = self.function_table.item(current_row, 0)
        if name_item:
            func_name = name_item.data(Qt.UserRole)
            func_info = self.function_loader.function_info.get(func_name, {})
            
            details = f"<b>File:</b> {func_name}<br>"
            details += f"<b>Category:</b> {func_info.get('category', 'Unknown')}<br>"
            details += f"<b>Description:</b> {func_info.get('doc', 'No description')}<br>"
            
            if func_info.get('params'):
                details += "<br><b>Parameters:</b><br>"
                for param_name, param_info in func_info['params'].items():
                    param_type = param_info['type'].__name__ if hasattr(param_info['type'], '__name__') else str(param_info['type'])
                    details += f"• {param_name} ({param_type})"
                    if param_info['default'] is not None:
                        details += f" = {param_info['default']}"
                    details += "<br>"
                    
            self.details_text.setHtml(details)
            
    def add_to_pipeline(self):
        """Add selected function to pipeline"""
        current_row = self.function_table.currentRow()  # Get index of currently selected row
        if current_row < 0:  # Check if no row is selected
            return  # Exit if nothing selected
            
        name_item = self.function_table.item(current_row, 0)  # Get function name item from first column
        if name_item:  # Check if item exists
            func_name = name_item.data(Qt.UserRole)  # Extract function name from stored data
            func_info = self.function_loader.function_info[func_name]  # Get function metadata
            
            # Create pipeline step
            pipeline_step = {  # Create dictionary representing this pipeline step
                'name': func_name,  # Store function name
                'params': {  # Create parameters dictionary with default values
                    name: p_info['default']   # Use parameter name as key and default value
                    for name, p_info in func_info['params'].items()  # Iterate through function parameters
                    if p_info['default'] is not None  # Only include parameters that have default values
                }
            }
            
            # Add to list
            item_text = self._format_pipeline_item(pipeline_step)  # Create display text for pipeline item
            list_item = QListWidgetItem(item_text)  # Create list widget item with formatted text
            list_item.setData(Qt.UserRole, pipeline_step)  # Store pipeline step data in item
            self.pipeline_list.addItem(list_item)  # Add item to pipeline list widget
            
    def _format_pipeline_item(self, step):
        """Format pipeline item text"""
        params_str = ", ".join(f"{k}={v}" for k, v in step['params'].items())  # Create parameter string with key=value pairs
        return f"{step['name']}" + (f" ({params_str})" if params_str else "")  # Return function name with parameters if any exist
        
    def edit_pipeline_params(self, item):
        """Edit parameters of pipeline item"""
        pipeline_step = item.data(Qt.UserRole)
        func_name = pipeline_step['name']
        func_info = self.function_loader.function_info[func_name]
        
        if not func_info['params']:
            QMessageBox.information(
                self, "No Parameters",
                f"'{func_name}' has no parameters to edit."
            )
            return
            
        # Create parameter dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Parameters - {func_name}")
        dialog.setMinimumWidth(400)
        
        form_layout = QFormLayout(dialog)
        widgets = {}
        
        for name, p_info in func_info['params'].items():
            current_val = pipeline_step['params'].get(name, p_info['default'])
            widget = None
            
            if p_info['type'] is bool:
                widget = QCheckBox()
                if current_val:
                    widget.setChecked(True)
            elif p_info['type'] is int:
                widget = QSpinBox()
                widget.setRange(-10000, 10000)
                if current_val is not None:
                    widget.setValue(current_val)
            elif p_info['type'] is float:
                widget = QDoubleSpinBox()
                widget.setRange(-10000.0, 10000.0)
                widget.setDecimals(4)
                if current_val is not None:
                    widget.setValue(current_val)
            else:
                widget = QLineEdit()
                if current_val is not None:
                    widget.setText(str(current_val))
                    
            if widget:
                param_label = QLabel(f"{name} ({p_info['type'].__name__}):")
                form_layout.addRow(param_label, widget)
                widgets[name] = widget
                
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form_layout.addRow(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # Update parameters
            for name, widget in widgets.items():
                if isinstance(widget, QCheckBox):
                    pipeline_step['params'][name] = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    pipeline_step['params'][name] = widget.value()
                else:
                    # Try to evaluate the string
                    text = widget.text()
                    try:
                        value = eval(text)
                    except:
                        value = text
                    pipeline_step['params'][name] = value
                    
            # Update item text
            item.setText(self._format_pipeline_item(pipeline_step))
            item.setData(Qt.UserRole, pipeline_step)
            
    def remove_from_pipeline(self):
        """Remove selected item from pipeline"""
        current_row = self.pipeline_list.currentRow()
        if current_row >= 0:
            self.pipeline_list.takeItem(current_row)
            
    def clear_pipeline(self):
        """Clear entire pipeline"""
        self.pipeline_list.clear()
        self.undo_btn.setEnabled(False)
    def load_image(self):
        """Load an image file"""
        path, _ = QFileDialog.getOpenFileName(  # Open file selection dialog
            self, "Open Image", "",  # Dialog parent, title, starting directory
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"  # File type filters
        )
        
        if path:  # Check if user selected a file (not cancelled)
            self.current_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Load image preserving original bit depth and channels
            if self.current_image is None:  # Check if image loading failed
                QMessageBox.critical(self, "Error", f"Failed to load image from {path}")  # Show error dialog
                return  # Exit if loading failed
                
            self.image_viewer.set_image(self.current_image)  # Display the loaded image in the viewer
            
            # Enable reset and toggle buttons
            self.reset_btn.setEnabled(True)  # Enable reset button now that we have an original image
            self.toggle_original_btn.setEnabled(True)  # Enable toggle button for comparing results
            
            # Update info
            h, w = self.current_image.shape[:2]  # Extract height and width from image shape
            c = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1  # Get channel count (1 for grayscale, 3 for color)
            self.image_info_label.setText(  # Update info label with image details
                f"Loaded: {os.path.basename(path)} | "  # Show filename without path
                f"Size: {w}×{h} | Channels: {c} | "  # Show dimensions and channel count
                f"Type: {self.current_image.dtype}"  # Show data type (uint8, uint16, etc.)
            )
            self.status_bar.showMessage("Image loaded successfully", 3000)  # Show success message for 3 seconds
            
    def save_image(self):
        """Save processed image"""
        if self.processed_image is None:
            QMessageBox.warning(self, "Warning", "No processed image to save")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)"
        )
        
        if path:
            try:
                cv2.imwrite(path, self.processed_image)
                self.status_bar.showMessage(f"Image saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {e}")
                
    def process_image(self):
        """Process the current image through the pipeline, and track recently used functions."""
        # 1) Pre-flight checks
        if self.current_image is None:  # Verify an image has been loaded
            QMessageBox.warning(self, "Warning", "Please load an image first")  # Show warning dialog
            return  # Exit if no image loaded
        if self.pipeline_list.count() == 0:  # Check if pipeline has any steps
            QMessageBox.warning(self, "Warning", "Pipeline is empty")  # Show warning dialog
            return  # Exit if pipeline is empty

        # 2) Gather pipeline steps
        pipeline_data = [  # Extract all pipeline step data from the UI list
            self.pipeline_list.item(i).data(Qt.UserRole)  # Get stored data for each pipeline item
            for i in range(self.pipeline_list.count())  # Iterate through all pipeline items
        ]

        # 3) Update 'recent_functions' (newest first, max 20)
        for step in pipeline_data:  # Process each step in the pipeline
            name = step['name']  # Extract function name from step
            if name in self.recent_functions:  # Check if function is already in recent list
                self.recent_functions.remove(name)  # Remove old entry to avoid duplicates
            self.recent_functions.insert(0, name)  # Add function to front of recent list
        self.recent_functions = self.recent_functions[:20]  # Keep only the 20 most recent functions

        # 4) Update UI to show we're processing
        self.process_btn.setEnabled(False)  # Disable process button during execution
        self.progress_bar.setVisible(True)  # Show progress bar
        self.executing_label.setVisible(True)  # Show currently executing script label
        self.current_script_label.setText("Starting pipeline...")  # Update status text

        # 5) Configure and start the worker thread
        self.worker.set_data(  # Pass data to background worker thread
            image=self.current_image,  # Provide the image to process
            pipeline=pipeline_data,  # Provide the pipeline steps
            functions=self.function_loader.functions  # Provide available functions dictionary
        )
        self.worker.start()  # Start the background processing thread

        
    def update_progress(self, percentage, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)
        self.current_script_label.setText(message)
        
    def update_executing_script(self, script_name):
        """Update the currently executing script display"""
        self.executing_label.setText(f"⚡ {script_name}")
        
    def on_processing_finished(self, result_image):
        """Handle processing completion"""
        self.processed_image = result_image  # Store the final processed image result
        self.image_viewer.set_image(result_image)  # Display the processed image in the viewer
        
        self.process_btn.setEnabled(True)  # Re-enable the process button for next pipeline run
        self.progress_bar.setVisible(False)  # Hide the progress bar
        self.executing_label.setVisible(False)  # Hide the executing script label
        
        # Enable undo button after successful processing
        self.undo_btn.setEnabled(True)  # Enable undo to allow reverting last step
        
        # Make sure toggle button shows correct state
        if self.toggle_original_btn.isChecked():  # Check if user was viewing original
            self.toggle_original_btn.setChecked(False)  # Reset to show processed result
        
        self.current_script_label.setText("✓ Processing complete!")  # Update status with success message
        self.status_bar.showMessage("Processing finished successfully!", 5000)  # Show success in status bar for 5 seconds
        
        # Update image info
        h, w = result_image.shape[:2]  # Extract height and width from result
        c = result_image.shape[2] if len(result_image.shape) > 2 else 1  # Get channel count
        self.image_info_label.setText(  # Update info label with processed image details
            f"Processed | Size: {w}×{h} | Channels: {c} | Type: {result_image.dtype}"  # Show dimensions and type
        )
        
    def on_processing_error(self, error_message):
        """Handle processing error"""
        QMessageBox.critical(self, "Processing Error", error_message)  # Show error dialog with detailed message
        
        self.process_btn.setEnabled(True)  # Re-enable process button to allow retry
        self.progress_bar.setVisible(False)  # Hide progress bar
        self.executing_label.setVisible(False)  # Hide executing script label
        
        self.current_script_label.setText("✗ Processing failed!")  # Update status with failure message
        self.status_bar.showMessage("An error occurred during processing", 5000)  # Show error in status bar for 5 seconds
        
    def zoom_in(self):
        """Zoom in the image"""
        if self.image_viewer.scale_factor < 10:  # Check if zoom level is below maximum (1000%)
            self.image_viewer.scale_factor *= 1.25  # Increase zoom by 25%
            self.image_viewer.update_display()  # Refresh the display with new zoom
            self.update_zoom_label()  # Update the zoom percentage label
            
    def zoom_out(self):
        """Zoom out the image"""
        if self.image_viewer.scale_factor > 0.1:  # Check if zoom level is above minimum (10%)
            self.image_viewer.scale_factor /= 1.25  # Decrease zoom by 20% (inverse of 1.25)
            self.image_viewer.update_display()  # Refresh the display with new zoom
            self.update_zoom_label()  # Update the zoom percentage label
            
    def zoom_fit(self):
        """Fit image to viewport"""
        self.image_viewer.zoom_to_fit()
        self.update_zoom_label()
        
    def zoom_reset(self):
        """Reset zoom to 100%"""
        self.image_viewer.reset_zoom()
        self.update_zoom_label()
        
    def update_zoom_label(self):
        """Update zoom percentage display"""
        zoom_percent = int(self.image_viewer.scale_factor * 100)
        self.zoom_label.setText(f"{zoom_percent}%")
        
    def save_pipeline(self):
        """Save pipeline configuration"""
        if self.pipeline_list.count() == 0:
            QMessageBox.warning(self, "Warning", "Pipeline is empty")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Pipeline", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if path:
            pipeline_data = []
            for i in range(self.pipeline_list.count()):
                item = self.pipeline_list.item(i)
                pipeline_data.append(item.data(Qt.UserRole))
                
            try:
                with open(path, 'w') as f:
                    json.dump(pipeline_data, f, indent=2)
                self.status_bar.showMessage(f"Pipeline saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save pipeline: {e}")
                
    def load_pipeline(self):
        """Load pipeline configuration"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pipeline", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if path:
            try:
                with open(path, 'r') as f:
                    pipeline_data = json.load(f)
                    
                self.pipeline_list.clear()
                
                for step in pipeline_data:
                    item_text = self._format_pipeline_item(step)
                    list_item = QListWidgetItem(item_text)
                    list_item.setData(Qt.UserRole, step)
                    self.pipeline_list.addItem(list_item)
                    
                self.status_bar.showMessage(f"Pipeline loaded from {path}", 3000)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load pipeline: {e}")


    def reset_to_original(self):
        """Reset the displayed image to the original loaded image"""
        if self.current_image is not None:
            self.processed_image = None
            self.image_viewer.set_image(self.current_image)
            
            # Update UI state
            self.undo_btn.setEnabled(False)
            self.toggle_original_btn.setChecked(False)
            
            # Update info
            h, w = self.current_image.shape[:2]
            c = self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1
            self.image_info_label.setText(
                f"Reset to Original | Size: {w}×{h} | Channels: {c} | "
                f"Type: {self.current_image.dtype}"
            )
            self.current_script_label.setText("Reset to original image")
            self.status_bar.showMessage("Reset to original image", 3000)
    
    def undo_last_step(self):
        """Remove the last step from pipeline and reprocess"""
        if self.pipeline_list.count() == 0:
            return
            
        # Remove last item from pipeline
        last_item = self.pipeline_list.takeItem(self.pipeline_list.count() - 1)
        
        # If pipeline is now empty, reset to original
        if self.pipeline_list.count() == 0:
            self.reset_to_original()
            return
            
        # Reprocess with updated pipeline
        self.status_bar.showMessage("Undoing last step and reprocessing...", 1000)
        self.process_image()
    
    def toggle_original_view(self, checked):
        """Toggle between original and processed image view"""
        if self.current_image is None:
            return
            
        if checked:
            # Show original image
            self.image_viewer.set_image(self.current_image)
            self.toggle_original_btn.setText("View Processed")
            self.image_info_label.setText(
                self.image_info_label.text() + " (Viewing Original)"
            )
        else:
            # Show processed image (if available)
            if self.processed_image is not None:
                self.image_viewer.set_image(self.processed_image)
            else:
                self.image_viewer.set_image(self.current_image)
            self.toggle_original_btn.setText("View Original")
            
            # Remove the "(Viewing Original)" text if present
            info_text = self.image_info_label.text()
            if "(Viewing Original)" in info_text:
                self.image_info_label.setText(
                    info_text.replace(" (Viewing Original)", "")
                )
                
    def _toggle_favorite(self):
        """Add or remove the selected function from favorites."""
        row = self.function_table.currentRow()
        if row < 0:
            return
        func_name = self.function_table.item(row, 0).data(Qt.UserRole)
        if func_name in self.favorite_functions:
            self.favorite_functions.remove(func_name)
        else:
            self.favorite_functions.add(func_name)
        # persist to disk immediately
        with open(self._favorites_file, "w") as f:
            json.dump(sorted(self.favorite_functions), f)
        self._apply_filters()
        self.status_bar.showMessage(
            f"{func_name} {'removed from' if func_name not in self.favorite_functions else 'added to'} favorites",
            3000
        )

    def _full_refresh_cache(self):
        """Delete the cleaned scripts cache and reload everything."""
        cleaned_dir = Path(self.function_loader.dir) / "cleaned"
        if cleaned_dir.exists():
            shutil.rmtree(cleaned_dir)
        self.load_functions()
        self.status_bar.showMessage("Cache fully refreshed", 3000)

    def _open_function_file(self, item):
        """Open the .py file for this function using the OS default editor."""
        func_name = item.data(Qt.UserRole)
        script_path = Path(self.function_loader.dir) / func_name

        if not script_path.exists():
            QMessageBox.warning(self, "File Not Found",
                                f"Cannot locate {script_path}")
            return

        try:
            if sys.platform.startswith("win"):
                # Windows: use the default file‐association opener
                os.startfile(str(script_path))

            elif sys.platform.startswith("darwin"):
                # macOS
                subprocess.Popen(["open", str(script_path)])

            else:
                # Linux / other UNIX
                subprocess.Popen(["xdg-open", str(script_path)])

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Opening File",
                f"Failed to open {script_path}:\n{e}"
            )



def main():
    """Main entry point"""
    app = QApplication(sys.argv)  # Create the main application object with command line arguments
    app.setStyle('Fusion')  # Use Fusion style for modern cross-platform appearance
    
    # Set application properties
    app.setApplicationName("Image Processing Pipeline Studio")  # Set application name for system integration
    app.setOrganizationName("OpenCV Practice")  # Set organization name for settings storage
    
    # Create and show main window
    window = MainWindow()  # Create the main application window
    window.show()  # Display the window on screen
    
    sys.exit(app.exec_())  # Start the event loop and exit when window closes


if __name__ == '__main__':  # Check if script is run directly (not imported)
    main()  # Call the main function to start the application
