#!/usr/bin/env python3
"""
Fiber Optic Core Detection Tool
Supports Windows and Linux (including Wayland)
Auto-installs missing dependencies
"""

import sys
import os
import subprocess
import importlib.util
import platform

def check_and_install_dependencies():
    """Check for required dependencies and install if missing"""
    
    # Define required packages with their import names and versions
    required_packages = {
        'opencv-python': {'import_name': 'cv2', 'version': '>=4.12.0'},
        'Pillow': {'import_name': 'PIL', 'version': '>=11.3.0'},
        'numpy': {'import_name': 'numpy', 'version': '>=1.20.0'}
    }
    
    optional_packages = {
        'pypylon': {'import_name': 'pypylon', 'version': None}
    }
    
    missing_packages = []
    
    print("Checking dependencies...")
    print("-" * 50)
    
    # Check required packages
    for package, info in required_packages.items():
        import_name = info['import_name']
        spec = importlib.util.find_spec(import_name)
        
        if spec is None:
            print(f"❌ {package} is not installed")
            missing_packages.append(package + (info['version'] if info['version'] else ''))
        else:
            try:
                module = importlib.import_module(import_name)
                if import_name == 'cv2':
                    version = module.__version__
                    print(f"✅ {package} is installed (version: {version})")
                elif import_name == 'PIL':
                    from PIL import __version__
                    print(f"✅ {package} is installed (version: {__version__})")
                elif import_name == 'numpy':
                    version = module.__version__
                    print(f"✅ {package} is installed (version: {version})")
            except Exception as e:
                print(f"⚠️  {package} is installed but cannot determine version")
    
    # Check optional packages
    print("\nOptional packages:")
    pypylon_available = False
    for package, info in optional_packages.items():
        import_name = info['import_name']
        spec = importlib.util.find_spec(import_name)
        
        if spec is None:
            print(f"⚠️  {package} is not installed (optional - for Basler camera support)")
            print(f"   To install: pip install pypylon")
        else:
            pypylon_available = True
            print(f"✅ {package} is installed")
    
    # Install missing packages
    if missing_packages:
        print("\n" + "=" * 50)
        print("Installing missing dependencies...")
        print("=" * 50)
        
        # Upgrade pip first
        print("\nUpgrading pip...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            print("✅ pip upgraded successfully")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to upgrade pip, continuing anyway...")
        
        # Install missing packages
        for package in missing_packages:
            print(f"\nInstalling {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}")
                print(f"Error: {e}")
                print("\nPlease install manually using:")
                print(f"  pip install {package}")
                sys.exit(1)
        
        print("\n" + "=" * 50)
        print("All dependencies installed! Restarting application...")
        print("=" * 50)
        
        # Restart the script
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        print("\n✅ All required dependencies are installed!")
        print("-" * 50)
        return pypylon_available

# Check dependencies before importing
PYLON_AVAILABLE = check_and_install_dependencies()

# Now import the required libraries
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple

# Import pypylon if available
if PYLON_AVAILABLE:
    try:
        from pypylon import pylon
    except ImportError:
        PYLON_AVAILABLE = False

@dataclass
class Circle:
    """Represents a circle with position and size"""
    x: int
    y: int
    diameter: int
    color: Tuple[int, int, int]
    locked: bool = False

class FiberOpticDetector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fiber Optic Core Detector")
        
        # Set window icon if on Windows
        if platform.system() == 'Windows':
            try:
                self.root.iconbitmap(default='')
            except:
                pass
        
        # Image and camera variables
        self.current_image = None
        self.display_image = None
        self.camera = None
        self.pylon_camera = None
        self.capture_thread = None
        self.running = False
        self.image_queue = queue.Queue(maxsize=2)
        
        # Circle parameters
        self.blue_circle = Circle(100, 100, 50, (255, 0, 0))
        self.red_circle = Circle(200, 200, 50, (0, 0, 255))
        self.use_hough = tk.BooleanVar(value=False)
        self.show_blue = tk.BooleanVar(value=True)
        self.show_red = tk.BooleanVar(value=True)
        
        # Mode selection
        self.mode = tk.StringVar(value="static")
        
        # Create GUI
        self.setup_gui()
        
        # Start update loop
        self.update_display()
        
    def setup_gui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Mode selection frame
        mode_frame = ttk.LabelFrame(main_frame, text="Input Mode", padding="5")
        mode_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(mode_frame, text="Static Image", variable=self.mode, 
                       value="static", command=self.change_mode).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(mode_frame, text="Camera", variable=self.mode, 
                       value="camera", command=self.change_mode).grid(row=0, column=1, padx=5)
        if PYLON_AVAILABLE:
            ttk.Radiobutton(mode_frame, text="Pylon Camera", variable=self.mode, 
                           value="pylon", command=self.change_mode).grid(row=0, column=2, padx=5)
        
        ttk.Button(mode_frame, text="Load Image", command=self.load_image).grid(row=0, column=3, padx=20)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Checkbutton(options_frame, text="Use Hough Circle Detection", 
                       variable=self.use_hough, command=self.toggle_hough).grid(row=0, column=0, padx=5)
        ttk.Checkbutton(options_frame, text="Show Blue Circle", 
                       variable=self.show_blue).grid(row=0, column=1, padx=5)
        ttk.Checkbutton(options_frame, text="Show Red Circle", 
                       variable=self.show_red).grid(row=0, column=2, padx=5)
        
        # Blue circle controls
        blue_frame = ttk.LabelFrame(main_frame, text="Blue Circle Controls", padding="5")
        blue_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.blue_diameter = self.create_slider(blue_frame, "Diameter:", 10, 500, 
                                              self.blue_circle.diameter, 0, 
                                              lambda v: self.update_circle('blue', 'diameter', v))
        self.blue_x = self.create_slider(blue_frame, "X Position:", 0, 1000, 
                                        self.blue_circle.x, 1,
                                        lambda v: self.update_circle('blue', 'x', v))
        self.blue_y = self.create_slider(blue_frame, "Y Position:", 0, 1000, 
                                        self.blue_circle.y, 2,
                                        lambda v: self.update_circle('blue', 'y', v))
        
        self.blue_lock_btn = ttk.Button(blue_frame, text="Lock Position", 
                                       command=lambda: self.toggle_lock('blue'))
        self.blue_lock_btn.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Red circle controls
        red_frame = ttk.LabelFrame(main_frame, text="Red Circle Controls", padding="5")
        red_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        self.red_diameter = self.create_slider(red_frame, "Diameter:", 10, 500, 
                                             self.red_circle.diameter, 0,
                                             lambda v: self.update_circle('red', 'diameter', v))
        self.red_x = self.create_slider(red_frame, "X Position:", 0, 1000, 
                                       self.red_circle.x, 1,
                                       lambda v: self.update_circle('red', 'x', v))
        self.red_y = self.create_slider(red_frame, "Y Position:", 0, 1000, 
                                       self.red_circle.y, 2,
                                       lambda v: self.update_circle('red', 'y', v))
        
        self.red_lock_btn = ttk.Button(red_frame, text="Lock Position", 
                                      command=lambda: self.toggle_lock('red'))
        self.red_lock_btn.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Image display
        self.canvas = tk.Canvas(main_frame, width=800, height=600, bg='black')
        self.canvas.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
    def create_slider(self, parent, label, min_val, max_val, initial, row, callback):
        """Create a labeled slider with value display"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=5)
        
        value_var = tk.IntVar(value=initial)
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=value_var,
                          orient=tk.HORIZONTAL, command=lambda v: self.slider_changed(value_var, callback))
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
        
        value_label = ttk.Label(parent, text=str(initial))
        value_label.grid(row=row, column=2, padx=5)
        
        # Store reference to update label
        slider.value_label = value_label
        slider.value_var = value_var
        
        parent.columnconfigure(1, weight=1)
        return slider
        
    def slider_changed(self, value_var, callback):
        """Handle slider value changes"""
        value = int(value_var.get())
        callback(value)
        
    def update_circle(self, circle_name, attribute, value):
        """Update circle parameters"""
        circle = self.blue_circle if circle_name == 'blue' else self.red_circle
        if not circle.locked or attribute == 'diameter':  # Allow diameter changes even when locked
            setattr(circle, attribute, int(value))
            # Update the label
            if circle_name == 'blue':
                if attribute == 'diameter':
                    self.blue_diameter.value_label.config(text=str(int(value)))
                elif attribute == 'x':
                    self.blue_x.value_label.config(text=str(int(value)))
                elif attribute == 'y':
                    self.blue_y.value_label.config(text=str(int(value)))
            else:
                if attribute == 'diameter':
                    self.red_diameter.value_label.config(text=str(int(value)))
                elif attribute == 'x':
                    self.red_x.value_label.config(text=str(int(value)))
                elif attribute == 'y':
                    self.red_y.value_label.config(text=str(int(value)))
                    
    def toggle_lock(self, circle_name):
        """Toggle circle lock state"""
        if circle_name == 'blue':
            self.blue_circle.locked = not self.blue_circle.locked
            btn_text = "Unlock Position" if self.blue_circle.locked else "Lock Position"
            self.blue_lock_btn.config(text=btn_text)
            # Disable/enable position sliders
            state = 'disabled' if self.blue_circle.locked else 'normal'
            self.blue_x.config(state=state)
            self.blue_y.config(state=state)
        else:
            self.red_circle.locked = not self.red_circle.locked
            btn_text = "Unlock Position" if self.red_circle.locked else "Lock Position"
            self.red_lock_btn.config(text=btn_text)
            # Disable/enable position sliders
            state = 'disabled' if self.red_circle.locked else 'normal'
            self.red_x.config(state=state)
            self.red_y.config(state=state)
            
    def toggle_hough(self):
        """Toggle Hough circle detection"""
        if self.use_hough.get() and self.current_image is not None:
            self.detect_circles()
            
    def detect_circles(self):
        """Use Hough circles to detect fiber core"""
        if self.current_image is None:
            return
            
        # Convert to grayscale
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                  param1=50, param2=30, minRadius=10, maxRadius=200)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            if len(circles[0]) > 0:
                # Use first detected circle for blue
                x, y, r = circles[0][0]
                if not self.blue_circle.locked:
                    self.blue_circle.x = int(x)
                    self.blue_circle.y = int(y)
                    self.blue_circle.diameter = int(r * 2)
                    # Update sliders
                    self.blue_x.value_var.set(x)
                    self.blue_y.value_var.set(y)
                    self.blue_diameter.value_var.set(r * 2)
                    
                if len(circles[0]) > 1 and not self.red_circle.locked:
                    # Use second detected circle for red
                    x, y, r = circles[0][1]
                    self.red_circle.x = int(x)
                    self.red_circle.y = int(y)
                    self.red_circle.diameter = int(r * 2)
                    # Update sliders
                    self.red_x.value_var.set(x)
                    self.red_y.value_var.set(y)
                    self.red_diameter.value_var.set(r * 2)
                    
    def load_image(self):
        """Load a static image from file"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), 
                      ("All files", "*.*")]
        )
        if filename:
            self.current_image = cv2.imread(filename)
            if self.current_image is not None:
                self.update_slider_ranges()
                self.status_label.config(text=f"Loaded: {os.path.basename(filename)}")
                if self.use_hough.get():
                    self.detect_circles()
            else:
                messagebox.showerror("Error", "Failed to load image")
                
    def update_slider_ranges(self):
        """Update slider ranges based on image size"""
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            # Update X position sliders
            self.blue_x.config(to=w)
            self.red_x.config(to=w)
            # Update Y position sliders
            self.blue_y.config(to=h)
            self.red_y.config(to=h)
            # Update diameter sliders
            max_diameter = min(w, h)
            self.blue_diameter.config(to=max_diameter)
            self.red_diameter.config(to=max_diameter)
            
    def change_mode(self):
        """Change input mode"""
        mode = self.mode.get()
        self.stop_capture()
        
        if mode == "camera":
            self.start_camera()
        elif mode == "pylon":
            self.start_pylon_camera()
        else:  # static
            self.status_label.config(text="Static mode - Load an image")
            
    def start_camera(self):
        """Start regular camera capture"""
        try:
            # Try different camera indices
            for idx in [0, 1, 2]:
                self.camera = cv2.VideoCapture(idx)
                if self.camera.isOpened():
                    break
            
            if not self.camera.isOpened():
                raise Exception("Cannot open camera")
            
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            self.status_label.config(text="Camera running")
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            self.mode.set("static")
            
    def start_pylon_camera(self):
        """Start Pylon camera capture"""
        if not PYLON_AVAILABLE:
            messagebox.showerror("Error", "pypylon is not installed")
            self.mode.set("static")
            return
            
        try:
            # Create camera instance
            self.pylon_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.pylon_camera.Open()
            
            # Configure camera
            self.pylon_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            
            self.running = True
            self.capture_thread = threading.Thread(target=self.pylon_capture_loop, daemon=True)
            self.capture_thread.start()
            self.status_label.config(text="Pylon camera running")
        except Exception as e:
            messagebox.showerror("Pylon Camera Error", str(e))
            self.mode.set("static")
            
    def capture_loop(self):
        """Capture loop for regular camera"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    self.image_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
                    
    def pylon_capture_loop(self):
        """Capture loop for Pylon camera"""
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        
        while self.running and self.pylon_camera.IsGrabbing():
            grab_result = self.pylon_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = converter.Convert(grab_result)
                frame = image.GetArray()
                try:
                    self.image_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
            grab_result.Release()
            
    def stop_capture(self):
        """Stop camera capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        if self.pylon_camera:
            self.pylon_camera.StopGrabbing()
            self.pylon_camera.Close()
            self.pylon_camera = None
            
    def draw_circles(self, image):
        """Draw circles on the image"""
        output = image.copy()
        
        if self.show_blue.get():
            cv2.circle(output, (self.blue_circle.x, self.blue_circle.y), 
                      self.blue_circle.diameter // 2, self.blue_circle.color, 2)
            # Draw center point
            cv2.circle(output, (self.blue_circle.x, self.blue_circle.y), 
                      3, self.blue_circle.color, -1)
                      
        if self.show_red.get():
            cv2.circle(output, (self.red_circle.x, self.red_circle.y), 
                      self.red_circle.diameter // 2, self.red_circle.color, 2)
            # Draw center point
            cv2.circle(output, (self.red_circle.x, self.red_circle.y), 
                      3, self.red_circle.color, -1)
                      
        return output
        
    def update_display(self):
        """Update the display with current image and circles"""
        # Get latest frame if in camera mode
        if self.mode.get() in ["camera", "pylon"]:
            try:
                frame = self.image_queue.get_nowait()
                self.current_image = frame
                if self.use_hough.get():
                    self.detect_circles()
                # Update slider ranges on first frame
                if hasattr(self, 'first_frame'):
                    self.update_slider_ranges()
                    delattr(self, 'first_frame')
            except queue.Empty:
                pass
                
        if self.current_image is not None:
            # Draw circles on image
            display = self.draw_circles(self.current_image)
            
            # Resize to fit canvas
            h, w = display.shape[:2]
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Canvas initialized
                scale = min(canvas_width / w, canvas_height / h, 1.0)
                new_width = int(w * scale)
                new_height = int(h * scale)
                display = cv2.resize(display, (new_width, new_height))
                
                # Convert to PhotoImage
                display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                self.photo = self.create_photo_image(display_rgb)
                
                # Update canvas
                self.canvas.delete("all")
                x = (canvas_width - new_width) // 2
                y = (canvas_height - new_height) // 2
                self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
                
        # Schedule next update
        self.root.after(30, self.update_display)
        
    def create_photo_image(self, cv_img):
        """Convert OpenCV image to PhotoImage"""
        from PIL import Image, ImageTk
        image = Image.fromarray(cv_img)
        return ImageTk.PhotoImage(image=image)
        
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_capture()
        self.root.destroy()

def main():
    """Main entry point"""
    print("\n" + "=" * 50)
    print("Fiber Optic Core Detection Tool")
    print("=" * 50 + "\n")
    
    app = FiberOpticDetector()
    app.run()

if __name__ == "__main__":
    main()
