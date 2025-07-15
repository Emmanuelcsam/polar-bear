#!/usr/bin/env python3
"""
Computer Vision Tutorial Script
Based on OpenCV tutorial covering image processing, face detection, and video processing
"""

import sys
import subprocess
import os
import time
from datetime import datetime

def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def install_package(package_name, import_name=None):
    """Install a package using pip if not already installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        log(f"✓ {package_name} is already installed")
        return True
    except ImportError:
        log(f"✗ {package_name} not found. Installing...")
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            # Install the package with upgrade flag to get latest version
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            log(f"✓ Successfully installed {package_name}")
            return True
        except subprocess.CalledProcessError:
            log(f"✗ Failed to install {package_name}")
            return False

# Check and install required packages
log("Checking required packages...")
packages = [
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("requests", "requests")
]

for package, import_name in packages:
    if not install_package(package, import_name):
        log("Failed to install required packages. Exiting...")
        sys.exit(1)

# Now import the packages
import cv2
import numpy as np
import requests
from PIL import Image
import urllib.request

log("All packages imported successfully")

# Create directories for assets
log("Creating directories for assets...")
os.makedirs("haar_cascades", exist_ok=True)
os.makedirs("sample_images", exist_ok=True)
os.makedirs("output", exist_ok=True)
log("✓ Directories created")

# Download Haar Cascade files if not present
def download_haar_cascades():
    """Download required Haar Cascade classifier files"""
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    cascades = {
        "haarcascade_frontalface_default.xml": "Face detection",
        "haarcascade_eye.xml": "Eye detection",
        "haarcascade_fullbody.xml": "Full body detection",
        "haarcascade_car.xml": "Car detection"
    }
    
    for filename, description in cascades.items():
        filepath = os.path.join("haar_cascades", filename)
        if not os.path.exists(filepath):
            log(f"Downloading {description} cascade...")
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
                log(f"✓ Downloaded {filename}")
            except Exception as e:
                log(f"✗ Failed to download {filename}: {e}")
        else:
            log(f"✓ {filename} already exists")

download_haar_cascades()

# Create a sample image if modi.jpg doesn't exist
def create_sample_image():
    """Create a sample image for testing"""
    log("Creating sample test image...")
    # Create a simple face-like image using shapes
    img = np.ones((600, 500, 3), dtype=np.uint8) * 255
    
    # Draw a face outline (circle)
    cv2.circle(img, (250, 250), 150, (200, 180, 160), -1)
    
    # Draw eyes
    cv2.circle(img, (200, 220), 30, (50, 50, 50), -1)
    cv2.circle(img, (300, 220), 30, (50, 50, 50), -1)
    
    # Draw pupils
    cv2.circle(img, (200, 220), 10, (0, 0, 0), -1)
    cv2.circle(img, (300, 220), 10, (0, 0, 0), -1)
    
    # Draw nose
    points = np.array([[250, 250], [230, 290], [270, 290]], np.int32)
    cv2.fillPoly(img, [points], (150, 130, 110))
    
    # Draw mouth
    cv2.ellipse(img, (250, 330), (60, 30), 0, 0, 180, (100, 50, 50), 3)
    
    cv2.imwrite("sample_images/modi.jpg", img)
    log("✓ Sample image created: sample_images/modi.jpg")
    return img

# Main Computer Vision Tutorial Functions
class ComputerVisionTutorial:
    def __init__(self):
        log("Initializing Computer Vision Tutorial...")
        self.create_sample_image_if_needed()
        
    def create_sample_image_if_needed(self):
        """Create sample image if it doesn't exist"""
        if not os.path.exists("sample_images/modi.jpg"):
            create_sample_image()
    
    def basic_image_operations(self):
        """Demonstrate basic image operations"""
        log("\n=== BASIC IMAGE OPERATIONS ===")
        
        # Read image
        log("Reading image with cv2.imread()...")
        img_color = cv2.imread("sample_images/modi.jpg", 1)  # 1 for color
        img_bw = cv2.imread("sample_images/modi.jpg", 0)     # 0 for black and white
        
        if img_color is None:
            log("✗ Failed to read image")
            return
        
        log(f"✓ Color image shape: {img_color.shape}")
        log(f"✓ Black and white image shape: {img_bw.shape}")
        
        # Resize image
        log("\nResizing image...")
        original_height, original_width = img_color.shape[:2]
        log(f"Original size: {original_width}x{original_height}")
        
        # Resize to 50% of original size
        new_width = int(original_width * 0.5)
        new_height = int(original_height * 0.5)
        resized_img = cv2.resize(img_color, (new_width, new_height))
        log(f"✓ Resized to: {new_width}x{new_height}")
        
        # Save images
        cv2.imwrite("output/color_image.jpg", img_color)
        cv2.imwrite("output/bw_image.jpg", img_bw)
        cv2.imwrite("output/resized_image.jpg", resized_img)
        log("✓ Saved images to output folder")
        
        # Show images (with timeout to avoid blocking)
        log("\nDisplaying images (press any key to continue)...")
        cv2.imshow("Color Image", img_color)
        cv2.imshow("Black and White", img_bw)
        cv2.imshow("Resized Image", resized_img)
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        log("✓ Closed all windows")
    
    def face_detection_image(self):
        """Demonstrate face detection on images"""
        log("\n=== FACE DETECTION ON IMAGE ===")
        
        # Load cascade classifiers
        log("Loading Haar Cascade classifiers...")
        face_cascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("haar_cascades/haarcascade_eye.xml")
        
        if face_cascade.empty() or eye_cascade.empty():
            log("✗ Failed to load cascade classifiers")
            return
        
        log("✓ Cascade classifiers loaded successfully")
        
        # Read and process image
        log("\nReading and processing image...")
        img = cv2.imread("sample_images/modi.jpg")
        img = cv2.resize(img, (500, 600))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        log("✓ Image converted to grayscale")
        
        # Detect faces
        log("\nDetecting faces...")
        faces = face_cascade.detectMultiScale(gray, 1.05, 5)
        log(f"✓ Found {len(faces)} face(s)")
        
        # Draw rectangles around faces and detect eyes
        for i, (x, y, w, h) in enumerate(faces):
            log(f"  Face {i+1}: Position ({x}, {y}), Size ({w}x{h})")
            
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            # Detect eyes within face
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5)
            log(f"    Found {len(eyes)} eye(s) in face {i+1}")
            
            for j, (ex, ey, ew, eh) in enumerate(eyes):
                log(f"      Eye {j+1}: Position ({ex}, {ey}), Size ({ew}x{eh})")
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Save and display result
        cv2.imwrite("output/face_detection_result.jpg", img)
        log("\n✓ Saved face detection result")
        
        cv2.imshow("Face Detection Result", img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        log("✓ Face detection demonstration complete")
    
    def video_capture_demo(self):
        """Demonstrate video capture and processing"""
        log("\n=== VIDEO CAPTURE DEMONSTRATION ===")
        
        # Try to open webcam
        log("Attempting to open webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            log("✗ No webcam detected. Creating sample video frames instead...")
            self.create_sample_video()
            return
        
        log("✓ Webcam opened successfully")
        log("Press 'q' to quit video capture")
        
        # Load face cascade for video
        face_cascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                log("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.05, 5)
            
            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            
            # Add text info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame
            cv2.imshow("Video - Face Detection", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                log("User pressed 'q' - stopping video capture")
                break
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                log(f"Processed {frame_count} frames, FPS: {fps:.2f}, Faces detected: {len(faces)}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        log(f"\n✓ Video capture complete. Total frames: {frame_count}, Time: {total_time:.2f}s")
    
    def create_sample_video(self):
        """Create sample video frames when no webcam is available"""
        log("Creating sample video demonstration...")
        
        face_cascade = cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/sample_video.avi', fourcc, 20.0, (640, 480))
        
        # Generate 100 frames with moving face
        for i in range(100):
            # Create blank frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
            
            # Draw moving face
            x_pos = 100 + int(300 * np.sin(i * 0.1))
            y_pos = 200
            
            # Face
            cv2.circle(frame, (x_pos, y_pos), 80, (200, 180, 160), -1)
            # Eyes
            cv2.circle(frame, (x_pos-30, y_pos-20), 15, (50, 50, 50), -1)
            cv2.circle(frame, (x_pos+30, y_pos-20), 15, (50, 50, 50), -1)
            # Mouth
            cv2.ellipse(frame, (x_pos, y_pos+30), (40, 20), 0, 0, 180, (100, 50, 50), 3)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {i+1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Detect and mark face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 5)
            
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 3)
            
            # Write frame
            out.write(frame)
            
            if i % 20 == 0:
                log(f"Generated {i+1} frames...")
        
        out.release()
        log("✓ Sample video created: output/sample_video.avi")
    
    def advanced_detection_demo(self):
        """Demonstrate detection of different objects"""
        log("\n=== ADVANCED OBJECT DETECTION ===")
        
        # Create test image with multiple objects
        log("Creating test image with multiple objects...")
        img = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        
        # Draw multiple faces at different positions
        face_positions = [(200, 200), (600, 200), (1000, 200)]
        for x, y in face_positions:
            # Face circle
            cv2.circle(img, (x, y), 80, (200, 180, 160), -1)
            # Eyes
            cv2.circle(img, (x-30, y-20), 15, (50, 50, 50), -1)
            cv2.circle(img, (x+30, y-20), 15, (50, 50, 50), -1)
            # Mouth
            cv2.ellipse(img, (x, y+30), (40, 20), 0, 0, 180, (100, 50, 50), 3)
        
        # Draw body shapes
        body_positions = [(200, 500), (600, 500), (1000, 500)]
        for x, y in body_positions:
            # Head
            cv2.circle(img, (x, y), 40, (200, 180, 160), -1)
            # Body
            cv2.rectangle(img, (x-50, y+40), (x+50, y+200), (100, 100, 200), -1)
            # Arms
            cv2.rectangle(img, (x-80, y+50), (x-50, y+150), (100, 100, 200), -1)
            cv2.rectangle(img, (x+50, y+50), (x+80, y+150), (100, 100, 200), -1)
        
        cv2.imwrite("output/multi_object_test.jpg", img)
        log("✓ Test image created")
        
        # Load all cascade classifiers
        cascades = {
            "Face": cv2.CascadeClassifier("haar_cascades/haarcascade_frontalface_default.xml"),
            "Full Body": cv2.CascadeClassifier("haar_cascades/haarcascade_fullbody.xml")
        }
        
        # Detect each type of object
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_copy = img.copy()
        
        colors = {"Face": (255, 0, 0), "Full Body": (0, 255, 0)}
        
        for name, cascade in cascades.items():
            if not cascade.empty():
                log(f"\nDetecting {name}...")
                objects = cascade.detectMultiScale(gray, 1.05, 5)
                log(f"✓ Found {len(objects)} {name.lower()}(s)")
                
                for i, (x, y, w, h) in enumerate(objects):
                    cv2.rectangle(img_copy, (x, y), (x+w, y+h), colors[name], 3)
                    cv2.putText(img_copy, name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 2)
        
        cv2.imwrite("output/multi_detection_result.jpg", img_copy)
        log("\n✓ Multi-object detection complete")
        
        # Display result
        cv2.imshow("Multi-Object Detection", img_copy)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    
    def color_conversion_demo(self):
        """Demonstrate color space conversions"""
        log("\n=== COLOR CONVERSION DEMONSTRATION ===")
        
        # Read image
        img = cv2.imread("sample_images/modi.jpg")
        if img is None:
            log("✗ Failed to read image")
            return
        
        img = cv2.resize(img, (400, 400))
        
        # Different color conversions
        conversions = {
            "Original (BGR)": img,
            "RGB": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "Grayscale": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            "HSV": cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            "LAB": cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        }
        
        # Save all conversions
        for name, converted in conversions.items():
            filename = f"output/color_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
            cv2.imwrite(filename, converted)
            log(f"✓ Saved {name} conversion")
        
        log("✓ Color conversion demonstration complete")
    
    def run_all_demos(self):
        """Run all tutorial demonstrations"""
        log("\n" + "="*50)
        log("COMPUTER VISION TUTORIAL - COMPLETE DEMONSTRATION")
        log("="*50)
        
        # Run all demonstrations
        self.basic_image_operations()
        time.sleep(1)
        
        self.face_detection_image()
        time.sleep(1)
        
        self.color_conversion_demo()
        time.sleep(1)
        
        self.advanced_detection_demo()
        time.sleep(1)
        
        # Video capture last (as it's interactive)
        self.video_capture_demo()
        
        log("\n" + "="*50)
        log("✓ TUTORIAL COMPLETE!")
        log("Check the 'output' folder for all generated images and videos")
        log("="*50)

# Main execution
if __name__ == "__main__":
    log("Starting Computer Vision Tutorial Script")
    log(f"OpenCV Version: {cv2.__version__}")
    log(f"Python Version: {sys.version}")
    
    # Create and run tutorial
    tutorial = ComputerVisionTutorial()
    
    try:
        tutorial.run_all_demos()
    except KeyboardInterrupt:
        log("\n✗ Tutorial interrupted by user")
    except Exception as e:
        log(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    log("\nScript execution completed")
