#!/usr/bin/env python3
"""
Advanced Computer Vision Course Implementation
Based on Murtaza's Workshop Tutorial
Includes: Hand Tracking, Pose Estimation, Face Detection, Face Mesh
Projects: Volume Control, Finger Counter, AI Trainer, Virtual Painter, Virtual Mouse
"""

import sys
import subprocess
import importlib
import os
import time
from datetime import datetime

# Logging function
def log(message, level="INFO"):
    """Print timestamped log messages"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

# Check and install required packages
def check_and_install_packages():
    """Check for required packages and install if missing"""
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'autopy': 'autopy',
        'pycaw': 'pycaw'
    }
    
    log("Checking for required packages...")
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'cv2':
                import cv2
            else:
                importlib.import_module(module_name)
            log(f"✓ {package_name} is already installed")
        except ImportError:
            log(f"✗ {package_name} not found. Installing...", "WARNING")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                log(f"✓ Successfully installed {package_name}", "SUCCESS")
            except subprocess.CalledProcessError as e:
                log(f"✗ Failed to install {package_name}: {e}", "ERROR")
                if package_name == 'autopy':
                    log("Note: autopy might require additional system dependencies", "WARNING")
                sys.exit(1)

# Run the package check
check_and_install_packages()

# Now import all required modules
log("Importing required modules...")
import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Try to import optional modules
try:
    import autopy
    AUTOPY_AVAILABLE = True
    log("✓ autopy imported successfully")
except ImportError:
    AUTOPY_AVAILABLE = False
    log("✗ autopy not available - Virtual Mouse project will be limited", "WARNING")

try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    PYCAW_AVAILABLE = True
    log("✓ pycaw imported successfully")
except ImportError:
    PYCAW_AVAILABLE = False
    log("✗ pycaw not available - Volume Control project will be limited", "WARNING")

log("All imports completed")

# ==================== HAND TRACKING MODULE ====================
class HandDetector:
    """Hand Detection and Tracking Module"""
    
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        log(f"Initializing HandDetector with maxHands={maxHands}, detectionCon={detectionCon}")
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        log("HandDetector initialized successfully")
    
    def findHands(self, img, draw=True):
        """Find hands in the image"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        """Find position of hand landmarks"""
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def fingersUp(self):
        """Check which fingers are up"""
        fingers = []
        if len(self.lmList) != 0:
            # Thumb
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # 4 Fingers
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True):
        """Find distance between two landmarks"""
        if len(self.lmList) != 0:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            
            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]
        return 0, img, []

# ==================== POSE ESTIMATION MODULE ====================
class PoseDetector:
    """Pose Detection Module"""
    
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        log(f"Initializing PoseDetector with detectionCon={detectionCon}")
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        log("PoseDetector initialized successfully")
    
    def findPose(self, img, draw=True):
        """Find pose in the image"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, 
                                         self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPosition(self, img, draw=True):
        """Find position of pose landmarks"""
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        """Find angle between three points"""
        if len(self.lmList) != 0:
            # Get the landmarks
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]
            
            # Calculate the angle
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - 
                               math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360
            
            # Draw
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
                cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return angle
        return 0

# ==================== FACE DETECTION MODULE ====================
class FaceDetector:
    """Face Detection Module"""
    
    def __init__(self, minDetectionCon=0.5):
        log(f"Initializing FaceDetector with minDetectionCon={minDetectionCon}")
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        log("FaceDetector initialized successfully")
    
    def findFaces(self, img, draw=True):
        """Find faces in the image"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([bbox, detection.score])
                
                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                               (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                               2, (255, 0, 255), 2)
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l=30, t=5):
        """Draw fancy bounding box"""
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # Top Left
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

# ==================== FACE MESH MODULE ====================
class FaceMeshDetector:
    """Face Mesh Detection Module"""
    
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        log(f"Initializing FaceMeshDetector with maxFaces={maxFaces}")
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        log("FaceMeshDetector initialized successfully")
    
    def findFaceMesh(self, img, draw=True):
        """Find face mesh in the image"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                             self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

# ==================== PROJECT IMPLEMENTATIONS ====================

def gesture_volume_control():
    """Project 1: Gesture Volume Control"""
    log("Starting Gesture Volume Control project")
    
    if not PYCAW_AVAILABLE:
        log("Volume control requires pycaw library which is not available", "ERROR")
        return
    
    # Camera setup
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    
    detector = HandDetector(detectionCon=0.7)
    
    # Volume setup
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    
    log(f"Volume range: {minVol} to {maxVol}")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Get thumb and index finger positions
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Convert length to volume
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])
            
            volume.SetMasterVolumeLevel(vol, None)
            
            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        
        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 3)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        
        cv2.imshow("Gesture Volume Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log("Gesture Volume Control ended")

def finger_counter():
    """Project 2: Finger Counter"""
    log("Starting Finger Counter project")
    
    # Setup
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    # Create header images folder if it doesn't exist
    if not os.path.exists("FingerImages"):
        os.makedirs("FingerImages")
        log("Created FingerImages folder")
        # Create placeholder images
        for i in range(6):
            img = np.ones((200, 200, 3), np.uint8) * 255
            cv2.putText(img, str(i), (50, 150), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 0), 25)
            cv2.imwrite(f"FingerImages/{i}.jpg", img)
        log("Created placeholder finger images")
    
    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    overlayList = []
    
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    
    pTime = 0
    detector = HandDetector(detectionCon=0.75)
    
    tipIds = [4, 8, 12, 16, 20]
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            totalFingers = fingers.count(1)
            
            # Display the finger image
            h, w, c = overlayList[totalFingers].shape
            img[0:h, 0:w] = overlayList[totalFingers]
            
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                        10, (255, 0, 0), 25)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        
        cv2.imshow("Finger Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log("Finger Counter ended")

def ai_personal_trainer():
    """Project 3: AI Personal Trainer"""
    log("Starting AI Personal Trainer project")
    
    cap = cv2.VideoCapture(0)
    detector = PoseDetector()
    count = 0
    dir = 0
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.resize(img, (1280, 720))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        
        if len(lmList) != 0:
            # Right arm curl detection
            angle = detector.findAngle(img, 12, 14, 16)
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (210, 310), (650, 100))
            
            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)
                if dir == 1:
                    count += 0.5
                    dir = 0
            
            # Draw Bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
            
            # Draw Curl Count
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                        (255, 0, 0), 25)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                    (255, 0, 0), 5)
        
        cv2.imshow("AI Personal Trainer", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log("AI Personal Trainer ended")

def ai_virtual_painter():
    """Project 4: AI Virtual Painter"""
    log("Starting AI Virtual Painter project")
    
    # Create header folder if it doesn't exist
    if not os.path.exists("Header"):
        os.makedirs("Header")
        log("Created Header folder")
        # Create placeholder header images
        colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 0)]
        for i in range(4):
            img = np.ones((125, 1280, 3), np.uint8) * 200
            cv2.rectangle(img, (i*320, 0), ((i+1)*320, 125), colors[i], -1)
            cv2.imwrite(f"Header/{i+1}.jpg", img)
        log("Created placeholder header images")
    
    folderPath = "Header"
    myList = os.listdir(folderPath)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    
    header = overlayList[0]
    drawColor = (255, 0, 255)
    brushThickness = 15
    eraserThickness = 50
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    detector = HandDetector(detectionCon=0.85)
    xp, yp = 0, 0
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1)
        
        # Find Hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Tip of index and middle fingers
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            
            # Check which fingers are up
            fingers = detector.fingersUp()
            
            # Selection Mode - Two fingers are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
                
                # Checking for the click
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
            
            # Drawing Mode - Index finger is up
            if fingers[1] and fingers[2] == False:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                
                if drawColor == (0, 0, 0):
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                
                xp, yp = x1, y1
        
        # Merge canvas and image
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        
        # Setting the header image
        img[0:125, 0:1280] = header
        
        cv2.imshow("AI Virtual Painter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log("AI Virtual Painter ended")

def ai_virtual_mouse():
    """Project 5: AI Virtual Mouse"""
    log("Starting AI Virtual Mouse project")
    
    if not AUTOPY_AVAILABLE:
        log("Virtual mouse requires autopy library which is not available", "ERROR")
        return
    
    # Variables
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction
    smoothening = 7
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    
    detector = HandDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    log(f"Screen size: {wScr}x{hScr}")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Find hand Landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        
        # Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            
            # Check which fingers are up
            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                          (255, 0, 255), 2)
            
            # Only Index Finger: Moving Mode
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                
                # Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                
                # Move Mouse
                try:
                    autopy.mouse.move(wScr - clocX, clocY)
                except:
                    pass
                
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
            
            # Both Index and middle fingers are up: Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1:
                # Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                
                # Click mouse if distance short
                if length < 40:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()
        
        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        cv2.imshow("AI Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log("AI Virtual Mouse ended")

def test_modules():
    """Test all modules with webcam"""
    log("Starting module testing")
    
    cap = cv2.VideoCapture(0)
    
    # Test options
    tests = {
        '1': ('Hand Tracking', HandDetector()),
        '2': ('Pose Estimation', PoseDetector()),
        '3': ('Face Detection', FaceDetector()),
        '4': ('Face Mesh', FaceMeshDetector())
    }
    
    print("\nSelect module to test:")
    for key, (name, _) in tests.items():
        print(f"{key}. {name}")
    print("q. Quit")
    
    choice = input("\nEnter choice: ")
    
    if choice == 'q':
        return
    
    if choice not in tests:
        log("Invalid choice", "ERROR")
        return
    
    module_name, detector = tests[choice]
    log(f"Testing {module_name}")
    
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Process based on module type
        if choice == '1':  # Hand Tracking
            img = detector.findHands(img)
            lmList = detector.findPosition(img)
            if len(lmList) != 0:
                fingers = detector.fingersUp()
                cv2.putText(img, f'Fingers: {fingers}', (10, 150), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        elif choice == '2':  # Pose Estimation
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            if len(lmList) != 0:
                cv2.putText(img, f'Landmarks: {len(lmList)}', (10, 150), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        elif choice == '3':  # Face Detection
            img, bboxs = detector.findFaces(img)
            cv2.putText(img, f'Faces: {len(bboxs)}', (10, 150), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        elif choice == '4':  # Face Mesh
            img, faces = detector.findFaceMesh(img)
            if len(faces) != 0:
                cv2.putText(img, f'Face Points: {len(faces[0])}', (10, 150), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        
        cv2.imshow(f"{module_name} Test", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    log(f"{module_name} test ended")

def main_menu():
    """Main menu for selecting projects"""
    while True:
        print("\n" + "="*60)
        print("ADVANCED COMPUTER VISION COURSE")
        print("Based on Murtaza's Workshop")
        print("="*60)
        print("\nSelect an option:")
        print("\n--- MODULES ---")
        print("1. Test Hand Tracking Module")
        print("2. Test Pose Estimation Module")
        print("3. Test Face Detection Module")
        print("4. Test Face Mesh Module")
        print("\n--- PROJECTS ---")
        print("5. Gesture Volume Control")
        print("6. Finger Counter")
        print("7. AI Personal Trainer")
        print("8. AI Virtual Painter")
        print("9. AI Virtual Mouse")
        print("\nq. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == 'q':
            log("Exiting program")
            break
        elif choice in ['1', '2', '3', '4']:
            test_modules()
        elif choice == '5':
            gesture_volume_control()
        elif choice == '6':
            finger_counter()
        elif choice == '7':
            ai_personal_trainer()
        elif choice == '8':
            ai_virtual_painter()
        elif choice == '9':
            ai_virtual_mouse()
        else:
            log("Invalid choice. Please try again.", "WARNING")

if __name__ == "__main__":
    log("Starting Advanced Computer Vision Course")
    log("Based on Murtaza's Workshop Tutorial")
    log("="*60)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        log("\nProgram interrupted by user", "WARNING")
    except Exception as e:
        log(f"Unexpected error: {e}", "ERROR")
    finally:
        cv2.destroyAllWindows()
        log("Program ended")