
import cv2
from pypylon import pylon
import threading
import time
import os

# --- Pylon Grabber Class from pylon_grabber.py ---
class PylonGrabber:
    def __init__(self, exposure_time=5000):
        self.exposure_time = exposure_time
        self.camera = None
        self.converter = None
        self.latest_frame = None
        self.grabbing = False
        self.lock = threading.Lock()
        self.thread = None

    def open(self):
        try:
            # Get the transport layer factory.
            tlFactory = pylon.TlFactory.GetInstance()
            # Get all attached devices and exit application if no device is found.
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No camera present.")
            # Create an instant camera object with the first found device.
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
            self.camera.Open()
            # Print the model name of the camera.
            print("Using device ", self.camera.GetDeviceInfo().GetModelName())
            # Set exposure time
            self.camera.ExposureTime.SetValue(self.exposure_time)
            # The parameter MaxNumBuffer can be used to control the count of buffers
            # allocated for grabbing. The default value of this parameter is 10.
            self.camera.MaxNumBuffer = 10
            # Create a pylon ImageFormatConverter
            self.converter = pylon.ImageFormatConverter()
            # Specifying the output pixel format
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            return True
        except pylon.RuntimeException as e:
            print(f"Error opening camera: {e}")
            return False

    def start_grabbing(self):
        if self.camera and not self.grabbing:
            self.grabbing = True
            self.thread = threading.Thread(target=self._grab_loop)
            self.thread.daemon = True
            self.thread.start()

    def _grab_loop(self):
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        while self.grabbing:
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                    with self.lock:
                        self.latest_frame = img
                grabResult.Release()
            except pylon.GenericException as e:
                if self.grabbing:
                    print(f"An exception occurred during grabbing: {e}")
                break
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def stop_grabbing(self):
        self.grabbing = False
        if self.thread:
            self.thread.join()
        if self.camera and self.camera.IsOpen():
            self.camera.Close()

# --- Detection Class from detection.py ---
class Detection:
    def __init__(self, conf_threshold=0.5, nms_threshold=0.4, model_path='yolov3.weights', config_path='yolov3.cfg', classes_path='coco.names'):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = self.load_class_names(classes_path)
        # Check for GPU
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is available. Using GPU.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print("CUDA not available. Using CPU.")

    def load_class_names(self, file_name):
        with open(file_name, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            # New way for OpenCV 4.x
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except TypeError:
             # Old way for OpenCV 3.x
            return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def detect(self, image):
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                final_boxes.append((self.classes[class_ids[i]], confidences[i], (x, y, w, h)))
        return final_boxes

    def draw_boxes(self, image, boxes):
        for (label, conf, box) in boxes:
            x, y, w, h = box
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

# --- Main Application Logic ---
def main():
    # --- Configuration ---
    # Make sure these files are in the same directory as the script,
    # or provide the full path.
    # You may need to download yolov3.weights, yolov3.cfg, and coco.names
    # if you don't have them.
    YOLO_MODEL_PATH = 'yolov3.weights'
    YOLO_CONFIG_PATH = 'yolov3.cfg'
    YOLO_CLASSES_PATH = 'coco.names'
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    EXPOSURE_TIME = 5000 # in microseconds

    # --- Prerequisite Check ---
    required_files = [YOLO_MODEL_PATH, YOLO_CONFIG_PATH, YOLO_CLASSES_PATH]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file not found: {f}")
            print("Please download the YOLOv3 model files ('yolov3.weights', 'yolov3.cfg', 'coco.names') and place them in the same directory as this script.")
            return

    # --- Initialization ---
    print("Initializing...")
    try:
        detector = Detection(CONF_THRESHOLD, NMS_THRESHOLD, YOLO_MODEL_PATH, YOLO_CONFIG_PATH, YOLO_CLASSES_PATH)
    except cv2.error as e:
        print(f"Error loading YOLO model: {e}")
        return

    grabber = PylonGrabber(exposure_time=EXPOSURE_TIME)
    if not grabber.open():
        print("Could not open camera. Exiting.")
        return

    # --- Start Grabbing Thread ---
    print("Starting camera...")
    grabber.start_grabbing()
    time.sleep(2) # Give camera time to start up

    print("Starting detection loop. Press 'q' to quit.")
    while True:
        frame = grabber.get_frame()

        if frame is not None:
            # --- Detection ---
            detection_boxes = detector.detect(frame)
            display_frame = detector.draw_boxes(frame.copy(), detection_boxes)

            # --- Display ---
            cv2.imshow("Pylon Detection", display_frame)

        # --- Exit Condition ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # --- Cleanup ---
    print("Cleaning up...")
    grabber.stop_grabbing()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
