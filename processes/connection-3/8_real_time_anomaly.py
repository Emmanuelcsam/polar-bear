# 8_real_time_anomaly.py
import cv2
import numpy as np

def run_real_time_detection():
    """Captures from webcam and highlights changes from a static background."""
    print("--- Real-Time Anomaly Detector Started ---")
    cap = cv2.VideoCapture(0) # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    print("Capturing background... Please step out of frame and press 'b'.")
    background = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('b'):
                background = gray_frame
                print("Background captured! Now detecting movement (anomalies).")

            if background is not None:
                # Calculate difference from the background
                frame_delta = cv2.absdiff(background, gray_frame)
                thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

                # Find contours of moving objects and draw boxes
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 500:
                        continue # Ignore small movements
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow("Real-Time Anomalies (q to quit)", frame)

            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("--- Real-Time Detector Stopped ---")
    return True

if __name__ == "__main__":
    run_real_time_detection()
