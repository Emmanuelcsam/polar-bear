# Computer-side script to receive and save images
# Run this on your computer, not on the Pico
# Usage: python save_images.py

import serial
import base64
import os
from datetime import datetime

# Configuration - CHANGE THESE TO MATCH YOUR SETUP
SERIAL_PORT = "COM3"  # Windows: "COM3", "COM4", etc.
                      # Linux: "/dev/ttyACM0", "/dev/ttyUSB0", etc.
                      # macOS: "/dev/tty.usbmodem14201", etc.
BAUD_RATE = 115200
OUTPUT_FOLDER = "received_images"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Open serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    print(f"Saving images to {os.path.abspath(OUTPUT_FOLDER)}/")
    print("Waiting for images...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}")
    print(f"Details: {e}")
    print("\nPlease check:")
    print("1. The correct port name (check Device Manager on Windows)")
    print("2. Pico 2 B is connected via USB")
    print("3. No other program (like Thonny) is using the port")
    exit(1)

current_file = None
file_data = bytearray()
total_files_received = 0

try:
    while True:
        if ser.in_waiting:
            try:
                line = ser.readline().decode('utf-8').strip()
            except UnicodeDecodeError:
                continue  # Skip malformed data
            
            if line.startswith("FILE_START:"):
                filename = line[11:]
                current_file = filename
                file_data = bytearray()
                print(f"\n📥 Receiving {filename}...")
                
            elif line.startswith("FILE_DATA:") and current_file:
                encoded_data = line[10:]
                try:
                    decoded = base64.b64decode(encoded_data)
                    file_data.extend(decoded)
                    # Show progress
                    print(f"\r   Progress: {len(file_data)} bytes", end="")
                except Exception as e:
                    print(f"\n   ⚠️ Error decoding data chunk: {e}")
                    
            elif line == "FILE_END" and current_file:
                # Save file with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(current_file)
                output_filename = f"{timestamp}_{name}{ext}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                try:
                    with open(output_path, 'wb') as f:
                        f.write(file_data)
                    
                    print(f"\n✅ Saved {output_filename} ({len(file_data):,} bytes)")
                    total_files_received += 1
                    print(f"   Total files received: {total_files_received}")
                except Exception as e:
                    print(f"\n❌ Error saving file: {e}")
                
                current_file = None
                file_data = bytearray()
                
            else:
                # Print other messages from Pico
                if line and not line.startswith("FILE_"):
                    print(f"[Pico] {line}")

except KeyboardInterrupt:
    print(f"\n\nStopping... Received {total_files_received} files total")
except Exception as e:
    print(f"\nError: {e}")
finally:
    ser.close()
    print("Serial connection closed")