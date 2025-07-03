# Complete Guide: Raspberry Pi Pico 2 Image Transfer System for Dimension Scope EC200KC V2

## Project Overview
This guide details how to build a system that captures images from a Dimension Scope Easy Check V2 EC200KC V2 microscope and transfers them to your computer using two Raspberry Pi Pico 2s.

## System Architecture
- **Pico 2 A**: Connected to the microscope's SD card slot, monitors for new images, reads them, and transmits data
- **Pico 2 B**: Connected to your computer via USB, receives image data and saves it to your computer
- **Communication**: UART serial connection between the two Picos
- **Image Format**: The microscope likely saves images as JPEG or BMP files

## Required Components
1. 2x Raspberry Pi Pico 2
2. 1x MicroSD card module (for Pico 2 A)
3. Jumper wires
4. 2x USB cables (one for each Pico)
5. Breadboards (optional but recommended)
6. MicroSD card (FAT32 formatted)

## Part 1: Setting Up the Development Environment

### Step 1: Install Thonny IDE
1. Go to https://thonny.org/
2. Download the version for your operating system
3. Install following the default settings

### Step 2: Install MicroPython on Both Pico 2s
1. Download the latest MicroPython firmware for Pico from https://micropython.org/download/RPI_PICO2/
2. For each Pico 2:
   - Hold down the BOOTSEL button while plugging in the USB cable
   - The Pico will appear as a drive called "RPI-RP2"
   - Drag and drop the .uf2 file onto this drive
   - The Pico will automatically reboot with MicroPython installed

### Step 3: Configure Thonny for Pico
1. Open Thonny
2. Go to Tools → Options → Interpreter
3. Select "MicroPython (Raspberry Pi Pico)"
4. Select the correct COM/Serial port
5. Click OK

## Part 2: Hardware Connections

### Pico 2 A (SD Card Reader) Connections:

#### SD Card Module to Pico 2 A:
```
SD Card Module    →    Pico 2 A
VCC              →    3V3 (Pin 36)
GND              →    GND (Pin 38)
MISO             →    GPIO 4 (Pin 6)
MOSI             →    GPIO 3 (Pin 5)
SCK              →    GPIO 2 (Pin 4)
CS               →    GPIO 5 (Pin 7)
```

#### UART Connection (Pico 2 A side):
```
Pico 2 A         →    Connection
GPIO 0 (TX)      →    To Pico 2 B RX
GPIO 1 (RX)      →    To Pico 2 B TX
GND              →    To Pico 2 B GND
```

### Pico 2 B (Computer Interface) Connections:

#### UART Connection (Pico 2 B side):
```
Pico 2 B         →    Connection
GPIO 1 (RX)      →    From Pico 2 A TX
GPIO 0 (TX)      →    From Pico 2 A RX
GND              →    From Pico 2 A GND
```

## Part 3: Software Installation

### Step 1: Install SD Card Driver on Pico 2 A
1. Open Thonny and connect to Pico 2 A
2. Create a new file
3. Go to https://github.com/micropython/micropython-lib/blob/master/micropython/drivers/storage/sdcard/sdcard.py
4. Copy all the code
5. In Thonny, paste the code
6. Save to Pico as "sdcard.py"

### Step 2: Create Helper Libraries

#### On Pico 2 A, create "image_utils.py":
```python
import struct

class ImageReader:
    def __init__(self):
        self.chunk_size = 512  # Bytes per chunk
        
    def read_jpeg_header(self, file):
        """Read JPEG header to get file size"""
        file.seek(0)
        header = file.read(2)
        if header != b'\xff\xd8':
            return None
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Return to start
        return size
    
    def read_bmp_header(self, file):
        """Read BMP header to get file size"""
        file.seek(0)
        header = file.read(2)
        if header != b'BM':
            return None
        file.seek(2)
        size = struct.unpack('<I', file.read(4))[0]
        file.seek(0)
        return size
    
    def detect_image_type(self, filename):
        """Detect if file is JPEG or BMP"""
        with open(filename, 'rb') as f:
            header = f.read(2)
            if header == b'\xff\xd8':
                return 'JPEG'
            elif header == b'BM':
                return 'BMP'
        return None
```

Save this file to Pico 2 A.

#### On both Picos, create "protocol.py":
```python
import struct
import time

# Protocol constants
START_MARKER = b'IMG>'
END_MARKER = b'<IMG'
ACK = b'ACK'
NACK = b'NCK'
CHUNK_SIZE = 512

class Protocol:
    @staticmethod
    def create_header(filename, filesize, image_type):
        """Create header packet"""
        header = START_MARKER
        header += struct.pack('<I', filesize)
        header += struct.pack('<B', len(filename))
        header += filename.encode()
        header += struct.pack('<B', len(image_type))
        header += image_type.encode()
        return header
    
    @staticmethod
    def create_data_packet(chunk_num, data):
        """Create data packet"""
        packet = b'DAT>'
        packet += struct.pack('<I', chunk_num)
        packet += struct.pack('<H', len(data))
        packet += data
        checksum = sum(data) & 0xFF
        packet += struct.pack('<B', checksum)
        return packet
    
    @staticmethod
    def create_end_packet():
        """Create end packet"""
        return END_MARKER
    
    @staticmethod
    def parse_header(data):
        """Parse header packet"""
        if not data.startswith(START_MARKER):
            return None
        
        pos = len(START_MARKER)
        filesize = struct.unpack('<I', data[pos:pos+4])[0]
        pos += 4
        
        filename_len = data[pos]
        pos += 1
        filename = data[pos:pos+filename_len].decode()
        pos += filename_len
        
        type_len = data[pos]
        pos += 1
        image_type = data[pos:pos+type_len].decode()
        
        return {
            'filename': filename,
            'filesize': filesize,
            'type': image_type
        }
```

Save this file to both Pico 2 A and Pico 2 B.

## Part 4: Main Scripts

### Pico 2 A Script (main.py):
```python
from machine import Pin, SPI, UART
import sdcard
import os
import time
import gc
from image_utils import ImageReader
from protocol import Protocol

# Initialize UART for communication with Pico B
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# Initialize SPI for SD card
spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
          sck=Pin(2), mosi=Pin(3), miso=Pin(4))
cs = Pin(5, Pin.OUT)

# Mount SD card
sd = sdcard.SDCard(spi, cs)
os.mount(sd, '/sd')

# Initialize components
reader = ImageReader()
protocol = Protocol()

# Keep track of processed files
processed_files = set()

def scan_for_new_images():
    """Scan SD card for new image files"""
    try:
        files = os.listdir('/sd')
        new_files = []
        
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.bmp')):
                if filename not in processed_files:
                    new_files.append(filename)
        
        return new_files
    except Exception as e:
        print(f"Error scanning SD card: {e}")
        return []

def send_image(filepath):
    """Send an image file over UART"""
    filename = filepath.split('/')[-1]
    
    print(f"Processing {filename}...")
    
    # Detect image type
    image_type = reader.detect_image_type(filepath)
    if not image_type:
        print(f"Unknown image type: {filename}")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            # Get file size
            f.seek(0, 2)
            filesize = f.tell()
            f.seek(0)
            
            print(f"File size: {filesize} bytes")
            
            # Send header
            header = protocol.create_header(filename, filesize, image_type)
            uart.write(header)
            time.sleep(0.1)
            
            # Wait for ACK
            response = uart.read(3)
            if response != protocol.ACK:
                print("Header ACK not received")
                return False
            
            # Send file in chunks
            chunk_num = 0
            bytes_sent = 0
            
            while bytes_sent < filesize:
                chunk = f.read(protocol.CHUNK_SIZE)
                if not chunk:
                    break
                
                # Create and send data packet
                packet = protocol.create_data_packet(chunk_num, chunk)
                uart.write(packet)
                
                # Wait for ACK
                start_time = time.time()
                ack_received = False
                
                while time.time() - start_time < 2.0:  # 2 second timeout
                    if uart.any():
                        response = uart.read(3)
                        if response == protocol.ACK:
                            ack_received = True
                            break
                
                if not ack_received:
                    print(f"Chunk {chunk_num} ACK timeout")
                    return False
                
                bytes_sent += len(chunk)
                chunk_num += 1
                
                # Progress update
                if chunk_num % 10 == 0:
                    print(f"Sent {bytes_sent}/{filesize} bytes ({100*bytes_sent//filesize}%)")
                
                gc.collect()  # Clean up memory
            
            # Send end marker
            uart.write(protocol.create_end_packet())
            time.sleep(0.1)
            
            # Wait for final ACK
            response = uart.read(3)
            if response == protocol.ACK:
                print(f"Successfully sent {filename}")
                processed_files.add(filename)
                return True
            else:
                print("Final ACK not received")
                return False
                
    except Exception as e:
        print(f"Error sending file: {e}")
        return False

# Main loop
print("SD Card Image Monitor Started")
print("Monitoring for new images...")

while True:
    try:
        # Scan for new images
        new_images = scan_for_new_images()
        
        if new_images:
            print(f"Found {len(new_images)} new images")
            
            for image in new_images:
                filepath = f'/sd/{image}'
                success = send_image(filepath)
                
                if success:
                    print(f"✓ {image} sent successfully")
                else:
                    print(f"✗ Failed to send {image}")
                
                time.sleep(1)  # Brief pause between files
        
        time.sleep(2)  # Check every 2 seconds
        
    except KeyboardInterrupt:
        print("Stopping...")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
        time.sleep(5)

# Cleanup
os.umount('/sd')
print("SD card unmounted")
```

### Pico 2 B Script (receiver.py):
```python
from machine import Pin, UART
import time
import gc
from protocol import Protocol

# Initialize UART
uart = UART(0, baudrate=115200, tx=Pin(0), rx=Pin(1))

# Initialize LED for status
led = Pin(25, Pin.OUT)

protocol = Protocol()

# Buffer for receiving data
buffer = bytearray()
receiving_file = False
current_file = None
expected_chunks = 0
received_chunks = 0
file_data = bytearray()

def save_to_computer(filename, data):
    """Send data to computer via USB serial"""
    # This will print to the USB serial connection
    # The computer-side Python script will capture this
    print(f"FILE_START:{filename}")
    
    # Send data in base64 encoded chunks to avoid issues
    import ubinascii
    chunk_size = 512
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        encoded = ubinascii.b2a_base64(chunk)
        print(f"FILE_DATA:{encoded.decode().strip()}")
        time.sleep(0.01)
    
    print("FILE_END")

def process_buffer():
    """Process received data in buffer"""
    global buffer, receiving_file, current_file, file_data
    global expected_chunks, received_chunks
    
    # Check for header
    if not receiving_file and len(buffer) >= len(Protocol.START_MARKER):
        start_idx = buffer.find(Protocol.START_MARKER)
        if start_idx != -1:
            # Try to parse header
            header_data = buffer[start_idx:]
            if len(header_data) >= 20:  # Minimum header size
                try:
                    header_info = protocol.parse_header(header_data)
                    if header_info:
                        current_file = header_info
                        receiving_file = True
                        file_data = bytearray()
                        expected_chunks = (header_info['filesize'] + protocol.CHUNK_SIZE - 1) // protocol.CHUNK_SIZE
                        received_chunks = 0
                        
                        print(f"Receiving {header_info['filename']} ({header_info['filesize']} bytes)")
                        
                        # Send ACK
                        uart.write(protocol.ACK)
                        
                        # Clear buffer up to end of header
                        buffer = buffer[start_idx + 20:]
                        
                except Exception as e:
                    print(f"Header parse error: {e}")
    
    # Check for data packets
    if receiving_file:
        while True:
            data_idx = buffer.find(b'DAT>')
            if data_idx == -1 or len(buffer) < data_idx + 11:
                break
            
            # Parse data packet
            packet_start = data_idx
            chunk_num = int.from_bytes(buffer[data_idx+4:data_idx+8], 'little')
            data_len = int.from_bytes(buffer[data_idx+8:data_idx+10], 'little')
            
            if len(buffer) < packet_start + 11 + data_len:
                break  # Not enough data yet
            
            # Extract data and checksum
            chunk_data = buffer[data_idx+10:data_idx+10+data_len]
            checksum = buffer[data_idx+10+data_len]
            
            # Verify checksum
            calc_checksum = sum(chunk_data) & 0xFF
            if calc_checksum == checksum:
                file_data.extend(chunk_data)
                received_chunks += 1
                uart.write(protocol.ACK)
                
                # Progress indicator
                led.toggle()
                
                if received_chunks % 10 == 0:
                    print(f"Progress: {received_chunks}/{expected_chunks} chunks")
            else:
                print(f"Checksum error in chunk {chunk_num}")
                uart.write(protocol.NACK)
            
            # Remove processed packet from buffer
            buffer = buffer[data_idx+11+data_len:]
    
    # Check for end marker
    if receiving_file and protocol.END_MARKER in buffer:
        print(f"File complete: {current_file['filename']}")
        
        # Save file
        save_to_computer(current_file['filename'], file_data)
        
        # Send final ACK
        uart.write(protocol.ACK)
        
        # Reset state
        receiving_file = False
        current_file = None
        file_data = bytearray()
        buffer = bytearray()
        
        led.off()
        gc.collect()

# Main loop
print("Image Receiver Ready")
print("Waiting for images...")

while True:
    try:
        # Check for incoming data
        if uart.any():
            # Read available data
            new_data = uart.read(uart.any())
            buffer.extend(new_data)
            
            # Process buffer
            process_buffer()
        
        time.sleep(0.001)  # Small delay to prevent CPU hogging
        
    except KeyboardInterrupt:
        print("Stopping receiver...")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)

led.off()
print("Receiver stopped")
```

### Computer-Side Python Script (save_images.py):
Save this on your computer, not on the Pico.

```python
import serial
import base64
import os
from datetime import datetime

# Configuration
SERIAL_PORT = "COM3"  # Change this to your Pico B's port
BAUD_RATE = 115200
OUTPUT_FOLDER = "received_images"

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Open serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
print(f"Connected to {SERIAL_PORT}")
print(f"Saving images to {OUTPUT_FOLDER}/")
print("Waiting for images...")

current_file = None
file_data = bytearray()

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            
            if line.startswith("FILE_START:"):
                filename = line[11:]
                current_file = filename
                file_data = bytearray()
                print(f"\nReceiving {filename}...")
                
            elif line.startswith("FILE_DATA:") and current_file:
                encoded_data = line[10:]
                try:
                    decoded = base64.b64decode(encoded_data)
                    file_data.extend(decoded)
                except:
                    print("Error decoding data chunk")
                    
            elif line == "FILE_END" and current_file:
                # Save file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{timestamp}_{current_file}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                with open(output_path, 'wb') as f:
                    f.write(file_data)
                
                print(f"✓ Saved {output_filename} ({len(file_data)} bytes)")
                
                current_file = None
                file_data = bytearray()
                
            else:
                # Print other messages
                if line and not line.startswith("FILE_"):
                    print(f"[Pico B] {line}")

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    ser.close()
    print("Serial connection closed")
```

## Part 5: Step-by-Step Setup Instructions

### Step 1: Prepare the Hardware
1. Connect the SD card module to Pico 2 A using the wiring diagram above
2. Connect the two Picos together via UART (TX to RX, RX to TX, GND to GND)
3. Do NOT connect both Picos to USB yet

### Step 2: Setup Pico 2 A (SD Card Reader)
1. Connect only Pico 2 A to your computer via USB
2. Open Thonny and ensure it's connected to Pico 2 A
3. Upload these files to Pico 2 A:
   - `sdcard.py` (the driver)
   - `image_utils.py`
   - `protocol.py`
   - `main.py` (the Pico 2 A script)
4. Insert a FAT32 formatted SD card into the module
5. Run the script to test - you should see "SD Card Image Monitor Started"
6. Stop the script and disconnect Pico 2 A

### Step 3: Setup Pico 2 B (Computer Interface)
1. Connect only Pico 2 B to your computer via USB
2. Open Thonny and ensure it's connected to Pico 2 B
3. Upload these files to Pico 2 B:
   - `protocol.py`
   - Save the receiver script as `main.py`
4. Test run the script - you should see "Image Receiver Ready"
5. Stop the script but keep Pico 2 B connected

### Step 4: Setup Computer Script
1. Install Python on your computer if not already installed
2. Install PySerial: `pip install pyserial`
3. Save the `save_images.py` script on your computer
4. Edit the `SERIAL_PORT` variable to match your Pico 2 B's port
5. Create a folder called "received_images" in the same directory

### Step 5: Full System Test
1. Close Thonny completely
2. Run the computer script: `python save_images.py`
3. Connect Pico 2 A to a power source (USB charger or power bank)
4. The system should now be running:
   - Pico 2 A monitors the SD card
   - When you save an image from the microscope, it detects and sends it
   - Pico 2 B receives the data and forwards to computer
   - Computer script saves the image file

## Troubleshooting

### Common Issues:

1. **"SD card not found" error**:
   - Check wiring connections
   - Ensure SD card is FAT32 formatted
   - Try a different SD card
   - Reduce SPI speed in the code

2. **No communication between Picos**:
   - Verify TX→RX and RX→TX connections
   - Check that both grounds are connected
   - Try swapping TX/RX connections
   - Reduce baud rate to 9600 for testing

3. **Computer doesn't receive files**:
   - Ensure Thonny is closed
   - Check serial port in device manager
   - Try a different USB cable
   - Verify Python script has correct port

4. **Incomplete file transfers**:
   - Add longer delays between chunks
   - Reduce chunk size to 256 bytes
   - Check available RAM with `gc.mem_free()`
   - Process smaller images first

5. **"Memory allocation failed"**:
   - Reduce chunk size
   - Add more `gc.collect()` calls
   - Process one file at a time
   - Use smaller test images

## Performance Optimization

1. **For faster transfers**:
   - Increase baud rate to 460800 or 921600
   - Increase chunk size to 1024 bytes
   - Remove progress print statements
   - Use hardware flow control if needed

2. **For larger images**:
   - Implement compression before sending
   - Send only changed pixels for similar images
   - Use binary protocols instead of base64

3. **For reliability**:
   - Add retry mechanisms for failed chunks
   - Implement CRC32 instead of simple checksum
   - Add file resume capability
   - Log all transfers for debugging

## Testing the System

1. **Initial Test**:
   - Place a small test image (< 100KB) on the SD card
   - Run the system and verify it transfers correctly

2. **Microscope Integration**:
   - Connect the SD card to the microscope
   - Take a test image
   - Verify Pico 2 A detects and transfers it

3. **Performance Test**:
   - Time the transfer of different sized images
   - Optimize based on your typical image sizes

## Next Steps

Once the basic system is working, you can enhance it with:
- LCD display on Pico 2 A showing transfer status
- Wireless communication (using Pico W)
- Automatic image organization by date/time
- Image preview on an OLED display
- Web interface for remote monitoring
- Batch processing multiple images
- Automatic backup to cloud storage

This system provides a solid foundation for automated image capture and transfer from your microscope to your computer.