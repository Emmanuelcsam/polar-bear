# Main script for Pico 2 A - SD Card Image Monitor and Sender
# Save this as main.py on Pico 2 A

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