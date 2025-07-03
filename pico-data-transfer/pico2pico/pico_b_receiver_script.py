# Main script for Pico 2 B - Image Receiver and Computer Interface
# Save this as main.py on Pico 2 B

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