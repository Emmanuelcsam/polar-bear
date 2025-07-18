# Protocol library for image transfer between Picos
# Save this as protocol.py on BOTH Pico 2 A and Pico 2 B

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
        """Create header packet for file transfer
        
        Format:
        - Start marker (4 bytes): 'IMG>'
        - File size (4 bytes): unsigned int
        - Filename length (1 byte)
        - Filename (variable)
        - Image type length (1 byte)
        - Image type (variable)
        """
        header = START_MARKER
        header += struct.pack('<I', filesize)
        header += struct.pack('<B', len(filename))
        header += filename.encode()
        header += struct.pack('<B', len(image_type))
        header += image_type.encode()
        return header
    
    @staticmethod
    def create_data_packet(chunk_num, data):
        """Create data packet
        
        Format:
        - Marker (4 bytes): 'DAT>'
        - Chunk number (4 bytes): unsigned int
        - Data length (2 bytes): unsigned short
        - Data (variable)
        - Checksum (1 byte): simple sum & 0xFF
        """
        packet = b'DAT>'
        packet += struct.pack('<I', chunk_num)
        packet += struct.pack('<H', len(data))
        packet += data
        checksum = sum(data) & 0xFF
        packet += struct.pack('<B', checksum)
        return packet
    
    @staticmethod
    def create_end_packet():
        """Create end-of-transmission packet"""
        return END_MARKER
    
    @staticmethod
    def parse_header(data):
        """Parse header packet and extract file information
        
        Returns dict with keys: filename, filesize, type
        Returns None if parsing fails
        """
        if not data.startswith(START_MARKER):
            return None
        
        try:
            pos = len(START_MARKER)
            
            # Extract file size
            filesize = struct.unpack('<I', data[pos:pos+4])[0]
            pos += 4
            
            # Extract filename
            filename_len = data[pos]
            pos += 1
            filename = data[pos:pos+filename_len].decode()
            pos += filename_len
            
            # Extract image type
            type_len = data[pos]
            pos += 1
            image_type = data[pos:pos+type_len].decode()
            
            return {
                'filename': filename,
                'filesize': filesize,
                'type': image_type
            }
        except:
            return None
    
    @staticmethod
    def parse_data_packet(data):
        """Parse data packet
        
        Returns tuple: (chunk_num, chunk_data, valid)
        """
        if not data.startswith(b'DAT>'):
            return None, None, False
        
        try:
            # Extract chunk number
            chunk_num = struct.unpack('<I', data[4:8])[0]
            
            # Extract data length
            data_len = struct.unpack('<H', data[8:10])[0]
            
            # Extract data
            chunk_data = data[10:10+data_len]
            
            # Verify checksum
            checksum = data[10+data_len]
            calc_checksum = sum(chunk_data) & 0xFF
            
            return chunk_num, chunk_data, (checksum == calc_checksum)
        except:
            return None, None, False