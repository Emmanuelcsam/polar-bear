# Image utilities for reading and identifying image files
# Save this as image_utils.py on Pico 2 A only

import struct

class ImageReader:
    def __init__(self):
        self.chunk_size = 512  # Bytes per chunk
        
    def read_jpeg_header(self, file):
        """Read JPEG header to get file size
        
        JPEG files start with FF D8 (Start of Image marker)
        """
        file.seek(0)
        header = file.read(2)
        if header != b'\xff\xd8':
            return None
        
        # Get file size by seeking to end
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Return to start
        return size
    
    def read_bmp_header(self, file):
        """Read BMP header to get file size and image info
        
        BMP header structure:
        - Signature (2 bytes): 'BM'
        - File size (4 bytes): total file size
        - Reserved (4 bytes)
        - Data offset (4 bytes): offset to pixel data
        - Header size (4 bytes)
        - Width (4 bytes)
        - Height (4 bytes)
        """
        file.seek(0)
        header = file.read(2)
        if header != b'BM':
            return None
        
        # Read file size from header
        file.seek(2)
        size = struct.unpack('<I', file.read(4))[0]
        file.seek(0)
        return size
    
    def detect_image_type(self, filename):
        """Detect if file is JPEG or BMP based on file header
        
        Returns: 'JPEG', 'BMP', or None
        """
        try:
            with open(filename, 'rb') as f:
                header = f.read(4)
                
                # Check for JPEG
                if header[:2] == b'\xff\xd8':
                    return 'JPEG'
                
                # Check for BMP
                elif header[:2] == b'BM':
                    return 'BMP'
                
                # Check for PNG (optional, for future use)
                elif header == b'\x89PNG':
                    return 'PNG'
                
            return None
        except:
            return None
    
    def get_image_info(self, filename):
        """Get basic information about an image file
        
        Returns dict with: type, size, dimensions (if available)
        """
        info = {
            'type': None,
            'size': 0,
            'width': None,
            'height': None
        }
        
        try:
            with open(filename, 'rb') as f:
                # Detect type
                info['type'] = self.detect_image_type(filename)
                
                if info['type'] == 'BMP':
                    # Read BMP header for dimensions
                    f.seek(18)
                    info['width'] = struct.unpack('<I', f.read(4))[0]
                    info['height'] = struct.unpack('<I', f.read(4))[0]
                
                # Get file size
                f.seek(0, 2)
                info['size'] = f.tell()
                
        except Exception as e:
            print(f"Error reading image info: {e}")
            
        return info