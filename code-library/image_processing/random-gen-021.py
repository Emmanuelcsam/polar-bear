#!/usr/bin/env python3
"""
Enhanced Pixel Reader - Works independently and with connector
Demonstrates full integration with the connector system
"""

from PIL import Image
import json
import time
import os

# Import the script wrapper for connector integration
try:
    from script_wrapper import (
        expose_function, expose_variable, collaborative_mode,
        get_shared_variable, set_shared_variable, send_message,
        log_to_connector, config, is_connected, broadcast_result
    )
    WRAPPER_AVAILABLE = True
except ImportError:
    # Fallback for independent operation
    WRAPPER_AVAILABLE = False
    
    # Create dummy decorators and functions
    def expose_function(func):
        return func
    
    def collaborative_mode(func):
        return func
    
    def expose_variable(name, value, writable=True):
        pass
    
    def get_shared_variable(name, default=None):
        return default
    
    def set_shared_variable(name, value):
        pass
    
    def send_message(msg_type, data):
        pass
    
    def log_to_connector(message, level="INFO"):
        print(f"[{level}] {message}")
    
    def is_connected():
        return False
    
    def broadcast_result(name, value):
        pass
    
    class DummyConfig:
        def set_parameter(self, name, value):
            pass
        def get_parameter(self, name, default=None):
            return default
    
    config = DummyConfig()


# Configuration parameters
DEFAULT_OUTPUT_FILE = 'pixel_data.json'
DEFAULT_IMAGE_FORMAT = 'L'  # Grayscale

# Expose configuration to connector
config.set_parameter('output_file', DEFAULT_OUTPUT_FILE)
config.set_parameter('image_format', DEFAULT_IMAGE_FORMAT)
config.set_parameter('verbose', True)

# Module-level variables exposed to connector
last_processed_image = None
last_pixel_count = 0
processing_time = 0.0

expose_variable('last_processed_image', last_processed_image)
expose_variable('last_pixel_count', last_pixel_count)
expose_variable('processing_time', processing_time)


@expose_function
def read_pixels(image_path, format_mode=None):
    """Read pixels from an image file"""
    global last_processed_image, last_pixel_count, processing_time
    
    start_time = time.time()
    
    # Use configuration or default
    if format_mode is None:
        format_mode = config.get_parameter('image_format', DEFAULT_IMAGE_FORMAT)
    
    log_to_connector(f"Reading pixels from {image_path} in {format_mode} mode")
    
    try:
        img = Image.open(image_path).convert(format_mode)
        pixels = list(img.getdata())
        
        # Update module variables
        last_processed_image = image_path
        last_pixel_count = len(pixels)
        processing_time = time.time() - start_time
        
        # Expose updated variables
        expose_variable('last_processed_image', last_processed_image)
        expose_variable('last_pixel_count', last_pixel_count)
        expose_variable('processing_time', processing_time)
        
        # Prepare data
        data = {
            'timestamp': time.time(),
            'image': image_path,
            'pixels': pixels,
            'size': img.size,
            'format': format_mode,
            'processing_time': processing_time
        }
        
        # Save to file if configured
        output_file = config.get_parameter('output_file', DEFAULT_OUTPUT_FILE)
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(data, f)
            log_to_connector(f"Saved pixel data to {output_file}")
        
        # Broadcast result if connected
        broadcast_result('pixel_data', data)
        
        # Send completion message
        send_message('pixel_read_complete', {
            'image': image_path,
            'pixel_count': len(pixels),
            'size': img.size
        })
        
        log_to_connector(f"Successfully read {len(pixels)} pixels from {image_path}")
        return pixels
        
    except Exception as e:
        log_to_connector(f"Error reading pixels: {e}", "ERROR")
        send_message('pixel_read_error', {
            'image': image_path,
            'error': str(e)
        })
        raise


@expose_function
def get_pixel_statistics(pixels):
    """Calculate statistics from pixel data"""
    if not pixels:
        return {}
    
    stats = {
        'count': len(pixels),
        'min': min(pixels),
        'max': max(pixels),
        'mean': sum(pixels) / len(pixels),
        'unique_values': len(set(pixels))
    }
    
    # Share statistics
    set_shared_variable('last_pixel_stats', stats)
    broadcast_result('pixel_statistics', stats)
    
    return stats


@expose_function
@collaborative_mode
def process_directory(directory='.', pattern='*.jpg,*.jpeg,*.png,*.bmp'):
    """Process all images in a directory"""
    log_to_connector(f"Processing directory: {directory}")
    
    results = []
    extensions = [ext.strip() for ext in pattern.split(',')]
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext.replace('*', '')) for ext in extensions):
            try:
                filepath = os.path.join(directory, file)
                pixels = read_pixels(filepath)
                stats = get_pixel_statistics(pixels)
                
                results.append({
                    'file': file,
                    'path': filepath,
                    'pixel_count': len(pixels),
                    'stats': stats
                })
                
                # Allow other scripts to process this data
                send_message('image_processed', {
                    'file': file,
                    'stats': stats
                })
                
            except Exception as e:
                log_to_connector(f"Failed to process {file}: {e}", "ERROR")
                results.append({
                    'file': file,
                    'error': str(e)
                })
    
    # Share final results
    set_shared_variable('directory_scan_results', results)
    broadcast_result('directory_processed', {
        'directory': directory,
        'files_processed': len(results),
        'results': results
    })
    
    return results


@expose_function
def set_output_format(format_type):
    """Set the output format for pixel reading"""
    valid_formats = ['L', 'RGB', 'RGBA', 'CMYK', '1']
    if format_type in valid_formats:
        config.set_parameter('image_format', format_type)
        log_to_connector(f"Output format set to: {format_type}")
        return True
    else:
        log_to_connector(f"Invalid format: {format_type}. Valid formats: {valid_formats}", "ERROR")
        return False


def main():
    """Main function for independent execution"""
    print(f"[PIXEL_READER] Starting (Connected: {is_connected()})")
    
    if is_connected():
        print("[PIXEL_READER] Running in collaborative mode")
        # In collaborative mode, wait for commands
        import time
        while True:
            time.sleep(1)
            # The connector will call our exposed functions
    else:
        print("[PIXEL_READER] Running in independent mode")
        
        # Example usage - reads the first image it finds
        image_found = False
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    pixels = read_pixels(file)
                    stats = get_pixel_statistics(pixels)
                    
                    print(f"[PIXEL_READER] Statistics: {stats}")
                    
                    # Print pixels if verbose
                    if config.get_parameter('verbose', True):
                        for p in pixels:
                            print(p)
                    
                    image_found = True
                    break
                except Exception as e:
                    print(f"[PIXEL_READER] Error: {e}")
        
        if not image_found:
            print("[PIXEL_READER] No image found in current directory")
            # Try processing entire directory
            results = process_directory()
            print(f"[PIXEL_READER] Processed {len(results)} files")


if __name__ == "__main__":
    # Auto-expose module contents if wrapper is available
    if WRAPPER_AVAILABLE:
        from script_wrapper import auto_expose_module
        auto_expose_module()
    
    main()