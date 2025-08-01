#!/usr/bin/env python3
"""
Create GUI-compatible wrappers
from existing OpenCV scripts.
"""

import os
import re
import ast
from pathlib import Path
import shutil

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

class ScriptCleaner:
    def __init__(self, source_dir=".", output_dir="scripts", verbose=True):
        # Convert source directory string to Path object for cross-platform compatibility
        self.source_dir = Path(source_dir)
        # Convert output directory string to Path object for cross-platform compatibility
        self.output_dir = Path(output_dir)
        # Create output directory if it doesn't exist (exist_ok=True prevents errors if directory already exists)
        self.output_dir.mkdir(exist_ok=True)
        # Store verbose flag to control output detail level
        self.verbose = verbose
        
        # Define regex patterns to detect different types of image processing operations in scripts
        # Each key represents an operation type, each value is a list of regex patterns to match that operation
        self.operation_patterns = {
            # Pattern for Gaussian blur operations (both OpenCV function and text mentions)
            'gaussian_blur': [r'cv2\.GaussianBlur', r'gaussian.*blur'],
            # Pattern for median filter operations
            'median_blur': [r'cv2\.medianBlur', r'median.*filter'],
            # Pattern for bilateral filtering operations
            'bilateral_filter': [r'cv2\.bilateralFilter'],
            # Pattern for Canny edge detection operations
            'canny_edge': [r'cv2\.Canny', r'canny.*edge'],
            # Pattern for Sobel edge detection operations
            'sobel_edge': [r'cv2\.Sobel', r'sobel'],
            # Pattern for Laplacian edge detection operations
            'laplacian_edge': [r'cv2\.Laplacian', r'laplacian'],
            # Pattern for threshold operations (excluding adaptive threshold)
            'threshold': [r'cv2\.threshold', r'thresh'],
            # Pattern for adaptive threshold operations
            'adaptive_threshold': [r'cv2\.adaptiveThreshold'],
            # Pattern for morphological operations (erosion, dilation, etc.)
            'morphology': [r'cv2\.morphologyEx', r'cv2\.erode', r'cv2\.dilate'],
            # Pattern for circle detection operations
            'circle_detection': [r'cv2\.HoughCircles', r'circle.*detect'],
            # Pattern for contour detection operations
            'contour': [r'cv2\.findContours', r'contour'],
            # Pattern for histogram equalization operations
            'histogram': [r'cv2\.equalizeHist', r'histogram'],
            # Pattern for CLAHE (Contrast Limited Adaptive Histogram Equalization) operations
            'clahe': [r'cv2\.createCLAHE', r'clahe'],
            # Pattern for grayscale conversion operations
            'grayscale': [r'cv2\.cvtColor.*GRAY', r'grayscale', r'gray'],
            # Pattern for colormap application operations
            'colormap': [r'cv2\.applyColorMap', r'colormap'],
            # Pattern for custom filtering operations
            'filter2D': [r'cv2\.filter2D'],
            # Pattern for masking operations
            'mask': [r'mask', r'bitwise_and'],
        }
        
    def detect_encoding(self, file_path):
        """Detect file encoding to handle Unicode errors"""
        # If chardet library is not available, default to UTF-8 encoding
        if not CHARDET_AVAILABLE:
            return 'utf-8'
            
        try:
            # Open file in binary mode to read raw bytes for encoding detection
            with open(file_path, 'rb') as f:
                # Read all bytes from the file
                raw_data = f.read()
                # Use chardet to detect the encoding of the file
                result = chardet.detect(raw_data)
                # Return the detected encoding
                return result['encoding']
        except:
            # If detection fails, fall back to UTF-8
            return 'utf-8'
            
    def read_script_safely(self, file_path):
        """Read script with proper encoding handling"""
        # List of common encodings to try in order of preference
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # Try detected encoding first by calling the detect_encoding method
        detected_encoding = self.detect_encoding(file_path)
        if detected_encoding:
            # Insert detected encoding at the beginning of the list to try it first
            encodings.insert(0, detected_encoding)
            
        # Try each encoding until one works
        for encoding in encodings:
            try:
                # Open file with current encoding, ignore errors to handle problematic characters
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    # Read the entire file content
                    content = f.read()
                    
                # Clean any remaining Unicode issues by encoding to ASCII and back
                # This removes any non-ASCII characters that might cause issues
                content = content.encode('ascii', 'ignore').decode('ascii')
                # Return the cleaned content
                return content
            except:
                # If this encoding fails, continue to the next one
                continue
                
        # Last resort - read as binary and decode
        # Open file in binary mode
        with open(file_path, 'rb') as f:
            # Read all bytes from the file
            content = f.read()
            # Remove null bytes that might cause issues
            content = content.replace(b'\x00', b'')
            # Decode as ASCII, ignoring any problematic characters
            return content.decode('ascii', 'ignore')
            
    def extract_imports(self, content):
        """Extract import statements from the script"""
        # Create a set to store unique import module names
        imports = set()
        # Define regex patterns to match different types of import statements
        import_patterns = [
            # Pattern for "import module" statements
            r'import\s+(\w+)',
            # Pattern for "from module import" statements
            r'from\s+(\w+)\s+import',
            # Pattern for "import module as alias" statements
            r'import\s+(\w+)\s+as\s+\w+'
        ]
        
        # Search for matches using each pattern
        for pattern in import_patterns:
            # Find all matches of the current pattern in the content
            matches = re.findall(pattern, content)
            # Add all found module names to the imports set
            imports.update(matches)
            
        # Always include these essential libraries for image processing
        imports.update(['cv2', 'numpy'])
        
        # Return sorted list of unique import names
        return sorted(imports)
        
    def detect_operations(self, content):
        """Detect which operations are performed in the script"""
        # List to store detected operation names
        detected = []
        
        # Iterate through each operation type and its patterns
        for op_name, patterns in self.operation_patterns.items():
            # Check each pattern for the current operation
            for pattern in patterns:
                # Search for the pattern in the content (case-insensitive)
                if re.search(pattern, content, re.IGNORECASE):
                    # If pattern is found, add operation name to detected list
                    detected.append(op_name)
                    # Break inner loop since we found this operation
                    break
                    
        # Return list of detected operations
        return detected
        
    def extract_parameters(self, content):
        """Try to extract parameters used in the script"""
        # Dictionary to store extracted parameter names and values
        params = {}
        
        # Define regex patterns for common parameter types
        # Each key is a parameter name, each value is a list of patterns to find that parameter
        param_patterns = {
            # Pattern for kernel size parameters (e.g., "kernel_size = 5")
            'kernel_size': [r'kernel.*size\s*=\s*(\d+)', r'\((\d+),\s*\1\)'],
            # Pattern for threshold parameters (e.g., "threshold = 127")
            'threshold': [r'threshold\s*=\s*(\d+)', r'thresh.*=\s*(\d+)'],
            # Pattern for sigma parameters (e.g., "sigma = 1.5")
            'sigma': [r'sigma\s*=\s*(\d+\.?\d*)', r'GaussianBlur.*,\s*(\d+\.?\d*)\)'],
            # Pattern for iteration parameters (e.g., "iterations = 3")
            'iterations': [r'iterations\s*=\s*(\d+)'],
            # Pattern for CLAHE clip limit parameters (e.g., "clipLimit = 2.0")
            'clip_limit': [r'clipLimit\s*=\s*(\d+\.?\d*)'],
        }
        
        # Search for each parameter type in the content
        for param_name, patterns in param_patterns.items():
            # Check each pattern for the current parameter
            for pattern in patterns:
                # Search for the pattern in the content
                match = re.search(pattern, content)
                if match:
                    try:
                        # Convert matched value to float
                        value = float(match.group(1))
                        # Store as integer if it's a whole number, otherwise as float
                        params[param_name] = int(value) if value.is_integer() else value
                        # Break inner loop since we found this parameter
                        break
                    except:
                        # If conversion fails, continue to next pattern
                        pass
                        
        # Return dictionary of extracted parameters
        return params
        
    def generate_wrapper(self, script_name, content):
        """Generate a clean wrapper for the script"""
        # Clean the script name by replacing spaces with underscores and removing parentheses
        clean_name = script_name.replace(' ', '_').replace('(', '').replace(')', '')
        
        # Extract information from the script content
        # Get list of imported modules
        imports = self.extract_imports(content)
        # Detect which operations are performed in the script
        operations = self.detect_operations(content)
        # Extract parameter values used in the script
        parameters = self.extract_parameters(content)
        
        # Generate description for the wrapper
        description = f"Processed from {script_name}"
        # Add detected operations to description if any were found
        if operations:
            description += f" - Detected operations: {', '.join(operations[:3])}"
            
        # Start building the wrapper code with docstring and essential imports
        wrapper = f'''"""{description}"""
import cv2
import numpy as np
'''
        
        # Add any additional imports that aren't standard ones
        # Filter out common standard libraries to avoid unnecessary imports
        additional_imports = [imp for imp in imports if imp not in ['cv2', 'numpy', 'os', 'sys']]
        if additional_imports:
            # Add imports for common image processing libraries
            for imp in additional_imports:
                if imp in ['matplotlib', 'PIL', 'skimage']:  # Common image processing libraries
                    wrapper += f"import {imp}\n"
                    
        # Start defining the process_image function with its signature
        wrapper += "\ndef process_image(image: np.ndarray"
        
        # Add detected parameters as function parameters with default values
        if parameters:
            for param_name, default_value in parameters.items():
                wrapper += f", {param_name}: float = {default_value}"
                
        # Complete the function signature and start the docstring
        wrapper += ") -> np.ndarray:\n"
        wrapper += f'    """\n    {description}\n    \n    Args:\n        image: Input image\n'
        
        # Add parameter descriptions to the docstring
        if parameters:
            for param_name in parameters:
                # Convert parameter name to readable description
                wrapper += f'        {param_name}: {param_name.replace("_", " ").capitalize()}\n'
                
        # Complete the docstring
        wrapper += '    \n    Returns:\n        Processed image\n    """\n'
        
        # Generate the actual processing code based on detected operations
        wrapper += self._generate_processing_code(operations, parameters)
        
        # Return the complete wrapper code
        return wrapper
        
    def _generate_processing_code(self, operations, parameters):
        """Generate the actual processing code based on detected operations"""
        # Start with error handling and result initialization
        code = "    try:\n        result = image.copy()\n        \n"
        
        # Generate code for each detected operation
        if 'grayscale' in operations:
            # Add grayscale conversion code
            code += "        # Convert to grayscale if needed\n"
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n        \n"
            
        if 'gaussian_blur' in operations:
            # Add Gaussian blur code
            code += "        # Apply Gaussian blur\n"
            if 'kernel_size' in parameters:
                # Use detected kernel size parameter
                code += "        kernel_size = int(kernel_size)\n"
                # Ensure kernel size is odd (required for Gaussian blur)
                code += "        if kernel_size % 2 == 0:\n"
                code += "            kernel_size += 1\n"
                code += "        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), "
                code += f"{parameters.get('sigma', 0)})\n        \n"
            else:
                # Use default kernel size
                code += "        result = cv2.GaussianBlur(result, (5, 5), 0)\n        \n"
                
        if 'median_blur' in operations:
            # Add median blur code
            code += "        # Apply median blur\n"
            code += f"        result = cv2.medianBlur(result, {parameters.get('kernel_size', 5)})\n        \n"
            
        if 'canny_edge' in operations:
            # Add Canny edge detection code
            code += "        # Apply Canny edge detection\n"
            # Convert to grayscale if needed for edge detection
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            # Apply Canny with detected or default threshold values
            code += f"        result = cv2.Canny(result, {parameters.get('threshold', 50)}, "
            code += f"{parameters.get('threshold', 50) * 3})\n        \n"
            
        if 'threshold' in operations and 'adaptive_threshold' not in operations:
            # Add simple threshold code (not adaptive)
            code += "        # Apply threshold\n"
            # Convert to grayscale if needed
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            # Apply threshold with detected or default value
            code += f"        _, result = cv2.threshold(result, {parameters.get('threshold', 127)}, 255, cv2.THRESH_BINARY)\n        \n"
            
        if 'adaptive_threshold' in operations:
            # Add adaptive threshold code
            code += "        # Apply adaptive threshold\n"
            # Convert to grayscale if needed
            code += "        if len(result.shape) == 3:\n"
            code += "            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)\n"
            # Apply adaptive threshold with default parameters
            code += "        result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "
            code += "cv2.THRESH_BINARY, 11, 2)\n        \n"
            
        if 'morphology' in operations:
            # Add morphological operations code
            code += "        # Apply morphological operation\n"
            # Create elliptical structuring element
            code += "        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))\n"
            # Apply morphological closing operation
            code += "        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)\n        \n"
            
        if 'circle_detection' in operations:
            # Add circle detection code
            code += "        # Detect circles\n"
            # Ensure we have a BGR image for drawing circles
            code += "        if len(result.shape) == 2:\n"
            code += "            display = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)\n"
            code += "        else:\n"
            code += "            display = result.copy()\n"
            # Convert to grayscale for circle detection
            code += "        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) if len(display.shape) == 3 else display\n"
            # Detect circles using Hough transform
            code += "        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30)\n"
            # Draw detected circles on the image
            code += "        if circles is not None:\n"
            code += "            circles = np.uint16(np.around(circles))\n"
            code += "            for i in circles[0, :]:\n"
            code += "                cv2.circle(display, (i[0], i[1]), i[2], (0, 255, 0), 2)\n"
            # Use the display image with drawn circles as result
            code += "        result = display\n        \n"
            
        if 'histogram' in operations:
            # Add histogram equalization code
            code += "        # Apply histogram equalization\n"
            # For color images, equalize only the Y channel in YCrCb color space
            code += "        if len(result.shape) == 3:\n"
            code += "            ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)\n"
            code += "            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])\n"
            code += "            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)\n"
            # For grayscale images, apply histogram equalization directly
            code += "        else:\n"
            code += "            result = cv2.equalizeHist(result)\n        \n"
            
        if 'clahe' in operations:
            # Add CLAHE (Contrast Limited Adaptive Histogram Equalization) code
            code += "        # Apply CLAHE\n"
            # Create CLAHE object with detected or default clip limit
            code += f"        clahe = cv2.createCLAHE(clipLimit={parameters.get('clip_limit', 2.0)}, tileGridSize=(8,8))\n"
            # For color images, apply CLAHE to L channel in LAB color space
            code += "        if len(result.shape) == 3:\n"
            code += "            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)\n"
            code += "            lab[:, :, 0] = clahe.apply(lab[:, :, 0])\n"
            code += "            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)\n"
            # For grayscale images, apply CLAHE directly
            code += "        else:\n"
            code += "            result = clahe.apply(result)\n        \n"
            
        if not operations:
            # If no operations were detected, add placeholder code
            code += "        # Add your processing logic here\n"
            code += "        # This is a placeholder - modify based on the original script\n        \n"
            
        # Add return statement and error handling
        code += "        return result\n        \n"
        code += "    except Exception as e:\n"
        code += f'        print(f"Error in processing: {{e}}")\n'
        code += "        return image\n"
        
        # Return the generated processing code
        return code
        
    def clean_script(self, script_path):
        """Clean a single script and create a wrapper"""
        try:
            # Read the script safely using the encoding detection method
            content = self.read_script_safely(script_path)
            
            # Clean common issues like hardcoded paths and problematic lines
            content = self.clean_common_issues(content)
            
            # Check if the script already has a process_image function and doesn't contain hardcoded paths
            if 'def process_image' in content and 'hardcoded' not in content.lower():
                # If it already has the right structure, just save it as is
                output_path = self.output_dir / script_path.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, "Already has process_image function"
                
            # Generate a new wrapper for the script
            wrapper = self.generate_wrapper(script_path.name, content)
            
            # Save the generated wrapper to the output directory
            output_path = self.output_dir / script_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(wrapper)
                
            return True, "Successfully created wrapper"
            
        except Exception as e:
            # Return error information if processing fails
            return False, str(e)
            
    def clean_common_issues(self, content):
        """Clean common issues in scripts"""
        # Remove hardcoded Windows paths (e.g., C:\Users\...)
        content = re.sub(r'[A-Za-z]:[\\\/][^\'";\n]+', '', content)
        # Remove hardcoded Unix paths (e.g., /home/...)
        content = re.sub(r'\/home\/[^\'";\n]+', '', content)
        # Remove hardcoded macOS paths (e.g., /Users/...)
        content = re.sub(r'\/Users\/[^\'";\n]+', '', content)
        
        # Define patterns for lines that should be removed (GUI-specific or problematic)
        lines_to_remove = [
            r'img_path\s*=.*',  # Image path assignments
            r'image_path\s*=.*',  # Image path assignments
            r'base_path\s*=.*',  # Base path assignments
            r'cv2\.imshow\(.*\)',  # OpenCV display windows
            r'cv2\.waitKey\(.*\)',  # OpenCV wait key calls
            r'cv2\.destroyAllWindows\(.*\)',  # OpenCV window destruction
            r'plt\.show\(.*\)',  # Matplotlib display calls
        ]
        
        # Remove each type of problematic line
        for pattern in lines_to_remove:
            content = re.sub(pattern, '', content)
            
        # Return the cleaned content
        return content
        
    def clean_all_scripts(self):
        """Clean all scripts in the source directory"""
        # Dictionary to store results of the cleaning process
        results = {
            'success': [],  # Successfully processed scripts
            'failed': [],   # Scripts that failed to process
            'skipped': []   # Scripts that were skipped
        }
        
        # Get all Python files in the source directory
        script_files = list(self.source_dir.glob("*.py"))
        
        # If no Python files found, print message and return empty results
        if not script_files:
            print(f"No Python files found in {self.source_dir}")
            return results
            
        # Print the number of scripts found for processing
        print(f"Found {len(script_files)} scripts to process\n")
        
        # Process each script file
        for script_path in script_files:
            # Skip certain files that shouldn't be processed
            if script_path.name.startswith('_') or script_path.name in [
                'setup_gui.py', 'image_processor_gui.py', 'script_cleaner.py'
            ]:
                # Add to skipped list
                results['skipped'].append(script_path.name)
                if self.verbose:
                    print(f"⏭️  Skipped: {script_path.name}")
                continue
                
            # Print processing message if verbose mode is enabled
            if self.verbose:
                print(f"🔧 Processing: {script_path.name}...", end=' ')
                
            # Attempt to clean the current script
            success, message = self.clean_script(script_path)
            
            # Handle the result of the cleaning attempt
            if success:
                # Add to success list if cleaning was successful
                results['success'].append(script_path.name)
                if self.verbose:
                    print(f"✅ {message}")
            else:
                # Add to failed list if cleaning failed
                results['failed'].append((script_path.name, message))
                if self.verbose:
                    print(f"❌ Failed: {message}")
                    
        # Print summary of the cleaning process
        print("\n" + "="*60)
        print("CLEANING SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {len(results['success'])} scripts")
        print(f"❌ Failed: {len(results['failed'])} scripts")
        print(f"⏭️  Skipped: {len(results['skipped'])} scripts")
        
        # Print details of failed scripts if any and verbose mode is enabled
        if results['failed'] and self.verbose:
            print("\nFailed scripts:")
            for name, error in results['failed']:
                print(f"  - {name}: {error}")
                
        # Return the results dictionary
        return results


def main():
    import argparse
    
    # Create argument parser for command-line interface
    parser = argparse.ArgumentParser(
        description="Clean scripts with Unicode errors and create GUI-compatible wrappers"
    )
    # Add argument for source directory (defaults to current directory)
    parser.add_argument(
        '--source', 
        default='.', 
        help='Source directory containing scripts (default: current directory)'
    )
    # Add argument for output directory (defaults to 'scripts')
    parser.add_argument(
        '--output', 
        default='scripts', 
        help='Output directory for cleaned scripts (default: scripts)'
    )
    # Add flag for quiet mode (suppresses detailed output)
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Suppress detailed output'
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Print header information
    print("🧹 Script Cleaner")
    print("=" * 60)
    print(f"Source directory: {args.source}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    # Create ScriptCleaner instance with parsed arguments
    cleaner = ScriptCleaner(
        source_dir=args.source,
        output_dir=args.output,
        verbose=not args.quiet  # Invert quiet flag for verbose setting
    )
    
    # Run the cleaning process and store results
    results = cleaner.clean_all_scripts()
    
    # Print success message and next steps if any scripts were successfully processed
    if results['success']:
        print(f"\n✅ Cleaning complete! Cleaned scripts are in '{args.output}' directory")
        print("\nNext steps:")
        print("1. Review the cleaned scripts in the output directory")
        print("2. Run the Image Processing GUI:")
        print("   python image_processor_gui.py")
        print("\nThe GUI will automatically load all scripts from the 'scripts' directory")


if __name__ == "__main__":
    main()
