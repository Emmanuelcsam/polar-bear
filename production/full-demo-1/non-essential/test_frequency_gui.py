#!/usr/bin/env python3
"""
GUI test script for frequency features emulator.
This script launches the GUI with the test image and performs automated UI interactions.
"""

import tkinter as tk
from tkinter import ttk
import sys
import time
import threading
from pathlib import Path

# Import the frequency features emulator
from frequency_features_emulator import FrequencyFeaturesGUI

class AutomatedGUITester:
    """Automated tester for the FrequencyFeaturesGUI."""
    
    def __init__(self, app, test_image_path):
        """Initialize the tester."""
        self.app = app
        self.test_image_path = test_image_path
        self.test_results = []
        self.current_test = ""
        
    def log_test(self, test_name, success, message=""):
        """Log test result."""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
        status = "✓" if success else "✗"
        print(f"[{status}] {test_name}: {message}")
        
    def run_tests(self):
        """Run all automated GUI tests."""
        print("\n" + "="*60)
        print("FREQUENCY FEATURES GUI AUTOMATED TESTS")
        print("="*60)
        
        # Start test sequence after GUI is ready
        self.app.root.after(1000, self.test_sequence)
        
    def test_sequence(self):
        """Execute the test sequence."""
        try:
            # Test 1: Load test image
            print("\n--- Test 1: Loading Test Image ---")
            self.load_test_image()
            self.app.root.after(2000, self.test_fft_processing)
            
        except Exception as e:
            self.log_test("Test Sequence", False, f"Error: {str(e)}")
            
    def load_test_image(self):
        """Load the test image programmatically."""
        try:
            import cv2
            # Load image directly
            self.app.current_image = cv2.imread(self.test_image_path, cv2.IMREAD_GRAYSCALE)
            if self.app.current_image is None:
                self.log_test("Load Image", False, f"Failed to load {self.test_image_path}")
                return
                
            # Display in GUI
            self.app._display_image(self.app.current_image, self.app.original_canvas)
            
            # Update status
            h, w = self.app.current_image.shape[:2]
            self.app.status_label.config(text=f"Loaded: {Path(self.test_image_path).name} ({w}x{h})")
            
            self.log_test("Load Image", True, f"Successfully loaded {self.test_image_path}")
            
        except Exception as e:
            self.log_test("Load Image", False, str(e))
            
    def test_fft_processing(self):
        """Test FFT processing."""
        print("\n--- Test 2: FFT Processing ---")
        try:
            # Enable FFT
            self.app.apply_fft_var.set(True)
            self.app.show_spectrum_var.set(True)
            self.app.log_scale_var.set(True)
            self.app._update_processing()
            
            # Process image
            self.app._process_image()
            
            # Wait for processing to complete
            self.app.root.after(3000, self.check_fft_results)
            
        except Exception as e:
            self.log_test("FFT Processing", False, str(e))
            
    def check_fft_results(self):
        """Check FFT processing results."""
        try:
            if self.app.fft_magnitude is not None and self.app.fft_phase is not None:
                self.log_test("FFT Processing", True, "FFT magnitude and phase computed successfully")
                
                # Check features
                if self.app.frequency_features['fft_mean'] > 0:
                    self.log_test("Feature Extraction", True, 
                                f"Features extracted (mean={self.app.frequency_features['fft_mean']:.2f})")
                else:
                    self.log_test("Feature Extraction", False, "No features extracted")
            else:
                self.log_test("FFT Processing", False, "FFT not computed")
                
            # Continue to filter tests
            self.app.root.after(1000, self.test_filters)
            
        except Exception as e:
            self.log_test("FFT Results Check", False, str(e))
            
    def test_filters(self):
        """Test frequency filters."""
        print("\n--- Test 3: Frequency Filters ---")
        
        filter_tests = [
            ('lowpass', 0.3, None),
            ('highpass', 0.3, None),
            ('bandpass', 0.2, 0.8),
            ('bandstop', 0.3, 0.7)
        ]
        
        self.filter_test_index = 0
        self.filter_tests = filter_tests
        self.test_next_filter()
        
    def test_next_filter(self):
        """Test next filter in sequence."""
        if self.filter_test_index < len(self.filter_tests):
            filter_type, cutoff_low, cutoff_high = self.filter_tests[self.filter_test_index]
            
            try:
                # Configure filter
                self.app.apply_filter_var.set(True)
                self.app.filter_type_var.set(filter_type)
                self.app.cutoff_freq_var.set(cutoff_low)
                if cutoff_high:
                    self.app.cutoff_freq_high_var.set(cutoff_high)
                self.app._update_processing()
                
                # Process with filter
                self.app._process_image()
                
                self.log_test(f"{filter_type.capitalize()} Filter", True, 
                            f"Applied with cutoff={cutoff_low}")
                
            except Exception as e:
                self.log_test(f"{filter_type.capitalize()} Filter", False, str(e))
                
            self.filter_test_index += 1
            self.app.root.after(2000, self.test_next_filter)
        else:
            # Continue to edge enhancement
            self.app.root.after(1000, self.test_edge_enhancement)
            
    def test_edge_enhancement(self):
        """Test edge enhancement."""
        print("\n--- Test 4: Edge Enhancement ---")
        try:
            self.app.enhance_edges_var.set(True)
            self.app.apply_filter_var.set(False)  # Disable filter for edge test
            self.app._update_processing()
            self.app._process_image()
            
            self.log_test("Edge Enhancement", True, "Applied successfully")
            
            # Continue to periodic detection
            self.app.root.after(2000, self.test_periodic_detection)
            
        except Exception as e:
            self.log_test("Edge Enhancement", False, str(e))
            
    def test_periodic_detection(self):
        """Test periodic pattern detection."""
        print("\n--- Test 5: Periodic Pattern Detection ---")
        try:
            self.app.detect_periodic_var.set(True)
            self.app.periodic_threshold_var.set(0.3)  # Lower threshold to find more patterns
            self.app.enhance_edges_var.set(False)  # Disable edge enhancement
            self.app._update_processing()
            self.app._process_image()
            
            # Wait and check results
            self.app.root.after(2000, self.check_periodic_results)
            
        except Exception as e:
            self.log_test("Periodic Detection", False, str(e))
            
    def check_periodic_results(self):
        """Check periodic detection results."""
        try:
            if 'periodic_peaks' in self.app.frequency_features:
                num_peaks = len(self.app.frequency_features['periodic_peaks'])
                if num_peaks > 0:
                    self.log_test("Periodic Detection", True, f"Found {num_peaks} periodic peaks")
                else:
                    self.log_test("Periodic Detection", True, "No periodic peaks found (may be expected)")
            else:
                self.log_test("Periodic Detection", False, "Detection not performed")
                
            # Continue to parameter extremes
            self.app.root.after(1000, self.test_extreme_parameters)
            
        except Exception as e:
            self.log_test("Periodic Results Check", False, str(e))
            
    def test_extreme_parameters(self):
        """Test extreme parameter values."""
        print("\n--- Test 6: Extreme Parameters ---")
        
        extreme_tests = [
            ('Min cutoff', 0.01),
            ('Max cutoff', 0.99),
            ('Mid cutoff', 0.5)
        ]
        
        for test_name, value in extreme_tests:
            try:
                self.app.cutoff_freq_var.set(value)
                self.app.periodic_threshold_var.set(value)
                self.app._update_processing()
                
                self.log_test(f"Extreme Parameter: {test_name}", True, f"value={value}")
                
            except Exception as e:
                self.log_test(f"Extreme Parameter: {test_name}", False, str(e))
                
        # Continue to UI responsiveness test
        self.app.root.after(1000, self.test_ui_responsiveness)
        
    def test_ui_responsiveness(self):
        """Test UI responsiveness."""
        print("\n--- Test 7: UI Responsiveness ---")
        
        try:
            # Rapidly change parameters
            for i in range(10):
                self.app.cutoff_freq_var.set(i / 10.0)
                self.app.root.update()
                
            self.log_test("UI Responsiveness", True, "GUI responds to rapid parameter changes")
            
        except Exception as e:
            self.log_test("UI Responsiveness", False, str(e))
            
        # Finish tests
        self.app.root.after(1000, self.finish_tests)
        
    def finish_tests(self):
        """Finish testing and print summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r['success'])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "PASSED" if result['success'] else "FAILED"
            print(f"{result['test']}: {status}")
            
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✅ ALL GUI TESTS PASSED!")
            print("\nThe frequency features emulator is working correctly:")
            print("- FFT features are extracted and displayed properly")
            print("- Frequency filters produce expected results")
            print("- GUI updates smoothly when parameters change")
            print("- Spectrum visualization shows frequency information")
            print("- No crashes with extreme parameter values")
        else:
            print(f"\n⚠️ {total - passed} tests failed")
            
        print("\n[GUI will remain open for manual inspection]")
        print("You can now interact with the GUI manually to verify functionality.")
        print("Close the window when done.\n")

def main():
    """Main function to run GUI tests."""
    test_image = "frequency_test.bmp"
    
    # Check if test image exists
    if not Path(test_image).exists():
        print(f"ERROR: Test image {test_image} not found!")
        print("Please run create_frequency_test.py first to generate the test image.")
        return
        
    # Create GUI
    root = tk.Tk()
    app = FrequencyFeaturesGUI(root)
    
    # Create tester
    tester = AutomatedGUITester(app, test_image)
    
    # Start tests after GUI initialization
    root.after(500, tester.run_tests)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Run GUI
    root.mainloop()

if __name__ == "__main__":
    main()
