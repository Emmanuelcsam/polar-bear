#!/usr/bin/env python3
"""
Show all available emulators for the BMP Video Analysis System.
This script provides a menu to launch different analysis emulators.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
from pathlib import Path


class EmulatorLauncher:
    """
    GUI for launching different analysis emulators.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("BMP Video Analysis - Emulator Launcher")
        self.root.geometry("600x500")
        self.root.resizable(False, False)

        self._create_widgets()

    def _create_widgets(self):
        """Create the GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame,
                               text="BMP Video Analysis - Emulator Launcher",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Description
        desc_label = ttk.Label(main_frame,
                              text="Select an emulator to launch for real-time analysis:",
                              font=('Arial', 10))
        desc_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        # Emulator buttons
        emulators = [
            {
                "name": "Basic BMP Video Emulator",
                "description": "Basic video emulation with Hough circles detection",
                "script": "bmp_video_emulator.py",
                "color": "#4CAF50"
            },
            {
                "name": "Blob Detection Emulator",
                "description": "Real-time blob detection and analysis",
                "script": "blob_detection_emulator.py",
                "color": "#2196F3"
            },
            {
                "name": "Scratch Detection Emulator",
                "description": "Hough lines detection for scratch analysis",
                "script": "scratch_detection_emulator.py",
                "color": "#FF9800"
            },
            {
                "name": "SSIM Detection Emulator",
                "description": "Structural similarity detection for defects",
                "script": "ssim_detection_emulator.py",
                "color": "#9C27B0"
            },
            {
                "name": "Statistical Features Emulator",
                "description": "Statistical analysis and feature extraction",
                "script": "statistical_features_emulator.py",
                "color": "#607D8B"
            },
            {
                "name": "Frequency Features Emulator",
                "description": "Frequency domain analysis and filtering",
                "script": "frequency_features_emulator.py",
                "color": "#795548"
            },
            {
                "name": "Morphological Features Emulator",
                "description": "Morphological operations and shape analysis",
                "script": "morphological_features_emulator.py",
                "color": "#E91E63"
            }
        ]

        # Create buttons for each emulator
        for i, emulator in enumerate(emulators):
            # Create frame for each emulator
            emulator_frame = ttk.LabelFrame(main_frame, text=emulator["name"], padding="10")
            emulator_frame.grid(row=i+2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
            main_frame.columnconfigure(0, weight=1)

            # Description label
            desc = ttk.Label(emulator_frame, text=emulator["description"])
            desc.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

            # Launch button
            launch_btn = ttk.Button(emulator_frame,
                                   text="Launch",
                                   command=lambda script=emulator["script"]: self._launch_emulator(script))
            launch_btn.grid(row=0, column=1, sticky=tk.E)

            emulator_frame.columnconfigure(0, weight=1)

        # Test all button
        test_frame = ttk.Frame(main_frame)
        test_frame.grid(row=len(emulators)+3, column=0, columnspan=2, pady=20)

        test_all_btn = ttk.Button(test_frame,
                                 text="Run All Tests",
                                 command=self._run_all_tests,
                                 style="Accent.TButton")
        test_all_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Exit button
        exit_btn = ttk.Button(test_frame,
                             text="Exit",
                             command=self.root.quit)
        exit_btn.pack(side=tk.RIGHT)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var,
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=len(emulators)+4, column=0, columnspan=2,
                       sticky=(tk.W, tk.E), pady=(20, 0))

    def _launch_emulator(self, script_name):
        """Launch the specified emulator script."""
        script_path = Path(script_name)

        if not script_path.exists():
            messagebox.showerror("Error", f"Script not found: {script_name}")
            return

        try:
            self.status_var.set(f"Launching {script_name}...")
            self.root.update()

            # Launch the script
            subprocess.Popen([sys.executable, str(script_path)],
                           cwd=Path.cwd())

            self.status_var.set(f"Launched {script_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {script_name}: {e}")
            self.status_var.set("Ready")

    def _run_all_tests(self):
        """Run all test scripts."""
        test_scripts = [
            "test_morphological_features.py",
            "test_all_emulators.py"
        ]

        self.status_var.set("Running all tests...")
        self.root.update()

        for script in test_scripts:
            script_path = Path(script)
            if script_path.exists():
                try:
                    subprocess.run([sys.executable, str(script_path)],
                                 cwd=Path.cwd(), check=True)
                except subprocess.CalledProcessError as e:
                    messagebox.showwarning("Warning",
                                         f"Test script {script} failed: {e}")

        self.status_var.set("All tests completed")
        messagebox.showinfo("Tests Complete", "All test scripts have been executed.")


def main():
    """Main function to run the emulator launcher."""
    root = tk.Tk()
    app = EmulatorLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
