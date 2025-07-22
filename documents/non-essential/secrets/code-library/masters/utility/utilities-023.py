#!/usr/bin/env python3
"""Fix Qt environment issues before importing OpenCV"""
import os
import sys

def fix_qt_environment():
    """Set environment variables to fix Qt plugin conflicts"""
    # Remove OpenCV's Qt plugin path to avoid conflicts
    os.environ['QT_PLUGIN_PATH'] = ''
    
    # Unset platform plugin path if it exists
    if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
        del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']
    
    # Force xcb platform for better compatibility
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    print("Qt environment fixed for OpenCV compatibility")

# Apply fix immediately when module is imported
fix_qt_environment()