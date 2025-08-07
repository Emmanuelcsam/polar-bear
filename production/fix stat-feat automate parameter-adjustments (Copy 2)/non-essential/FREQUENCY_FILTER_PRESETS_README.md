# Frequency Filter Presets - User Guide

## Overview
The frequency filter preset system provides quick access to common filtering scenarios and allows users to save custom configurations for repeated use.

## Features

### 1. Built-in Presets
Three pre-configured filtering scenarios are available:

#### **Noise Removal** 
- **Type:** Lowpass filter
- **Cutoff:** 0.2
- **Purpose:** Removes high-frequency noise while preserving main image features
- **Use Case:** Cleaning up noisy images, removing sensor noise, smoothing

#### **Edge Enhancement**
- **Type:** Highpass filter  
- **Cutoff:** 0.1
- **Purpose:** Enhances edges and fine details by removing low frequencies
- **Use Case:** Sharpening images, edge detection preprocessing, detail extraction

#### **Pattern Isolation**
- **Type:** Bandpass filter
- **Cutoff:** 0.3
- **Purpose:** Isolates periodic patterns in a specific frequency range
- **Use Case:** Detecting repeating patterns, texture analysis, interference pattern detection

### 2. Custom Presets
Save your own filter configurations for later use:

- **Save Current Settings:** Store current filter parameters with a custom name
- **Load Preset:** Apply saved configurations instantly
- **Delete Preset:** Remove unwanted custom presets
- **Persistence:** Custom presets are saved to `frequency_filter_presets.json`

### 3. Import/Export
Share presets between systems or backup your configurations:

- **Export:** Save all custom presets to a JSON file
- **Import:** Load presets from an external JSON file
- **Format:** Standard JSON format for easy editing

### 4. Reset to Defaults
Quickly return to the default filter settings:
- Filter Type: Lowpass
- Cutoff Frequency: 0.3
- Filter Applied: Off

## User Interface

### Control Tabs
The interface is organized into two tabs:

1. **Manual Controls Tab**
   - Direct control over filter parameters
   - Real-time slider for cutoff frequency
   - Filter type selection dropdown
   - Reset to defaults button

2. **Presets Tab**
   - Built-in preset buttons
   - Custom preset management
   - Import/Export functionality

### Using the Presets

#### Applying a Built-in Preset:
1. Click on the "Presets" tab
2. Click one of the preset buttons (Noise Removal, Edge Enhancement, Pattern Isolation)
3. The filter will be applied immediately

#### Creating a Custom Preset:
1. Adjust filter settings in the Manual Controls tab
2. Switch to the Presets tab
3. Click "Save Current"
4. Enter a name for your preset
5. Click OK to save

#### Loading a Custom Preset:
1. Select a preset from the list in the Presets tab
2. Click "Load Selected"
3. The saved settings will be applied

#### Managing Custom Presets:
- **Delete:** Select preset and click "Delete Selected"
- **Export:** Click "Export All" to save presets to file
- **Import:** Click "Import" to load presets from file

## Technical Details

### Preset Storage Format
Custom presets are stored in JSON format:
```json
{
  "My Custom Filter": {
    "filter_type": "bandpass",
    "cutoff_freq": 0.25,
    "apply_filter": true
  }
}
```

### File Locations
- **Auto-save file:** `frequency_filter_presets.json` (in application directory)
- **Export files:** User-specified location with `.json` extension

### Parameter Ranges
- **Cutoff Frequency:** 0.01 to 0.99 (normalized frequency)
- **Filter Types:** lowpass, highpass, bandpass

## Tips and Best Practices

1. **Testing Presets:** Use the "Generate Test" button to create a test image with various frequency components

2. **Fine-tuning:** Start with a built-in preset and adjust parameters to create custom variations

3. **Naming Convention:** Use descriptive names for custom presets (e.g., "Strong Noise Reduction", "Subtle Edge Enhance")

4. **Backup:** Regularly export your custom presets to avoid losing configurations

5. **Sharing:** Export presets to share optimal settings with colleagues

## Keyboard Shortcuts
Currently, the interface uses standard GUI interactions. Keyboard shortcuts may be added in future versions.

## Troubleshooting

### Preset Not Loading
- Check if the preset file exists
- Verify JSON format is valid
- Ensure file permissions allow reading

### Changes Not Saving
- Check write permissions for `frequency_filter_presets.json`
- Verify disk space is available

### Filter Not Applying
- Ensure "Apply Filter" checkbox is checked
- Verify image is loaded
- Check parameter values are within valid ranges

## Examples

### Example 1: Removing Sensor Noise
1. Load your noisy image
2. Click "Noise Removal" preset
3. Fine-tune cutoff if needed (lower = more smoothing)
4. Save as custom preset if you like the result

### Example 2: Enhancing Document Scans
1. Load scanned document
2. Click "Edge Enhancement" preset
3. Adjust cutoff for optimal text clarity
4. Save configuration for batch processing

### Example 3: Analyzing Periodic Patterns
1. Load image with repeating patterns
2. Click "Pattern Isolation" preset
3. Adjust cutoff to isolate specific frequencies
4. Use for texture analysis or defect detection

## Future Enhancements
Potential improvements for future versions:
- Preset categories/groups
- Preset preview thumbnails
- Batch processing with presets
- Preset recommendation based on image analysis
- Cloud sync for presets
- Preset sharing community

## Version History
- **v1.0** - Initial implementation with basic preset functionality
  - Built-in presets for common scenarios
  - Custom preset management
  - Import/Export capabilities
  - Real-time parameter updates
