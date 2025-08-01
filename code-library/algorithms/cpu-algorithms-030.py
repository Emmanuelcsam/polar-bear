
from dataclasses import dataclass
from typing import Optional

@dataclass
class DefectMeasurement:
    """Data structure for defect measurements."""
    value_px: Optional[float] = None # Measurement in pixels.
    value_um: Optional[float] = None # Measurement in micrometers.

if __name__ == '__main__':
    # Example of how to use the DefectMeasurement dataclass

    # 1. Measurement with both pixel and micron values
    area_measurement = DefectMeasurement(value_px=150.5, value_um=75.25)
    print(f"Area Measurement: {area_measurement}")
    print(f"Area in pixels: {area_measurement.value_px} px^2")
    print(f"Area in microns: {area_measurement.value_um} µm^2")

    # 2. Measurement with only pixel values
    length_measurement_px_only = DefectMeasurement(value_px=25.0)
    print(f"Length Measurement (pixels only): {length_measurement_px_only}")
    
    # 3. An empty measurement
    empty_measurement = DefectMeasurement()
    print(f"Empty Measurement: {empty_measurement}")
    print(f"Value in pixels: {'Not available' if empty_measurement.value_px is None else empty_measurement.value_px}")
