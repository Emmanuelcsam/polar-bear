from dataclasses import dataclass
from typing import Optional

@dataclass
class DefectMeasurement:
    """Data structure for defect measurements."""
    value_px: Optional[float] = None # Measurement in pixels.
    value_um: Optional[float] = None # Measurement in micrometers.

if __name__ == '__main__':
    # Example of creating an instance of DefectMeasurement
    measurement = DefectMeasurement(value_px=10.5, value_um=2.1)
    print(f"Created DefectMeasurement instance: {measurement}")
