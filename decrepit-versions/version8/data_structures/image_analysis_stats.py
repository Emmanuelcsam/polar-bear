from dataclasses import dataclass
from typing import Optional

@dataclass
class ImageAnalysisStats:
    """Statistics for a single image analysis."""
    total_defects: int = 0 # Total number of defects found.
    core_defects: int = 0 # Number of defects in the core.
    cladding_defects: int = 0 # Number of defects in the cladding.
    ferrule_defects: int = 0 # Number of defects in the ferrule_contact zone.
    adhesive_defects: int = 0 # Number of defects in the adhesive zone.
    processing_time_s: float = 0.0 # Time taken to process the image in seconds.
    status: str = "Pending" # Pass/Fail/Review status.
    microns_per_pixel: Optional[float] = None # Calculated conversion ratio for this image (µm/px).

if __name__ == '__main__':
    # Example of creating an instance of ImageAnalysisStats
    stats = ImageAnalysisStats(
        total_defects=5,
        core_defects=1,
        cladding_defects=4,
        processing_time_s=1.23,
        status="Review"
    )
    print(f"Created ImageAnalysisStats instance: {stats}")
