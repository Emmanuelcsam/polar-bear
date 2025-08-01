
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
    # Example of how to use the ImageAnalysisStats dataclass

    # 1. Create stats for a successfully processed image with defects
    stats1 = ImageAnalysisStats(
        total_defects=5,
        core_defects=1,
        cladding_defects=4,
        processing_time_s=1.25,
        status="Review",
        microns_per_pixel=0.5
    )
    print(f"Image Stats 1: {stats1}")
    print(f"Status: {stats1.status}")
    print(f"Microns per Pixel: {stats1.microns_per_pixel}")

    # 2. Create stats for an image that failed processing
    stats_error = ImageAnalysisStats(
        status="Error",
        processing_time_s=0.1
    )
    print(f"Image Stats (Error): {stats_error}")

    # 3. Create default stats (e.g., at the beginning of processing)
    default_stats = ImageAnalysisStats()
    print(f"Default Stats: {default_stats}")
    print(f"Initial total defects: {default_stats.total_defects}")
