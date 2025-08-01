from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class DefectInfo:
    """Information about a detected defect"""
    defect_id: int
    zone_name: str
    defect_type: str  # 'scratch', 'dig', 'contamination', etc.
    centroid_px: Tuple[int, int]
    area_px: float
    area_um: Optional[float] = None
    major_dimension_px: float = 0
    major_dimension_um: Optional[float] = None
    minor_dimension_px: float = 0
    minor_dimension_um: Optional[float] = None
    confidence_score: float = 0.0
    detection_methods: List[str] = field(default_factory=list)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
    eccentricity: float = 0.0
    solidity: float = 0.0
    orientation: float = 0.0

if __name__ == '__main__':
    defect = DefectInfo(
        defect_id=1,
        zone_name='core',
        defect_type='scratch',
        centroid_px=(100, 120),
        area_px=50.5,
    )
    print("Created a sample DefectInfo instance:")
    print(defect)
