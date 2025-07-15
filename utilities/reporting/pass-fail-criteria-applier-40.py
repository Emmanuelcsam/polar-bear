from typing import Dict, Any, List

from .defect_info import DefectInfo

def apply_pass_fail_criteria(defects: List[DefectInfo], 
                             pixels_per_micron: float or None) -> Dict[str, Any]:
    """Apply IEC 61300-3-35 based pass/fail criteria"""
    # Zone-specific criteria (in microns)
    zone_criteria = {
        'core': {'max_scratch_width_um': 3, 'max_dig_diameter_um': 2, 'max_count': 5},
        'cladding': {'max_scratch_width_um': 5, 'max_dig_diameter_um': 5, 'max_count': None},
        'ferrule': {'max_defect_um': 25, 'max_count': None},
        'adhesive': {'max_defect_um': 50, 'max_count': None}
    }
    
    status = 'PASS'
    failures = []
    
    if not pixels_per_micron:
        return {
            'status': 'INCONCLUSIVE',
            'failures': ['Cannot determine pass/fail status without pixel to micron conversion.'],
            'total_defects': len(defects)
        }

    defects_by_zone = {name: [] for name in zone_criteria.keys()}
    for defect in defects:
        if defect.zone_name in defects_by_zone:
            defects_by_zone[defect.zone_name].append(defect)
    
    for zone_name, criteria in zone_criteria.items():
        zone_defects = defects_by_zone.get(zone_name, [])
        
        if criteria.get('max_count') and len(zone_defects) > criteria['max_count']:
            status = 'FAIL'
            failures.append(f"{zone_name}: Too many defects ({len(zone_defects)} > {criteria['max_count']})")
        
        for defect in zone_defects:
            if defect.defect_type == 'scratch' and 'max_scratch_width_um' in criteria:
                if defect.minor_dimension_um > criteria['max_scratch_width_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} scratch width > {criteria['max_scratch_width_um']}µm")
            elif defect.defect_type == 'dig' and 'max_dig_diameter_um' in criteria:
                if defect.major_dimension_um > criteria['max_dig_diameter_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} dig diameter > {criteria['max_dig_diameter_um']}µm")
            elif 'max_defect_um' in criteria:
                 if defect.major_dimension_um > criteria['max_defect_um']:
                    status = 'FAIL'
                    failures.append(f"{zone_name} defect size > {criteria['max_defect_um']}µm")

    return {
        'status': status,
        'failures': list(set(failures)), # Unique failures
        'defects_by_zone': {k: len(v) for k, v in defects_by_zone.items()},
        'total_defects': len(defects)
    }

if __name__ == '__main__':
    defects = [
        DefectInfo(1, 'core', 'dig', (1,1), 10, major_dimension_um=2.5),
        DefectInfo(2, 'cladding', 'scratch', (2,2), 20, minor_dimension_um=6.0),
    ]
    pixels_per_micron = 2.0

    print("Applying pass/fail criteria...")
    result = apply_pass_fail_criteria(defects, pixels_per_micron)
    print(result)

    defects_fail = [
        DefectInfo(1, 'core', 'dig', (1,1), 10, major_dimension_um=3.5),
    ]
    print("\nApplying pass/fail criteria to a failing case...")
    result_fail = apply_pass_fail_criteria(defects_fail, pixels_per_micron)
    print(result_fail)
