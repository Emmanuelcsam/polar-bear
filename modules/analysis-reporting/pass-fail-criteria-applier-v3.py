

import pandas as pd
from typing import List, Tuple

def apply_pass_fail_criteria(defects_df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Apply IEC-61300 based pass/fail criteria (from test3.py)
    """
    zones_criteria = {
        "core": {"max_defect_um": 3},
        "cladding": {"max_defect_um": 10},
        "ferrule": {"max_defect_um": 20}
    }
    
    status = "PASS"
    failure_reasons = []
    
    if defects_df.empty:
        return status, failure_reasons

    for zone_name, zone_params in zones_criteria.items():
        zone_defects = defects_df[defects_df["zone"] == zone_name]
        
        if zone_defects.empty:
            continue
        
        # Check dig sizes
        digs = zone_defects[zone_defects["type"] == "dig"]
        if not digs.empty:
            max_dig_diameter = digs["diameter_um"].max()
            if max_dig_diameter > zone_params["max_defect_um"]:
                status = "FAIL"
                failure_reasons.append(
                    f"{zone_name}: Dig diameter {max_dig_diameter:.1f}μm exceeds limit {zone_params['max_defect_um']}μm"
                )
        
        # Check scratch lengths
        scratches = zone_defects[zone_defects["type"] == "scratch"]
        if not scratches.empty:
            max_scratch_length = scratches["length_um"].max()
            # Scratches are often evaluated by width, but here we use length as a proxy
            if max_scratch_length > zone_params["max_defect_um"] * 5: # Allow longer scratches
                status = "FAIL"
                failure_reasons.append(
                    f"{zone_name}: Scratch length {max_scratch_length:.1f}μm exceeds limit"
                )
        
        # Check total defect count (example criteria)
        if zone_name == "core" and len(zone_defects) > 5:
            status = "FAIL"
            failure_reasons.append(f"Core: Too many defects ({len(zone_defects)} > 5)")
            
    return status, failure_reasons

if __name__ == '__main__':
    # Create dummy defect data for demonstration
    
    # --- Test Case 1: PASS ---
    print("--- Test Case 1: Should PASS ---")
    pass_data = {
        'type': ['dig', 'scratch'],
        'zone': ['cladding', 'cladding'],
        'diameter_um': [8.0, None],
        'length_um': [None, 45.0]
    }
    pass_df = pd.DataFrame(pass_data)
    pass_status, pass_reasons = apply_pass_fail_criteria(pass_df)
    print(f"Status: {pass_status}")
    print(f"Reasons: {pass_reasons}")
    assert pass_status == "PASS"
    
    # --- Test Case 2: FAIL due to large dig in core ---
    print("\n--- Test Case 2: Should FAIL (Core Dig) ---")
    fail_data_dig = {
        'type': ['dig'],
        'zone': ['core'],
        'diameter_um': [5.0],
        'length_um': [None]
    }
    fail_df_dig = pd.DataFrame(fail_data_dig)
    fail_status_dig, fail_reasons_dig = apply_pass_fail_criteria(fail_df_dig)
    print(f"Status: {fail_status_dig}")
    print(f"Reasons: {fail_reasons_dig}")
    assert fail_status_dig == "FAIL"

    # --- Test Case 3: FAIL due to too many defects in core ---
    print("\n--- Test Case 3: Should FAIL (Too many core defects) ---")
    fail_data_count = {
        'type': ['dig'] * 6,
        'zone': ['core'] * 6,
        'diameter_um': [1.0] * 6,
        'length_um': [None] * 6
    }
    fail_df_count = pd.DataFrame(fail_data_count)
    fail_status_count, fail_reasons_count = apply_pass_fail_criteria(fail_df_count)
    print(f"Status: {fail_status_count}")
    print(f"Reasons: {fail_reasons_count}")
    assert fail_status_count == "FAIL"
    
    print("\nScript finished successfully.")

