#!/usr/bin/env python3
"""
Pass/Fail Rules Engine Module
============================
Comprehensive pass/fail evaluation system for fiber optic inspection
based on IEC 61300-3-35 standards and customizable rules.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

class PassFailRulesEngine:
    """
    Advanced rules engine for fiber optic inspection pass/fail evaluation.
    """
    
    def __init__(self, rules_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the rules engine with configuration.
        
        Args:
            rules_config: Dictionary containing rules configuration
        """
        self.rules_config = rules_config or self._get_default_rules()
        
    def _get_default_rules(self) -> Dict[str, Any]:
        """Get default IEC 61300-3-35 compliant rules."""
        return {
            "single_mode_pc": {
                "Core": {
                    "max_scratches": 0,
                    "max_defects": 0,
                    "max_defect_size_um": 3,
                    "max_total_defect_area_um2": 0,
                    "critical_zone": True
                },
                "Cladding": {
                    "max_scratches": 5,
                    "max_scratches_gt_5um": 0,
                    "max_defects": 5,
                    "max_defect_size_um": 10,
                    "max_total_defect_area_um2": 100,
                    "critical_zone": False
                }
            },
            "multi_mode_pc": {
                "Core": {
                    "max_scratches": 1,
                    "max_scratch_length_um": 10,
                    "max_defects": 3,
                    "max_defect_size_um": 5,
                    "max_total_defect_area_um2": 25,
                    "critical_zone": True
                },
                "Cladding": {
                    "max_scratches": "unlimited",
                    "max_defects": "unlimited",
                    "max_defect_size_um": 20,
                    "max_total_defect_area_um2": 200,
                    "critical_zone": False
                }
            }
        }
    
    def evaluate_zone_rules(self, 
                           defects: List[Dict[str, Any]], 
                           zone_name: str,
                           fiber_type: str,
                           zone_rules: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Evaluate pass/fail rules for a specific zone.
        
        Args:
            defects: List of defects in the zone
            zone_name: Name of the zone
            fiber_type: Type of fiber
            zone_rules: Rules configuration for the zone
            
        Returns:
            Tuple of (status, failure_reasons)
        """
        status = "PASS"
        failure_reasons = []
        
        # Separate defects by classification
        scratches = [d for d in defects if d.get("classification") == "Scratch"]
        pits_digs = [d for d in defects if d.get("classification") in ["Pit", "Dig"]]
        all_non_scratch = [d for d in defects if d.get("classification") != "Scratch"]
        
        # Check critical zone rule (any defect fails)
        if zone_rules.get("critical_zone", False) and len(defects) > 0:
            status = "FAIL"
            failure_reasons.append(f"Zone '{zone_name}': Critical zone contains {len(defects)} defect(s)")
            return status, failure_reasons
        
        # Check scratch count
        max_scratches = zone_rules.get("max_scratches")
        if isinstance(max_scratches, int) and len(scratches) > max_scratches:
            status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many scratches ({len(scratches)} > {max_scratches})"
            )
        
        # Check pit/dig count
        max_defects = zone_rules.get("max_defects")
        if isinstance(max_defects, int) and len(pits_digs) > max_defects:
            status = "FAIL"
            failure_reasons.append(
                f"Zone '{zone_name}': Too many pits/digs ({len(pits_digs)} > {max_defects})"
            )
        
        # Check individual defect sizes
        max_defect_size_um = zone_rules.get("max_defect_size_um")
        max_scratch_length_um = zone_rules.get("max_scratch_length_um")
        
        for defect in defects:
            defect_type = defect.get("classification", "Unknown")
            primary_dimension_um = defect.get("length_um")
            
            if primary_dimension_um is None:
                continue
            
            # Determine size limit based on defect type
            if defect_type == "Scratch":
                size_limit = max_scratch_length_um if max_scratch_length_um is not None else max_defect_size_um
            else:
                size_limit = max_defect_size_um
            
            if size_limit is not None and primary_dimension_um > size_limit:
                status = "FAIL"
                reason = (f"Zone '{zone_name}': {defect_type} '{defect.get('defect_id', 'Unknown')}' "
                         f"size ({primary_dimension_um:.2f}µm) exceeds limit ({size_limit}µm)")
                failure_reasons.append(reason)
        
        # Check total defect area
        max_total_area = zone_rules.get("max_total_defect_area_um2")
        if max_total_area is not None:
            total_area = sum(d.get("area_um2", 0) for d in defects)
            if total_area > max_total_area:
                status = "FAIL"
                failure_reasons.append(
                    f"Zone '{zone_name}': Total defect area ({total_area:.2f}µm²) exceeds limit ({max_total_area}µm²)"
                )
        
        # Check scratches greater than threshold
        scratches_gt_threshold = zone_rules.get("max_scratches_gt_5um")
        if scratches_gt_threshold is not None:
            large_scratches = [s for s in scratches if s.get("length_um", 0) > 5.0]
            if len(large_scratches) > scratches_gt_threshold:
                status = "FAIL"
                failure_reasons.append(
                    f"Zone '{zone_name}': Too many scratches > 5µm ({len(large_scratches)} > {scratches_gt_threshold})"
                )
        
        return status, failure_reasons
    
    def apply_pass_fail_rules(self, 
                             characterized_defects: List[Dict[str, Any]],
                             fiber_type: str) -> Tuple[str, List[str]]:
        """
        Apply comprehensive pass/fail rules to characterized defects.
        
        Args:
            characterized_defects: List of characterized defect dictionaries
            fiber_type: Fiber type key (e.g., "single_mode_pc")
            
        Returns:
            Tuple of (overall_status, failure_reasons)
        """
        overall_status = "PASS"
        all_failure_reasons = []
        
        # Get rules for this fiber type
        fiber_rules = self.rules_config.get(fiber_type)
        if not fiber_rules:
            error_msg = f"No pass/fail rules defined for fiber type '{fiber_type}'"
            logging.error(error_msg)
            return "ERROR_CONFIG", [error_msg]
        
        # Group defects by zone
        defects_by_zone = {}
        for defect in characterized_defects:
            zone_name = defect.get("zone", "Unknown")
            if zone_name not in defects_by_zone:
                defects_by_zone[zone_name] = []
            defects_by_zone[zone_name].append(defect)
        
        # Evaluate each zone
        for zone_name, zone_rules in fiber_rules.items():
            zone_defects = defects_by_zone.get(zone_name, [])
            
            zone_status, zone_reasons = self.evaluate_zone_rules(
                zone_defects, zone_name, fiber_type, zone_rules
            )
            
            if zone_status == "FAIL":
                overall_status = "FAIL"
                all_failure_reasons.extend(zone_reasons)
        
        # Log results
        if overall_status == "PASS":
            logging.info(f"Pass/Fail evaluation for '{fiber_type}': PASS")
        else:
            logging.warning(f"Pass/Fail evaluation for '{fiber_type}': FAIL - {len(all_failure_reasons)} reason(s)")
        
        return overall_status, all_failure_reasons
    
    def get_zone_statistics(self, 
                           defects: List[Dict[str, Any]],
                           zone_name: str) -> Dict[str, Any]:
        """
        Calculate detailed statistics for a zone.
        
        Args:
            defects: List of defects in the zone
            zone_name: Name of the zone
            
        Returns:
            Dictionary with zone statistics
        """
        zone_defects = [d for d in defects if d.get("zone") == zone_name]
        
        if not zone_defects:
            return {
                "zone_name": zone_name,
                "total_defects": 0,
                "scratches": 0,
                "pits_digs": 0,
                "total_area_um2": 0.0,
                "max_defect_size_um": 0.0,
                "avg_defect_size_um": 0.0
            }
        
        # Separate by type
        scratches = [d for d in zone_defects if d.get("classification") == "Scratch"]
        pits_digs = [d for d in zone_defects if d.get("classification") in ["Pit", "Dig"]]
        
        # Calculate statistics
        total_area = sum(d.get("area_um2", 0) for d in zone_defects)
        defect_sizes = [d.get("length_um", d.get("length_px", 0)) for d in zone_defects]
        
        stats = {
            "zone_name": zone_name,
            "total_defects": len(zone_defects),
            "scratches": len(scratches),
            "pits_digs": len(pits_digs),
            "total_area_um2": float(total_area),
            "max_defect_size_um": float(max(defect_sizes) if defect_sizes else 0),
            "avg_defect_size_um": float(sum(defect_sizes) / len(defect_sizes) if defect_sizes else 0)
        }
        
        return stats
    
    def generate_detailed_report(self, 
                                characterized_defects: List[Dict[str, Any]],
                                fiber_type: str) -> Dict[str, Any]:
        """
        Generate a comprehensive inspection report.
        
        Args:
            characterized_defects: List of characterized defects
            fiber_type: Fiber type key
            
        Returns:
            Detailed inspection report dictionary
        """
        # Apply pass/fail rules
        overall_status, failure_reasons = self.apply_pass_fail_rules(characterized_defects, fiber_type)
        
        # Calculate zone statistics
        zone_stats = {}
        fiber_rules = self.rules_config.get(fiber_type, {})
        
        for zone_name in fiber_rules.keys():
            zone_stats[zone_name] = self.get_zone_statistics(characterized_defects, zone_name)
        
        # Overall statistics
        total_defects = len(characterized_defects)
        total_scratches = len([d for d in characterized_defects if d.get("classification") == "Scratch"])
        total_pits_digs = len([d for d in characterized_defects if d.get("classification") in ["Pit", "Dig"]])
        
        report = {
            "fiber_type": fiber_type,
            "overall_status": overall_status,
            "failure_reasons": failure_reasons,
            "summary": {
                "total_defects": total_defects,
                "total_scratches": total_scratches,
                "total_pits_digs": total_pits_digs,
                "pass_fail_status": overall_status
            },
            "zone_statistics": zone_stats,
            "detailed_defects": characterized_defects
        }
        
        return report

def create_custom_rules(base_fiber_type: str = "single_mode_pc",
                       custom_modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create custom pass/fail rules based on a base type with modifications.
    
    Args:
        base_fiber_type: Base fiber type to start from
        custom_modifications: Dictionary of modifications to apply
        
    Returns:
        Custom rules configuration
    """
    engine = PassFailRulesEngine()
    base_rules = engine.rules_config.get(base_fiber_type, {})
    
    if not base_rules:
        raise ValueError(f"Base fiber type '{base_fiber_type}' not found")
    
    # Deep copy base rules
    import copy
    custom_rules = copy.deepcopy(base_rules)
    
    # Apply modifications
    if custom_modifications:
        for zone_name, zone_mods in custom_modifications.items():
            if zone_name in custom_rules:
                custom_rules[zone_name].update(zone_mods)
            else:
                custom_rules[zone_name] = zone_mods
    
    return custom_rules

if __name__ == "__main__":
    """Test the pass/fail rules engine"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Create test defects
    test_defects = [
        {
            "defect_id": "D1",
            "zone": "Core",
            "classification": "Scratch",
            "length_um": 2.5,
            "area_um2": 1.2,
            "confidence_score": 0.9
        },
        {
            "defect_id": "D2",
            "zone": "Cladding",
            "classification": "Pit",
            "length_um": 8.0,
            "area_um2": 15.0,
            "confidence_score": 0.8
        },
        {
            "defect_id": "D3",
            "zone": "Cladding",
            "classification": "Scratch",
            "length_um": 12.0,
            "area_um2": 8.0,
            "confidence_score": 0.7
        }
    ]
    
    print("Testing Pass/Fail Rules Engine...")
    
    # Test default rules engine
    engine = PassFailRulesEngine()
    
    # Test single-mode fiber rules
    print("\\nTesting single-mode PC rules:")
    status, reasons = engine.apply_pass_fail_rules(test_defects, "single_mode_pc")
    print(f"Status: {status}")
    if reasons:
        for reason in reasons:
            print(f"  - {reason}")
    
    # Test zone statistics
    print("\\nZone statistics:")
    core_stats = engine.get_zone_statistics(test_defects, "Core")
    cladding_stats = engine.get_zone_statistics(test_defects, "Cladding")
    print(f"Core: {core_stats}")
    print(f"Cladding: {cladding_stats}")
    
    # Test detailed report
    print("\\nDetailed report:")
    report = engine.generate_detailed_report(test_defects, "single_mode_pc")
    print(f"Overall status: {report['overall_status']}")
    print(f"Summary: {report['summary']}")
    
    # Test custom rules
    print("\\nTesting custom rules:")
    custom_mods = {
        "Core": {"max_defects": 1},  # Allow 1 defect in core
        "Cladding": {"max_scratches": 2}  # Reduce max scratches
    }
    
    custom_rules = create_custom_rules("single_mode_pc", custom_mods)
    custom_engine = PassFailRulesEngine({"custom_fiber": custom_rules})
    
    custom_status, custom_reasons = custom_engine.apply_pass_fail_rules(test_defects, "custom_fiber")
    print(f"Custom rules status: {custom_status}")
    if custom_reasons:
        for reason in custom_reasons:
            print(f"  - {reason}")
    
    print("Pass/Fail rules engine tests completed!")
