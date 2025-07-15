#!/usr/bin/env python3
"""
CSV Report Generator Module
===========================
Standalone module for generating detailed CSV reports of defect analysis results.

Usage:
    python csv_report_generator.py --defects defects.json --output report.csv
    python csv_report_generator.py --analysis results.json --output detailed_report.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CSVReportGenerator:
    """
    Standalone CSV report generator for fiber optic inspection results.
    """
    
    def __init__(self):
        """Initialize the CSV report generator."""
        pass
    
    def generate_defect_csv_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Generate a detailed CSV report of defects.
        
        Args:
            analysis_results: Dictionary containing analysis results with defects
            output_path: Path where the CSV file will be saved
            
        Returns:
            True if report was generated successfully, False otherwise
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            
            if not defects_list:
                logger.info(f"No defects to report for {output_path.name}. Creating empty CSV with headers.")
                return self._create_empty_csv(output_path)
            
            # Create DataFrame from defects
            df = pd.DataFrame(defects_list)
            
            # Define desired column order
            report_columns = [
                "defect_id", "zone", "classification", "confidence_score",
                "centroid_x_px", "centroid_y_px",
                "length_um", "width_um", "effective_diameter_um", "area_um2",
                "length_px", "width_px", "area_px",
                "aspect_ratio_oriented",
                "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                "rotated_rect_center_px", "rotated_rect_angle_deg"
            ]
            
            # Select only columns that exist in the DataFrame
            final_columns = [col for col in report_columns if col in df.columns]
            df_report = df[final_columns]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df_report.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Defect CSV report saved successfully to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save defect CSV report to '{output_path}': {e}")
            return False
    
    def _create_empty_csv(self, output_path: Path) -> bool:
        """
        Create an empty CSV file with standard headers.
        
        Args:
            output_path: Path where the empty CSV will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cols = [
                "defect_id", "zone", "classification", "confidence_score",
                "centroid_x_px", "centroid_y_px", "area_px", "length_px", "width_px",
                "aspect_ratio_oriented", "bbox_x_px", "bbox_y_px", "bbox_w_px", "bbox_h_px",
                "area_um2", "length_um", "width_um", "effective_diameter_um",
                "rotated_rect_center_px", "rotated_rect_angle_deg"
            ]
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame([], columns=cols)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Empty defect report CSV (with headers) saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save empty defect report to '{output_path}': {e}")
            return False
    
    def generate_summary_csv_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Path,
        include_metadata: bool = True
    ) -> bool:
        """
        Generate a summary CSV report with overall statistics.
        
        Args:
            analysis_results: Dictionary containing analysis results
            output_path: Path where the summary CSV will be saved
            include_metadata: Whether to include metadata in the report
            
        Returns:
            True if successful, False otherwise
        """
        try:
            defects_list = analysis_results.get("characterized_defects", [])
            
            # Calculate summary statistics
            summary_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_defects": len(defects_list),
                "overall_status": analysis_results.get("overall_status", "UNKNOWN"),
                "scratches": len([d for d in defects_list if d.get("classification") == "Scratch"]),
                "pits_digs": len([d for d in defects_list if d.get("classification") == "Pit/Dig"]),
                "other_defects": len([d for d in defects_list if d.get("classification") not in ["Scratch", "Pit/Dig"]]),
            }
            
            # Add failure reasons if available
            failure_reasons = analysis_results.get("failure_reasons", [])
            summary_data["failure_reasons"] = "; ".join(failure_reasons) if failure_reasons else "None"
            
            # Add zone-based statistics
            zone_stats = self._calculate_zone_statistics(defects_list)
            summary_data.update(zone_stats)
            
            # Add size statistics
            size_stats = self._calculate_size_statistics(defects_list)
            summary_data.update(size_stats)
            
            # Create DataFrame and save
            df_summary = pd.DataFrame([summary_data])
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_summary.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Summary CSV report saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save summary CSV report to '{output_path}': {e}")
            return False
    
    def _calculate_zone_statistics(self, defects_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics by zone."""
        zone_counts = {}
        zones = ["Core", "Cladding", "Adhesive", "Contact", "Unknown"]
        
        for zone in zones:
            zone_defects = [d for d in defects_list if d.get("zone") == zone]
            zone_counts[f"{zone.lower()}_defects"] = len(zone_defects)
        
        return zone_counts
    
    def _calculate_size_statistics(self, defects_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate size-based statistics."""
        if not defects_list:
            return {
                "avg_defect_area_um2": 0,
                "max_defect_area_um2": 0,
                "min_defect_area_um2": 0,
                "avg_confidence": 0,
                "max_confidence": 0,
                "min_confidence": 0
            }
        
        areas = [d.get("area_um2", 0) for d in defects_list if d.get("area_um2") is not None]
        confidences = [d.get("confidence_score", 0) for d in defects_list if d.get("confidence_score") is not None]
        
        return {
            "avg_defect_area_um2": sum(areas) / len(areas) if areas else 0,
            "max_defect_area_um2": max(areas) if areas else 0,
            "min_defect_area_um2": min(areas) if areas else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0
        }
    
    def generate_comparative_report(
        self,
        multiple_results: List[Dict[str, Any]],
        output_path: Path,
        image_names: Optional[List[str]] = None
    ) -> bool:
        """
        Generate a comparative CSV report for multiple images.
        
        Args:
            multiple_results: List of analysis results from multiple images
            output_path: Path where the comparative CSV will be saved
            image_names: Optional list of image names for identification
            
        Returns:
            True if successful, False otherwise
        """
        try:
            comparative_data = []
            
            for i, results in enumerate(multiple_results):
                image_name = image_names[i] if image_names and i < len(image_names) else f"Image_{i+1}"
                defects_list = results.get("characterized_defects", [])
                
                # Calculate statistics for this image
                summary_data = {
                    "image_name": image_name,
                    "total_defects": len(defects_list),
                    "overall_status": results.get("overall_status", "UNKNOWN"),
                    "scratches": len([d for d in defects_list if d.get("classification") == "Scratch"]),
                    "pits_digs": len([d for d in defects_list if d.get("classification") == "Pit/Dig"]),
                }
                
                # Add zone statistics
                zone_stats = self._calculate_zone_statistics(defects_list)
                summary_data.update(zone_stats)
                
                # Add size statistics
                size_stats = self._calculate_size_statistics(defects_list)
                summary_data.update(size_stats)
                
                comparative_data.append(summary_data)
            
            # Create DataFrame and save
            df_comparative = pd.DataFrame(comparative_data)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df_comparative.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Comparative CSV report saved to '{output_path}'.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save comparative CSV report to '{output_path}': {e}")
            return False
    
    def load_analysis_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from JSON file.
        
        Args:
            file_path: Path to JSON file containing analysis results
            
        Returns:
            Analysis results dictionary or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded analysis results from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load analysis results from {file_path}: {e}")
            return None
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample analysis data for testing."""
        return {
            "image_filename": "test_image.png",
            "overall_status": "FAIL",
            "total_defect_count": 3,
            "failure_reasons": ["Core defect exceeds size limit", "Critical scratch detected"],
            "characterized_defects": [
                {
                    "defect_id": "D1_S1",
                    "zone": "Core",
                    "classification": "Scratch",
                    "confidence_score": 0.95,
                    "centroid_x_px": 190,
                    "centroid_y_px": 140,
                    "area_px": 50,
                    "length_px": 22,
                    "width_px": 2.5,
                    "aspect_ratio_oriented": 8.8,
                    "bbox_x_px": 180,
                    "bbox_y_px": 130,
                    "bbox_w_px": 15,
                    "bbox_h_px": 25,
                    "area_um2": 12.5,
                    "length_um": 11.0,
                    "width_um": 1.25
                },
                {
                    "defect_id": "D1_L1",
                    "zone": "Cladding",
                    "classification": "Scratch",
                    "confidence_score": 0.88,
                    "centroid_x_px": 230,
                    "centroid_y_px": 170,
                    "area_px": 0,
                    "length_px": 10,
                    "width_px": 0,
                    "area_um2": 0,
                    "length_um": 5.0,
                    "width_um": 0
                },
                {
                    "defect_id": "D2_P2",
                    "zone": "Contact",
                    "classification": "Pit/Dig",
                    "confidence_score": 0.70,
                    "centroid_x_px": 150,
                    "centroid_y_px": 100,
                    "area_px": 20,
                    "length_px": 5,
                    "width_px": 4,
                    "bbox_x_px": 148,
                    "bbox_y_px": 98,
                    "bbox_w_px": 5,
                    "bbox_h_px": 5,
                    "area_um2": 5.0,
                    "effective_diameter_um": 2.51
                }
            ]
        }


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Generate CSV Reports for Fiber Optic Inspection")
    parser.add_argument("--analysis", help="Path to JSON file with analysis results")
    parser.add_argument("--output", required=True, help="Path for output CSV file")
    parser.add_argument("--report-type", choices=["detailed", "summary", "comparative"], 
                       default="detailed", help="Type of report to generate")
    parser.add_argument("--demo", action="store_true", help="Generate demo report with sample data")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CSVReportGenerator()
    
    # Load or create data
    if args.demo:
        logger.info("Using demo data")
        analysis_results = generator.create_sample_data()
    else:
        if not args.analysis:
            logger.error("Either --analysis or --demo must be specified")
            sys.exit(1)
        
        analysis_results = generator.load_analysis_results(args.analysis)
        if analysis_results is None:
            sys.exit(1)
    
    # Generate report based on type
    output_path = Path(args.output)
    
    if args.report_type == "detailed":
        success = generator.generate_defect_csv_report(analysis_results, output_path)
    elif args.report_type == "summary":
        success = generator.generate_summary_csv_report(analysis_results, output_path)
    elif args.report_type == "comparative":
        # For demo, use the same data multiple times
        if args.demo:
            multiple_results = [analysis_results, analysis_results]
            image_names = ["Sample_Image_1", "Sample_Image_2"]
        else:
            # For real use, you'd load multiple result files
            multiple_results = [analysis_results]
            image_names = ["Single_Image"]
        
        success = generator.generate_comparative_report(multiple_results, output_path, image_names)
    
    if success:
        logger.info(f"Successfully generated {args.report_type} CSV report: {output_path}")
    else:
        logger.error(f"Failed to generate {args.report_type} CSV report")
        sys.exit(1)


if __name__ == "__main__":
    main()
