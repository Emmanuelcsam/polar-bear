from typing import Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd

from utils import _log_message
from data_structures import ImageResult

def _generate_annotated_image(self, original_bgr_image: np.ndarray, image_res: ImageResult) -> Optional[np.ndarray]:
    """Generates an image with detected zones and defects annotated."""
    _log_message(f"Generating annotated image for {image_res.filename}...")
    annotated_image = original_bgr_image.copy()

    for zone_name, zone_info in image_res.detected_zones.items():
        zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None)
        if zone_def and zone_info.mask is not None:
            contours, _ = cv2.findContours(zone_info.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, zone_def.color_bgr, self.config.LINE_THICKNESS + 1)
            if contours:
                c = contours[0]
                text_pos = tuple(c[c[:, :, 1].argmin()][0])
                cv2.putText(annotated_image, zone_name, (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE * 1.2, zone_def.color_bgr, self.config.LINE_THICKNESS)

    for defect in image_res.defects:
        defect_color = self.config.DEFECT_COLORS.get(defect.defect_type, (255, 255, 255))
        x, y, w, h = defect.bounding_box_px
        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), defect_color, self.config.LINE_THICKNESS)
        if defect.contour is not None:
            cv2.drawContours(annotated_image, [defect.contour], -1, defect_color, self.config.LINE_THICKNESS)
        size_info = f"{defect.major_dimension.value_um:.1f}um" if defect.major_dimension.value_um is not None else (f"{defect.major_dimension.value_px:.0f}px" if defect.major_dimension.value_px is not None else "")
        label = f"ID{defect.defect_id}:{defect.defect_type[:3]}:{size_info} (C:{defect.confidence_score:.2f})"
        cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE, defect_color, self.config.LINE_THICKNESS)

    cv2.putText(annotated_image, f"File: {image_res.filename}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
    cv2.putText(annotated_image, f"Status: {image_res.stats.status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
    cv2.putText(annotated_image, f"Total Defects: {image_res.stats.total_defects}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SCALE*1.1, (230,230,230), self.config.LINE_THICKNESS)
    _log_message(f"Annotated image generated for {image_res.filename}.")
    return annotated_image

def _generate_defect_histogram(self, image_res: ImageResult) -> Optional[plt.Figure]:
    """Generates a polar histogram of defect distribution."""
    _log_message(f"Generating defect histogram for {image_res.filename}...")
    if not image_res.defects or not image_res.detected_zones.get("cladding"):
        _log_message("No defects or cladding center not found, skipping histogram.", level="WARNING")
        return None

    cladding_zone_info = image_res.detected_zones.get("cladding")
    if not cladding_zone_info or cladding_zone_info.center_px is None:
        _log_message("Cladding center is None, cannot generate polar histogram.", level="WARNING")
        return None
    fiber_center_x, fiber_center_y = cladding_zone_info.center_px

    angles, radii, defect_plot_colors = [], [], []
    for defect in image_res.defects:
        dx = defect.centroid_px[0] - fiber_center_x; dy = defect.centroid_px[1] - fiber_center_y
        angles.append(np.arctan2(dy, dx)); radii.append(np.sqrt(dx**2 + dy**2))
        bgr_color = self.config.DEFECT_COLORS.get(defect.defect_type, (0,0,0))
        rgb_color_normalized = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)
        defect_plot_colors.append(rgb_color_normalized)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.scatter(angles, radii, c=defect_plot_colors, s=50, alpha=0.75, edgecolors='k')

    for zone_name, zone_info in image_res.detected_zones.items():
        zone_def = next((zd for zd in self.active_zone_definitions if zd.name == zone_name), None)
        if zone_def and zone_info.radius_px > 0:
            plot_color_rgb = (zone_def.color_bgr[2]/255.0, zone_def.color_bgr[1]/255.0, zone_def.color_bgr[0]/255.0)
            ax.plot(np.linspace(0, 2 * np.pi, 100), [zone_info.radius_px] * 100, color=plot_color_rgb, linestyle='--', label=zone_name)
    ax.set_title(f"Defect Distribution: {image_res.filename}", va='bottom')
    max_r_display = cladding_zone_info.radius_px * 2.5 if cladding_zone_info else (max(radii) * 1.1 if radii else 100)
    ax.set_rmax(max_r_display)
    ax.legend()
    plt.tight_layout()
    _log_message(f"Defect histogram generated for {image_res.filename}.")
    return fig

def _save_individual_image_report_csv(self, image_res: ImageResult, image_output_dir: Path):
    """Saves a detailed CSV report for a single image's defects."""
    _log_message(f"Saving individual CSV report for {image_res.filename}...")
    report_path = image_output_dir / f"{Path(image_res.filename).stem}_defect_report.csv"
    image_res.report_csv_path = report_path
    fieldnames = [
        "Defect_ID", "Zone", "Type", "Centroid_X_px", "Centroid_Y_px",
        "Area_px2", "Area_um2", "Perimeter_px", "Perimeter_um",
        "Major_Dim_px", "Major_Dim_um", "Minor_Dim_px", "Minor_Dim_um",
        "Confidence", "Detection_Methods"
    ]
    with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for defect in image_res.defects:
            writer.writerow({
                "Defect_ID": defect.defect_id, "Zone": defect.zone_name, "Type": defect.defect_type,
                "Centroid_X_px": defect.centroid_px[0], "Centroid_Y_px": defect.centroid_px[1],
                "Area_px2": f"{defect.area.value_px:.2f}" if defect.area.value_px is not None else "N/A",
                "Area_um2": f"{defect.area.value_um:.2f}" if defect.area.value_um is not None else "N/A",
                "Perimeter_px": f"{defect.perimeter.value_px:.2f}" if defect.perimeter.value_px is not None else "N/A",
                "Perimeter_um": f"{defect.perimeter.value_um:.2f}" if defect.perimeter.value_um is not None else "N/A",
                "Major_Dim_px": f"{defect.major_dimension.value_px:.2f}" if defect.major_dimension.value_px is not None else "N/A",
                "Major_Dim_um": f"{defect.major_dimension.value_um:.2f}" if defect.major_dimension.value_um is not None else "N/A",
                "Minor_Dim_px": f"{defect.minor_dimension.value_px:.2f}" if defect.minor_dimension.value_px is not None else "N/A",
                "Minor_Dim_um": f"{defect.minor_dimension.value_um:.2f}" if defect.minor_dimension.value_um is not None else "N/A",
                "Confidence": f"{defect.confidence_score:.3f}",
                "Detection_Methods": "; ".join(defect.detection_methods)
            })
    _log_message(f"Individual CSV report saved to {report_path}")

def _save_image_artifacts(self, original_bgr_image: np.ndarray, image_res: ImageResult):
    """Saves all generated artifacts for a single image."""
    _log_message(f"Saving artifacts for {image_res.filename}...")
    image_specific_output_dir = self.output_dir_path / Path(image_res.filename).stem
    image_specific_output_dir.mkdir(parents=True, exist_ok=True)

    if self.config.SAVE_ANNOTATED_IMAGE:
        annotated_img = self._generate_annotated_image(original_bgr_image, image_res)
        if annotated_img is not None:
            annotated_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated_img)
            image_res.annotated_image_path = annotated_path
            _log_message(f"Annotated image saved to {annotated_path}")

    if self.config.DETAILED_REPORT_PER_IMAGE and image_res.defects:
        self._save_individual_image_report_csv(image_res, image_specific_output_dir)

    if self.config.SAVE_HISTOGRAM:
        histogram_fig = self._generate_defect_histogram(image_res)
        if histogram_fig:
            histogram_path = image_specific_output_dir / f"{Path(image_res.filename).stem}_histogram.png"
            histogram_fig.savefig(str(histogram_path), dpi=150)
            plt.close(histogram_fig)
            image_res.histogram_path = histogram_path
            _log_message(f"Defect histogram saved to {histogram_path}")

    if self.config.SAVE_DEFECT_MAPS and image_res.intermediate_defect_maps:
        maps_dir = image_specific_output_dir / "defect_maps"; maps_dir.mkdir(exist_ok=True)
        for map_name, defect_map_img in image_res.intermediate_defect_maps.items():
            if defect_map_img is not None:
                cv2.imwrite(str(maps_dir / f"{map_name}.png"), defect_map_img)
        _log_message(f"Intermediate defect maps saved to {maps_dir}")
    _log_message(f"Artifacts saved for {image_res.filename}.")

def _save_batch_summary_report_csv(self):
    """Saves a summary CSV report for the entire batch."""
    _log_message("Saving batch summary report...")
    if not self.batch_results_summary_list:
        _log_message("No batch results to save.", level="WARNING")
        return
    summary_path = self.output_dir_path / self.config.BATCH_SUMMARY_FILENAME
    try:
        summary_df = pd.DataFrame(self.batch_results_summary_list)
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        _log_message(f"Batch summary report saved to {summary_path}")
    except Exception as e:
        _log_message(f"Error saving batch summary report: {e}", level="ERROR")
