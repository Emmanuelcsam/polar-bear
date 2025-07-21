import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Union
import cv2

from ..utils.logger import get_logger
from ..config.config import get_config
from ..core.anomaly_detection import AnomalyResult


class ResultsExporter:
    """
    Exports analysis results in minimal file sizes.
    "for every image the program will spit out an anomaly or defect map which will 
    just be a matrix of the pixel intensities of the defects in a heightened contrast 
    on top of the matrix of the input image"
    "each result matrix will be the smallest files type and size possible"
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        # Create results directory
        self.results_dir = Path(self.config.RESULTS_PATH)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized ResultsExporter with output dir: {self.results_dir}")
    
    def export_anomaly_result(self, 
                            image_id: str,
                            anomaly_result: AnomalyResult,
                            input_tensor: torch.Tensor,
                            additional_data: Optional[Dict] = None) -> Path:
        """
        Export anomaly detection results.
        "it will export this into a results folder"
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_filename = f"{image_id}_{timestamp}_result"
        
        # Prepare data for export
        export_data = self._prepare_export_data(
            anomaly_result, 
            input_tensor,
            additional_data
        )
        
        # Export in compressed format
        if self.config.EXPORT_FORMAT == 'npz':
            output_path = self._export_npz(result_filename, export_data)
        elif self.config.EXPORT_FORMAT == 'npy':
            output_path = self._export_npy(result_filename, export_data)
        else:
            output_path = self._export_binary(result_filename, export_data)
        
        # Log export
        file_size = output_path.stat().st_size / 1024  # KB
        self.logger.info(f"Exported results to {output_path} (size: {file_size:.2f} KB)")
        
        return output_path
    
    def _prepare_export_data(self, 
                           anomaly_result: AnomalyResult,
                           input_tensor: torch.Tensor,
                           additional_data: Optional[Dict]) -> Dict:
        """
        Prepare data for export with minimal size.
        "a matrix of the pixel intensities of the defects in a heightened contrast 
        on top of the matrix of the input image"
        """
        # Convert tensors to numpy arrays
        input_np = input_tensor.cpu().numpy()
        defect_map_np = anomaly_result.defect_map.cpu().numpy()
        anomaly_heatmap_np = anomaly_result.anomaly_heatmap.cpu().numpy()
        
        # Create overlay with heightened contrast for defects
        overlay = self._create_defect_overlay(input_np, defect_map_np, anomaly_heatmap_np)
        
        # Quantize to reduce size (convert to uint8)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        defect_map_uint8 = (defect_map_np * 255).astype(np.uint8)
        heatmap_uint8 = (anomaly_heatmap_np * 255).astype(np.uint8)
        
        export_data = {
            'defect_overlay': overlay_uint8,
            'defect_map': defect_map_uint8,
            'anomaly_heatmap': heatmap_uint8,
            'defect_locations': np.array(anomaly_result.defect_locations, dtype=np.uint16),
            'confidence_scores': np.array(anomaly_result.confidence_scores, dtype=np.float16),
            'combined_score': np.float16(anomaly_result.combined_anomaly_score)
        }
        
        # Add defect types as encoded integers to save space
        defect_type_encoding = {dtype: i for i, dtype in enumerate(set(anomaly_result.defect_types))}
        export_data['defect_types_encoded'] = np.array(
            [defect_type_encoding[dtype] for dtype in anomaly_result.defect_types],
            dtype=np.uint8
        )
        export_data['defect_type_mapping'] = json.dumps(defect_type_encoding)
        
        # Add additional data if provided
        if additional_data:
            for key, value in additional_data.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                if isinstance(value, np.ndarray):
                    # Quantize float arrays to save space
                    if value.dtype == np.float32 or value.dtype == np.float64:
                        value = value.astype(np.float16)
                export_data[key] = value
        
        return export_data
    
    def _create_defect_overlay(self, 
                             input_image: np.ndarray,
                             defect_map: np.ndarray,
                             anomaly_heatmap: np.ndarray) -> np.ndarray:
        """
        Create overlay visualization with heightened contrast for defects.
        "a matrix of the pixel intensities of the defects in a heightened contrast"
        """
        # Ensure input is in [0, 1] range
        if input_image.max() > 1.0:
            input_image = input_image / 255.0
        
        # Convert to RGB if grayscale
        if input_image.ndim == 2:
            overlay = np.stack([input_image] * 3, axis=0)
        else:
            overlay = input_image.copy()
        
        # Apply heightened contrast to defect regions
        contrast_factor = 2.0  # Increase contrast
        
        # Create defect overlay in red channel
        defect_overlay = np.zeros_like(overlay)
        
        # Heighten contrast in defect regions
        for c in range(overlay.shape[0]):
            # Apply contrast enhancement only to defect regions
            enhanced = overlay[c] * (1 + contrast_factor * defect_map)
            enhanced = np.clip(enhanced, 0, 1)
            overlay[c] = enhanced
        
        # Add heatmap visualization in red channel
        overlay[0] = np.maximum(overlay[0], anomaly_heatmap * 0.7)
        
        return overlay
    
    def _export_npz(self, filename: str, data: Dict) -> Path:
        """
        Export as compressed NPZ file.
        "its easy for me to just copy and paste values in a visualizer so I don't 
        need any robust file type"
        """
        output_path = self.results_dir / f"{filename}.npz"
        
        # Save with compression
        np.savez_compressed(output_path, **data)
        
        return output_path
    
    def _export_npy(self, filename: str, data: Dict) -> Path:
        """Export as NPY files"""
        output_dir = self.results_dir / filename
        output_dir.mkdir(exist_ok=True)
        
        # Save each array separately
        for key, array in data.items():
            if isinstance(array, np.ndarray):
                np.save(output_dir / f"{key}.npy", array)
            else:
                # Save non-array data as JSON
                with open(output_dir / f"{key}.json", 'w') as f:
                    json.dump(array, f)
        
        return output_dir
    
    def _export_binary(self, filename: str, data: Dict) -> Path:
        """Export as raw binary for minimal size"""
        output_path = self.results_dir / f"{filename}.bin"
        
        # Create header with data structure info
        header = {
            'version': 1,
            'arrays': {}
        }
        
        # Write arrays consecutively
        with open(output_path, 'wb') as f:
            # Reserve space for header
            header_size_pos = f.tell()
            f.write(b'\x00' * 4)  # 4 bytes for header size
            
            # Write each array
            for key, array in data.items():
                if isinstance(array, np.ndarray):
                    header['arrays'][key] = {
                        'dtype': str(array.dtype),
                        'shape': array.shape,
                        'offset': f.tell()
                    }
                    array.tofile(f)
            
            # Write header
            header_json = json.dumps(header).encode('utf-8')
            header_pos = f.tell()
            f.write(header_json)
            
            # Update header size
            f.seek(header_size_pos)
            f.write(header_pos.to_bytes(4, 'little'))
        
        return output_path
    
    def export_batch_results(self, 
                           results: List[Dict[str, Union[str, AnomalyResult, torch.Tensor]]]) -> Path:
        """Export results for batch processing"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = self.results_dir / f"batch_{timestamp}"
        batch_dir.mkdir(exist_ok=True)
        
        manifest = {
            'timestamp': timestamp,
            'num_results': len(results),
            'results': []
        }
        
        for i, result in enumerate(results):
            image_id = result.get('image_id', f'image_{i}')
            anomaly_result = result['anomaly_result']
            input_tensor = result['input_tensor']
            
            # Export individual result
            output_path = self.export_anomaly_result(
                image_id,
                anomaly_result,
                input_tensor,
                result.get('additional_data')
            )
            
            manifest['results'].append({
                'image_id': image_id,
                'output_path': str(output_path.relative_to(self.results_dir)),
                'combined_score': float(anomaly_result.combined_anomaly_score),
                'num_defects': len(anomaly_result.defect_locations)
            })
        
        # Save manifest
        manifest_path = batch_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.logger.info(f"Exported batch results to {batch_dir}")
        
        return batch_dir
    
    def export_visualization(self, 
                           image_id: str,
                           input_tensor: torch.Tensor,
                           anomaly_result: AnomalyResult,
                           segmentation_masks: Optional[Dict[str, torch.Tensor]] = None) -> Path:
        """Export visual representation for quick viewing"""
        # Convert to numpy
        input_np = input_tensor.cpu().numpy()
        if input_np.shape[0] == 3:
            input_np = input_np.transpose(1, 2, 0)
        
        # Normalize to [0, 255]
        input_uint8 = (input_np * 255).astype(np.uint8)
        
        # Create visualization
        fig_height = input_uint8.shape[0]
        fig_width = input_uint8.shape[1] * 3  # 3 panels
        
        visualization = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
        
        # Panel 1: Original image
        if input_uint8.ndim == 2:
            visualization[:, :input_uint8.shape[1]] = np.stack([input_uint8] * 3, axis=2)
        else:
            visualization[:, :input_uint8.shape[1]] = input_uint8
        
        # Panel 2: Defect overlay
        defect_overlay = self._create_defect_overlay(
            input_np,
            anomaly_result.defect_map.cpu().numpy(),
            anomaly_result.anomaly_heatmap.cpu().numpy()
        )
        defect_overlay_uint8 = (defect_overlay.transpose(1, 2, 0) * 255).astype(np.uint8)
        visualization[:, input_uint8.shape[1]:2*input_uint8.shape[1]] = defect_overlay_uint8
        
        # Panel 3: Anomaly heatmap
        heatmap = anomaly_result.anomaly_heatmap.cpu().numpy()
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        visualization[:, 2*input_uint8.shape[1]:] = heatmap_colored
        
        # Save visualization
        vis_path = self.results_dir / f"{image_id}_visualization.png"
        cv2.imwrite(str(vis_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        return vis_path
    
    def export_summary_report(self, 
                            results: List[Dict],
                            output_name: str = "summary_report") -> Path:
        """Export summary report of analysis results"""
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_images': len(results),
            'total_defects': sum(len(r['anomaly_result'].defect_locations) for r in results),
            'defect_statistics': self._calculate_defect_statistics(results),
            'performance_metrics': self._calculate_performance_metrics(results)
        }
        
        # Save as JSON
        report_path = self.results_dir / f"{output_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_path = self.results_dir / f"{output_name}.csv"
        self._export_summary_csv(results, csv_path)
        
        return report_path
    
    def _calculate_defect_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics across all results"""
        all_scores = []
        defect_counts = {}
        
        for result in results:
            anomaly_result = result['anomaly_result']
            all_scores.append(anomaly_result.combined_anomaly_score)
            
            for dtype in anomaly_result.defect_types:
                defect_counts[dtype] = defect_counts.get(dtype, 0) + 1
        
        return {
            'mean_anomaly_score': np.mean(all_scores),
            'std_anomaly_score': np.std(all_scores),
            'max_anomaly_score': np.max(all_scores),
            'min_anomaly_score': np.min(all_scores),
            'defect_type_counts': defect_counts
        }
    
    def _calculate_performance_metrics(self, results: List[Dict]) -> Dict:
        """Calculate performance metrics if available"""
        metrics = {}
        
        if 'processing_time' in results[0]:
            times = [r['processing_time'] for r in results]
            metrics['avg_processing_time_ms'] = np.mean(times)
            metrics['total_processing_time_s'] = sum(times) / 1000
        
        return metrics
    
    def _export_summary_csv(self, results: List[Dict], csv_path: Path):
        """Export summary as CSV"""
        import csv
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Image ID', 'Anomaly Score', 'Num Defects', 
                'Defect Types', 'Processing Time (ms)'
            ])
            
            # Data rows
            for result in results:
                anomaly_result = result['anomaly_result']
                writer.writerow([
                    result.get('image_id', 'unknown'),
                    f"{anomaly_result.combined_anomaly_score:.4f}",
                    len(anomaly_result.defect_locations),
                    ', '.join(anomaly_result.defect_types),
                    result.get('processing_time', 'N/A')
                ])
        
        self.logger.info(f"Exported CSV summary to {csv_path}")