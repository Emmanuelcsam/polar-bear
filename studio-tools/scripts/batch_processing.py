#!/usr/bin/env python3
"""
Batch Processing and Production Tools for Ultimate Defect Detection System
==========================================================================

Advanced tools for production-scale fiber optic inspection including:
- Batch processing of multiple images
- Parallel processing optimization
- Database integration
- Quality tracking over time
- Automated report generation
"""

import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pandas as pd
import sqlite3
from datetime import datetime
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import yaml
import hashlib
import pickle
import warnings
from dataclasses import dataclass, asdict
import asyncio
import aiofiles
from functools import lru_cache

# Import the main inspection system
from ultimate_defect_detector import UltimateDefectDetector, DefectDetectionConfig
from integration_workflow import CompleteFiberInspectionSystem

warnings.filterwarnings('ignore')


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    input_directory: str
    output_directory: str
    file_patterns: List[str] = None
    max_workers: int = mp.cpu_count()
    parallel_mode: str = "process"  # "thread", "process", "async"
    save_intermediate: bool = False
    database_path: str = "inspection_results.db"
    generate_summary_report: bool = True
    quality_thresholds: Dict[str, float] = None
    processing_profile: str = "comprehensive"  # "fast", "balanced", "comprehensive"
    enable_caching: bool = True
    cache_directory: str = ".cache"
    
    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                "min_surface_quality": 70.0,
                "max_defects_core": 0,
                "max_defects_cladding": 5,
                "max_defect_density": 0.01
            }


class BatchProcessor:
    """
    Advanced batch processing system for fiber optic inspection
    """
    
    def __init__(self, config: Union[BatchConfig, str, dict]):
        """Initialize batch processor with configuration"""
        if isinstance(config, str):
            # Load from file
            config = self._load_config(config)
        elif isinstance(config, dict):
            config = BatchConfig(**config)
        
        self.config = config
        self.logger = self._setup_logging()
        self.database = DatabaseManager(self.config.database_path)
        self.cache = CacheManager(self.config.cache_directory) if self.config.enable_caching else None
        
        # Initialize inspection system
        self.inspection_config = self._get_inspection_config()
        
        # Setup output directory
        self.output_path = Path(self.config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_stats = {
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'total_time': 0,
            'average_time': 0,
            'defects_found': 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processing"""
        logger = logging.getLogger('BatchProcessor')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_path / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_config(self, config_path: str) -> BatchConfig:
        """Load configuration from file"""
        path = Path(config_path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        return BatchConfig(**data)
    
    def _get_inspection_config(self) -> dict:
        """Get inspection configuration based on processing profile"""
        profiles = {
            "fast": {
                "defect_detection": {
                    "use_all_methods": False,
                    "method_groups": {
                        "statistical": {"enabled": True, "methods": ["zscore"]},
                        "morphological": {"enabled": True, "methods": ["tophat"]},
                        "ml": {"enabled": False}
                    }
                }
            },
            "balanced": {
                "defect_detection": {
                    "use_all_methods": False,
                    "method_groups": {
                        "statistical": {"enabled": True, "methods": ["zscore", "mad"]},
                        "spatial": {"enabled": True, "methods": ["lbp"]},
                        "morphological": {"enabled": True, "methods": ["tophat", "blackhat"]},
                        "ml": {"enabled": True, "methods": ["isolation_forest"]}
                    }
                }
            },
            "comprehensive": {
                "defect_detection": {
                    "use_all_methods": True
                }
            }
        }
        
        base_config = {
            "image_processing": {
                "preprocessing": {
                    "illumination_correction": True,
                    "noise_reduction": True,
                    "contrast_enhancement": True
                }
            },
            "output": {
                "save_all_intermediate": self.config.save_intermediate,
                "generate_report": True
            }
        }
        
        # Merge with profile-specific config
        profile_config = profiles.get(self.config.processing_profile, profiles["balanced"])
        return {**base_config, **profile_config}
    
    def process_batch(self, input_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Process batch of images
        
        Args:
            input_directory: Override config input directory
            
        Returns:
            Summary of batch processing results
        """
        input_path = Path(input_directory or self.config.input_directory)
        
        # Collect image files
        image_files = self._collect_image_files(input_path)
        self.performance_stats['total_images'] = len(image_files)
        
        if not image_files:
            self.logger.warning(f"No image files found in {input_path}")
            return self.performance_stats
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process based on parallel mode
        start_time = datetime.now()
        
        if self.config.parallel_mode == "async":
            results = asyncio.run(self._process_async(image_files))
        elif self.config.parallel_mode == "process":
            results = self._process_parallel(image_files)
        else:  # thread or sequential
            results = self._process_threaded(image_files)
        
        end_time = datetime.now()
        self.performance_stats['total_time'] = (end_time - start_time).total_seconds()
        
        # Generate summary report
        if self.config.generate_summary_report:
            self._generate_batch_report(results)
        
        # Update database with batch summary
        self.database.save_batch_summary(self.performance_stats)
        
        return self.performance_stats
    
    def _collect_image_files(self, input_path: Path) -> List[Path]:
        """Collect all image files matching patterns"""
        image_files = []
        
        for pattern in self.config.file_patterns:
            image_files.extend(input_path.glob(pattern))
            image_files.extend(input_path.glob(f"**/{pattern}"))  # Recursive
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        
        return image_files
    
    def _process_parallel(self, image_files: List[Path]) -> List[Dict[str, Any]]:
        """Process images in parallel using processes"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_image, file): file 
                for file in image_files
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(image_files), desc="Processing images") as pbar:
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.performance_stats['processed'] += 1
                        
                        # Update defect count
                        if result and 'defects' in result:
                            self.performance_stats['defects_found'] += len(result['defects'])
                    
                    except Exception as e:
                        self.logger.error(f"Failed to process {file}: {str(e)}")
                        self.performance_stats['failed'] += 1
                        results.append({
                            'file': str(file),
                            'status': 'failed',
                            'error': str(e)
                        })
                    
                    pbar.update(1)
        
        return results
    
    def _process_threaded(self, image_files: List[Path]) -> List[Dict[str, Any]]:
        """Process images using thread pool"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self._process_single_image, file) for file in image_files]
            
            # Process with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                try:
                    result = future.result()
                    results.append(result)
                    self.performance_stats['processed'] += 1
                    
                    if result and 'defects' in result:
                        self.performance_stats['defects_found'] += len(result['defects'])
                
                except Exception as e:
                    self.logger.error(f"Processing failed: {str(e)}")
                    self.performance_stats['failed'] += 1
        
        return results
    
    async def _process_async(self, image_files: List[Path]) -> List[Dict[str, Any]]:
        """Process images asynchronously"""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_with_semaphore(file):
            async with semaphore:
                return await self._process_single_image_async(file)
        
        tasks = [process_with_semaphore(file) for file in image_files]
        
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing images"):
            try:
                result = await coro
                results.append(result)
                self.performance_stats['processed'] += 1
                
                if result and 'defects' in result:
                    self.performance_stats['defects_found'] += len(result['defects'])
            
            except Exception as e:
                self.logger.error(f"Async processing failed: {str(e)}")
                self.performance_stats['failed'] += 1
        
        return results
    
    def _process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image"""
        try:
            # Check cache
            if self.cache:
                cached_result = self.cache.get(str(image_path))
                if cached_result is not None:
                    self.logger.debug(f"Using cached result for {image_path}")
                    return cached_result
            
            # Create inspection system
            inspector = CompleteFiberInspectionSystem(self.inspection_config)
            
            # Process image
            start_time = datetime.now()
            results = inspector.inspect_fiber(str(image_path))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Add metadata
            results['processing_time'] = processing_time
            results['file_name'] = image_path.name
            results['file_path'] = str(image_path)
            
            # Save to database
            self.database.save_inspection_result(results)
            
            # Cache result
            if self.cache:
                self.cache.set(str(image_path), results)
            
            # Save individual report if needed
            if self.config.save_intermediate:
                self._save_individual_report(image_path, results)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
            raise
    
    async def _process_single_image_async(self, image_path: Path) -> Dict[str, Any]:
        """Process a single image asynchronously"""
        # Run CPU-intensive processing in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_single_image, image_path)
    
    def _save_individual_report(self, image_path: Path, results: Dict[str, Any]) -> None:
        """Save individual image report"""
        # Create output directory for this image
        output_dir = self.output_path / image_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Copy original image
        import shutil
        shutil.copy2(image_path, output_dir / f"original{image_path.suffix}")
    
    def _generate_batch_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate comprehensive batch processing report"""
        report_path = self.output_path / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_path.mkdir(exist_ok=True)
        
        # 1. Generate summary statistics
        self._generate_summary_stats(results, report_path)
        
        # 2. Generate quality distribution plots
        self._generate_quality_plots(results, report_path)
        
        # 3. Generate defect analysis
        self._generate_defect_analysis(results, report_path)
        
        # 4. Generate HTML dashboard
        self._generate_html_dashboard(results, report_path)
        
        # 5. Export to Excel
        self._export_to_excel(results, report_path)
        
        self.logger.info(f"Batch report generated at: {report_path}")
    
    def _generate_summary_stats(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Generate summary statistics"""
        # Filter out failed results
        successful_results = [r for r in results if r.get('status') != 'failed']
        
        # Calculate statistics
        stats = {
            'total_processed': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'total_defects': sum(len(r.get('defects', [])) for r in successful_results),
            'pass_rate': sum(1 for r in successful_results if r.get('pass_fail', {}).get('overall') == 'PASS') / len(successful_results) if successful_results else 0,
            'average_quality': np.mean([r.get('quality_metrics', {}).get('surface_quality_index', 0) for r in successful_results]),
            'average_processing_time': np.mean([r.get('processing_time', 0) for r in successful_results])
        }
        
        # Save as JSON
        with open(output_dir / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Pass/Fail pie chart
        ax = axes[0, 0]
        pass_count = stats['pass_rate'] * stats['successful']
        fail_count = stats['successful'] - pass_count
        ax.pie([pass_count, fail_count], labels=['Pass', 'Fail'], autopct='%1.1f%%', 
               colors=['#2ecc71', '#e74c3c'])
        ax.set_title('Pass/Fail Distribution')
        
        # Quality distribution
        ax = axes[0, 1]
        qualities = [r.get('quality_metrics', {}).get('surface_quality_index', 0) 
                    for r in successful_results]
        ax.hist(qualities, bins=20, color='#3498db', edgecolor='black')
        ax.set_xlabel('Surface Quality Index')
        ax.set_ylabel('Count')
        ax.set_title('Quality Distribution')
        ax.axvline(x=stats['average_quality'], color='red', linestyle='--', 
                  label=f'Mean: {stats["average_quality"]:.1f}')
        ax.legend()
        
        # Defects by region
        ax = axes[1, 0]
        region_defects = {'core': 0, 'cladding': 0, 'ferrule': 0}
        for r in successful_results:
            for defect in r.get('defects', []):
                region = defect.get('region', '').lower()
                if region in region_defects:
                    region_defects[region] += 1
        
        ax.bar(region_defects.keys(), region_defects.values(), color='#9b59b6')
        ax.set_xlabel('Region')
        ax.set_ylabel('Defect Count')
        ax.set_title('Defects by Region')
        
        # Processing time distribution
        ax = axes[1, 1]
        times = [r.get('processing_time', 0) for r in successful_results]
        ax.hist(times, bins=20, color='#1abc9c', edgecolor='black')
        ax.set_xlabel('Processing Time (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('Processing Time Distribution')
        ax.axvline(x=stats['average_processing_time'], color='red', linestyle='--',
                  label=f'Mean: {stats["average_processing_time"]:.1f}s')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_quality_plots(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Generate quality analysis plots"""
        # Create DataFrame for easier analysis
        data = []
        for r in results:
            if r.get('status') != 'failed':
                data.append({
                    'file': r.get('file_name', 'unknown'),
                    'quality_index': r.get('quality_metrics', {}).get('surface_quality_index', 0),
                    'total_defects': len(r.get('defects', [])),
                    'core_defects': len([d for d in r.get('defects', []) if d.get('region', '').lower() == 'core']),
                    'status': r.get('pass_fail', {}).get('overall', 'UNKNOWN'),
                    'processing_time': r.get('processing_time', 0)
                })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Quality vs Defects scatter plot
            plt.figure(figsize=(10, 6))
            colors = df['status'].map({'PASS': '#2ecc71', 'FAIL': '#e74c3c', 'UNKNOWN': '#95a5a6'})
            plt.scatter(df['total_defects'], df['quality_index'], c=colors, alpha=0.6, s=100)
            plt.xlabel('Total Defects')
            plt.ylabel('Surface Quality Index')
            plt.title('Quality Index vs Defect Count')
            
            # Add trend line
            z = np.polyfit(df['total_defects'], df['quality_index'], 1)
            p = np.poly1d(z)
            plt.plot(df['total_defects'], p(df['total_defects']), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'quality_vs_defects.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Box plots by status
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Quality by status
            df.boxplot(column='quality_index', by='status', ax=axes[0])
            axes[0].set_title('Quality Index by Status')
            axes[0].set_ylabel('Surface Quality Index')
            
            # Processing time by status
            df.boxplot(column='processing_time', by='status', ax=axes[1])
            axes[1].set_title('Processing Time by Status')
            axes[1].set_ylabel('Time (seconds)')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'status_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_defect_analysis(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Generate detailed defect analysis"""
        # Collect all defects
        all_defects = []
        for r in results:
            if r.get('status') != 'failed':
                for defect in r.get('defects', []):
                    defect_data = defect.copy()
                    defect_data['source_file'] = r.get('file_name', 'unknown')
                    all_defects.append(defect_data)
        
        if not all_defects:
            return
        
        # Create defect DataFrame
        df_defects = pd.DataFrame(all_defects)
        
        # Defect type distribution
        plt.figure(figsize=(12, 6))
        defect_types = df_defects['type'].value_counts()
        plt.bar(defect_types.index, defect_types.values, color='#e67e22')
        plt.xlabel('Defect Type')
        plt.ylabel('Count')
        plt.title('Defect Type Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'defect_types.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Defect size distribution by type
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by type and plot size distributions
        for defect_type in df_defects['type'].unique():
            type_data = df_defects[df_defects['type'] == defect_type]
            if 'area_um2' in type_data.columns:
                sizes = type_data['area_um2'].dropna()
                if len(sizes) > 0:
                    ax.hist(sizes, bins=20, alpha=0.5, label=defect_type)
        
        ax.set_xlabel('Defect Area (μm²)')
        ax.set_ylabel('Count')
        ax.set_title('Defect Size Distribution by Type')
        ax.legend()
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / 'defect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Heatmap of defect locations (simplified)
        if 'location_px' in df_defects.columns:
            plt.figure(figsize=(10, 10))
            
            # Extract x, y coordinates
            locations = df_defects['location_px'].dropna()
            if len(locations) > 0:
                x_coords = [loc[0] if isinstance(loc, (list, tuple)) else 0 for loc in locations]
                y_coords = [loc[1] if isinstance(loc, (list, tuple)) else 0 for loc in locations]
                
                plt.hexbin(x_coords, y_coords, gridsize=20, cmap='YlOrRd')
                plt.colorbar(label='Defect Count')
                plt.xlabel('X Position (pixels)')
                plt.ylabel('Y Position (pixels)')
                plt.title('Defect Location Heatmap')
                plt.tight_layout()
                plt.savefig(output_dir / 'defect_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _generate_html_dashboard(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Generate interactive HTML dashboard"""
        # Calculate statistics
        successful_results = [r for r in results if r.get('status') != 'failed']
        
        stats = {
            'total': len(results),
            'passed': sum(1 for r in successful_results if r.get('pass_fail', {}).get('overall') == 'PASS'),
            'failed_inspection': sum(1 for r in successful_results if r.get('pass_fail', {}).get('overall') == 'FAIL'),
            'failed_processing': len(results) - len(successful_results),
            'total_defects': sum(len(r.get('defects', [])) for r in successful_results),
            'avg_quality': np.mean([r.get('quality_metrics', {}).get('surface_quality_index', 0) for r in successful_results]) if successful_results else 0,
            'avg_time': np.mean([r.get('processing_time', 0) for r in successful_results]) if successful_results else 0
        }
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Processing Dashboard</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f0f2f5; 
                }}
                .dashboard {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stats-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 30px; 
                }}
                .stat-card {{ 
                    background: white; 
                    padding: 25px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.2s;
                }}
                .stat-card:hover {{ 
                    transform: translateY(-5px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }}
                .stat-value {{ 
                    font-size: 36px; 
                    font-weight: bold; 
                    margin: 10px 0; 
                }}
                .stat-label {{ 
                    color: #666; 
                    font-size: 14px; 
                }}
                .pass {{ color: #27ae60; }}
                .fail {{ color: #e74c3c; }}
                .warning {{ color: #f39c12; }}
                .section {{ 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .results-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 20px;
                }}
                .results-table th {{ 
                    background-color: #f8f9fa; 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 2px solid #dee2e6;
                    position: sticky;
                    top: 0;
                }}
                .results-table td {{ 
                    padding: 12px; 
                    border-bottom: 1px solid #dee2e6; 
                }}
                .results-table tr:hover {{ 
                    background-color: #f8f9fa; 
                }}
                .progress-bar {{
                    width: 100%;
                    height: 20px;
                    background-color: #e9ecef;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
                    transition: width 0.3s ease;
                }}
                .chart-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }}
                .chart-box {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 12px;
                    text-align: right;
                    margin-top: 20px;
                }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <h1>Fiber Optic Batch Processing Dashboard</h1>
                    <p>Comprehensive analysis of {stats['total']} fiber optic end faces</p>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Processed</div>
                        <div class="stat-value">{stats['total']}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 100%"></div>
                        </div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Passed</div>
                        <div class="stat-value pass">{stats['passed']}</div>
                        <div class="stat-label">{stats['passed']/stats['total']*100:.1f}%</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Failed Inspection</div>
                        <div class="stat-value fail">{stats['failed_inspection']}</div>
                        <div class="stat-label">{stats['failed_inspection']/stats['total']*100:.1f}%</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Processing Errors</div>
                        <div class="stat-value warning">{stats['failed_processing']}</div>
                        <div class="stat-label">{stats['failed_processing']/stats['total']*100:.1f}%</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Total Defects</div>
                        <div class="stat-value">{stats['total_defects']}</div>
                        <div class="stat-label">Across all samples</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Average Quality</div>
                        <div class="stat-value">{stats['avg_quality']:.1f}</div>
                        <div class="stat-label">Out of 100</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Avg Processing Time</div>
                        <div class="stat-value">{stats['avg_time']:.1f}s</div>
                        <div class="stat-label">Per image</div>
                    </div>
                    
                    <div class="stat-card">
                        <div class="stat-label">Total Time</div>
                        <div class="stat-value">{self.performance_stats['total_time']:.1f}s</div>
                        <div class="stat-label">Complete batch</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Quality Distribution</h2>
                    <div id="qualityChart" style="height: 400px;"></div>
                </div>
                
                <div class="section">
                    <h2>Defect Analysis</h2>
                    <div class="chart-container">
                        <div class="chart-box">
                            <div id="defectTypesChart" style="height: 300px;"></div>
                        </div>
                        <div class="chart-box">
                            <div id="defectRegionsChart" style="height: 300px;"></div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Individual Results</h2>
                    <table class="results-table">
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Status</th>
                                <th>Quality</th>
                                <th>Defects</th>
                                <th>Core</th>
                                <th>Cladding</th>
                                <th>Time (s)</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add individual results
        for r in results[:100]:  # Limit to first 100
            if r.get('status') != 'failed':
                status = r.get('pass_fail', {}).get('overall', 'UNKNOWN')
                status_class = status.lower()
                quality = r.get('quality_metrics', {}).get('surface_quality_index', 0)
                total_defects = len(r.get('defects', []))
                core_defects = len([d for d in r.get('defects', []) if d.get('region', '').lower() == 'core'])
                cladding_defects = len([d for d in r.get('defects', []) if d.get('region', '').lower() == 'cladding'])
                
                html_content += f"""
                            <tr>
                                <td>{r.get('file_name', 'unknown')}</td>
                                <td class="{status_class}">{status}</td>
                                <td>{quality:.1f}</td>
                                <td>{total_defects}</td>
                                <td>{core_defects}</td>
                                <td>{cladding_defects}</td>
                                <td>{r.get('processing_time', 0):.1f}</td>
                            </tr>
                """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
                
                <div class="timestamp">
                    Report generated by Ultimate Defect Detection System v3.0
                </div>
            </div>
            
            <script>
                // Quality distribution histogram
                var qualityData = [
        """
        
        # Add quality data
        qualities = [r.get('quality_metrics', {}).get('surface_quality_index', 0) 
                    for r in successful_results]
        
        html_content += f"""
                    {{
                        x: {qualities},
                        type: 'histogram',
                        nbinsx: 20,
                        marker: {{ color: '#3498db' }}
                    }}
                ];
                
                var qualityLayout = {{
                    title: 'Surface Quality Index Distribution',
                    xaxis: {{ title: 'Quality Index' }},
                    yaxis: {{ title: 'Count' }},
                    showlegend: false
                }};
                
                Plotly.newPlot('qualityChart', qualityData, qualityLayout);
                
                // Defect types chart
                var defectTypes = {{}};
                var defectRegions = {{'core': 0, 'cladding': 0, 'ferrule': 0}};
        """
        
        # Calculate defect statistics
        for r in successful_results:
            for defect in r.get('defects', []):
                dtype = defect.get('type', 'unknown')
                defectTypes[dtype] = (defectTypes[dtype] || 0) + 1
                
                region = defect.get('region', '').lower()
                if region in defectRegions:
                    defectRegions[region] += 1
        
        html_content += f"""
                {json.dumps(defectTypes)}.forEach(function(value, key) {{
                    defectTypes[key] = value;
                }});
                
                {json.dumps(defectRegions)}.forEach(function(value, key) {{
                    defectRegions[key] = value;
                }});
                
                // Defect types pie chart
                var typeData = [{{
                    values: Object.values(defectTypes),
                    labels: Object.keys(defectTypes),
                    type: 'pie',
                    hole: 0.4
                }}];
                
                var typeLayout = {{
                    title: 'Defect Types',
                    showlegend: true
                }};
                
                Plotly.newPlot('defectTypesChart', typeData, typeLayout);
                
                // Defect regions bar chart
                var regionData = [{{
                    x: Object.keys(defectRegions),
                    y: Object.values(defectRegions),
                    type: 'bar',
                    marker: {{ color: ['#e74c3c', '#f39c12', '#3498db'] }}
                }}];
                
                var regionLayout = {{
                    title: 'Defects by Region',
                    xaxis: {{ title: 'Region' }},
                    yaxis: {{ title: 'Count' }}
                }};
                
                Plotly.newPlot('defectRegionsChart', regionData, regionLayout);
            </script>
        </body>
        </html>
        """
        
        with open(output_dir / 'dashboard.html', 'w') as f:
            f.write(html_content)
    
    def _export_to_excel(self, results: List[Dict[str, Any]], output_dir: Path) -> None:
        """Export results to Excel with multiple sheets"""
        excel_path = output_dir / 'batch_results.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for r in results:
                if r.get('status') != 'failed':
                    summary_data.append({
                        'File': r.get('file_name', 'unknown'),
                        'Status': r.get('pass_fail', {}).get('overall', 'UNKNOWN'),
                        'Quality Index': r.get('quality_metrics', {}).get('surface_quality_index', 0),
                        'Total Defects': len(r.get('defects', [])),
                        'Core Defects': len([d for d in r.get('defects', []) if d.get('region', '').lower() == 'core']),
                        'Cladding Defects': len([d for d in r.get('defects', []) if d.get('region', '').lower() == 'cladding']),
                        'Processing Time': r.get('processing_time', 0)
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Defects sheet
            defect_data = []
            for r in results:
                if r.get('status') != 'failed':
                    for defect in r.get('defects', []):
                        defect_record = {
                            'File': r.get('file_name', 'unknown'),
                            'Defect ID': defect.get('id', ''),
                            'Type': defect.get('type', ''),
                            'Region': defect.get('region', ''),
                            'Confidence': defect.get('confidence', 0),
                            'Area (px)': defect.get('area_px', 0),
                            'Area (um2)': defect.get('area_um2', 0),
                            'Major Axis (um)': defect.get('major_axis_um', 0),
                            'Minor Axis (um)': defect.get('minor_axis_um', 0)
                        }
                        defect_data.append(defect_record)
            
            if defect_data:
                df_defects = pd.DataFrame(defect_data)
                df_defects.to_excel(writer, sheet_name='Defects', index=False)
            
            # Statistics sheet
            stats_data = {
                'Metric': [
                    'Total Images',
                    'Successfully Processed',
                    'Failed Processing',
                    'Pass Rate (%)',
                    'Average Quality Index',
                    'Total Defects Found',
                    'Average Processing Time (s)',
                    'Total Processing Time (s)'
                ],
                'Value': [
                    self.performance_stats['total_images'],
                    self.performance_stats['processed'],
                    self.performance_stats['failed'],
                    (sum(1 for r in results if r.get('pass_fail', {}).get('overall') == 'PASS') / 
                     len([r for r in results if r.get('status') != 'failed']) * 100) if results else 0,
                    np.mean([r.get('quality_metrics', {}).get('surface_quality_index', 0) 
                            for r in results if r.get('status') != 'failed']),
                    self.performance_stats['defects_found'],
                    np.mean([r.get('processing_time', 0) for r in results if r.get('status') != 'failed']),
                    self.performance_stats['total_time']
                ]
            }
            
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)


class DatabaseManager:
    """Manage inspection results in SQLite database"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Inspections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                inspection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                quality_index REAL,
                total_defects INTEGER,
                processing_time REAL,
                results_json TEXT
            )
        """)
        
        # Defects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER,
                defect_id TEXT,
                type TEXT,
                region TEXT,
                confidence REAL,
                area_px INTEGER,
                area_um2 REAL,
                location_x INTEGER,
                location_y INTEGER,
                FOREIGN KEY (inspection_id) REFERENCES inspections (id)
            )
        """)
        
        # Batch summaries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_images INTEGER,
                processed INTEGER,
                failed INTEGER,
                total_time REAL,
                defects_found INTEGER,
                summary_json TEXT
            )
        """)
        
        self.conn.commit()
    
    def save_inspection_result(self, result: Dict[str, Any]) -> int:
        """Save inspection result to database"""
        cursor = self.conn.cursor()
        
        # Insert inspection record
        cursor.execute("""
            INSERT INTO inspections 
            (file_path, file_name, status, quality_index, total_defects, processing_time, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get('file_path', ''),
            result.get('file_name', ''),
            result.get('pass_fail', {}).get('overall', 'UNKNOWN'),
            result.get('quality_metrics', {}).get('surface_quality_index', 0),
            len(result.get('defects', [])),
            result.get('processing_time', 0),
            json.dumps(result, default=str)
        ))
        
        inspection_id = cursor.lastrowid
        
        # Insert defect records
        for defect in result.get('defects', []):
            location = defect.get('location_px', (0, 0))
            cursor.execute("""
                INSERT INTO defects
                (inspection_id, defect_id, type, region, confidence, area_px, area_um2, location_x, location_y)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inspection_id,
                defect.get('id', ''),
                defect.get('type', ''),
                defect.get('region', ''),
                defect.get('confidence', 0),
                defect.get('area_px', 0),
                defect.get('area_um2', 0),
                location[0] if isinstance(location, (list, tuple)) else 0,
                location[1] if isinstance(location, (list, tuple)) else 0
            ))
        
        self.conn.commit()
        return inspection_id
    
    def save_batch_summary(self, stats: Dict[str, Any]) -> None:
        """Save batch processing summary"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO batch_summaries
            (total_images, processed, failed, total_time, defects_found, summary_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            stats.get('total_images', 0),
            stats.get('processed', 0),
            stats.get('failed', 0),
            stats.get('total_time', 0),
            stats.get('defects_found', 0),
            json.dumps(stats)
        ))
        
        self.conn.commit()
    
    def get_statistics(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics from database"""
        cursor = self.conn.cursor()
        
        # Build query
        query = "SELECT * FROM inspections WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND inspection_timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND inspection_timestamp <= ?"
            params.append(end_date)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Calculate statistics
        stats = {
            'total_inspections': len(results),
            'passed': sum(1 for r in results if r[3] == 'PASS'),
            'failed': sum(1 for r in results if r[3] == 'FAIL'),
            'average_quality': np.mean([r[4] for r in results if r[4]]) if results else 0,
            'total_defects': sum(r[5] for r in results if r[5]),
            'average_processing_time': np.mean([r[6] for r in results if r[6]]) if results else 0
        }
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()


class CacheManager:
    """Manage caching of inspection results"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key from file path"""
        # Use file path and modification time
        file_stat = Path(file_path).stat()
        key_string = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                # Invalid cache file
                cache_file.unlink()
        
        return None
    
    def set(self, file_path: str, result: Dict[str, Any]) -> None:
        """Cache result"""
        cache_key = self._get_cache_key(file_path)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    def clear(self) -> None:
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# Command-line interface
def main():
    """Command-line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process fiber optic images")
    parser.add_argument("input_dir", help="Input directory containing images")
    parser.add_argument("-o", "--output", default="batch_results", help="Output directory")
    parser.add_argument("-c", "--config", help="Configuration file (JSON or YAML)")
    parser.add_argument("-p", "--profile", choices=["fast", "balanced", "comprehensive"], 
                       default="balanced", help="Processing profile")
    parser.add_argument("-w", "--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--parallel", choices=["thread", "process", "async"], 
                       default="process", help="Parallel processing mode")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before processing")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = BatchConfig(**json.load(open(args.config)))
    else:
        config = BatchConfig(
            input_directory=args.input_dir,
            output_directory=args.output,
            processing_profile=args.profile,
            parallel_mode=args.parallel,
            enable_caching=not args.no_cache
        )
    
    if args.workers:
        config.max_workers = args.workers
    
    # Create processor
    processor = BatchProcessor(config)
    
    # Clear cache if requested
    if args.clear_cache and processor.cache:
        processor.cache.clear()
        print("Cache cleared")
    
    # Run batch processing
    print(f"Starting batch processing of {args.input_dir}")
    print(f"Profile: {config.processing_profile}")
    print(f"Workers: {config.max_workers}")
    print(f"Mode: {config.parallel_mode}")
    
    stats = processor.process_batch()
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images: {stats['total_images']}")
    print(f"Processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total defects: {stats['defects_found']}")
    print(f"Total time: {stats['total_time']:.1f} seconds")
    print(f"Average time: {stats['total_time']/stats['processed']:.1f} seconds/image")
    print(f"\nResults saved to: {config.output_directory}")


if __name__ == "__main__":
    main()
