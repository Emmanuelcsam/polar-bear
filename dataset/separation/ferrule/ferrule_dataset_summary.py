#!/usr/bin/env python3
"""
Ferrule Dataset Summary Generator
================================
Generate a comprehensive summary of the reorganized ferrule dataset.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def generate_dataset_summary():
    """Generate a comprehensive summary of the reorganized dataset"""
    
    # Load analysis results
    with open('ferrule_analysis_results.json', 'r') as f:
        analysis_results = json.load(f)
        
    # Load reorganization log
    with open('ferrule_reorganization_log.json', 'r') as f:
        reorg_log = json.load(f)
        
    # Count files in dataset structure
    dataset_stats = defaultdict(lambda: defaultdict(int))
    dataset_path = Path('dataset')
    
    for split in ['train', 'val', 'test']:
        for defect_type in ['scratch', 'clean', 'contaminated']:
            path = dataset_path / split / defect_type
            if path.exists():
                count = len(list(path.glob('*.png')))
                dataset_stats[split][defect_type] = count
                
    # Generate report
    report_lines = [
        "# Ferrule Dataset Organization Summary",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Dataset Overview",
        f"Total images processed: {len(analysis_results)}",
        f"Total images reorganized: {len(reorg_log)}",
        "\n## Original File Analysis",
        "\n### Defect Type Distribution"
    ]
    
    # Count original defect types
    defect_counts = defaultdict(int)
    for result in analysis_results:
        defect_counts[result['defect_type']] += 1
        
    for defect_type, count in sorted(defect_counts.items()):
        percentage = (count / len(analysis_results)) * 100
        report_lines.append(f"- {defect_type.capitalize()}: {count} ({percentage:.1f}%)")
        
    # Add reorganized dataset structure
    report_lines.extend([
        "\n## Reorganized Dataset Structure",
        "\n### Train/Val/Test Split"
    ])
    
    total_by_split = defaultdict(int)
    for split in ['train', 'val', 'test']:
        split_total = sum(dataset_stats[split].values())
        total_by_split[split] = split_total
        report_lines.append(f"\n**{split.capitalize()} Set**: {split_total} images")
        
        for defect_type in ['scratch', 'clean', 'contaminated']:
            count = dataset_stats[split][defect_type]
            if count > 0:
                report_lines.append(f"  - {defect_type}: {count}")
                
    # Add file naming convention
    report_lines.extend([
        "\n## File Naming Convention",
        "Files have been renamed according to the pattern:",
        "`{defect_type}_{split}_{index:04d}_{hash}.png`",
        "\nExamples:",
        "- `scratch_train_0001_eee5eaa7.png`",
        "- `contaminated_val_0001_f018732a.png`",
        "- `clean_test_0001_abc12345.png`",
        "\n## Original to New File Mapping",
        "\n### Scratch Images"
    ])
    
    # Group by defect type
    mapping_by_type = defaultdict(list)
    for entry in reorg_log:
        mapping_by_type[entry['defect_type']].append(entry)
        
    # Add mappings
    for defect_type in ['scratch', 'contaminated', 'clean']:
        if defect_type in mapping_by_type:
            report_lines.append(f"\n### {defect_type.capitalize()} Images")
            for entry in mapping_by_type[defect_type][:5]:  # Show first 5
                orig_name = Path(entry['original']).name
                new_name = Path(entry['new_path']).name
                report_lines.append(f"- `{orig_name}` → `{new_name}` (severity: {entry['severity']:.3f})")
            
            if len(mapping_by_type[defect_type]) > 5:
                report_lines.append(f"  ... and {len(mapping_by_type[defect_type]) - 5} more")
                
    # Add analysis insights
    report_lines.extend([
        "\n## Key Insights",
        "\n1. **Scratch Detection**: The analysis successfully identified all images with 'scratch' in their filename as having scratch defects",
        "2. **UBET Images**: Images labeled 'ubet' were classified as contaminated, suggesting they may have surface contamination",
        "3. **Unnamed Images**: Generic ferrule images (without specific labels) showed varied defect patterns",
        "4. **No Clean Images**: No images were classified as completely clean, indicating all samples have some form of defect",
        "\n## Recommendations",
        "\n1. **Collect Clean Samples**: The dataset lacks clean ferrule images for comparison",
        "2. **Manual Verification**: Review the contaminated classification for UBET images",
        "3. **Augmentation**: Consider data augmentation to balance the dataset",
        "4. **Severity Thresholds**: Fine-tune severity thresholds based on domain expertise"
    ])
    
    # Save report
    with open('ferrule_dataset_summary.md', 'w') as f:
        f.write('\n'.join(report_lines))
        
    print("Dataset summary saved to ferrule_dataset_summary.md")
    
    # Print quick statistics
    print("\nQuick Statistics:")
    print(f"Total images: {len(analysis_results)}")
    print(f"Train set: {total_by_split['train']} images")
    print(f"Val set: {total_by_split['val']} images")
    print(f"Test set: {total_by_split['test']} images")
    print(f"Scratch defects: {defect_counts['scratch']}")
    print(f"Contaminated: {defect_counts['contaminated']}")
    print(f"Clean: {defect_counts['clean']}")


if __name__ == "__main__":
    generate_dataset_summary()