import json
import csv
import os
import time
import pickle
import h5py
import numpy as np
from datetime import datetime
import zipfile
import xml.etree.ElementTree as ET
import yaml

def export_to_csv():
    """Export all data to CSV format"""
    
    print("[EXPORT] Exporting data to CSV format...")
    
    exported_files = []
    
    # Export pixel data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
        
        with open('pixel_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'value'])
            
            for i, pixel in enumerate(data.get('pixels', [])):
                writer.writerow([i, pixel])
        
        exported_files.append('pixel_data.csv')
        print("[EXPORT] Exported pixel_data.csv")
    
    # Export patterns
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
        
        if 'frequency' in patterns:
            with open('pattern_frequencies.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['value', 'frequency'])
                
                for value, freq in patterns['frequency'].items():
                    writer.writerow([value, freq])
            
            exported_files.append('pattern_frequencies.csv')
    
    # Export anomalies
    if os.path.exists('anomalies.json'):
        with open('anomalies.json', 'r') as f:
            anomalies = json.load(f)
        
        if 'z_score_anomalies' in anomalies:
            with open('anomalies.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['index', 'value', 'z_score'])
                
                for anom in anomalies['z_score_anomalies']:
                    writer.writerow([anom['index'], anom['value'], anom.get('z_score', '')])
            
            exported_files.append('anomalies.csv')
    
    print(f"[EXPORT] Exported {len(exported_files)} CSV files")
    return exported_files

def export_to_hdf5():
    """Export data to HDF5 format for efficient storage"""
    
    print("\n[EXPORT] Exporting data to HDF5 format...")
    
    with h5py.File('pixel_analysis_data.h5', 'w') as hf:
        # Metadata
        hf.attrs['created'] = datetime.now().isoformat()
        hf.attrs['version'] = '1.0'
        
        # Pixel data
        if os.path.exists('pixel_data.json'):
            with open('pixel_data.json', 'r') as f:
                data = json.load(f)
            
            if 'pixels' in data:
                pixel_group = hf.create_group('pixels')
                pixel_group.create_dataset('values', data=np.array(data['pixels']))
                pixel_group.attrs['size'] = data.get('size', [0, 0])
                pixel_group.attrs['timestamp'] = data.get('timestamp', 0)
        
        # Neural results
        if os.path.exists('neural_results.json'):
            with open('neural_results.json', 'r') as f:
                neural = json.load(f)
            
            if 'predictions' in neural:
                neural_group = hf.create_group('neural')
                neural_group.create_dataset('predictions', data=np.array(neural['predictions']))
                neural_group.attrs['training_loss'] = neural.get('training_loss', 0)
        
        # GPU results
        if os.path.exists('gpu_results.json'):
            with open('gpu_results.json', 'r') as f:
                gpu = json.load(f)
            
            gpu_group = hf.create_group('gpu')
            gpu_group.attrs['device'] = gpu.get('device', 'unknown')
            
            if 'performance' in gpu:
                perf = gpu['performance']
                gpu_group.attrs['total_time'] = perf.get('total_time', 0)
                gpu_group.attrs['pixels_per_second'] = perf.get('pixels_per_second', 0)
        
        # ML clustering
        if os.path.exists('ml_clustering.json'):
            with open('ml_clustering.json', 'r') as f:
                clustering = json.load(f)
            
            ml_group = hf.create_group('ml_clustering')
            
            if 'kmeans' in clustering:
                kmeans_group = ml_group.create_group('kmeans')
                kmeans_group.create_dataset('labels', data=np.array(clustering['kmeans']['labels']))
                kmeans_group.attrs['n_clusters'] = clustering['kmeans']['n_clusters']
    
    print("[EXPORT] Exported data to pixel_analysis_data.h5")
    
    # Print file info
    file_size = os.path.getsize('pixel_analysis_data.h5') / 1024
    print(f"[EXPORT] HDF5 file size: {file_size:.1f} KB")
    
    return 'pixel_analysis_data.h5'

def export_to_xml():
    """Export data to XML format"""
    
    print("\n[EXPORT] Exporting data to XML format...")
    
    root = ET.Element('PixelAnalysisData')
    root.set('version', '1.0')
    root.set('timestamp', datetime.now().isoformat())
    
    # Add pixel data
    if os.path.exists('pixel_data.json'):
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
        
        pixels_elem = ET.SubElement(root, 'Pixels')
        pixels_elem.set('count', str(len(data.get('pixels', []))))
        
        # Add statistics
        if 'pixels' in data:
            pixels_array = np.array(data['pixels'])
            stats_elem = ET.SubElement(pixels_elem, 'Statistics')
            ET.SubElement(stats_elem, 'Mean').text = str(np.mean(pixels_array))
            ET.SubElement(stats_elem, 'StdDev').text = str(np.std(pixels_array))
            ET.SubElement(stats_elem, 'Min').text = str(np.min(pixels_array))
            ET.SubElement(stats_elem, 'Max').text = str(np.max(pixels_array))
    
    # Add patterns
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
        
        patterns_elem = ET.SubElement(root, 'Patterns')
        
        if 'frequency' in patterns:
            for value, freq in list(patterns['frequency'].items())[:10]:  # Top 10
                pattern_elem = ET.SubElement(patterns_elem, 'Pattern')
                pattern_elem.set('value', str(value))
                pattern_elem.set('frequency', str(freq))
    
    # Add anomalies
    if os.path.exists('anomalies.json'):
        with open('anomalies.json', 'r') as f:
            anomalies = json.load(f)
        
        anomalies_elem = ET.SubElement(root, 'Anomalies')
        
        if 'z_score_anomalies' in anomalies:
            for anom in anomalies['z_score_anomalies'][:5]:  # First 5
                anom_elem = ET.SubElement(anomalies_elem, 'Anomaly')
                anom_elem.set('index', str(anom['index']))
                anom_elem.set('value', str(anom['value']))
                if 'z_score' in anom:
                    anom_elem.set('z_score', str(anom['z_score']))
    
    # Write XML
    tree = ET.ElementTree(root)
    tree.write('pixel_analysis_data.xml', encoding='utf-8', xml_declaration=True)
    
    print("[EXPORT] Exported data to pixel_analysis_data.xml")
    return 'pixel_analysis_data.xml'

def export_to_yaml():
    """Export configuration and results to YAML"""
    
    print("\n[EXPORT] Exporting data to YAML format...")
    
    export_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'version': '1.0',
            'modules': []
        },
        'results': {}
    }
    
    # Collect all results
    result_files = [
        'pixel_data.json',
        'patterns.json',
        'anomalies.json',
        'neural_results.json',
        'vision_results.json',
        'gpu_results.json',
        'ml_report.json'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            module_name = file.replace('.json', '')
            export_data['modules'].append(module_name)
            
            with open(file, 'r') as f:
                data = json.load(f)
            
            # Extract key information
            if module_name == 'pixel_data':
                export_data['results'][module_name] = {
                    'pixel_count': len(data.get('pixels', [])),
                    'size': data.get('size', [0, 0])
                }
            elif module_name == 'patterns':
                export_data['results'][module_name] = {
                    'pattern_count': len(data.get('frequency', {}))
                }
            elif module_name == 'neural_results':
                export_data['results'][module_name] = {
                    'training_loss': data.get('training_loss', 0),
                    'predictions_count': len(data.get('predictions', []))
                }
            else:
                # Generic summary
                export_data['results'][module_name] = {
                    'data_available': True,
                    'keys': list(data.keys())[:5]  # First 5 keys
                }
    
    # Write YAML
    with open('pixel_analysis_config.yaml', 'w') as f:
        yaml.dump(export_data, f, default_flow_style=False)
    
    print("[EXPORT] Exported configuration to pixel_analysis_config.yaml")
    return 'pixel_analysis_config.yaml'

def create_export_package():
    """Create a complete export package with all data"""
    
    print("\n[EXPORT] Creating complete export package...")
    
    # Create exports directory
    export_dir = f'pixel_analysis_export_{int(time.time())}'
    os.makedirs(export_dir, exist_ok=True)
    
    # Export to all formats
    csv_files = export_to_csv()
    hdf5_file = export_to_hdf5()
    xml_file = export_to_xml()
    yaml_file = export_to_yaml()
    
    # Copy all files to export directory
    import shutil
    
    # JSON files
    for file in os.listdir('.'):
        if file.endswith('.json'):
            shutil.copy(file, os.path.join(export_dir, file))
    
    # CSV files
    for file in csv_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(export_dir, file))
    
    # Other exports
    for file in [hdf5_file, xml_file, yaml_file]:
        if os.path.exists(file):
            shutil.move(file, os.path.join(export_dir, file))
    
    # Images
    image_count = 0
    for file in os.listdir('.'):
        if file.endswith(('.jpg', '.png', '.gif')):
            shutil.copy(file, os.path.join(export_dir, file))
            image_count += 1
            if image_count >= 10:  # Limit to 10 images
                break
    
    # Create README
    readme_content = f"""
Pixel Analysis Export Package
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents:
- JSON files: Original analysis results
- CSV files: Tabular data exports
- HDF5 file: Efficient binary storage
- XML file: Structured data export
- YAML file: Configuration and summary
- Images: Sample generated images

Usage:
- JSON: Use with any JSON parser
- CSV: Open in Excel or any spreadsheet software
- HDF5: Use with h5py in Python or HDF5 viewers
- XML: Parse with any XML library
- YAML: Human-readable configuration format

Generated by: Modular Pixel Analysis System
"""
    
    with open(os.path.join(export_dir, 'README.txt'), 'w') as f:
        f.write(readme_content)
    
    # Create ZIP archive
    zip_filename = f'{export_dir}.zip'
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, export_dir)
                zipf.write(file_path, arcname)
    
    # Clean up directory
    shutil.rmtree(export_dir)
    
    print(f"[EXPORT] Created export package: {zip_filename}")
    
    # Print package info
    zip_size = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"[EXPORT] Package size: {zip_size:.2f} MB")
    
    return zip_filename

def import_data(filename):
    """Import data from various formats"""
    
    print(f"\n[EXPORT] Importing data from {filename}...")
    
    if filename.endswith('.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"[EXPORT] Imported JSON data with {len(data)} keys")
        return data
        
    elif filename.endswith('.csv'):
        data = []
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"[EXPORT] Imported {len(data)} rows from CSV")
        return data
        
    elif filename.endswith('.h5'):
        data = {}
        with h5py.File(filename, 'r') as hf:
            def extract_group(group, prefix=''):
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        data[prefix + key] = group[key][()]
                    elif isinstance(group[key], h5py.Group):
                        extract_group(group[key], prefix + key + '/')
            
            extract_group(hf)
            
            # Extract attributes
            data['_attributes'] = dict(hf.attrs)
        
        print(f"[EXPORT] Imported HDF5 data with {len(data)} datasets")
        return data
        
    elif filename.endswith('.xml'):
        tree = ET.parse(filename)
        root = tree.getroot()
        
        data = {'_root': root.tag, '_attributes': root.attrib}
        
        def parse_element(elem, parent_dict):
            for child in elem:
                if len(child) == 0:  # Leaf node
                    parent_dict[child.tag] = {
                        'text': child.text,
                        'attributes': child.attrib
                    }
                else:
                    parent_dict[child.tag] = {'attributes': child.attrib}
                    parse_element(child, parent_dict[child.tag])
        
        parse_element(root, data)
        print(f"[EXPORT] Imported XML data")
        return data
        
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        print(f"[EXPORT] Imported YAML data")
        return data
        
    elif filename.endswith('.zip'):
        # Extract and import
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            print(f"[EXPORT] Extracted ZIP archive")
            
            # Import JSON files from archive
            imported_count = 0
            for file in os.listdir(temp_dir):
                if file.endswith('.json'):
                    src = os.path.join(temp_dir, file)
                    shutil.copy(src, file)
                    imported_count += 1
            
            print(f"[EXPORT] Imported {imported_count} JSON files from archive")
        
        return {'imported_files': imported_count}
    
    else:
        print(f"[EXPORT] Unknown file format: {filename}")
        return None

def main():
    """Main export/import functionality"""
    
    print("=== DATA EXPORT/IMPORT MODULE ===\n")
    
    # Create complete export package
    package_file = create_export_package()
    
    print("\n[EXPORT] Export complete!")
    print("[EXPORT] Created files:")
    print(f"  - {package_file} (complete package)")
    
    # Demonstrate import
    print("\n[EXPORT] Testing import functionality...")
    
    # Test importing different formats
    test_files = ['pixel_analysis_config.yaml']
    
    for test_file in test_files:
        if os.path.exists(test_file):
            data = import_data(test_file)
            if data:
                print(f"[EXPORT] Successfully imported {test_file}")

if __name__ == "__main__":
    main()