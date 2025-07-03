#!/usr/bin/env python3
"""
Enhanced Directory Deep Dive Analyzer with Neural Network Visualization
Analyzes directories, finds file relationships, and generates interactive reports.
"""

import os
import sys
import subprocess
import importlib.util

# Auto-install dependencies
def install_dependencies():
    """Check and install required dependencies automatically."""
    required_packages = {
        'networkx': 'networkx',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'jinja2': 'jinja2',
        'pyvis': 'pyvis'
    }
    
    print("Checking dependencies...")
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages detected: {', '.join(missing_packages)}")
        print("Installing missing dependencies...")
        
        for package in missing_packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}. Please install manually.")
                sys.exit(1)
        
        print("\nAll dependencies installed! Restarting script...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        print("✓ All dependencies are installed\n")

# Run dependency check
install_dependencies()

# Now import all required modules
import hashlib
import mimetypes
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import json
import re
import ast
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jinja2 import Template
from pyvis.network import Network
import fnmatch
import webbrowser

class EnhancedDirectoryAnalyzer:
    def __init__(self):
        self.root_path = None
        self.files_data = []
        self.file_hashes = defaultdict(list)
        self.file_graph = nx.DiGraph()
        self.file_connections = defaultdict(set)
        self.search_patterns = []
        self.date_filter = None
        self.stats = {
            'total_files': 0,
            'total_directories': 0,
            'total_size': 0,
            'file_types': Counter(),
            'extensions': Counter(),
            'duplicate_files': [],
            'largest_files': [],
            'oldest_files': [],
            'newest_files': [],
            'empty_directories': [],
            'hidden_files': 0,
            'python_scripts': 0,
            'pdfs': 0,
            'images': 0,
            'documents': 0,
            'videos': 0,
            'audio': 0,
            'archives': 0,
            'code_files': 0,
            'text_files': 0,
            'directory_depths': Counter(),
            'files_by_year': Counter(),
            'files_by_month': Counter(),
            'files_by_size_range': {
                '0-1KB': 0,
                '1KB-100KB': 0,
                '100KB-1MB': 0,
                '1MB-10MB': 0,
                '10MB-100MB': 0,
                '100MB-1GB': 0,
                '1GB+': 0
            },
            'pattern_matches': [],
            'date_filtered_files': [],
            'file_relationships': []
        }
    
    def get_user_inputs(self):
        """Interactive prompts for user configuration."""
        print("="*80)
        print("ENHANCED DIRECTORY ANALYZER WITH NEURAL NETWORK VISUALIZATION")
        print("="*80)
        
        # Get directory path
        while True:
            path = input("\nEnter the directory path to analyze (or press Enter for current directory): ").strip()
            if not path:
                path = "."
            
            if os.path.exists(path) and os.path.isdir(path):
                self.root_path = Path(path).resolve()
                print(f"✓ Will analyze: {self.root_path}")
                break
            else:
                print("✗ Invalid directory path. Please try again.")
        
        # Ask about file patterns
        print("\n" + "-"*40)
        if input("Do you want to search for specific file patterns? (y/n): ").lower() == 'y':
            print("Enter file patterns (e.g., *.log, test_*.py, *report*)")
            print("Press Enter with empty pattern when done.")
            while True:
                pattern = input("Pattern: ").strip()
                if not pattern:
                    break
                self.search_patterns.append(pattern)
            print(f"✓ Will search for: {', '.join(self.search_patterns)}")
        
        # Ask about date filtering
        print("\n" + "-"*40)
        if input("Do you want to filter files by modification date? (y/n): ").lower() == 'y':
            print("\nDate filter options:")
            print("1. Last N days")
            print("2. Specific date range")
            print("3. Files older than N days")
            
            choice = input("Choose option (1-3): ").strip()
            
            if choice == '1':
                days = int(input("Enter number of days: "))
                self.date_filter = {
                    'type': 'last_n_days',
                    'days': days,
                    'start_date': datetime.now() - timedelta(days=days),
                    'end_date': datetime.now()
                }
            elif choice == '2':
                start = input("Enter start date (YYYY-MM-DD): ")
                end = input("Enter end date (YYYY-MM-DD): ")
                self.date_filter = {
                    'type': 'range',
                    'start_date': datetime.strptime(start, '%Y-%m-%d'),
                    'end_date': datetime.strptime(end, '%Y-%m-%d')
                }
            elif choice == '3':
                days = int(input("Enter number of days: "))
                self.date_filter = {
                    'type': 'older_than',
                    'days': days,
                    'cutoff_date': datetime.now() - timedelta(days=days)
                }
        
        print("\n" + "-"*40)
        print("Configuration complete! Starting analysis...")
        print("Note: Complete analysis of all files may take significant time for large directories.")
        print("-"*40)
    
    def calculate_file_hash(self, filepath, chunk_size=8192):
        """Calculate MD5 hash of a file for duplicate detection."""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def extract_imports_from_python(self, filepath):
        """Extract import statements from Python files."""
        imports = set()
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
        except:
            pass
        return imports
    
    def find_file_references(self, filepath):
        """Find references to other files within text-based files."""
        references = set()
        extensions_to_check = {'.py', '.js', '.html', '.css', '.cpp', '.java', '.md', '.txt', '.yml', '.yaml', '.json'}
        
        if filepath.suffix.lower() not in extensions_to_check:
            return references
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Find file paths in various formats
            patterns = [
                r'(?:import|from|require|include)\s+["\']([^"\']+)["\']',  # Import statements
                r'(?:src|href)=["\']([^"\']+)["\']',  # HTML references
                r'["\']([A-Za-z0-9_\-./]+\.[A-Za-z0-9]+)["\']',  # General file references
                r'open\(["\']([^"\']+)["\']',  # File open statements
                r'read_file\(["\']([^"\']+)["\']',  # Common file reading patterns
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if '.' in match and not match.startswith('http'):
                        references.add(match)
        except:
            pass
        
        return references
    
    def build_file_relationship_graph(self):
        """Build a graph of file relationships and dependencies."""
        print("\nBuilding file relationship neural network...")
        
        # Add all files as nodes
        print(f"Adding {len(self.files_data)} nodes to graph...")
        for i, file_info in enumerate(self.files_data):
            filepath = Path(file_info['path'])
            self.file_graph.add_node(file_info['path'], 
                                   name=file_info['name'],
                                   type=file_info.get('category', 'other'),
                                   size=file_info['size'])
            if (i + 1) % 1000 == 0:
                print(f"  Added {i + 1} nodes...", end='\r')
        print(f"  ✓ Added {len(self.files_data)} nodes")
        
        # Build relationships
        print("Analyzing file relationships...")
        code_files = [f for f in self.files_data if f.get('category') == 'code' or f['extension'] == '.py']
        print(f"  Found {len(code_files)} code files to analyze")
        
        processed = 0
        total_connections = 0
        
        for i, file_info in enumerate(self.files_data):
            filepath = Path(file_info['path'])
            
            # Python imports
            if filepath.suffix == '.py':
                imports = self.extract_imports_from_python(filepath)
                for imp in imports:
                    # Find matching Python files
                    for other_file in self.files_data:
                        if other_file['path'] != file_info['path']:
                            if imp in other_file['name'] and other_file['name'].endswith('.py'):
                                self.file_graph.add_edge(file_info['path'], other_file['path'], 
                                                       relationship='imports')
                                self.file_connections[file_info['path']].add(other_file['path'])
                                total_connections += 1
            
            # File references (for all text-based files)
            if file_info.get('category') in ['code', 'document'] or filepath.suffix in ['.html', '.css', '.js', '.md']:
                references = self.find_file_references(filepath)
                for ref in references:
                    # Try to find the referenced file
                    ref_path = filepath.parent / ref
                    for other_file in self.files_data:
                        if Path(other_file['path']).name == Path(ref).name:
                            self.file_graph.add_edge(file_info['path'], other_file['path'], 
                                                   relationship='references')
                            self.file_connections[file_info['path']].add(other_file['path'])
                            total_connections += 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(self.files_data)} files, found {total_connections} connections...", end='\r')
        
        print(f"\n  ✓ Found {total_connections} direct file connections")
        
        # Same directory relationships
        print("  Adding same-directory relationships...")
        dir_groups = defaultdict(list)
        for file_info in self.files_data:
            dir_path = str(Path(file_info['path']).parent)
            dir_groups[dir_path].append(file_info['path'])
        
        same_dir_connections = 0
        for dir_path, files in dir_groups.items():
            if len(files) > 1:
                for i, file1 in enumerate(files):
                    for file2 in files[i+1:]:
                        self.file_graph.add_edge(file1, file2, 
                                               relationship='same_directory', weight=0.1)
                        self.file_graph.add_edge(file2, file1, 
                                               relationship='same_directory', weight=0.1)
                        same_dir_connections += 2
        
        print(f"  ✓ Added {same_dir_connections} same-directory relationships")
        print("✓ Neural network building complete!")
    
    def get_file_category(self, extension):
        """Categorize files based on extension."""
        ext = extension.lower()
        
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', 
                          '.go', '.rs', '.swift', '.kt', '.php', '.r', '.m', '.sh', '.bat'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico', '.webp'}
        document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.odt', '.rtf', '.tex'}
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a'}
        archive_extensions = {'.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'}
        
        if ext in code_extensions:
            return 'code'
        elif ext in image_extensions:
            return 'image'
        elif ext in document_extensions:
            return 'document'
        elif ext in video_extensions:
            return 'video'
        elif ext in audio_extensions:
            return 'audio'
        elif ext in archive_extensions:
            return 'archive'
        else:
            return 'other'
    
    def get_size_range(self, size):
        """Categorize file size into ranges."""
        if size < 1024:
            return '0-1KB'
        elif size < 102400:
            return '1KB-100KB'
        elif size < 1048576:
            return '100KB-1MB'
        elif size < 10485760:
            return '1MB-10MB'
        elif size < 104857600:
            return '10MB-100MB'
        elif size < 1073741824:
            return '100MB-1GB'
        else:
            return '1GB+'
    
    def check_date_filter(self, file_stat):
        """Check if file matches date filter criteria."""
        if not self.date_filter:
            return True
        
        mod_time = datetime.fromtimestamp(file_stat.st_mtime)
        
        if self.date_filter['type'] == 'last_n_days':
            return mod_time >= self.date_filter['start_date']
        elif self.date_filter['type'] == 'range':
            return self.date_filter['start_date'] <= mod_time <= self.date_filter['end_date']
        elif self.date_filter['type'] == 'older_than':
            return mod_time < self.date_filter['cutoff_date']
        
        return True
    
    def check_pattern_match(self, filename):
        """Check if filename matches any search patterns."""
        if not self.search_patterns:
            return True
        
        for pattern in self.search_patterns:
            if fnmatch.fnmatch(filename.lower(), pattern.lower()):
                return True
        return False
    
    def analyze_directory(self):
        """Main analysis function that walks through all directories."""
        print(f"\nAnalyzing directory: {self.root_path}")
        print("This may take a while for large directory structures...\n")
        
        # Progress tracking
        file_count = 0
        
        # Walk through directory tree
        for root, dirs, files in os.walk(self.root_path):
            # Calculate directory depth
            depth = len(Path(root).relative_to(self.root_path).parts)
            self.stats['directory_depths'][depth] += 1
            
            # Check for empty directories
            if not dirs and not files:
                self.stats['empty_directories'].append(root)
            
            self.stats['total_directories'] += 1
            
            # Analyze each file
            for filename in files:
                filepath = Path(root) / filename
                if self.analyze_file(filepath):
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"Processed {file_count} files...", end='\r')
        
        print(f"\nProcessed {file_count} files total.")
        
        # Post-processing
        self.find_duplicates()
        self.sort_files_by_size()
        self.sort_files_by_date()
        self.build_file_relationship_graph()
    
    def analyze_file(self, filepath):
        """Analyze individual file and collect statistics."""
        try:
            file_stat = filepath.stat()
            
            # Check pattern match
            if not self.check_pattern_match(filepath.name):
                return False
            
            # Check date filter
            if not self.check_date_filter(file_stat):
                return False
            
            file_info = {
                'path': str(filepath),
                'name': filepath.name,
                'size': file_stat.st_size,
                'modified': file_stat.st_mtime,
                'created': file_stat.st_ctime,
                'extension': filepath.suffix.lower(),
                'category': self.get_file_category(filepath.suffix.lower())
            }
            
            # Track pattern matches
            if self.search_patterns and self.check_pattern_match(filepath.name):
                self.stats['pattern_matches'].append(file_info)
            
            # Track date filtered files
            if self.date_filter:
                self.stats['date_filtered_files'].append(file_info)
            
            # Update basic stats
            self.stats['total_files'] += 1
            self.stats['total_size'] += file_info['size']
            self.stats['extensions'][file_info['extension']] += 1
            
            # Check if hidden file
            if filepath.name.startswith('.'):
                self.stats['hidden_files'] += 1
            
            # Categorize file
            self.stats['file_types'][file_info['category']] += 1
            
            # Count specific file types
            if file_info['extension'] == '.py':
                self.stats['python_scripts'] += 1
            elif file_info['extension'] == '.pdf':
                self.stats['pdfs'] += 1
            elif file_info['category'] == 'image':
                self.stats['images'] += 1
            elif file_info['category'] == 'document':
                self.stats['documents'] += 1
            elif file_info['category'] == 'video':
                self.stats['videos'] += 1
            elif file_info['category'] == 'audio':
                self.stats['audio'] += 1
            elif file_info['category'] == 'archive':
                self.stats['archives'] += 1
            elif file_info['category'] == 'code':
                self.stats['code_files'] += 1
            elif file_info['extension'] in ['.txt', '.md', '.log']:
                self.stats['text_files'] += 1
            
            # Size range
            size_range = self.get_size_range(file_info['size'])
            self.stats['files_by_size_range'][size_range] += 1
            
            # Time-based statistics
            created_date = datetime.fromtimestamp(file_info['created'])
            self.stats['files_by_year'][created_date.year] += 1
            self.stats['files_by_month'][created_date.strftime('%Y-%m')] += 1
            
            # Calculate hash for duplicate detection
            file_hash = self.calculate_file_hash(filepath)
            if file_hash:
                file_info['hash'] = file_hash
                self.file_hashes[file_hash].append(file_info)
            
            self.files_data.append(file_info)
            return True
            
        except Exception as e:
            print(f"\nError analyzing file {filepath}: {e}")
            return False
    
    def find_duplicates(self):
        """Identify duplicate files based on hash."""
        for hash_value, files in self.file_hashes.items():
            if len(files) > 1:
                self.stats['duplicate_files'].append({
                    'hash': hash_value,
                    'files': [f['path'] for f in files],
                    'size': files[0]['size'],
                    'count': len(files)
                })
    
    def sort_files_by_size(self):
        """Find largest files."""
        sorted_files = sorted(self.files_data, key=lambda x: x['size'], reverse=True)
        self.stats['largest_files'] = sorted_files[:20]
    
    def sort_files_by_date(self):
        """Find oldest and newest files."""
        sorted_by_created = sorted(self.files_data, key=lambda x: x['created'])
        self.stats['oldest_files'] = sorted_by_created[:10]
        self.stats['newest_files'] = sorted_by_created[-10:][::-1]
    
    def format_size(self, size):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"
    
    def generate_neural_network_visualization(self):
        """Generate interactive neural network visualization of file relationships."""
        print("\nGenerating neural network visualization...")
        
        # Create interactive network using pyvis
        net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white')
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
        
        # Add nodes with different colors for different file types
        color_map = {
            'code': '#3498db',      # Blue
            'document': '#e74c3c',  # Red
            'image': '#2ecc71',     # Green
            'video': '#f39c12',     # Orange
            'audio': '#9b59b6',     # Purple
            'archive': '#34495e',   # Dark gray
            'other': '#95a5a6'      # Light gray
        }
        
        # Only show files with connections or important files
        nodes_to_show = set()
        for edge in self.file_graph.edges():
            nodes_to_show.add(edge[0])
            nodes_to_show.add(edge[1])
        
        # Also add largest files
        for file_info in self.stats['largest_files'][:10]:
            nodes_to_show.add(file_info['path'])
        
        # Add nodes
        for node in nodes_to_show:
            if node in self.file_graph.nodes:
                node_data = self.file_graph.nodes[node]
                net.add_node(node, 
                           label=node_data['name'],
                           title=f"{node_data['name']}\nType: {node_data['type']}\nSize: {self.format_size(node_data['size'])}",
                           color=color_map.get(node_data['type'], '#95a5a6'),
                           size=20 + min(node_data['size'] / 1000000, 30))  # Size based on file size
        
        # Add edges
        for edge in self.file_graph.edges(data=True):
            if edge[0] in nodes_to_show and edge[1] in nodes_to_show:
                if edge[2].get('relationship') == 'imports':
                    net.add_edge(edge[0], edge[1], color='#e74c3c', width=3, title='imports')
                elif edge[2].get('relationship') == 'references':
                    net.add_edge(edge[0], edge[1], color='#3498db', width=2, title='references')
                elif edge[2].get('relationship') == 'same_directory':
                    net.add_edge(edge[0], edge[1], color='#95a5a6', width=0.5, title='same directory')
        
        # Save network
        net.save_graph('file_network.html')
        print("✓ Neural network visualization saved as 'file_network.html'")
    
    def generate_visualizations(self):
        """Generate various data visualizations using Plotly."""
        print("\nGenerating visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('File Type Distribution', 'Size Distribution', 
                          'Files by Year', 'Top Extensions',
                          'Directory Depth', 'Monthly File Creation'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. File type pie chart
        if self.stats['file_types']:
            fig.add_trace(
                go.Pie(labels=list(self.stats['file_types'].keys()),
                      values=list(self.stats['file_types'].values())),
                row=1, col=1
            )
        
        # 2. Size distribution bar chart
        size_labels = list(self.stats['files_by_size_range'].keys())
        size_values = list(self.stats['files_by_size_range'].values())
        fig.add_trace(
            go.Bar(x=size_labels, y=size_values, name='File Count'),
            row=1, col=2
        )
        
        # 3. Files by year
        years = sorted(self.stats['files_by_year'].keys())[-10:]  # Last 10 years
        year_counts = [self.stats['files_by_year'][y] for y in years]
        fig.add_trace(
            go.Bar(x=years, y=year_counts, name='Files'),
            row=2, col=1
        )
        
        # 4. Top extensions
        top_ext = self.stats['extensions'].most_common(10)
        ext_names = [e[0] if e[0] else 'No ext' for e in top_ext]
        ext_counts = [e[1] for e in top_ext]
        fig.add_trace(
            go.Bar(x=ext_names, y=ext_counts, name='Count'),
            row=2, col=2
        )
        
        # 5. Directory depth
        depths = sorted(self.stats['directory_depths'].keys())
        depth_counts = [self.stats['directory_depths'][d] for d in depths]
        fig.add_trace(
            go.Bar(x=depths, y=depth_counts, name='Directories'),
            row=3, col=1
        )
        
        # 6. Monthly file creation (last 12 months)
        months = sorted(self.stats['files_by_month'].keys())[-12:]
        month_counts = [self.stats['files_by_month'][m] for m in months]
        fig.add_trace(
            go.Scatter(x=months, y=month_counts, mode='lines+markers', name='Files'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=False, 
                         title_text="Directory Analysis Visualizations")
        
        # Save visualization
        fig.write_html('visualizations.html')
        print("✓ Visualizations saved as 'visualizations.html'")
    
    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        print("\nGenerating HTML report...")
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Directory Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2c3e50; margin-top: 30px; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        .duplicate-group { background: #ffe6e6; margin: 10px 0; padding: 10px; border-radius: 5px; }
        .file-path { font-family: monospace; font-size: 0.9em; color: #555; }
        .button-group { margin: 20px 0; }
        .button { display: inline-block; padding: 10px 20px; margin: 5px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; }
        .button:hover { background: #2980b9; }
        .pattern-match { background: #e8f5e9; padding: 5px; border-radius: 3px; }
        .date-filtered { background: #fff3e0; padding: 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Directory Analysis Report</h1>
        <p><strong>Analyzed Path:</strong> {{ root_path }}</p>
        <p><strong>Analysis Date:</strong> {{ analysis_date }}</p>
        
        <div class="button-group">
            <a href="visualizations.html" class="button">View Visualizations</a>
            <a href="file_network.html" class="button">View File Network</a>
            <a href="directory_analysis.json" class="button">Download JSON Data</a>
        </div>
        
        <h2>Summary Statistics</h2>
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{{ total_files|numberformat }}</div>
                <div class="stat-label">Total Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_directories|numberformat }}</div>
                <div class="stat-label">Total Directories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_size }}</div>
                <div class="stat-label">Total Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ duplicate_count|numberformat }}</div>
                <div class="stat-label">Duplicate Files</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ wasted_space }}</div>
                <div class="stat-label">Wasted Space</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ python_scripts|numberformat }}</div>
                <div class="stat-label">Python Scripts</div>
            </div>
        </div>
        
        {% if pattern_matches %}
        <h2>Pattern Match Results</h2>
        <p>Found {{ pattern_matches|length }} files matching patterns: {{ search_patterns|join(', ') }}</p>
        <table>
            <tr><th>File</th><th>Size</th><th>Modified</th></tr>
            {% for file in pattern_matches[:20] %}
            <tr>
                <td class="pattern-match">{{ file.name }}</td>
                <td>{{ file.size_formatted }}</td>
                <td>{{ file.modified_date }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        {% if date_filtered_files %}
        <h2>Date Filtered Files</h2>
        <p>Found {{ date_filtered_files|length }} files matching date criteria</p>
        <table>
            <tr><th>File</th><th>Size</th><th>Modified</th></tr>
            {% for file in date_filtered_files[:20] %}
            <tr>
                <td class="date-filtered">{{ file.name }}</td>
                <td>{{ file.size_formatted }}</td>
                <td>{{ file.modified_date }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <h2>Largest Files</h2>
        <table>
            <tr><th>File</th><th>Size</th><th>Path</th></tr>
            {% for file in largest_files %}
            <tr>
                <td>{{ file.name }}</td>
                <td>{{ file.size_formatted }}</td>
                <td class="file-path">{{ file.path }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>Duplicate Files</h2>
        <p>Total space wasted by duplicates: {{ wasted_space }}</p>
        {% for dup in duplicate_files[:10] %}
        <div class="duplicate-group">
            <strong>{{ dup.count }} copies</strong> ({{ dup.size_formatted }} each, {{ dup.wasted_formatted }} wasted)
            <ul>
            {% for path in dup.files[:3] %}
                <li class="file-path">{{ path }}</li>
            {% endfor %}
            {% if dup.files|length > 3 %}
                <li>... and {{ dup.files|length - 3 }} more</li>
            {% endif %}
            </ul>
        </div>
        {% endfor %}
        
        <h2>File Type Distribution</h2>
        <table>
            <tr><th>Extension</th><th>Count</th><th>Total Size</th></tr>
            {% for ext in top_extensions %}
            <tr>
                <td>{{ ext.name }}</td>
                <td>{{ ext.count|numberformat }}</td>
                <td>{{ ext.size }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>File Connections</h2>
        <p>Found {{ total_connections }} file relationships</p>
        <p>Files with most connections:</p>
        <table>
            <tr><th>File</th><th>Connections</th><th>Type</th></tr>
            {% for conn in top_connected_files %}
            <tr>
                <td>{{ conn.name }}</td>
                <td>{{ conn.connection_count }}</td>
                <td>{{ conn.type }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
        """
        
        # Prepare template data
        total_duplicate_size = sum(d['size'] * (d['count'] - 1) for d in self.stats['duplicate_files'])
        
        # Get file connections
        connection_counts = Counter()
        for source, targets in self.file_connections.items():
            connection_counts[source] = len(targets)
        
        top_connected = []
        for path, count in connection_counts.most_common(10):
            file_info = next((f for f in self.files_data if f['path'] == path), None)
            if file_info:
                top_connected.append({
                    'name': file_info['name'],
                    'connection_count': count,
                    'type': file_info['category']
                })
        
        # Format data for template
        template_data = {
            'root_path': str(self.root_path),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': self.stats['total_files'],
            'total_directories': self.stats['total_directories'],
            'total_size': self.format_size(self.stats['total_size']),
            'duplicate_count': len(self.stats['duplicate_files']),
            'wasted_space': self.format_size(total_duplicate_size),
            'python_scripts': self.stats['python_scripts'],
            'search_patterns': self.search_patterns,
            'pattern_matches': [{
                'name': f['name'],
                'size_formatted': self.format_size(f['size']),
                'modified_date': datetime.fromtimestamp(f['modified']).strftime('%Y-%m-%d %H:%M')
            } for f in self.stats['pattern_matches']],
            'date_filtered_files': [{
                'name': f['name'],
                'size_formatted': self.format_size(f['size']),
                'modified_date': datetime.fromtimestamp(f['modified']).strftime('%Y-%m-%d %H:%M')
            } for f in self.stats['date_filtered_files']],
            'largest_files': [{
                'name': f['name'],
                'path': f['path'],
                'size_formatted': self.format_size(f['size'])
            } for f in self.stats['largest_files'][:10]],
            'duplicate_files': [{
                'count': d['count'],
                'size_formatted': self.format_size(d['size']),
                'wasted_formatted': self.format_size(d['size'] * (d['count'] - 1)),
                'files': d['files']
            } for d in sorted(self.stats['duplicate_files'], 
                            key=lambda x: x['size'] * (x['count'] - 1), 
                            reverse=True)],
            'top_extensions': [{
                'name': ext[0] if ext[0] else 'No extension',
                'count': ext[1],
                'size': self.format_size(sum(f['size'] for f in self.files_data if f['extension'] == ext[0]))
            } for ext in self.stats['extensions'].most_common(15)],
            'total_connections': sum(len(targets) for targets in self.file_connections.values()),
            'top_connected_files': top_connected
        }
        
        # Add custom filter
        def numberformat(value):
            return f"{value:,}"
        
        # Render template
        template = Template(html_template)
        template.globals['numberformat'] = numberformat
        html_content = template.render(**template_data)
        
        # Save report
        with open('directory_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✓ HTML report saved as 'directory_report.html'")
    
    def generate_report(self):
        """Generate and display analysis report."""
        print("\n" + "="*80)
        print("DIRECTORY ANALYSIS COMPLETE")
        print("="*80)
        
        print("\nGenerating output files...")
        
        # Generate visualizations
        print("  1/4 Creating interactive charts...", end='')
        self.generate_visualizations()
        print(" ✓")
        
        print("  2/4 Creating neural network visualization...", end='')
        self.generate_neural_network_visualization()
        print(" ✓")
        
        print("  3/4 Creating HTML report...", end='')
        self.generate_html_report()
        print(" ✓")
        
        # Export JSON data
        print("  4/4 Exporting raw data...", end='')
        self.export_json()
        print(" ✓")
        
        # Print summary
        print(f"\nAnalyzed Path: {self.root_path}")
        print(f"Total Files: {self.stats['total_files']:,}")
        print(f"Total Size: {self.format_size(self.stats['total_size'])}")
        print(f"Duplicate Files: {len(self.stats['duplicate_files'])}")
        
        if self.search_patterns:
            print(f"\nPattern Matches: {len(self.stats['pattern_matches'])} files")
        
        if self.date_filter:
            print(f"Date Filtered Files: {len(self.stats['date_filtered_files'])} files")
        
        print(f"\nFile Connections: {sum(len(targets) for targets in self.file_connections.values())} relationships found")
        
        print("\n" + "="*80)
        print("GENERATED FILES:")
        print("="*80)
        print("1. directory_report.html    - Main analysis report")
        print("2. visualizations.html      - Interactive charts")
        print("3. file_network.html        - Neural network visualization")
        print("4. directory_analysis.json  - Raw data export")
        
        # Ask if user wants to open the report
        if input("\nWould you like to open the HTML report in your browser? (y/n): ").lower() == 'y':
            webbrowser.open('directory_report.html')
    
    def export_json(self, output_file='directory_analysis.json'):
        """Export detailed results to JSON file."""
        export_data = {
            'analysis_date': datetime.now().isoformat(),
            'root_path': str(self.root_path),
            'search_patterns': self.search_patterns,
            'date_filter': self.date_filter,
            'statistics': {
                'total_files': self.stats['total_files'],
                'total_directories': self.stats['total_directories'],
                'total_size': self.stats['total_size'],
                'total_size_formatted': self.format_size(self.stats['total_size']),
                'file_types': dict(self.stats['file_types']),
                'extensions': dict(self.stats['extensions'].most_common(50)),
                'duplicate_count': len(self.stats['duplicate_files']),
                'empty_directories_count': len(self.stats['empty_directories']),
                'file_connections': len(self.file_connections)
            },
            'duplicates': self.stats['duplicate_files'][:100],
            'largest_files': [{
                'path': f['path'],
                'size': f['size'],
                'size_formatted': self.format_size(f['size'])
            } for f in self.stats['largest_files'][:50]],
            'pattern_matches': self.stats['pattern_matches'][:100] if self.search_patterns else [],
            'file_relationships': [
                {
                    'source': source,
                    'targets': list(targets)
                } for source, targets in list(self.file_connections.items())[:100]
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Raw data exported to {output_file}")


def main():
    analyzer = EnhancedDirectoryAnalyzer()
    analyzer.get_user_inputs()
    analyzer.analyze_directory()
    analyzer.generate_report()


if __name__ == "__main__":
    main()
