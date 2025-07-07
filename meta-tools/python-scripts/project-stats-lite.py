#!/usr/bin/env python3
"""Enhanced Directory Analyzer - Minimal but Robust Version"""

import os, sys, subprocess, importlib.util, hashlib, re, ast, json, webbrowser, fnmatch
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Auto-install dependencies
def ensure_deps():
    deps = {'networkx': 'networkx', 'matplotlib': 'matplotlib', 'plotly': 'plotly',
            'pandas': 'pandas', 'numpy': 'numpy', 'jinja2': 'jinja2', 'pyvis': 'pyvis'}
    missing = [pkg for mod, pkg in deps.items() if not importlib.util.find_spec(mod)]
    if missing:
        print(f"Installing: {', '.join(missing)}...")
        for pkg in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_deps()

import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
from jinja2 import Template

class DirectoryAnalyzer:
    def __init__(self):
        self.root_path = None
        self.files = []
        self.hashes = defaultdict(list)
        self.graph = nx.DiGraph()
        self.connections = defaultdict(set)
        self.patterns = []
        self.date_filter = None
        self.stats = defaultdict(lambda: 0)
        self.lists = defaultdict(list)
        self.counters = defaultdict(Counter)
        
    def setup(self):
        """Get user configuration"""
        print("="*60 + "\nDIRECTORY ANALYZER\n" + "="*60)
        
        # Directory
        path = input("\nDirectory to analyze (Enter for current): ").strip() or "."
        self.root_path = Path(path).resolve()
        if not self.root_path.is_dir():
            print("Invalid directory!"); sys.exit(1)
        print(f"✓ Analyzing: {self.root_path}")
        
        # Patterns
        if input("\nSearch patterns? (y/n): ").lower() == 'y':
            print("Enter patterns (empty to finish):")
            while p := input("Pattern: ").strip():
                self.patterns.append(p)
        
        # Date filter
        if input("\nFilter by date? (y/n): ").lower() == 'y':
            print("1. Last N days\n2. Date range\n3. Older than N days")
            choice = input("Choice (1-3): ")
            if choice == '1':
                days = int(input("Days: "))
                self.date_filter = lambda t: t >= datetime.now() - timedelta(days=days)
            elif choice == '2':
                start = datetime.strptime(input("Start (YYYY-MM-DD): "), '%Y-%m-%d')
                end = datetime.strptime(input("End (YYYY-MM-DD): "), '%Y-%m-%d')
                self.date_filter = lambda t: start <= t <= end
            elif choice == '3':
                days = int(input("Days: "))
                self.date_filter = lambda t: t < datetime.now() - timedelta(days=days)
    
    def analyze(self):
        """Main analysis"""
        print("\nAnalyzing...")
        count = 0
        
        for root, dirs, files in os.walk(self.root_path):
            self.stats['dirs'] += 1
            if not dirs and not files:
                self.lists['empty_dirs'].append(root)
            
            for name in files:
                path = Path(root) / name
                if self._process_file(path):
                    count += 1
                    print(f"\rProcessed {count} files...", end='')
        
        print(f"\n✓ Processed {count} files")
        self._find_duplicates()
        self._build_relationships()
    
    def _process_file(self, path):
        """Process single file"""
        try:
            # Check patterns
            if self.patterns and not any(fnmatch.fnmatch(path.name.lower(), p.lower()) for p in self.patterns):
                return False
            
            # Get stats
            stat = path.stat()
            
            # Check date
            if self.date_filter and not self.date_filter(datetime.fromtimestamp(stat.st_mtime)):
                return False
            
            # File info
            ext = path.suffix.lower()
            cat = self._categorize(ext)
            info = {
                'path': str(path), 'name': path.name, 'size': stat.st_size,
                'modified': stat.st_mtime, 'ext': ext, 'cat': cat
            }
            
            # Stats
            self.stats['files'] += 1
            self.stats['size'] += stat.st_size
            self.counters['exts'][ext] += 1
            self.counters['cats'][cat] += 1
            self.counters['years'][datetime.fromtimestamp(stat.st_mtime).year] += 1
            
            # Size ranges
            sizes = [(1024, '0-1KB'), (102400, '1-100KB'), (1048576, '100KB-1MB'),
                    (10485760, '1-10MB'), (104857600, '10-100MB'), (1073741824, '100MB-1GB')]
            for limit, label in sizes:
                if stat.st_size < limit:
                    self.counters['sizes'][label] += 1
                    break
            else:
                self.counters['sizes']['1GB+'] += 1
            
            # Hash for duplicates
            if stat.st_size < 100*1024*1024:  # Only hash files < 100MB
                h = hashlib.md5()
                with open(path, 'rb') as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
                info['hash'] = h.hexdigest()
                self.hashes[info['hash']].append(info)
            
            self.files.append(info)
            
            # Track patterns/date matches
            if self.patterns:
                self.lists['pattern_matches'].append(info)
            if self.date_filter:
                self.lists['date_matches'].append(info)
                
            return True
        except Exception as e:
            print(f"\nError with {path}: {e}")
            return False
    
    def _categorize(self, ext):
        """Categorize by extension"""
        cats = {
            'code': {'.py','.js','.java','.cpp','.c','.h','.cs','.go','.rs','.php'},
            'image': {'.jpg','.jpeg','.png','.gif','.bmp','.svg','.webp'},
            'doc': {'.pdf','.doc','.docx','.txt','.md','.rtf'},
            'video': {'.mp4','.avi','.mkv','.mov','.wmv','.webm'},
            'audio': {'.mp3','.wav','.flac','.aac','.ogg','.m4a'},
            'archive': {'.zip','.rar','.7z','.tar','.gz','.bz2'}
        }
        for cat, exts in cats.items():
            if ext in exts:
                return cat
        return 'other'
    
    def _find_duplicates(self):
        """Find duplicate files"""
        for h, files in self.hashes.items():
            if len(files) > 1:
                self.lists['duplicates'].append({
                    'hash': h, 'files': files, 'count': len(files),
                    'size': files[0]['size'], 'waste': files[0]['size'] * (len(files)-1)
                })
    
    def _build_relationships(self):
        """Build file relationship graph"""
        print("Building relationships...")
        
        # Add nodes
        for f in self.files:
            self.graph.add_node(f['path'], **f)
        
        # Find connections
        code_files = [f for f in self.files if f['cat'] == 'code' or f['ext'] == '.py']
        
        for f in code_files:
            path = Path(f['path'])
            
            # Python imports
            if f['ext'] == '.py':
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                        tree = ast.parse(file.read())
                    
                    imports = set()
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.update(n.name.split('.')[0] for n in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            imports.add(node.module.split('.')[0])
                    
                    # Find matching files
                    for imp in imports:
                        for other in self.files:
                            if other['path'] != f['path'] and imp in other['name']:
                                self.graph.add_edge(f['path'], other['path'], type='import')
                                self.connections[f['path']].add(other['path'])
                except:
                    pass
            
            # File references
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                    content = file.read()
                
                # Find references
                refs = re.findall(r'["\']([A-Za-z0-9_\-./]+\.[A-Za-z0-9]+)["\']', content)
                for ref in refs:
                    if '.' in ref and not ref.startswith('http'):
                        for other in self.files:
                            if Path(other['path']).name == Path(ref).name:
                                self.graph.add_edge(f['path'], other['path'], type='reference')
                                self.connections[f['path']].add(other['path'])
            except:
                pass
        
        print(f"✓ Found {sum(len(v) for v in self.connections.values())} connections")
    
    def generate_outputs(self):
        """Generate all outputs"""
        print("\nGenerating outputs...")
        self._visualizations()
        self._network()
        self._html_report()
        self._json_export()
        
        print("\n" + "="*60)
        print("GENERATED FILES:")
        print("="*60)
        print("1. report.html         - Main report")
        print("2. charts.html         - Visualizations")
        print("3. network.html        - File network")
        print("4. analysis.json       - Raw data")
        
        if input("\nOpen report? (y/n): ").lower() == 'y':
            webbrowser.open('report.html')
    
    def _visualizations(self):
        """Create charts"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('File Types', 'Size Distribution', 'Top Extensions', 'Files by Year'),
            specs=[[{'type':'pie'}, {'type':'bar'}], [{'type':'bar'}, {'type':'bar'}]]
        )
        
        # File types pie
        if self.counters['cats']:
            fig.add_trace(go.Pie(labels=list(self.counters['cats'].keys()),
                                values=list(self.counters['cats'].values())), row=1, col=1)
        
        # Size distribution
        fig.add_trace(go.Bar(x=list(self.counters['sizes'].keys()),
                            y=list(self.counters['sizes'].values())), row=1, col=2)
        
        # Top extensions
        top_ext = self.counters['exts'].most_common(10)
        fig.add_trace(go.Bar(x=[e[0] or 'none' for e in top_ext],
                            y=[e[1] for e in top_ext]), row=2, col=1)
        
        # Files by year
        years = sorted(self.counters['years'].keys())[-10:]
        fig.add_trace(go.Bar(x=years, y=[self.counters['years'][y] for y in years]), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        fig.write_html('charts.html')
        print("✓ Created charts.html")
    
    def _network(self):
        """Create network visualization"""
        net = Network(height='800px', width='100%', bgcolor='#222', font_color='white')
        net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=250)
        
        # Get connected nodes
        nodes = set()
        for edge in self.graph.edges():
            nodes.update(edge)
        
        # Add top 10 largest files too
        for f in sorted(self.files, key=lambda x: x['size'], reverse=True)[:10]:
            nodes.add(f['path'])
        
        # Colors by type
        colors = {'code':'#3498db', 'doc':'#e74c3c', 'image':'#2ecc71',
                 'video':'#f39c12', 'audio':'#9b59b6', 'archive':'#34495e', 'other':'#95a5a6'}
        
        # Add nodes
        for node in nodes:
            if node in self.graph.nodes:
                data = self.graph.nodes[node]
                net.add_node(node, label=data['name'], color=colors.get(data['cat'], '#95a5a6'),
                           title=f"{data['name']}\n{self._format_size(data['size'])}")
        
        # Add edges
        for a, b, data in self.graph.edges(data=True):
            if a in nodes and b in nodes:
                net.add_edge(a, b, color='#e74c3c' if data.get('type')=='import' else '#3498db')
        
        net.save_graph('network.html')
        print("✓ Created network.html")
    
    def _html_report(self):
        """Generate HTML report"""
        template = """
<!DOCTYPE html>
<html>
<head>
    <title>Directory Analysis</title>
    <style>
        body {font-family:Arial,sans-serif; margin:20px; background:#f5f5f5;}
        .container {max-width:1200px; margin:0 auto; background:white; padding:20px; box-shadow:0 0 10px rgba(0,0,0,0.1);}
        h1 {color:#333; border-bottom:2px solid #3498db; padding-bottom:10px;}
        h2 {color:#2c3e50; margin-top:30px;}
        .stats {display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:20px; margin:20px 0;}
        .stat {background:#ecf0f1; padding:20px; border-radius:8px; text-align:center;}
        .stat-value {font-size:2em; font-weight:bold; color:#3498db;}
        .stat-label {color:#7f8c8d; margin-top:5px;}
        table {width:100%; border-collapse:collapse; margin:20px 0;}
        th,td {padding:10px; text-align:left; border-bottom:1px solid #ddd;}
        th {background:#3498db; color:white;}
        tr:hover {background:#f5f5f5;}
        .btn {display:inline-block; padding:10px 20px; margin:5px; background:#3498db; color:white; text-decoration:none; border-radius:5px;}
        .btn:hover {background:#2980b9;}
    </style>
</head>
<body>
    <div class="container">
        <h1>Directory Analysis Report</h1>
        <p><strong>Path:</strong> {{path}}</p>
        <p><strong>Date:</strong> {{date}}</p>
        
        <div style="margin:20px 0;">
            <a href="charts.html" class="btn">View Charts</a>
            <a href="network.html" class="btn">View Network</a>
            <a href="analysis.json" class="btn">Download Data</a>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{{files}}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{dirs}}</div>
                <div class="stat-label">Directories</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{size}}</div>
                <div class="stat-label">Total Size</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{dups}}</div>
                <div class="stat-label">Duplicates</div>
            </div>
        </div>
        
        {% if patterns %}
        <h2>Pattern Matches ({{pattern_count}})</h2>
        <table>
            <tr><th>File</th><th>Size</th></tr>
            {% for f in pattern_files %}
            <tr><td>{{f.name}}</td><td>{{f.size}}</td></tr>
            {% endfor %}
        </table>
        {% endif %}
        
        <h2>Largest Files</h2>
        <table>
            <tr><th>File</th><th>Size</th><th>Path</th></tr>
            {% for f in largest %}
            <tr><td>{{f.name}}</td><td>{{f.size}}</td><td style="font-family:monospace;font-size:0.9em;">{{f.path}}</td></tr>
            {% endfor %}
        </table>
        
        {% if duplicates %}
        <h2>Duplicate Files</h2>
        <p>Wasted space: {{waste}}</p>
        {% for d in duplicates %}
        <div style="background:#ffe6e6;margin:10px 0;padding:10px;border-radius:5px;">
            <strong>{{d.count}} copies ({{d.size}} each)</strong>
            <ul>
            {% for p in d.paths[:3] %}
                <li style="font-family:monospace;font-size:0.9em;">{{p}}</li>
            {% endfor %}
            {% if d.paths|length > 3 %}<li>...and {{d.paths|length - 3}} more</li>{% endif %}
            </ul>
        </div>
        {% endfor %}
        {% endif %}
    </div>
</body>
</html>"""
        
        largest = sorted(self.files, key=lambda x: x['size'], reverse=True)[:10]
        waste = sum(d['waste'] for d in self.lists['duplicates'])
        
        data = {
            'path': str(self.root_path),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'files': f"{self.stats['files']:,}",
            'dirs': f"{self.stats['dirs']:,}",
            'size': self._format_size(self.stats['size']),
            'dups': len(self.lists['duplicates']),
            'patterns': bool(self.patterns),
            'pattern_count': len(self.lists['pattern_matches']),
            'pattern_files': [{'name':f['name'], 'size':self._format_size(f['size'])} 
                             for f in self.lists['pattern_matches'][:20]],
            'largest': [{'name':f['name'], 'size':self._format_size(f['size']), 'path':f['path']} 
                       for f in largest],
            'duplicates': [{'count':d['count'], 'size':self._format_size(d['size']), 
                          'paths':[f['path'] for f in d['files']]} 
                         for d in sorted(self.lists['duplicates'], key=lambda x: x['waste'], reverse=True)[:10]],
            'waste': self._format_size(waste)
        }
        
        with open('report.html', 'w') as f:
            f.write(Template(template).render(**data))
        print("✓ Created report.html")
    
    def _json_export(self):
        """Export JSON data"""
        data = {
            'date': datetime.now().isoformat(),
            'path': str(self.root_path),
            'stats': dict(self.stats),
            'counters': {k: dict(v.most_common(20)) for k, v in self.counters.items()},
            'duplicates': self.lists['duplicates'][:50],
            'largest': sorted(self.files, key=lambda x: x['size'], reverse=True)[:50],
            'connections': {k: list(v) for k, v in list(self.connections.items())[:50]}
        }
        with open('analysis.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("✓ Created analysis.json")
    
    def _format_size(self, size):
        """Format bytes to human readable"""
        for unit in ['B','KB','MB','GB','TB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

def main():
    analyzer = DirectoryAnalyzer()
    analyzer.setup()
    analyzer.analyze()
    analyzer.generate_outputs()

if __name__ == "__main__":
    main()