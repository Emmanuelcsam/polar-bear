#!/usr/bin/env python3
import os
import json
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import webbrowser


class FileNode:
    def __init__(self, name: str, path: str, is_dir: bool = False):
        self.name = name
        self.path = path
        self.is_dir = is_dir
        self.size = 0
        self.modified_time = None
        self.children: List[FileNode] = []
        self.file_type = None
        self.depth = 0
        
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'path': self.path,
            'is_dir': self.is_dir,
            'size': self.size,
            'modified_time': self.modified_time,
            'file_type': self.file_type,
            'depth': self.depth,
            'children': [child.to_dict() for child in self.children]
        }


class FileTreeCrawler:
    def __init__(self, root_path: str = "."):
        self.root_path = os.path.abspath(root_path)
        self.tree_data = None
        self.file_count = 0
        self.dir_count = 0
        self.total_size = 0
        
    def crawl(self) -> FileNode:
        self.file_count = 0
        self.dir_count = 0
        self.total_size = 0
        self.tree_data = self._crawl_directory(self.root_path, 0)
        return self.tree_data
    
    def _crawl_directory(self, path: str, depth: int) -> FileNode:
        name = os.path.basename(path) or os.path.basename(os.path.dirname(path))
        node = FileNode(name, path, is_dir=True)
        node.depth = depth
        self.dir_count += 1
        
        try:
            stat = os.stat(path)
            node.modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                
                if os.path.isdir(item_path):
                    if not item.startswith('.'):
                        child_node = self._crawl_directory(item_path, depth + 1)
                        node.children.append(child_node)
                        node.size += child_node.size
                else:
                    child_node = self._create_file_node(item_path, depth + 1)
                    node.children.append(child_node)
                    node.size += child_node.size
                    
        except PermissionError:
            pass
            
        return node
    
    def _create_file_node(self, path: str, depth: int) -> FileNode:
        name = os.path.basename(path)
        node = FileNode(name, path, is_dir=False)
        node.depth = depth
        self.file_count += 1
        
        try:
            stat = os.stat(path)
            node.size = stat.st_size
            node.modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
            self.total_size += node.size
            
            ext = os.path.splitext(path)[1].lower()
            node.file_type = self._get_file_type_by_extension(ext)
                
        except (PermissionError, OSError):
            pass
            
        return node
    
    def _get_file_type_by_extension(self, ext: str) -> str:
        type_mapping = {
            '.py': 'code',
            '.js': 'code',
            '.html': 'code',
            '.css': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
            '.h': 'code',
            '.go': 'code',
            '.rs': 'code',
            '.txt': 'text',
            '.md': 'text',
            '.json': 'data',
            '.xml': 'data',
            '.yaml': 'data',
            '.yml': 'data',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.svg': 'image',
            '.mp4': 'video',
            '.avi': 'video',
            '.mov': 'video',
            '.mp3': 'audio',
            '.wav': 'audio',
            '.flac': 'audio'
        }
        return type_mapping.get(ext, 'other')
    
    def get_stats(self) -> Dict:
        return {
            'total_files': self.file_count,
            'total_dirs': self.dir_count,
            'total_size': self.total_size,
            'root_path': self.root_path
        }


class FileTreeExporter:
    def __init__(self, tree_data: FileNode):
        self.tree_data = tree_data
        
    def export_to_text(self) -> str:
        lines = []
        self._build_text_tree(self.tree_data, lines, "", True)
        return '\n'.join(lines)
    
    def _build_text_tree(self, node: FileNode, lines: List[str], prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        icon = "📁 " if node.is_dir else "📄 "
        size_str = f" ({self._format_size(node.size)})" if not node.is_dir else ""
        
        if node.depth == 0:
            lines.append(f"{icon}{node.name}/")
        else:
            lines.append(f"{prefix}{connector}{icon}{node.name}{'/' if node.is_dir else ''}{size_str}")
        
        if node.is_dir:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                is_last_child = i == len(node.children) - 1
                child_prefix = prefix + extension if node.depth > 0 else ""
                self._build_text_tree(child, lines, child_prefix, is_last_child)
    
    def export_to_markdown(self) -> str:
        lines = ["# File Tree Structure\n"]
        lines.append(f"**Root Path:** `{self.tree_data.path}`\n")
        lines.append("## Directory Structure\n")
        lines.append("```")
        lines.append(self.export_to_text())
        lines.append("```")
        return '\n'.join(lines)
    
    def _format_size(self, size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"


class HTMLGenerator:
    def __init__(self, tree_data: FileNode, stats: Dict):
        self.tree_data = tree_data
        self.stats = stats
        
    def generate(self) -> str:
        tree_json = json.dumps(self.tree_data.to_dict())
        stats_json = json.dumps(self.stats)
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Tree Visualizer</title>
    <style>
        {self._get_css()}
    </style>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.155.0/build/three.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌳 File Tree Visualizer</h1>
            <div class="stats">
                <span>📁 Directories: <strong id="dirCount">0</strong></span>
                <span>📄 Files: <strong id="fileCount">0</strong></span>
                <span>💾 Total Size: <strong id="totalSize">0</strong></span>
            </div>
        </header>
        
        <nav class="view-tabs">
            <button class="tab-btn active" onclick="showView('tree', event)">🌲 Tree View</button>
            <button class="tab-btn" onclick="showView('network', event)">🕸️ Network View</button>
            <button class="tab-btn" onclick="showView('galaxy', event)">🌌 Galaxy View</button>
            <button class="tab-btn" onclick="showView('export', event)">📤 Export</button>
        </nav>
        
        <main>
            <div id="tree-view" class="view-panel active">
                <div class="tree-controls">
                    <button onclick="expandAll()">➕ Expand All</button>
                    <button onclick="collapseAll()">➖ Collapse All</button>
                </div>
                <div id="tree-container"></div>
            </div>
            
            <div id="network-view" class="view-panel">
                <svg id="network-svg"></svg>
            </div>
            
            <div id="galaxy-view" class="view-panel">
                <div id="galaxy-container"></div>
            </div>
            
            <div id="export-view" class="view-panel">
                <h2>Export Options</h2>
                <div class="export-buttons">
                    <button onclick="exportAsText()">📄 Export as Text</button>
                    <button onclick="exportAsMarkdown()">📝 Export as Markdown</button>
                    <button onclick="copyToClipboard()">📋 Copy to Clipboard</button>
                </div>
                <textarea id="export-content" readonly></textarea>
            </div>
        </main>
    </div>
    
    <script>
        const treeData = {tree_json};
        const stats = {stats_json};
        {self._get_javascript()}
    </script>
</body>
</html>'''
    
    def _get_css(self) -> str:
        return '''
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .stats {
            display: flex;
            gap: 30px;
            justify-content: center;
            font-size: 1.1em;
        }
        
        .stats span {
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
        }
        
        .view-tabs {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 30px;
        }
        
        .tab-btn {
            padding: 12px 24px;
            border: none;
            background: rgba(255,255,255,0.1);
            color: #fff;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .tab-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }
        
        .tab-btn.active {
            background: rgba(255,255,255,0.3);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .view-panel {
            display: none;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            min-height: 600px;
            backdrop-filter: blur(10px);
        }
        
        .view-panel.active {
            display: block;
        }
        
        .tree-controls {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        
        .tree-controls button {
            padding: 8px 16px;
            border: none;
            background: rgba(255,255,255,0.2);
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
        }
        
        .tree-controls button:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-1px);
        }
        
        #tree-container {
            overflow: auto;
            max-height: 600px;
        }
        
        .tree-node {
            margin: 5px 0;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .tree-node:hover {
            transform: translateX(5px);
        }
        
        .tree-node.directory::before {
            content: "📁 ";
        }
        
        .tree-node.file::before {
            content: "📄 ";
        }
        
        .tree-children {
            margin-left: 25px;
            border-left: 2px solid rgba(255,255,255,0.2);
            padding-left: 10px;
        }
        
        #network-svg, #galaxy-container {
            width: 100%;
            height: 600px;
            border-radius: 10px;
            background: rgba(0,0,0,0.2);
        }
        
        .export-buttons {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }
        
        .export-buttons button {
            padding: 10px 20px;
            border: none;
            background: rgba(255,255,255,0.2);
            color: #fff;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .export-buttons button:hover {
            background: rgba(255,255,255,0.3);
        }
        
        #export-content {
            width: 100%;
            height: 500px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            resize: none;
        }
        
        .node {
            cursor: pointer;
        }
        
        .node circle {
            stroke: #fff;
            stroke-width: 2px;
        }
        
        .node text {
            font-size: 12px;
            fill: #fff;
        }
        
        .link {
            fill: none;
            stroke: rgba(255,255,255,0.3);
            stroke-width: 1px;
        }
        '''
    
    def _get_javascript(self) -> str:
        return '''
        // Update stats
        document.getElementById('dirCount').textContent = stats.total_dirs;
        document.getElementById('fileCount').textContent = stats.total_files;
        document.getElementById('totalSize').textContent = formatSize(stats.total_size);
        
        // View switching
        function showView(viewName, event) {
            document.querySelectorAll('.view-panel').forEach(panel => {
                panel.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            document.getElementById(viewName + '-view').classList.add('active');
            event.target.classList.add('active');
            
            if (viewName === 'tree' && !document.getElementById('tree-container').innerHTML) {
                renderTreeView();
            } else if (viewName === 'network' && !document.getElementById('network-svg').innerHTML) {
                renderNetworkView();
            } else if (viewName === 'galaxy' && !window.galaxyScene) {
                renderGalaxyView();
            }
        }
        
        // Tree View
        function renderTreeView() {
            const container = document.getElementById('tree-container');
            container.innerHTML = '';
            renderTreeNode(treeData, container);
        }
        
        function renderTreeNode(node, parent) {
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'tree-node ' + (node.is_dir ? 'directory' : 'file');
            nodeDiv.textContent = node.name + (node.is_dir ? '/' : ' (' + formatSize(node.size) + ')');
            parent.appendChild(nodeDiv);
            
            if (node.children && node.children.length > 0) {
                const childrenDiv = document.createElement('div');
                childrenDiv.className = 'tree-children';
                parent.appendChild(childrenDiv);
                
                node.children.forEach(child => {
                    renderTreeNode(child, childrenDiv);
                });
                
                nodeDiv.onclick = () => {
                    childrenDiv.style.display = childrenDiv.style.display === 'none' ? 'block' : 'none';
                };
            }
        }
        
        // Network View
        function renderNetworkView() {
            const width = document.getElementById('network-svg').clientWidth;
            const height = 600;
            
            const svg = d3.select('#network-svg')
                .attr('width', width)
                .attr('height', height);
            
            const root = d3.hierarchy(treeData);
            const links = root.links();
            const nodes = root.descendants();
            
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links).id(d => d.id).distance(50))
                .force('charge', d3.forceManyBody().strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2));
            
            const link = svg.append('g')
                .selectAll('line')
                .data(links)
                .enter().append('line')
                .attr('class', 'link');
            
            const node = svg.append('g')
                .selectAll('g')
                .data(nodes)
                .enter().append('g')
                .attr('class', 'node')
                .call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
            
            node.append('circle')
                .attr('r', d => d.data.is_dir ? 8 : 5)
                .attr('fill', d => d.data.is_dir ? '#4CAF50' : '#2196F3');
            
            node.append('text')
                .attr('dx', 12)
                .attr('dy', 4)
                .text(d => d.data.name);
            
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);
                
                node.attr('transform', d => `translate(${d.x},${d.y})`);
            });
            
            function dragstarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }
            
            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }
            
            function dragended(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }
        
        // Galaxy View
        function renderGalaxyView() {
            const container = document.getElementById('galaxy-container');
            
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 600, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            
            renderer.setSize(container.clientWidth, 600);
            container.appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            const pointLight = new THREE.PointLight(0xffffff, 0.8);
            pointLight.position.set(50, 50, 50);
            scene.add(pointLight);
            
            // Create galaxy structure
            const group = new THREE.Group();
            
            function createNodeSphere(node, position, depth = 0) {
                const geometry = new THREE.SphereGeometry(node.is_dir ? 2 : 1, 32, 32);
                const material = new THREE.MeshPhongMaterial({
                    color: node.is_dir ? 0x4CAF50 : 0x2196F3,
                    emissive: node.is_dir ? 0x2E7D32 : 0x1565C0,
                    emissiveIntensity: 0.3
                });
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.copy(position);
                group.add(sphere);
                
                if (node.children) {
                    const angleStep = (Math.PI * 2) / node.children.length;
                    node.children.forEach((child, index) => {
                        const angle = angleStep * index;
                        const radius = 20 + depth * 10;
                        const childPos = new THREE.Vector3(
                            position.x + Math.cos(angle) * radius,
                            position.y + (Math.random() - 0.5) * 10,
                            position.z + Math.sin(angle) * radius
                        );
                        
                        // Create connection line
                        const points = [position, childPos];
                        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
                        const lineMaterial = new THREE.LineBasicMaterial({ 
                            color: 0xffffff, 
                            opacity: 0.3,
                            transparent: true 
                        });
                        const line = new THREE.Line(lineGeometry, lineMaterial);
                        group.add(line);
                        
                        createNodeSphere(child, childPos, depth + 1);
                    });
                }
            }
            
            createNodeSphere(treeData, new THREE.Vector3(0, 0, 0));
            scene.add(group);
            
            camera.position.z = 100;
            
            // Animation
            function animate() {
                requestAnimationFrame(animate);
                group.rotation.y += 0.002;
                renderer.render(scene, camera);
            }
            
            animate();
            window.galaxyScene = scene;
            
            // Mouse controls
            let mouseX = 0, mouseY = 0;
            container.addEventListener('mousemove', (event) => {
                const rect = container.getBoundingClientRect();
                mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                camera.position.x = mouseX * 20;
                camera.position.y = mouseY * 20;
                camera.lookAt(scene.position);
            });
        }
        
        // Export functions
        function exportAsText() {
            const content = generateTextTree(treeData, '', true);
            document.getElementById('export-content').value = content;
        }
        
        function exportAsMarkdown() {
            let content = '# File Tree Structure\\n\\n';
            content += '**Root Path:** `' + stats.root_path + '`\\n\\n';
            content += '## Directory Structure\\n\\n';
            content += '```\\n';
            content += generateTextTree(treeData, '', true);
            content += '\\n```';
            document.getElementById('export-content').value = content;
        }
        
        function generateTextTree(node, prefix, isLast) {
            let result = '';
            const connector = isLast ? '└── ' : '├── ';
            const icon = node.is_dir ? '📁 ' : '📄 ';
            const sizeStr = !node.is_dir ? ' (' + formatSize(node.size) + ')' : '';
            
            if (node.depth === 0) {
                result += icon + node.name + '/\\n';
            } else {
                result += prefix + connector + icon + node.name + (node.is_dir ? '/' : '') + sizeStr + '\\n';
            }
            
            if (node.children) {
                const extension = isLast ? '    ' : '│   ';
                node.children.forEach((child, index) => {
                    const isLastChild = index === node.children.length - 1;
                    const childPrefix = node.depth > 0 ? prefix + extension : '';
                    result += generateTextTree(child, childPrefix, isLastChild);
                });
            }
            
            return result;
        }
        
        // Utility functions
        function formatSize(bytes) {
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            let size = bytes;
            let unitIndex = 0;
            
            while (size >= 1024 && unitIndex < units.length - 1) {
                size /= 1024;
                unitIndex++;
            }
            
            return size.toFixed(1) + units[unitIndex];
        }
        
        // Copy to clipboard function
        function copyToClipboard() {
            const content = document.getElementById('export-content').value;
            if (content) {
                navigator.clipboard.writeText(content).then(() => {
                    alert('Copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy:', err);
                });
            } else {
                alert('Please generate an export first!');
            }
        }
        
        // Expand/Collapse all functions
        function expandAll() {
            const allChildren = document.querySelectorAll('.tree-children');
            allChildren.forEach(child => {
                child.style.display = 'block';
            });
        }
        
        function collapseAll() {
            const allChildren = document.querySelectorAll('.tree-children');
            allChildren.forEach(child => {
                child.style.display = 'none';
            });
        }
        
        // Initialize tree view on load
        renderTreeView();
        '''


def main():
    print("🌳 File Tree Visualizer")
    print("=" * 50)
    
    crawler = FileTreeCrawler()
    print("Crawling directory structure...")
    tree_data = crawler.crawl()
    stats = crawler.get_stats()
    
    print(f"\n📊 Statistics:")
    print(f"   📁 Total directories: {stats['total_dirs']}")
    print(f"   📄 Total files: {stats['total_files']}")
    print(f"   💾 Total size: {stats['total_size']:,} bytes")
    
    print("\n🎨 Generating visualizations...")
    html_generator = HTMLGenerator(tree_data, stats)
    html_content = html_generator.generate()
    
    # Save HTML file in current directory
    html_filename = "file_tree_visualization.html"
    with open(html_filename, 'w') as f:
        f.write(html_content)
    
    html_path = os.path.abspath(html_filename)
    print(f"\n✅ Visualization created: {html_path}")
    print("🌐 Opening in browser...")
    webbrowser.open(f'file://{html_path}')
    
    exporter = FileTreeExporter(tree_data)
    
    text_output = exporter.export_to_text()
    with open("file_tree.txt", "w") as f:
        f.write(text_output)
    print("📄 Text export saved to: file_tree.txt")
    
    md_output = exporter.export_to_markdown()
    with open("file_tree.md", "w") as f:
        f.write(md_output)
    print("📝 Markdown export saved to: file_tree.md")


if __name__ == "__main__":
    main()