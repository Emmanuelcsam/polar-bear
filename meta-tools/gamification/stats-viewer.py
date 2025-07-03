#!/usr/bin/env python3
"""
Project Stats Viewer - Web-based viewer for project statistics
Provides a simple web interface to browse and visualize project data
"""

import os
import json
import http.server
import socketserver
import webbrowser
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import threading

class StatsViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for stats viewer"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_home()
        elif parsed_path.path == '/api/projects':
            self.serve_projects_list()
        elif parsed_path.path == '/api/stats':
            self.serve_project_stats(parsed_path)
        elif parsed_path.path == '/api/report':
            self.serve_report(parsed_path)
        else:
            # Try to serve static files
            super().do_GET()
    
    def serve_home(self):
        """Serve the main HTML page"""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Project Stats Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .projects-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .project-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .project-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .project-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }
        .project-path {
            font-size: 0.875rem;
            color: #7f8c8d;
            font-family: monospace;
            margin-bottom: 1rem;
        }
        .stats-summary {
            display: flex;
            justify-content: space-between;
            margin-top: 1rem;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #3498db;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #95a5a6;
            text-transform: uppercase;
        }
        .health-bar {
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            margin-top: 1rem;
            overflow: hidden;
        }
        .health-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .health-excellent { background: #27ae60; }
        .health-good { background: #f39c12; }
        .health-poor { background: #e74c3c; }
        .loading {
            text-align: center;
            padding: 3rem;
            color: #7f8c8d;
        }
        .report-viewer {
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-top: 2rem;
        }
        .back-button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        .back-button:hover {
            background: #2980b9;
        }
        pre {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
        .timeline-chart {
            height: 300px;
            margin: 2rem 0;
        }
        .no-projects {
            text-align: center;
            padding: 3rem;
            color: #7f8c8d;
        }
        .search-box {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Project Stats Viewer</h1>
    </div>
    
    <div class="container">
        <div id="content">
            <div class="loading">Loading projects...</div>
        </div>
    </div>
    
    <script>
        // Load and display projects
        async function loadProjects() {
            try {
                const response = await fetch('/api/projects');
                const projects = await response.json();
                displayProjects(projects);
            } catch (error) {
                document.getElementById('content').innerHTML = 
                    '<div class="no-projects">Error loading projects: ' + error.message + '</div>';
            }
        }
        
        function displayProjects(projects) {
            if (projects.length === 0) {
                document.getElementById('content').innerHTML = 
                    '<div class="no-projects">No projects found with stats data.</div>';
                return;
            }
            
            let html = '<input type="text" class="search-box" placeholder="Search projects..." onkeyup="filterProjects(this.value)">';
            html += '<div class="projects-grid" id="projects-grid">';
            
            projects.forEach(project => {
                const healthClass = project.health >= 90 ? 'health-excellent' : 
                                  project.health >= 70 ? 'health-good' : 'health-poor';
                
                html += `
                    <div class="project-card" onclick="viewProject('${project.path}')" data-name="${project.name.toLowerCase()}">
                        <div class="project-name">${project.name}</div>
                        <div class="project-path">${project.path}</div>
                        
                        <div class="stats-summary">
                            <div class="stat-item">
                                <div class="stat-value">${project.files.toLocaleString()}</div>
                                <div class="stat-label">Files</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${project.size}</div>
                                <div class="stat-label">Size</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">${project.commits || '-'}</div>
                                <div class="stat-label">Commits</div>
                            </div>
                        </div>
                        
                        <div class="health-bar">
                            <div class="health-fill ${healthClass}" style="width: ${project.health}%"></div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            document.getElementById('content').innerHTML = html;
        }
        
        function filterProjects(searchTerm) {
            const cards = document.querySelectorAll('.project-card');
            const term = searchTerm.toLowerCase();
            
            cards.forEach(card => {
                const name = card.getAttribute('data-name');
                if (name.includes(term)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
        
        async function viewProject(projectPath) {
            try {
                const response = await fetch(`/api/stats?path=${encodeURIComponent(projectPath)}`);
                const stats = await response.json();
                displayProjectStats(stats);
            } catch (error) {
                alert('Error loading project stats: ' + error.message);
            }
        }
        
        function displayProjectStats(stats) {
            let html = '<a href="/" class="back-button">← Back to Projects</a>';
            html += '<div class="report-viewer">';
            html += '<h2>' + stats.name + '</h2>';
            
            // Display available reports
            html += '<h3>Available Reports</h3>';
            html += '<ul>';
            stats.reports.forEach(report => {
                html += `<li><a href="#" onclick="viewReport('${stats.path}', '${report.file}')">${report.type} - ${report.date}</a></li>`;
            });
            html += '</ul>';
            
            // Show latest dashboard if available
            if (stats.latest_dashboard) {
                html += '<h3>Latest Dashboard</h3>';
                html += '<pre>' + JSON.stringify(stats.latest_dashboard, null, 2) + '</pre>';
            }
            
            html += '</div>';
            document.getElementById('content').innerHTML = html;
        }
        
        async function viewReport(projectPath, reportFile) {
            try {
                const response = await fetch(`/api/report?path=${encodeURIComponent(projectPath)}&file=${encodeURIComponent(reportFile)}`);
                const report = await response.json();
                
                let html = '<a href="#" onclick="viewProject(\'' + projectPath + '\')" class="back-button">← Back</a>';
                html += '<div class="report-viewer">';
                html += '<h2>' + reportFile + '</h2>';
                html += '<pre>' + JSON.stringify(report, null, 2) + '</pre>';
                html += '</div>';
                
                document.getElementById('content').innerHTML = html;
            } catch (error) {
                alert('Error loading report: ' + error.message);
            }
        }
        
        // Load projects on page load
        loadProjects();
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def serve_projects_list(self):
        """Serve list of projects with stats"""
        projects = []
        
        # Search for projects with .project-stats directories
        search_paths = [
            Path.home(),  # User home
            Path.cwd(),   # Current directory
            Path.cwd().parent,  # Parent directory
        ]
        
        # Also check common project directories
        for base in [Path.home(), Path.cwd()]:
            for subdir in ['projects', 'Projects', 'code', 'Code', 'dev', 'Development']:
                path = base / subdir
                if path.exists():
                    search_paths.append(path)
        
        # Remove duplicates
        search_paths = list(set(search_paths))
        
        for search_path in search_paths:
            try:
                # Look for .project-stats directories
                for stats_dir in search_path.rglob('.project-stats'):
                    if stats_dir.is_dir():
                        project_info = self.get_project_info(stats_dir.parent)
                        if project_info:
                            projects.append(project_info)
            except:
                pass
        
        # Remove duplicates based on path
        seen_paths = set()
        unique_projects = []
        for project in projects:
            if project['path'] not in seen_paths:
                seen_paths.add(project['path'])
                unique_projects.append(project)
        
        self.send_json_response(unique_projects)
    
    def get_project_info(self, project_path):
        """Get basic info about a project"""
        stats_dir = project_path / '.project-stats'
        
        if not stats_dir.exists():
            return None
        
        info = {
            'name': project_path.name,
            'path': str(project_path),
            'files': 0,
            'size': '0 B',
            'health': 0,
            'commits': 0
        }
        
        # Try to read latest dashboard
        latest_dashboard = stats_dir / 'latest_dashboard.json'
        if latest_dashboard.exists():
            try:
                with open(latest_dashboard, 'r') as f:
                    data = json.load(f)
                
                overview = data.get('overview', {})
                info['files'] = overview.get('total_files', 0)
                info['size'] = self.format_size(overview.get('total_size', 0))
                info['health'] = data.get('health_score', 0)
                info['commits'] = overview.get('commits', 0)
            except:
                pass
        
        return info
    
    def serve_project_stats(self, parsed_path):
        """Serve detailed stats for a project"""
        params = parse_qs(parsed_path.query)
        project_path = params.get('path', [''])[0]
        
        if not project_path:
            self.send_error(400, "Missing project path")
            return
        
        project_path = Path(project_path)
        stats_dir = project_path / '.project-stats'
        
        if not stats_dir.exists():
            self.send_error(404, "Project stats not found")
            return
        
        stats = {
            'name': project_path.name,
            'path': str(project_path),
            'reports': []
        }
        
        # List all reports
        for report_file in stats_dir.glob('*.json'):
            report_type = 'Unknown'
            
            if 'snapshot' in report_file.name:
                report_type = 'Snapshot'
            elif 'dashboard' in report_file.name:
                report_type = 'Dashboard'
            elif 'health' in report_file.name:
                report_type = 'Health Report'
            elif 'timeline' in report_file.name:
                report_type = 'Timeline'
            elif 'duplicates' in report_file.name:
                report_type = 'Duplicates'
            elif 'code_analysis' in report_file.name:
                report_type = 'Code Analysis'
            
            stats['reports'].append({
                'file': report_file.name,
                'type': report_type,
                'date': datetime.fromtimestamp(report_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
                'size': report_file.stat().st_size
            })
        
        # Sort reports by date
        stats['reports'].sort(key=lambda x: x['date'], reverse=True)
        
        # Include latest dashboard
        latest_dashboard = stats_dir / 'latest_dashboard.json'
        if latest_dashboard.exists():
            try:
                with open(latest_dashboard, 'r') as f:
                    stats['latest_dashboard'] = json.load(f)
            except:
                pass
        
        self.send_json_response(stats)
    
    def serve_report(self, parsed_path):
        """Serve a specific report file"""
        params = parse_qs(parsed_path.query)
        project_path = params.get('path', [''])[0]
        report_file = params.get('file', [''])[0]
        
        if not project_path or not report_file:
            self.send_error(400, "Missing parameters")
            return
        
        file_path = Path(project_path) / '.project-stats' / report_file
        
        if not file_path.exists():
            self.send_error(404, "Report not found")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.send_json_response(data)
        except Exception as e:
            self.send_error(500, f"Error reading report: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def format_size(self, size):
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass

def start_server(port=8888):
    """Start the web server"""
    handler = StatsViewerHandler
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"📊 Project Stats Viewer")
        print(f"🌐 Server running at: http://localhost:{port}")
        print(f"📁 Serving from: {Path.cwd()}")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Open browser
        def open_browser():
            webbrowser.open(f'http://localhost:{port}')
        
        # Open browser after a short delay
        timer = threading.Timer(1.0, open_browser)
        timer.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based viewer for project statistics')
    parser.add_argument('--port', type=int, default=8888, help='Port to run server on (default: 8888)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    if args.no_browser:
        # Monkey patch to disable browser opening
        webbrowser.open = lambda x: None
    
    try:
        start_server(args.port)
    except OSError as e:
        if 'Address already in use' in str(e):
            print(f"❌ Port {args.port} is already in use!")
            print(f"   Try a different port: python {sys.argv[0]} --port 8889")
        else:
            print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
