#!/usr/bin/env python3
import os
import sys

# Add current directory to path to import the visualizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_tree_visualizer import FileTreeCrawler, HTMLGenerator, FileTreeExporter
import webbrowser

def test_small_tree():
    print("Testing with small directory...")
    crawler = FileTreeCrawler("test_tree")
    tree_data = crawler.crawl()
    stats = crawler.get_stats()
    
    print(f"Files: {stats['total_files']}, Dirs: {stats['total_dirs']}")
    
    html_generator = HTMLGenerator(tree_data, stats)
    html_content = html_generator.generate()
    
    # Save HTML file in current directory
    html_filename = "test_tree_visualization.html"
    with open(html_filename, "w") as f:
        f.write(html_content)
    
    html_path = os.path.abspath(html_filename)
    print(f"Test HTML created: {html_path}")
    print(f"HTML size: {len(html_content)} bytes")
    
    webbrowser.open(f'file://{html_path}')

if __name__ == "__main__":
    test_small_tree()