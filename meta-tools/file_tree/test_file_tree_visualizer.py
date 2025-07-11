#!/usr/bin/env python3
import unittest
import os
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from file_tree_visualizer import FileNode, FileTreeCrawler, FileTreeExporter, HTMLGenerator


class TestFileNode(unittest.TestCase):
    def test_file_node_creation(self):
        node = FileNode("test.txt", "/path/to/test.txt", is_dir=False)
        self.assertEqual(node.name, "test.txt")
        self.assertEqual(node.path, "/path/to/test.txt")
        self.assertFalse(node.is_dir)
        self.assertEqual(node.size, 0)
        self.assertIsNone(node.modified_time)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.file_type)
        self.assertEqual(node.depth, 0)
    
    def test_directory_node_creation(self):
        node = FileNode("test_dir", "/path/to/test_dir", is_dir=True)
        self.assertEqual(node.name, "test_dir")
        self.assertEqual(node.path, "/path/to/test_dir")
        self.assertTrue(node.is_dir)
        self.assertEqual(len(node.children), 0)
    
    def test_node_to_dict(self):
        parent = FileNode("parent", "/parent", is_dir=True)
        child = FileNode("child.txt", "/parent/child.txt", is_dir=False)
        parent.children.append(child)
        
        result = parent.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['name'], "parent")
        self.assertEqual(result['path'], "/parent")
        self.assertTrue(result['is_dir'])
        self.assertEqual(len(result['children']), 1)
        self.assertEqual(result['children'][0]['name'], "child.txt")


class TestFileTreeCrawler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create test directory structure
        os.makedirs(os.path.join(self.test_dir, "subdir1"))
        os.makedirs(os.path.join(self.test_dir, "subdir2", "subsubdir"))
        
        # Create test files
        with open(os.path.join(self.test_dir, "file1.txt"), "w") as f:
            f.write("Test content 1")
        with open(os.path.join(self.test_dir, "file2.py"), "w") as f:
            f.write("print('hello')")
        with open(os.path.join(self.test_dir, "subdir1", "file3.md"), "w") as f:
            f.write("# Markdown")
        with open(os.path.join(self.test_dir, "subdir2", "file4.json"), "w") as f:
            f.write('{"key": "value"}')
        with open(os.path.join(self.test_dir, "subdir2", "subsubdir", "file5.txt"), "w") as f:
            f.write("Nested file")
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_crawler_initialization(self):
        crawler = FileTreeCrawler(self.test_dir)
        self.assertEqual(crawler.root_path, os.path.abspath(self.test_dir))
        self.assertIsNone(crawler.tree_data)
        self.assertEqual(crawler.file_count, 0)
        self.assertEqual(crawler.dir_count, 0)
        self.assertEqual(crawler.total_size, 0)
    
    def test_crawl_directory(self):
        crawler = FileTreeCrawler(self.test_dir)
        tree_data = crawler.crawl()
        
        self.assertIsNotNone(tree_data)
        self.assertTrue(tree_data.is_dir)
        self.assertEqual(crawler.file_count, 5)
        self.assertEqual(crawler.dir_count, 4)  # root + 3 subdirs
        self.assertGreater(crawler.total_size, 0)
    
    def test_crawl_structure(self):
        crawler = FileTreeCrawler(self.test_dir)
        tree_data = crawler.crawl()
        
        # Check root has correct children
        child_names = [child.name for child in tree_data.children]
        self.assertIn("file1.txt", child_names)
        self.assertIn("file2.py", child_names)
        self.assertIn("subdir1", child_names)
        self.assertIn("subdir2", child_names)
        
        # Check subdir structure
        subdir2 = next(child for child in tree_data.children if child.name == "subdir2")
        self.assertTrue(subdir2.is_dir)
        subdir2_children = [child.name for child in subdir2.children]
        self.assertIn("file4.json", subdir2_children)
        self.assertIn("subsubdir", subdir2_children)
    
    def test_file_type_detection(self):
        crawler = FileTreeCrawler(self.test_dir)
        tree_data = crawler.crawl()
        
        # Find specific files and check their types
        file2 = next(child for child in tree_data.children if child.name == "file2.py")
        self.assertEqual(file2.file_type, "code")
        
        subdir1 = next(child for child in tree_data.children if child.name == "subdir1")
        file3 = next(child for child in subdir1.children if child.name == "file3.md")
        self.assertEqual(file3.file_type, "text")
    
    def test_get_stats(self):
        crawler = FileTreeCrawler(self.test_dir)
        crawler.crawl()
        stats = crawler.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_files'], 5)
        self.assertEqual(stats['total_dirs'], 4)
        self.assertGreater(stats['total_size'], 0)
        self.assertEqual(stats['root_path'], os.path.abspath(self.test_dir))
    
    def test_get_file_type_by_extension(self):
        crawler = FileTreeCrawler()
        
        self.assertEqual(crawler._get_file_type_by_extension('.py'), 'code')
        self.assertEqual(crawler._get_file_type_by_extension('.js'), 'code')
        self.assertEqual(crawler._get_file_type_by_extension('.txt'), 'text')
        self.assertEqual(crawler._get_file_type_by_extension('.jpg'), 'image')
        self.assertEqual(crawler._get_file_type_by_extension('.mp4'), 'video')
        self.assertEqual(crawler._get_file_type_by_extension('.mp3'), 'audio')
        self.assertEqual(crawler._get_file_type_by_extension('.xyz'), 'other')


class TestFileTreeExporter(unittest.TestCase):
    def setUp(self):
        # Create a simple tree structure for testing
        self.root = FileNode("root", "/root", is_dir=True)
        self.root.depth = 0
        
        file1 = FileNode("file1.txt", "/root/file1.txt", is_dir=False)
        file1.size = 1024
        file1.depth = 1
        
        subdir = FileNode("subdir", "/root/subdir", is_dir=True)
        subdir.depth = 1
        
        file2 = FileNode("file2.py", "/root/subdir/file2.py", is_dir=False)
        file2.size = 2048
        file2.depth = 2
        
        file3 = FileNode("file3.txt", "/root/subdir/file3.txt", is_dir=False)
        file3.size = 512
        file3.depth = 2
        
        subdir.children.extend([file2, file3])
        self.root.children.extend([file1, subdir])
        
        self.exporter = FileTreeExporter(self.root)
    
    def test_export_to_text(self):
        text_output = self.exporter.export_to_text()
        
        self.assertIsInstance(text_output, str)
        self.assertIn("📁 root/", text_output)
        self.assertIn("📄 file1.txt", text_output)
        self.assertIn("📁 subdir/", text_output)
        self.assertIn("📄 file2.py", text_output)
        self.assertIn("1.0KB", text_output)
        self.assertIn("2.0KB", text_output)
    
    def test_export_to_markdown(self):
        md_output = self.exporter.export_to_markdown()
        
        self.assertIsInstance(md_output, str)
        self.assertIn("# File Tree Structure", md_output)
        self.assertIn("**Root Path:**", md_output)
        self.assertIn("## Directory Structure", md_output)
        self.assertIn("```", md_output)
        self.assertIn("📁 root/", md_output)
    
    def test_format_size(self):
        exporter = FileTreeExporter(self.root)
        
        self.assertEqual(exporter._format_size(512), "512.0B")
        self.assertEqual(exporter._format_size(1024), "1.0KB")
        self.assertEqual(exporter._format_size(1536), "1.5KB")
        self.assertEqual(exporter._format_size(1048576), "1.0MB")
        self.assertEqual(exporter._format_size(1073741824), "1.0GB")
    
    def test_build_text_tree_structure(self):
        text_output = self.exporter.export_to_text()
        lines = text_output.split('\n')
        
        # Check structure
        self.assertTrue(lines[0].startswith("📁 root/"))
        self.assertTrue(any("├──" in line for line in lines))
        self.assertTrue(any("└──" in line for line in lines))
        # Check indentation for nested items
        self.assertTrue(any("    " in line for line in lines))


class TestHTMLGenerator(unittest.TestCase):
    def setUp(self):
        self.root = FileNode("root", "/root", is_dir=True)
        self.stats = {
            'total_files': 10,
            'total_dirs': 5,
            'total_size': 1048576,
            'root_path': '/root'
        }
        self.generator = HTMLGenerator(self.root, self.stats)
    
    def test_html_generation(self):
        html = self.generator.generate()
        
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<title>File Tree Visualizer</title>", html)
        self.assertIn("d3.v7.min.js", html)
        self.assertIn("three.min.js", html)
    
    def test_html_contains_data(self):
        html = self.generator.generate()
        
        # Check that tree data is embedded
        self.assertIn("const treeData =", html)
        self.assertIn("const stats =", html)
        self.assertIn('"total_files": 10', html)
        self.assertIn('"total_dirs": 5', html)
    
    def test_html_contains_views(self):
        html = self.generator.generate()
        
        self.assertIn("tree-view", html)
        self.assertIn("network-view", html)
        self.assertIn("galaxy-view", html)
        self.assertIn("export-view", html)
    
    def test_html_contains_functions(self):
        html = self.generator.generate()
        
        self.assertIn("showView", html)
        self.assertIn("renderTreeView", html)
        self.assertIn("renderNetworkView", html)
        self.assertIn("renderGalaxyView", html)
        self.assertIn("exportAsText", html)
        self.assertIn("exportAsMarkdown", html)
    
    def test_css_generation(self):
        css = self.generator._get_css()
        
        self.assertIsInstance(css, str)
        self.assertIn("body {", css)
        self.assertIn(".container {", css)
        self.assertIn(".view-panel {", css)
        self.assertIn("#tree-container {", css)
    
    def test_javascript_generation(self):
        js = self.generator._get_javascript()
        
        self.assertIsInstance(js, str)
        self.assertIn("function showView", js)
        self.assertIn("function renderTreeView", js)
        self.assertIn("function formatSize", js)
        self.assertIn("d3.hierarchy", js)
        self.assertIn("THREE.Scene", js)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create a more complex test structure
        os.makedirs(os.path.join(self.test_dir, "src", "components"))
        os.makedirs(os.path.join(self.test_dir, "tests"))
        os.makedirs(os.path.join(self.test_dir, "docs"))
        
        # Create various file types
        with open(os.path.join(self.test_dir, "README.md"), "w") as f:
            f.write("# Test Project\n\nThis is a test.")
        with open(os.path.join(self.test_dir, "src", "main.py"), "w") as f:
            f.write("def main():\n    print('Hello')")
        with open(os.path.join(self.test_dir, "src", "components", "widget.js"), "w") as f:
            f.write("class Widget {}")
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_full_workflow(self):
        # Crawl
        crawler = FileTreeCrawler(self.test_dir)
        tree_data = crawler.crawl()
        stats = crawler.get_stats()
        
        self.assertIsNotNone(tree_data)
        self.assertGreater(stats['total_files'], 0)
        self.assertGreater(stats['total_dirs'], 0)
        
        # Export
        exporter = FileTreeExporter(tree_data)
        text_output = exporter.export_to_text()
        md_output = exporter.export_to_markdown()
        
        self.assertIn("README.md", text_output)
        self.assertIn("main.py", text_output)
        self.assertIn("widget.js", text_output)
        self.assertIn("# File Tree Structure", md_output)
        
        # Generate HTML
        generator = HTMLGenerator(tree_data, stats)
        html = generator.generate()
        
        self.assertIn("File Tree Visualizer", html)
        self.assertIn("const treeData =", html)
        
    def test_hidden_files_excluded(self):
        # Create hidden directory
        os.makedirs(os.path.join(self.test_dir, ".hidden"))
        with open(os.path.join(self.test_dir, ".hidden", "secret.txt"), "w") as f:
            f.write("secret")
        
        crawler = FileTreeCrawler(self.test_dir)
        tree_data = crawler.crawl()
        
        # Check that hidden directory is not in the tree
        child_names = [child.name for child in tree_data.children]
        self.assertNotIn(".hidden", child_names)
    
    def test_permission_error_handling(self):
        # This test is tricky to implement cross-platform
        # We'll just ensure the crawler doesn't crash on permission errors
        crawler = FileTreeCrawler("/root")  # Likely to have permission issues
        try:
            tree_data = crawler.crawl()
            # If we get here, it handled permission errors gracefully
            self.assertIsNotNone(tree_data)
        except Exception as e:
            self.fail(f"Crawler should handle permission errors gracefully, but raised: {e}")


if __name__ == '__main__':
    unittest.main()