#!/usr/bin/env python3
"""
File Tree Crawler Script

This script recursively crawls through all directories and subdirectories
starting from the current directory and creates a comprehensive file tree
structure saved to a text file.

Author: GitHub Copilot
Date: August 7, 2025
"""

import os
import argparse
from pathlib import Path
from datetime import datetime


def get_file_tree(root_path, max_depth=None, include_files=True, include_hidden=False, exclude_venv=True):
    """
    Generate a file tree structure starting from root_path.
    
    Args:
        root_path (str): The root directory to start crawling from
        max_depth (int, optional): Maximum depth to crawl (None for unlimited)
        include_files (bool): Whether to include files in the tree
        include_hidden (bool): Whether to include hidden files/directories
        exclude_venv (bool): Whether to exclude Python virtual environment directories
    
    Returns:
        list: List of strings representing the file tree
    """
    tree_lines = []
    root_path = Path(root_path).resolve()
    
    def should_include(path):
        """Check if a path should be included based on settings."""
        if not include_hidden and path.name.startswith('.'):
            return False
        
        # Exclude Python virtual environment directories if enabled
        if exclude_venv:
            venv_names = {
                'venv', '.venv', 'env', '.env', 'virtualenv', '.virtualenv',
                'pyenv', '.pyenv', 'conda', '.conda', 'miniconda', '.miniconda',
                'anaconda', '.anaconda', 'pipenv', '.pipenv', '__pycache__',
                '.pytest_cache', '.tox', '.nox', 'site-packages'
            }
            
            if path.is_dir() and path.name.lower() in venv_names:
                return False
            
            # Also check for common virtual environment patterns
            path_lower = path.name.lower()
            if path.is_dir() and (
                path_lower.endswith('_env') or 
                path_lower.endswith('-env') or
                path_lower.endswith('_venv') or
                path_lower.endswith('-venv') or
                path_lower.startswith('venv_') or
                path_lower.startswith('venv-') or
                path_lower.startswith('env_') or
                path_lower.startswith('env-')
            ):
                return False
        
        return True
    
    def crawl_directory(current_path, prefix="", depth=0):
        """Recursively crawl directory structure."""
        if max_depth is not None and depth > max_depth:
            return
        
        try:
            # Get all items in the current directory
            items = sorted(current_path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            items = [item for item in items if should_include(item)]
            
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                
                # Determine the appropriate tree characters
                if is_last:
                    current_prefix = "└── "
                    next_prefix = prefix + "    "
                else:
                    current_prefix = "├── "
                    next_prefix = prefix + "│   "
                
                # Add the current item to the tree
                if item.is_file():
                    if include_files:
                        try:
                            size = item.stat().st_size
                            size_str = format_file_size(size)
                            tree_lines.append(f"{prefix}{current_prefix}{item.name} ({size_str})")
                        except (OSError, PermissionError):
                            tree_lines.append(f"{prefix}{current_prefix}{item.name} (access denied)")
                else:
                    # It's a directory
                    try:
                        dir_count = len([x for x in item.iterdir() if should_include(x)])
                        tree_lines.append(f"{prefix}{current_prefix}{item.name}/ ({dir_count} items)")
                        
                        # Recursively crawl subdirectory
                        crawl_directory(item, next_prefix, depth + 1)
                    except (OSError, PermissionError):
                        tree_lines.append(f"{prefix}{current_prefix}{item.name}/ (access denied)")
                        
        except (OSError, PermissionError) as e:
            tree_lines.append(f"{prefix}Error accessing directory: {e}")
    
    # Add root directory
    tree_lines.append(f"{root_path.name}/")
    crawl_directory(root_path)
    
    return tree_lines


def format_file_size(size_bytes):
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_directory_stats(root_path):
    """Get basic statistics about the directory structure."""
    root_path = Path(root_path)
    total_dirs = 0
    total_files = 0
    total_size = 0
    
    try:
        for item in root_path.rglob('*'):
            try:
                if item.is_file():
                    total_files += 1
                    total_size += item.stat().st_size
                elif item.is_dir():
                    total_dirs += 1
            except (OSError, PermissionError):
                continue
    except (OSError, PermissionError):
        pass
    
    return total_dirs, total_files, total_size


def ask_yes_no(prompt, default="y"):
    """Ask a yes/no question with a default answer."""
    while True:
        if default.lower() == "y":
            answer = input(f"{prompt} [Y/n]: ").strip()
            answer = answer if answer else "y"
        else:
            answer = input(f"{prompt} [y/N]: ").strip()
            answer = answer if answer else "n"
        
        if answer.lower() in ['y', 'yes']:
            return True
        elif answer.lower() in ['n', 'no']:
            return False
        else:
            print("Please answer yes (y) or no (n).")


def interactive_mode():
    """Run the file tree crawler in interactive mode."""
    print("=" * 60)
    print("   INTERACTIVE FILE TREE CRAWLER")
    print("=" * 60)
    print()
    
    print("Let's configure your file tree crawler:")
    print()
    
    # Root directory
    print("1. Root Directory")
    root_path = input("Enter the root directory to start crawling from (press Enter for current directory): ").strip()
    root_path = root_path if root_path else "."
    
    # Output file
    print()
    print("2. Output File")
    output_file = input("Enter the output file name (press Enter for 'file_tree_structure.txt'): ").strip()
    output_file = output_file if output_file else "file_tree_structure.txt"
    
    # Maximum depth
    print()
    print("3. Maximum Depth")
    max_depth_str = input("Enter maximum depth to crawl (press Enter for unlimited): ").strip()
    max_depth = int(max_depth_str) if max_depth_str.isdigit() else None
    
    # Include files
    print()
    print("4. File Inclusion")
    include_files = ask_yes_no("Include files in the tree structure?", "y")
    
    # Include hidden files
    print()
    print("5. Hidden Files")
    include_hidden = ask_yes_no("Include hidden files and directories?", "n")
    
    # Include virtual environments
    print()
    print("6. Virtual Environments")
    include_venv = ask_yes_no("Include Python virtual environment directories?", "n")
    
    # Include statistics
    print()
    print("7. Statistics")
    include_stats = ask_yes_no("Include directory statistics in the output?", "y")
    
    # Show configuration summary
    print()
    print("=" * 60)
    print("   CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Root directory: {root_path}")
    print(f"Output file: {output_file}")
    print(f"Max depth: {'unlimited' if max_depth is None else max_depth}")
    print(f"Include files: {'yes' if include_files else 'no'}")
    print(f"Include hidden: {'yes' if include_hidden else 'no'}")
    print(f"Include virtual envs: {'yes' if include_venv else 'no'}")
    print(f"Include statistics: {'yes' if include_stats else 'no'}")
    print("=" * 60)
    print()
    
    # Confirm before running
    if ask_yes_no("Run the file tree crawler with these settings?", "y"):
        return {
            'root': root_path,
            'output': output_file,
            'max_depth': max_depth,
            'no_files': not include_files,
            'include_hidden': include_hidden,
            'include_venv': include_venv,
            'stats': include_stats
        }
    else:
        print("Operation cancelled.")
        return None


def main():
    """Main function to handle command line arguments and execute the crawl."""
    parser = argparse.ArgumentParser(
        description="Create a comprehensive file tree structure of directories and subdirectories"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (ask questions instead of using arguments)"
    )
    
    parser.add_argument(
        "--root", 
        default=".", 
        help="Root directory to start crawling from (default: current directory)"
    )
    
    parser.add_argument(
        "--output", 
        default="file_tree_structure.txt", 
        help="Output file name (default: file_tree_structure.txt)"
    )
    
    parser.add_argument(
        "--max-depth", 
        type=int, 
        help="Maximum depth to crawl (default: unlimited)"
    )
    
    parser.add_argument(
        "--no-files", 
        action="store_true", 
        help="Exclude files, show only directories"
    )
    
    parser.add_argument(
        "--include-hidden", 
        action="store_true", 
        help="Include hidden files and directories"
    )
    
    parser.add_argument(
        "--include-venv", 
        action="store_true", 
        help="Include Python virtual environment directories (excluded by default)"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true", 
        help="Include directory statistics in the output"
    )
    
    args = parser.parse_args()
    
    # Check if we should run in interactive mode
    # Run interactive mode if --interactive is specified OR if no arguments are provided
    if args.interactive or len([arg for arg in vars(args).values() if arg not in [False, None, '.', 'file_tree_structure.txt']]) == 0:
        config = interactive_mode()
        if config is None:
            return 1
        
        # Apply interactive configuration
        args.root = config['root']
        args.output = config['output']
        args.max_depth = config['max_depth']
        args.no_files = config['no_files']
        args.include_hidden = config['include_hidden']
        args.include_venv = config['include_venv']
        args.stats = config['stats']
    
    root_path = Path(args.root).resolve()
    
    if not root_path.exists():
        print(f"Error: Directory '{root_path}' does not exist!")
        return 1
    
    if not root_path.is_dir():
        print(f"Error: '{root_path}' is not a directory!")
        return 1
    
    print(f"Crawling directory structure starting from: {root_path}")
    print(f"Output file: {args.output}")
    
    # Generate the file tree
    include_files = not args.no_files
    exclude_venv = not args.include_venv  # Default is to exclude venv, unless --include-venv is specified
    tree_lines = get_file_tree(
        root_path, 
        max_depth=args.max_depth,
        include_files=include_files,
        include_hidden=args.include_hidden,
        exclude_venv=exclude_venv
    )
    
    # Get statistics if requested
    stats_lines = []
    if args.stats:
        print("Calculating directory statistics...")
        total_dirs, total_files, total_size = get_directory_stats(root_path)
        stats_lines = [
            "",
            "=" * 60,
            "DIRECTORY STATISTICS",
            "=" * 60,
            f"Total directories: {total_dirs:,}",
            f"Total files: {total_files:,}",
            f"Total size: {format_file_size(total_size)}",
            f"Average file size: {format_file_size(total_size / max(total_files, 1))}",
        ]
    
    # Write to output file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            # Write header
            f.write("FILE TREE STRUCTURE\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Root directory: {root_path}\n")
            f.write(f"Max depth: {'Unlimited' if args.max_depth is None else args.max_depth}\n")
            f.write(f"Include files: {'Yes' if include_files else 'No'}\n")
            f.write(f"Include hidden: {'Yes' if args.include_hidden else 'No'}\n")
            f.write(f"Exclude virtual environments: {'Yes' if exclude_venv else 'No'}\n")
            f.write("=" * 60 + "\n\n")
            
            # Write tree structure
            for line in tree_lines:
                f.write(line + "\n")
            
            # Write statistics if requested
            for line in stats_lines:
                f.write(line + "\n")
        
        print(f"File tree successfully written to: {args.output}")
        print(f"Total lines in tree: {len(tree_lines):,}")
        
    except (OSError, PermissionError) as e:
        print(f"Error writing output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
