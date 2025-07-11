#!/usr/bin/env python3
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(message, color=Colors.CYAN):
    """Print colored log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {message}{Colors.END}")

def check_and_install_requirements():
    """Check if required modules are installed and install if missing"""
    log("Checking system requirements...", Colors.YELLOW)
    
    # List of required modules (all are built-in for this script)
    required_modules = {
        'os': 'Built-in module',
        'sys': 'Built-in module',
        'pathlib': 'Built-in module',
        'datetime': 'Built-in module'
    }
    
    log("All required modules are built-in Python modules ✓", Colors.GREEN)
    time.sleep(0.5)

def format_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}PB"

def get_directory_choice():
    """Interactive prompt to get directory path"""
    log("File Tree Visualizer Ready!", Colors.HEADER)
    print("\nPlease choose a directory to scan:")
    print("1. Current directory")
    print("2. Enter custom path")
    print("3. Home directory")
    
    while True:
        choice = input(f"\n{Colors.YELLOW}Enter your choice (1-3): {Colors.END}").strip()
        
        if choice == '1':
            path = os.getcwd()
            log(f"Selected current directory: {path}", Colors.GREEN)
            return path
        elif choice == '2':
            custom_path = input(f"{Colors.YELLOW}Enter directory path: {Colors.END}").strip()
            if os.path.isdir(custom_path):
                path = os.path.abspath(custom_path)
                log(f"Selected directory: {path}", Colors.GREEN)
                return path
            else:
                log("Invalid directory path! Please try again.", Colors.RED)
        elif choice == '3':
            path = str(Path.home())
            log(f"Selected home directory: {path}", Colors.GREEN)
            return path
        else:
            log("Invalid choice! Please enter 1, 2, or 3.", Colors.RED)

def get_scan_options():
    """Get scanning options from user"""
    options = {}
    
    print(f"\n{Colors.YELLOW}Configuration Options:{Colors.END}")
    
    # Include hidden files
    include_hidden = input("Include hidden files/folders? (y/n) [n]: ").strip().lower()
    options['include_hidden'] = include_hidden == 'y'
    log(f"Include hidden files: {'Yes' if options['include_hidden'] else 'No'}", Colors.CYAN)
    
    # Max depth
    max_depth_input = input("Maximum depth to scan (0 for unlimited) [0]: ").strip()
    try:
        options['max_depth'] = int(max_depth_input) if max_depth_input else 0
        log(f"Maximum depth: {'Unlimited' if options['max_depth'] == 0 else options['max_depth']}", Colors.CYAN)
    except ValueError:
        options['max_depth'] = 0
        log("Invalid input, using unlimited depth", Colors.YELLOW)
    
    # File size info
    show_size = input("Show file sizes? (y/n) [y]: ").strip().lower()
    options['show_size'] = show_size != 'n'
    log(f"Show file sizes: {'Yes' if options['show_size'] else 'No'}", Colors.CYAN)
    
    # Sort files
    sort_files = input("Sort files alphabetically? (y/n) [y]: ").strip().lower()
    options['sort_files'] = sort_files != 'n'
    log(f"Sort files: {'Yes' if options['sort_files'] else 'No'}", Colors.CYAN)
    
    return options

def scan_directory(path, options, current_depth=0, prefix="", is_last=True, output_lines=None, stats=None):
    """Recursively scan directory and build tree structure"""
    if output_lines is None:
        output_lines = []
    if stats is None:
        stats = {'files': 0, 'dirs': 0, 'total_size': 0, 'errors': 0}
    
    # Check max depth
    if options['max_depth'] > 0 and current_depth > options['max_depth']:
        return output_lines, stats
    
    try:
        # Get directory contents
        items = os.listdir(path)
        
        # Filter hidden files if needed
        if not options['include_hidden']:
            items = [item for item in items if not item.startswith('.')]
        
        # Sort if requested
        if options['sort_files']:
            items.sort(key=str.lower)
        
        # Separate directories and files
        dirs = []
        files = []
        
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                dirs.append(item)
            else:
                files.append(item)
        
        # Process directories first, then files
        all_items = dirs + files
        
        for i, item in enumerate(all_items):
            item_path = os.path.join(path, item)
            is_last_item = (i == len(all_items) - 1)
            
            # Determine tree characters
            if current_depth == 0:
                tree_char = ""
                new_prefix = ""
            else:
                tree_char = "└── " if is_last_item else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")
            
            # Check if directory
            is_directory = os.path.isdir(item_path)
            
            # Get file size if it's a file
            size_str = ""
            if not is_directory and options['show_size']:
                try:
                    size = os.path.getsize(item_path)
                    stats['total_size'] += size
                    size_str = f" ({format_size(size)})"
                except:
                    size_str = " (error)"
            
            # Build output line
            icon = "📁" if is_directory else "📄"
            name = item + "/" if is_directory else item
            line = f"{prefix}{tree_char}{icon} {name}{size_str}"
            output_lines.append(line)
            
            # Log progress
            if is_directory:
                stats['dirs'] += 1
                if current_depth < 3:  # Only log first few levels to avoid spam
                    log(f"Scanning directory: {item}", Colors.BLUE)
            else:
                stats['files'] += 1
            
            # Recursively scan subdirectories
            if is_directory:
                scan_directory(item_path, options, current_depth + 1, new_prefix, is_last_item, output_lines, stats)
                
    except PermissionError:
        stats['errors'] += 1
        error_line = f"{prefix}└── ❌ [Permission Denied]"
        output_lines.append(error_line)
        log(f"Permission denied: {path}", Colors.RED)
    except Exception as e:
        stats['errors'] += 1
        error_line = f"{prefix}└── ❌ [Error: {str(e)}]"
        output_lines.append(error_line)
        log(f"Error scanning {path}: {str(e)}", Colors.RED)
    
    return output_lines, stats

def save_tree_to_file(root_path, output_lines, stats, options):
    """Save the tree structure to a text file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"file_tree_{timestamp}.txt"
    
    log(f"Saving tree to file: {filename}", Colors.YELLOW)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("FILE TREE VISUALIZATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Root Path: {root_path}\n")
            f.write(f"Options: Hidden files: {'Yes' if options['include_hidden'] else 'No'}, ")
            f.write(f"Max depth: {'Unlimited' if options['max_depth'] == 0 else options['max_depth']}, ")
            f.write(f"Show sizes: {'Yes' if options['show_size'] else 'No'}, ")
            f.write(f"Sorted: {'Yes' if options['sort_files'] else 'No'}\n")
            f.write("-" * 80 + "\n\n")
            
            # Write tree structure
            root_name = os.path.basename(root_path) or os.path.basename(os.path.dirname(root_path))
            f.write(f"📁 {root_name}/\n")
            for line in output_lines:
                f.write(line + "\n")
            
            # Write statistics
            f.write("\n" + "-" * 80 + "\n")
            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Directories: {stats['dirs']:,}\n")
            f.write(f"Total Files: {stats['files']:,}\n")
            f.write(f"Total Size: {format_size(stats['total_size'])} ({stats['total_size']:,} bytes)\n")
            if stats['errors'] > 0:
                f.write(f"Errors Encountered: {stats['errors']}\n")
            f.write("=" * 80 + "\n")
        
        log(f"Successfully saved tree to {filename}", Colors.GREEN)
        return filename
    except Exception as e:
        log(f"Error saving file: {str(e)}", Colors.RED)
        return None

def display_summary(stats):
    """Display scan summary in terminal"""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.END}")
    print(f"{Colors.HEADER}SCAN COMPLETE{Colors.END}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.END}")
    print(f"{Colors.GREEN}📁 Directories scanned: {stats['dirs']:,}{Colors.END}")
    print(f"{Colors.GREEN}📄 Files found: {stats['files']:,}{Colors.END}")
    print(f"{Colors.GREEN}💾 Total size: {format_size(stats['total_size'])} ({stats['total_size']:,} bytes){Colors.END}")
    if stats['errors'] > 0:
        print(f"{Colors.YELLOW}⚠️  Errors encountered: {stats['errors']}{Colors.END}")

def main():
    """Main function"""
    print(f"{Colors.HEADER}{'=' * 50}")
    print("🌳 SIMPLE FILE TREE VISUALIZER 🌳")
    print(f"{'=' * 50}{Colors.END}\n")
    
    # Check requirements
    check_and_install_requirements()
    
    # Get directory choice
    root_path = get_directory_choice()
    
    # Get scan options
    options = get_scan_options()
    
    # Start scanning
    print(f"\n{Colors.YELLOW}Starting directory scan...{Colors.END}")
    log("Beginning recursive directory scan", Colors.CYAN)
    
    start_time = time.time()
    output_lines, stats = scan_directory(root_path, options)
    end_time = time.time()
    
    scan_duration = end_time - start_time
    log(f"Scan completed in {scan_duration:.2f} seconds", Colors.GREEN)
    
    # Save to file
    output_file = save_tree_to_file(root_path, output_lines, stats, options)
    
    # Display summary
    display_summary(stats)
    
    if output_file:
        print(f"\n{Colors.CYAN}📝 Tree saved to: {output_file}{Colors.END}")
        
        # Ask if user wants to view the file
        view_file = input(f"\n{Colors.YELLOW}Would you like to view the generated file? (y/n): {Colors.END}").strip().lower()
        if view_file == 'y':
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    print(f"\n{Colors.HEADER}FILE CONTENTS:{Colors.END}")
                    print("-" * 50)
                    content = f.read()
                    # Show first 50 lines if file is too long
                    lines = content.split('\n')
                    if len(lines) > 50:
                        print('\n'.join(lines[:50]))
                        print(f"\n{Colors.YELLOW}... (showing first 50 lines of {len(lines)} total) ...{Colors.END}")
                    else:
                        print(content)
            except Exception as e:
                log(f"Error reading file: {str(e)}", Colors.RED)
    
    print(f"\n{Colors.GREEN}✅ All done! Thank you for using File Tree Visualizer!{Colors.END}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation cancelled by user.{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        sys.exit(1)