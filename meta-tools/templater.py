#!/usr/bin/env python3
"""
Enhanced Subfolder and File Generator Script
Creates multiple subfolders and files with advanced features and robust error handling
"""

import os
import sys
import json
import shlex
import traceback
from pathlib import Path
from datetime import datetime



def clean_path(path_input):
    """Clean and normalize path input, handling quotes and special characters"""
    if not path_input:
        return ""

    # Remove leading/trailing whitespace
    path_input = path_input.strip()

    # Handle various quote types
    quotes = ['"', "'", '"', '"', ''', ''']
    for quote in quotes:
        if path_input.startswith(quote) and path_input.endswith(quote):
            path_input = path_input[1:-1]

    # Normalize path separators
    path_input = path_input.replace('\\', '/')

    # Expand user home directory
    if path_input.startswith('~'):
        path_input = os.path.expanduser(path_input)

    # Handle relative paths
    if not os.path.isabs(path_input):
        path_input = os.path.abspath(path_input)

    return path_input

def get_default_content(file_extension):
    """Return default content based on file extension"""
    templates = {
        '.py': '''#!/usr/bin/env python3
"""
Module description here
Created on: {date}
"""

def main():
    """Main function"""
    pass

if __name__ == "__main__":
    main()
''',
        '.js': '''/**
 * JavaScript file
 * Created on: {date}
 */

function main() {
    console.log("Hello World!");
}

main();
''',
        '.html': '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Page</title>
</head>
<body>
    <h1>Hello World!</h1>
</body>
</html>
''',
        '.css': '''/*
 * CSS Stylesheet
 * Created on: {date}
 */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
}
''',
        '.json': '''{
    "name": "example",
    "version": "1.0.0",
    "created": "{date}"
}
''',
        '.md': '''# Title

Created on: {date}

## Description

Add your content here.
''',
        '.sh': '''#!/bin/bash
# Bash script
# Created on: {date}

echo "Hello World!"
''',
        '.bat': '''@echo off
REM Batch script
REM Created on: {date}

echo Hello World!
pause
''',
        '.xml': '''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <created>{date}</created>
    <content>Hello World!</content>
</root>
''',
        '.txt': '''Text file created on {date}

Add your content here.
''',
        '.gitignore': '''# Created on {date}

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
''',
        '.env': '''# Environment variables
# Created on {date}

# Add your environment variables here
# EXAMPLE_VAR=value
''',
        '.yml': '''# YAML configuration
# Created on: {date}

name: example
version: 1.0.0
settings:
  debug: false
''',
        '.yaml': '''# YAML configuration
# Created on: {date}

name: example
version: 1.0.0
settings:
  debug: false
'''
    }

    # Format with current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = templates.get(file_extension.lower(), '')

    if content:
        return content.format(date=current_date)
    return ''

def create_item(base_path, item_name, add_content=True):
    """Create a file or folder based on the item name"""
    try:
        # Clean the item name
        item_name = item_name.strip()
        if not item_name:
            return False, "Empty item name"

        full_path = Path(base_path) / item_name

        # Check if it's a file (has extension) or folder
        if '.' in Path(item_name).name:
            # It's a file
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists
            if full_path.exists():
                overwrite = input(f"  ⚠ File '{full_path}' already exists. Overwrite? (y/n): ").lower()
                if overwrite != 'y':
                    return False, "Skipped existing file"

            # Create the file with content if applicable
            if add_content:
                file_extension = full_path.suffix
                content = get_default_content(file_extension)

                try:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    # Try without content if writing fails
                    full_path.touch(exist_ok=True)
                    return True, f"Created file (empty due to write error): {full_path}"
            else:
                full_path.touch(exist_ok=True)

            return True, f"Created file: {full_path}"
        else:
            # It's a folder
            if full_path.exists():
                return False, f"Folder already exists: {full_path}"

            full_path.mkdir(parents=True, exist_ok=True)
            return True, f"Created folder: {full_path}"

    except PermissionError:
        return False, f"Permission denied: {item_name}"
    except OSError as e:
        return False, f"OS error creating {item_name}: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error creating {item_name}: {str(e)}"

def parse_items_input(items_input):
    """Parse input string into individual items, handling quotes properly"""
    try:
        # Try using shlex for proper quote handling
        items = shlex.split(items_input)
        return items
    except ValueError:
        # If shlex fails, fall back to simple splitting
        items = []
        current_item = ""
        in_quotes = False
        quote_char = None

        for i, char in enumerate(items_input):
            if char in ['"', "'"] and (i == 0 or items_input[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current_item += char
            elif char == " " and not in_quotes:
                if current_item:
                    items.append(current_item)
                    current_item = ""
            else:
                current_item += char

        if current_item:
            items.append(current_item)

        return items



def draw_tree(directory, prefix="", is_last=True, max_depth=5, current_depth=0):
    """Draw directory tree structure"""
    if current_depth >= max_depth:
        return

    try:
        contents = list(Path(directory).iterdir())
        contents.sort(key=lambda x: (x.is_file(), x.name.lower()))

        for i, path in enumerate(contents):
            is_last_item = i == len(contents) - 1
            current_prefix = "└── " if is_last_item else "├── "
            print(prefix + current_prefix + path.name)

            if path.is_dir():
                extension = "    " if is_last_item else "│   "
                draw_tree(path, prefix + extension, is_last_item, max_depth, current_depth + 1)
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
    except Exception as e:
        print(prefix + f"└── [Error: {str(e)}]")

def display_tree(base_path):
    """Display directory tree for the given path"""
    print("\n" + "=" * 50)
    print("Directory Structure:")
    print("=" * 50)
    print(os.path.basename(base_path) + "/")
    draw_tree(base_path)
    print("=" * 50)

def safe_input(prompt, default=""):
    """Safely get user input with error handling"""
    try:
        if default:
            response = input(f"{prompt} [{default}]: ").strip()
            return response if response else default
        else:
            return input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput interrupted.")
        return default
    except Exception as e:
        print(f"Input error: {str(e)}")
        return default

def main():
    """Main function to run the script"""
    print("=" * 50)
    print("Enhanced Subfolder and File Generator v2.0")
    print("=" * 50)

    while True:
        try:
            # Get working path
            print("What path are you working with? [press enter for current path]")
            working_path = safe_input("Path")

            # Clean the path
            working_path = clean_path(working_path)

            if not working_path:
                working_path = os.getcwd()
                print(f"Using current directory: {working_path}")
            else:
                # Validate path
                if not os.path.exists(working_path):
                    create_base = safe_input(f"Path '{working_path}' doesn't exist. Create it? (y/n)").lower()
                    if create_base == 'y':
                        try:
                            Path(working_path).mkdir(parents=True, exist_ok=True)
                            print(f"Created base path: {working_path}")
                        except Exception as e:
                            print(f"Error creating path: {e}")
                            continue
                    else:
                        print("Please enter a valid path.")
                        continue

            # Check if path is writable
            try:
                test_file = Path(working_path) / '.test_write_permission'
                test_file.touch()
                test_file.unlink()
            except Exception:
                print(f"Warning: No write permission in {working_path}")
                continue_anyway = safe_input("Continue anyway? (y/n)").lower()
                if continue_anyway != 'y':
                    continue

            print("\n" + "-" * 50)
            print("What files do you want to make?")
            print("(generates files in current directory unless slashes are used)")
            print("(if no file type indicated it will just create a folder)")
            print("(separate responses by spaces, use quotes for names with spaces)")
            print("-" * 50)

            items_input = safe_input("Items")

            if not items_input:
                print("No items specified. Please try again.")
                continue

            # Parse items
            items = parse_items_input(items_input)

            if not items:
                print("No valid items to create.")
                continue

            print(f"\nGenerating {len(items)} items in {working_path}")
            print("-" * 50)

            # Track results
            success_count = 0
            error_count = 0

            # Create each item (always add content to files)
            for item in items:
                try:
                    success, message = create_item(working_path, item, add_content=True)
                    if success:
                        print(f"  ✓ {message}")
                        success_count += 1
                    else:
                        print(f"  ⚠ {message}")
                        error_count += 1
                except Exception as e:
                    print(f"  ✗ Unexpected error with {item}: {str(e)}")
                    error_count += 1

            print(f"\nSummary: {success_count} created, {error_count} skipped/failed")

            # Show directory tree if items were created
            if success_count > 0:
                try:
                    display_tree(working_path)
                except Exception as e:
                    print(f"Error displaying tree: {e}")

            print("\n" + "=" * 50)
            continue_choice = safe_input("Files and Folders have been generated. Would you like to generate more? (y/n)").lower()

            if continue_choice != 'y':
                print("\nThank you for using the Enhanced Subfolder and File Generator!")
                break
            else:
                print("\n" + "=" * 50)

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nUnexpected error in main loop: {str(e)}")
            print("Debug info:")
            traceback.print_exc()
            retry = safe_input("\nWould you like to try again? (y/n)", "y").lower()
            if retry != 'y':
                break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("\nDebug information:")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
