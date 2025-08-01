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
    if not path_input:  # Check if input is None, empty string, or falsy
        return ""  # Return empty string for invalid input

    # Remove leading/trailing whitespace
    path_input = path_input.strip()  # Strip spaces, tabs, newlines from both ends

    # Handle various quote types
    quotes = ['"', "'", '"', '"', ''', ''']  # List of quote characters including smart quotes
    for quote in quotes:  # Iterate through each quote type
        if path_input.startswith(quote) and path_input.endswith(quote):  # Check if path is wrapped in matching quotes
            path_input = path_input[1:-1]  # Remove first and last character (the quotes)

    # Normalize path separators
    path_input = path_input.replace('\\', '/')  # Convert Windows backslashes to forward slashes

    # Expand user home directory
    if path_input.startswith('~'):  # Check if path starts with tilde (home directory shortcut)
        path_input = os.path.expanduser(path_input)  # Convert ~ to actual home directory path

    # Handle relative paths
    if not os.path.isabs(path_input):  # Check if path is not already absolute
        path_input = os.path.abspath(path_input)  # Convert relative path to absolute path

    return path_input  # Return the cleaned and normalized path

def get_default_content(file_extension):
    """Return default content based on file extension"""
    templates = {  # Dictionary mapping file extensions to their template content
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
''',  # Python template with shebang, docstring, and main function structure
        '.js': '''/**
 * JavaScript file
 * Created on: {date}
 */

function main() {
    console.log("Hello World!");
}

main();
''',  # JavaScript template with comment block and main function
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
''',  # HTML5 template with responsive viewport and basic structure
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
''',  # CSS template with reset styles and basic body styling
        '.json': '''{
    "name": "example",
    "version": "1.0.0",
    "created": "{date}"
}
''',  # JSON template with common package.json-like structure
        '.md': '''# Title

Created on: {date}

## Description

Add your content here.
''',  # Markdown template with basic heading structure
        '.sh': '''#!/bin/bash
# Bash script
# Created on: {date}

echo "Hello World!"
''',  # Bash script template with shebang and basic command
        '.bat': '''@echo off
REM Batch script
REM Created on: {date}

echo Hello World!
pause
''',  # Windows batch file template with echo off and pause
        '.xml': '''<?xml version="1.0" encoding="UTF-8"?>
<root>
    <created>{date}</created>
    <content>Hello World!</content>
</root>
''',  # XML template with declaration and basic structure
        '.txt': '''Text file created on {date}

Add your content here.
''',  # Plain text template with timestamp
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
''',  # Git ignore template with common patterns for Python projects
        '.env': '''# Environment variables
# Created on {date}

# Add your environment variables here
# EXAMPLE_VAR=value
''',  # Environment variables template with example format
        '.yml': '''# YAML configuration
# Created on: {date}

name: example
version: 1.0.0
settings:
  debug: false
''',  # YAML template with basic configuration structure
        '.yaml': '''# YAML configuration
# Created on: {date}

name: example
version: 1.0.0
settings:
  debug: false
'''  # Alternative YAML extension with same structure
    }

    # Format with current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get current timestamp in readable format
    content = templates.get(file_extension.lower(), '')  # Look up template by extension, default to empty string

    if content:  # Check if template was found for this extension
        return content.format(date=current_date)  # Replace {date} placeholder with actual date
    return ''  # Return empty string if no template exists for this extension

def create_item(base_path, item_name, add_content=True):
    """Create a file or folder based on the item name"""
    try:
        # Clean the item name
        item_name = item_name.strip()  # Remove whitespace from start and end of item name
        if not item_name:  # Check if item name is empty after stripping
            return False, "Empty item name"  # Return failure status and error message

        full_path = Path(base_path) / item_name  # Combine base path with item name using pathlib

        # Check if it's a file (has extension) or folder
        if '.' in Path(item_name).name:  # Check if the filename portion contains a dot (indicating file extension)
            # It's a file
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)  # Create all parent directories in the path

            # Check if file already exists
            if full_path.exists():  # Check if file already exists at target location
                overwrite = input(f"  ⚠ File '{full_path}' already exists. Overwrite? (y/n): ").lower()  # Ask user for overwrite permission
                if overwrite != 'y':  # If user doesn't confirm overwrite
                    return False, "Skipped existing file"  # Return failure status with skip message

            # Create the file with content if applicable
            if add_content:  # Check if content should be added to the file
                file_extension = full_path.suffix  # Get the file extension (e.g., '.py', '.txt')
                content = get_default_content(file_extension)  # Get template content for this file type

                try:
                    with open(full_path, 'w', encoding='utf-8') as f:  # Open file for writing with UTF-8 encoding
                        f.write(content)  # Write the template content to the file
                except Exception as e:  # Handle any file writing errors
                    # Try without content if writing fails
                    full_path.touch(exist_ok=True)  # Create empty file if content writing fails
                    return True, f"Created file (empty due to write error): {full_path}"  # Return success with warning
            else:
                full_path.touch(exist_ok=True)  # Create empty file when no content is requested

            return True, f"Created file: {full_path}"  # Return success status with file path
        else:
            # It's a folder
            if full_path.exists():  # Check if folder already exists
                return False, f"Folder already exists: {full_path}"  # Return failure for existing folder

            full_path.mkdir(parents=True, exist_ok=True)  # Create the folder and any missing parent directories
            return True, f"Created folder: {full_path}"  # Return success status with folder path

    except PermissionError:  # Handle permission denied errors
        return False, f"Permission denied: {item_name}"  # Return failure with permission error message
    except OSError as e:  # Handle operating system related errors
        return False, f"OS error creating {item_name}: {str(e)}"  # Return failure with OS error details
    except Exception as e:  # Handle any other unexpected errors
        return False, f"Unexpected error creating {item_name}: {str(e)}"  # Return failure with generic error message

def parse_items_input(items_input):
    """Parse input string into individual items, handling quotes properly"""
    try:
        # Try using shlex for proper quote handling
        items = shlex.split(items_input)  # Use shell-like lexical analysis to split input respecting quotes
        return items  # Return the parsed list of items
    except ValueError:  # Handle cases where shlex parsing fails (unmatched quotes, etc.)
        # If shlex fails, fall back to simple splitting
        items = []  # Initialize empty list to store parsed items
        current_item = ""  # Track the current item being built
        in_quotes = False  # Flag to track if we're inside quoted text
        quote_char = None  # Store the type of quote character we're inside

        for i, char in enumerate(items_input):  # Iterate through each character with its index
            if char in ['"', "'"] and (i == 0 or items_input[i-1] != '\\'):  # Check if char is quote and not escaped
                if not in_quotes:  # If we're not already inside quotes
                    in_quotes = True  # Mark that we're entering quoted text
                    quote_char = char  # Remember which quote character opened this section
                elif char == quote_char:  # If this quote matches the opening quote
                    in_quotes = False  # Mark that we're exiting quoted text
                    quote_char = None  # Clear the quote character tracker
                else:
                    current_item += char  # Add mismatched quote as regular character
            elif char == " " and not in_quotes:  # If space encountered outside of quotes
                if current_item:  # If we have accumulated characters for current item
                    items.append(current_item)  # Add completed item to list
                    current_item = ""  # Reset for next item
            else:
                current_item += char  # Add character to current item being built

        if current_item:  # If there's a remaining item after processing all characters
            items.append(current_item)  # Add final item to list

        return items  # Return the manually parsed list of items



def draw_tree(directory, prefix="", is_last=True, max_depth=5, current_depth=0):
    """Draw directory tree structure"""
    if current_depth >= max_depth:  # Check if we've reached maximum recursion depth
        return  # Exit early to prevent infinite recursion

    try:
        contents = list(Path(directory).iterdir())  # Get all files and folders in current directory
        contents.sort(key=lambda x: (x.is_file(), x.name.lower()))  # Sort with folders first, then files, alphabetically

        for i, path in enumerate(contents):  # Iterate through sorted directory contents with index
            is_last_item = i == len(contents) - 1  # Check if this is the last item in the directory
            current_prefix = "└── " if is_last_item else "├── "  # Choose tree branch character based on position
            print(prefix + current_prefix + path.name)  # Print the tree branch with item name

            if path.is_dir():  # Check if current item is a directory
                extension = "    " if is_last_item else "│   "  # Choose prefix for next level based on position
                draw_tree(path, prefix + extension, is_last_item, max_depth, current_depth + 1)  # Recursively draw subdirectory
    except PermissionError:  # Handle directories without read permission
        print(prefix + "└── [Permission Denied]")  # Display error message in tree format
    except Exception as e:  # Handle any other errors accessing directory
        print(prefix + f"└── [Error: {str(e)}]")  # Display generic error message in tree format

def display_tree(base_path):
    """Display directory tree for the given path"""
    print("\n" + "=" * 50)  # Print blank line followed by separator line
    print("Directory Structure:")  # Print header for tree display
    print("=" * 50)  # Print separator line
    print(os.path.basename(base_path) + "/")  # Print root directory name with trailing slash
    draw_tree(base_path)  # Call recursive function to draw the tree structure
    print("=" * 50)  # Print closing separator line

def safe_input(prompt, default=""):
    """Safely get user input with error handling"""
    try:
        if default:  # Check if a default value was provided
            response = input(f"{prompt} [{default}]: ").strip()  # Show prompt with default in brackets, strip whitespace
            return response if response else default  # Return user input or default if empty
        else:
            return input(f"{prompt}: ").strip()  # Show prompt without default, strip whitespace from response
    except (EOFError, KeyboardInterrupt):  # Handle Ctrl+C or EOF signals
        print("\nInput interrupted.")  # Inform user that input was interrupted
        return default  # Return default value when interrupted
    except Exception as e:  # Handle any other unexpected input errors
        print(f"Input error: {str(e)}")  # Display error message to user
        return default  # Return default value on error

def main():
    """Main function to run the script"""
    print("=" * 50)  # Print header separator line
    print("Enhanced Subfolder and File Generator v2.0")  # Print application title
    print("=" * 50)  # Print header separator line

    while True:  # Start infinite loop for multiple operations
        try:
            # Get working path
            print("What path are you working with? [press enter for current path]")  # Show instructions to user
            working_path = safe_input("Path")  # Get path input from user

            # Clean the path
            working_path = clean_path(working_path)  # Normalize and clean the input path

            if not working_path:  # Check if no path was provided after cleaning
                working_path = os.getcwd()  # Use current working directory as default
                print(f"Using current directory: {working_path}")  # Inform user of default choice
            else:
                # Validate path
                if not os.path.exists(working_path):  # Check if the specified path exists
                    create_base = safe_input(f"Path '{working_path}' doesn't exist. Create it? (y/n)").lower()  # Ask to create missing path
                    if create_base == 'y':  # If user confirms creation
                        try:
                            Path(working_path).mkdir(parents=True, exist_ok=True)  # Create directory and all parent directories
                            print(f"Created base path: {working_path}")  # Confirm successful creation
                        except Exception as e:  # Handle directory creation errors
                            print(f"Error creating path: {e}")  # Display error message
                            continue  # Skip to next iteration of main loop
                    else:
                        print("Please enter a valid path.")  # Inform user to try again
                        continue  # Skip to next iteration of main loop

            # Check if path is writable
            try:
                test_file = Path(working_path) / '.test_write_permission'  # Create test file path
                test_file.touch()  # Attempt to create test file
                test_file.unlink()  # Remove test file after successful creation
            except Exception:  # Handle write permission errors
                print(f"Warning: No write permission in {working_path}")  # Warn user about permission issues
                continue_anyway = safe_input("Continue anyway? (y/n)").lower()  # Ask if user wants to proceed
                if continue_anyway != 'y':  # If user doesn't want to continue
                    continue  # Skip to next iteration of main loop

            print("\n" + "-" * 50)  # Print section separator
            print("What files do you want to make?")  # Prompt for file/folder list
            print("(generates files in current directory unless slashes are used)")  # Explain relative path behavior
            print("(if no file type indicated it will just create a folder)")  # Explain folder creation logic
            print("(separate responses by spaces, use quotes for names with spaces)")  # Explain input format
            print("-" * 50)  # Print section separator

            items_input = safe_input("Items")  # Get list of items to create from user

            if not items_input:  # Check if no items were specified
                print("No items specified. Please try again.")  # Inform user to provide input
                continue  # Skip to next iteration of main loop

            # Parse items
            items = parse_items_input(items_input)  # Parse input string into individual items

            if not items:  # Check if parsing resulted in empty list
                print("No valid items to create.")  # Inform user that parsing failed
                continue  # Skip to next iteration of main loop

            print(f"\nGenerating {len(items)} items in {working_path}")  # Show count and location of items to create
            print("-" * 50)  # Print separator line

            # Track results
            success_count = 0  # Initialize counter for successful creations
            error_count = 0  # Initialize counter for failed creations

            # Create each item (always add content to files)
            for item in items:  # Iterate through each item to create
                try:
                    success, message = create_item(working_path, item, add_content=True)  # Attempt to create item with content
                    if success:  # Check if creation was successful
                        print(f"  ✓ {message}")  # Print success message with checkmark
                        success_count += 1  # Increment success counter
                    else:
                        print(f"  ⚠ {message}")  # Print warning message with warning symbol
                        error_count += 1  # Increment error counter
                except Exception as e:  # Handle unexpected errors during creation
                    print(f"  ✗ Unexpected error with {item}: {str(e)}")  # Print error message with X symbol
                    error_count += 1  # Increment error counter

            print(f"\nSummary: {success_count} created, {error_count} skipped/failed")  # Display final statistics

            # Show directory tree if items were created
            if success_count > 0:  # Check if any items were successfully created
                try:
                    display_tree(working_path)  # Display directory tree structure
                except Exception as e:  # Handle errors in tree display
                    print(f"Error displaying tree: {e}")  # Show error message if tree display fails

            print("\n" + "=" * 50)  # Print section separator
            continue_choice = safe_input("Files and Folders have been generated. Would you like to generate more? (y/n)").lower()  # Ask if user wants to continue

            if continue_choice != 'y':  # Check if user doesn't want to continue
                print("\nThank you for using the Enhanced Subfolder and File Generator!")  # Print goodbye message
                break  # Exit the main loop
            else:
                print("\n" + "=" * 50)  # Print separator for next iteration

        except KeyboardInterrupt:  # Handle Ctrl+C interruption
            print("\n\nOperation cancelled by user.")  # Inform user of cancellation
            break  # Exit the main loop
        except Exception as e:  # Handle any other unexpected errors
            print(f"\nUnexpected error in main loop: {str(e)}")  # Display error message
            print("Debug info:")  # Label for debug information
            traceback.print_exc()  # Print full stack trace for debugging
            retry = safe_input("\nWould you like to try again? (y/n)", "y").lower()  # Ask if user wants to retry
            if retry != 'y':  # If user doesn't want to retry
                break  # Exit the main loop

if __name__ == "__main__":  # Check if script is being run directly (not imported)
    try:
        main()  # Call the main function to start the application
    except Exception as e:  # Handle any unhandled exceptions at the top level
        print(f"\nFatal error: {str(e)}")  # Display fatal error message
        print("\nDebug information:")  # Label for debug section
        traceback.print_exc()  # Print full stack trace for debugging
        input("\nPress Enter to exit...")  # Wait for user input before closing
        sys.exit(1)  # Exit with error code 1 to indicate failure
