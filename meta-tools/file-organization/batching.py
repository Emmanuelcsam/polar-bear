import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def move_files_to_folder(args):
    folder, files, use_copy = args  # Unpack tuple argument for parallel processing
    print(f"Starting to {'copy' if use_copy else 'move'} {len(files)} files to {folder}")  # Status message with operation type and count
    os.makedirs(folder, exist_ok=True)  # Create destination folder, skip if already exists
    for file in files:  # Process each file in the current chunk
        try:
            if use_copy:
                shutil.copy(file, os.path.join(folder, os.path.basename(file)))  # Copy file to destination with original filename
            else:
                shutil.move(file, os.path.join(folder, os.path.basename(file)))  # Move file to destination with original filename
        except Exception as e:
            print(f"Error processing {file}: {e}")  # Log any file operation errors without stopping
    print(f"Finished processing to {folder}")  # Completion message for this chunk

def get_images(current_dir, extensions, recursive):
    images = []  # Initialize empty list to store found image file paths
    if recursive:
        for root, _, files in os.walk(current_dir):  # Walk through all subdirectories recursively
            for f in files:  # Check each file in current directory
                if f.lower().endswith(tuple(extensions)):  # Check if file extension matches allowed image types
                    images.append(os.path.join(root, f))  # Add full path to images list
    else:
        for f in os.listdir(current_dir):  # Only scan current directory, no subdirectories
            if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(tuple(extensions)):  # Verify it's a file and has valid extension
                images.append(os.path.join(current_dir, f))  # Add full path to images list
    return images  # Return complete list of found image file paths

def main():
    current_dir = os.getcwd()  # Get current working directory as base location
    print(f"Current directory: {current_dir}")  # Display working directory to user

    # Interactive configuration
    chunk_size_str = input("Enter chunk size (default 1000): ") or "1000"  # Get user input for batch size, default to 1000
    chunk_size = int(chunk_size_str)  # Convert string input to integer

    extensions_str = input("Enter file extensions separated by space (default png jpg jpeg): ") or "png jpg jpeg"  # Get allowed file types from user
    extensions = ['.' + ext.lower().lstrip('.') for ext in extensions_str.split()]  # Normalize extensions with dots and lowercase

    recursive_str = input("Scan recursively? (y/n, default n): ") or "n"  # Ask if subdirectories should be included
    recursive = recursive_str.lower() == 'y'  # Convert to boolean for recursive scanning

    copy_str = input("Copy instead of move? (y/n, default n): ") or "n"  # Ask whether to copy or move files
    use_copy = copy_str.lower() == 'y'  # Convert to boolean for operation type

    # Get images
    images = get_images(current_dir, extensions, recursive)  # Collect all matching image files based on user settings
    print(f"Found {len(images)} image files with extensions {extensions}")  # Display count of discovered files

    # Sort them for consistent ordering (optional)
    images.sort()  # Alphabetically sort file paths for predictable chunking

    # Split into chunks
    chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]  # Divide images into batches of specified size
    print(f"Creating {len(chunks)} chunks")  # Display number of batches that will be created

    tasks = []  # Initialize list to hold parallel processing tasks
    for idx, chunk in enumerate(chunks, 1):  # Iterate through chunks with 1-based indexing
        folder = os.path.join(current_dir, f"chunk_{idx}")  # Create folder name for current batch
        tasks.append((folder, chunk, use_copy))  # Package arguments for parallel worker function

    # Use ThreadPoolExecutor for parallel processing (IO-bound task)
    with ThreadPoolExecutor(max_workers=8) as executor:  # Create thread pool with 8 workers for concurrent file operations
        executor.map(move_files_to_folder, tasks)  # Execute file operations in parallel across all chunks

    print("All chunks processed")  # Final completion message

if __name__ == "__main__":
    main()  # Execute main function only when script is run directly, not imported
