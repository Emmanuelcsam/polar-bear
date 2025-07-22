import os
import shutil
from concurrent.futures import ThreadPoolExecutor

def move_files_to_folder(args):
    folder, files, use_copy = args
    print(f"Starting to {'copy' if use_copy else 'move'} {len(files)} files to {folder}")
    os.makedirs(folder, exist_ok=True)
    for file in files:
        try:
            if use_copy:
                shutil.copy(file, os.path.join(folder, os.path.basename(file)))
            else:
                shutil.move(file, os.path.join(folder, os.path.basename(file)))
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print(f"Finished processing to {folder}")

def get_images(current_dir, extensions, recursive):
    images = []
    if recursive:
        for root, _, files in os.walk(current_dir):
            for f in files:
                if f.lower().endswith(tuple(extensions)):
                    images.append(os.path.join(root, f))
    else:
        for f in os.listdir(current_dir):
            if os.path.isfile(os.path.join(current_dir, f)) and f.lower().endswith(tuple(extensions)):
                images.append(os.path.join(current_dir, f))
    return images

def main():
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")

    # Interactive configuration
    chunk_size_str = input("Enter chunk size (default 1000): ") or "1000"
    chunk_size = int(chunk_size_str)

    extensions_str = input("Enter file extensions separated by space (default png jpg jpeg): ") or "png jpg jpeg"
    extensions = ['.' + ext.lower().lstrip('.') for ext in extensions_str.split()]

    recursive_str = input("Scan recursively? (y/n, default n): ") or "n"
    recursive = recursive_str.lower() == 'y'

    copy_str = input("Copy instead of move? (y/n, default n): ") or "n"
    use_copy = copy_str.lower() == 'y'

    # Get images
    images = get_images(current_dir, extensions, recursive)
    print(f"Found {len(images)} image files with extensions {extensions}")

    # Sort them for consistent ordering (optional)
    images.sort()

    # Split into chunks
    chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]
    print(f"Creating {len(chunks)} chunks")

    tasks = []
    for idx, chunk in enumerate(chunks, 1):
        folder = os.path.join(current_dir, f"chunk_{idx}")
        tasks.append((folder, chunk, use_copy))

    # Use ThreadPoolExecutor for parallel processing (IO-bound task)
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(move_files_to_folder, tasks)

    print("All chunks processed")

if __name__ == "__main__":
    main()
