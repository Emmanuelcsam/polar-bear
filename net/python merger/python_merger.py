import os

def merge_python_files(directory, output_filename):
    """
    Finds all .py files in a given directory and its subdirectories,
    and merges them into a single text file.

    Args:
        directory (str): The path to the directory to search.
        output_filename (str): The name of the output file to create.
    """
    # Check if the provided directory path is valid
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found. Please run the script again.")
        return

    try:
        # Open the output file in write mode
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            print(f"Searching for .py files in '{directory}'...")
            
            file_count = 0
            # os.walk recursively goes through the directory tree
            for foldername, subfolders, filenames in os.walk(directory):
                for filename in filenames:
                    # Check if the file has a .py extension
                    if filename.endswith('.py'):
                        file_count += 1
                        file_path = os.path.join(foldername, filename)
                        
                        # Create a distinct header for each file in the merged document
                        header = f"\n{'='*40}\n# File: {file_path}\n{'='*40}\n\n"
                        outfile.write(header)
                        
                        try:
                            # Open and read the content of the python file
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                outfile.write(infile.read())
                        except Exception as e:
                            # Write an error message if a file can't be read
                            outfile.write(f"# ERROR: Could not read file. Reason: {e}\n")

        if file_count > 0:
            print(f"\nSuccess! Merged {file_count} Python files into '{output_filename}'")
        else:
            print(f"\nFinished. No .py files were found in '{directory}'.")


    except IOError as e:
        print(f"Error: Could not write to the output file. Reason: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Interactive Questions ---
    
    # Question 1: Get the directory to search
    target_dir_input = input("Enter the path to the directory you want to search (or press Enter for the current directory): ")
    
    # FIX: Strip leading/trailing whitespace and quote characters from the input
    target_dir = target_dir_input.strip().strip('\'"')

    if not target_dir:
        target_dir = '.' # Default to the current directory
    
    # Question 2: Get the desired name for the output file
    output_file = input("Enter the name for the merged text file (or press Enter for 'merged_scripts.txt'): ")
    if not output_file:
        output_file = 'merged_scripts.txt' # Default output filename
    
    # Ensure the output file has a .txt extension if the user forgets
    if not output_file.lower().endswith('.txt'):
        output_file += '.txt'

    print("-" * 20)
    
    # Run the main function with the user's answers
    merge_python_files(target_dir, output_file)
