import logging
import os
import shutil
import subprocess
import keyword
import sys

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('organizer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def install_if_missing(module_name, pip_name):
    try:
        __import__(module_name)
        logger.info(f"{module_name} is already installed.")
    except ImportError:
        logger.info(f"{module_name} not found. Installing latest version of {pip_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', pip_name])
        logger.info(f"Installed/Upgraded {pip_name}.")

# Check and install required libraries
required_modules = [
    ('numpy', 'numpy'),
    ('sklearn', 'scikit-learn'),
    ('sentence_transformers', 'sentence-transformers')
]
for module, pip_name in required_modules:
    install_if_missing(module, pip_name)

# Now import the modules after installation
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def get_folder_name(group_contents):
    if not group_contents:
        return "empty_group"
    
    stop_words = keyword.kwlist + ['print', 'len', 'str', 'int', 'list', 'dict', 'set', 'open', 'file', 'import', 'from']
    vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=r'\b\w+\b')
    try:
        tfidf_matrix = vectorizer.fit_transform(group_contents)
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)
        feature_names = vectorizer.get_feature_names_out()
        top_indices = np.argsort(avg_tfidf)[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices if feature_names[i].isalnum()]
        folder_name = '_'.join(top_words[:5])  # Take up to 5 alphanumeric words
        if not folder_name:
            folder_name = "misc_group"
    except ValueError:
        folder_name = "generic_group"
    
    logger.info(f"Generated folder name: {folder_name}")
    return folder_name

def organize(base_dir, files):
    if len(files) == 0:
        logger.info(f"No files to organize in {base_dir}.")
        return
    
    logger.info(f"Organizing {len(files)} files in {base_dir}.")
    
    # Read contents
    contents = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                content = fp.read()
                contents.append(content)
            logger.info(f"Read content from {f}.")
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            continue
    
    if not contents:
        return
    
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(contents)
    logger.info("Computed embeddings for files.")
    
    # Determine number of clusters
    num_clusters = max(1, len(files) // 15 + (1 if len(files) % 15 > 0 else 0))
    logger.info(f"Clustering into {num_clusters} clusters.")
    
    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    logger.info("Performed clustering.")
    
    # Group files and contents
    groups = [[] for _ in range(num_clusters)]
    group_contents_list = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        groups[label].append(files[i])
        group_contents_list[label].append(contents[i])
    
    # Process each group
    for group_idx, group in enumerate(groups):
        if not group:
            continue
        
        g_contents = group_contents_list[group_idx]
        folder_name = get_folder_name(g_contents)
        
        # Create unique subdir
        subdir = os.path.join(base_dir, folder_name)
        count = 1
        orig_name = folder_name
        while os.path.exists(subdir):
            subdir = os.path.join(base_dir, f"{orig_name}_{count}")
            count += 1
        os.makedirs(subdir)
        logger.info(f"Created subfolder: {subdir}")
        
        # Move files to subdir
        new_files = []
        for f in group:
            new_path = os.path.join(subdir, os.path.basename(f))
            shutil.move(f, new_path)
            logger.info(f"Moved {f} to {new_path}")
            new_files.append(new_path)
        
        # Recurse if necessary (but since we clustered based on size, recurse will check again)
        organize(subdir, new_files)

def main():
    logger.info("Starting script organization process.")
    
    # Ask for directory
    while True:
        dir_path = input("Enter the directory path to organize (press Enter for current directory): ").strip()
        if not dir_path:
            dir_path = os.getcwd()
        logger.info(f"User entered directory: {dir_path}")
        
        if os.path.isdir(dir_path):
            break
        else:
            print("Invalid directory. Please try again.")
            logger.warning(f"Invalid directory entered: {dir_path}")
    
    # Collect Python scripts
    py_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.py')]
    logger.info(f"Found {len(py_files)} Python scripts in {dir_path}.")
    
    if not py_files:
        logger.info("No Python scripts found. Exiting.")
        return
    
    # Start organization
    organize(dir_path, py_files)
    
    logger.info("Organization process completed.")

if __name__ == "__main__":
    main()
