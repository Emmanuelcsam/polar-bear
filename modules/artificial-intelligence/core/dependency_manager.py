
import ast
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# List of standard library modules to ignore.
# This list can be expanded.
STANDARD_LIBS = {
    "argparse", "collections", "cv2", "datetime", "hashlib", "itertools", "json",
    "logging", "numpy", "os", "pathlib", "random", "re", "shutil", "sqlite3",
    "subprocess", "sys", "time", "torch", "typing", "unittest", "warnings",
    "tensorflow", "skimage"
}

def get_imported_libraries(file_path: Path) -> set:
    """
    Parses a Python file and returns a set of imported library names.
    """
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
    except (SyntaxError, UnicodeDecodeError) as e:
        logger.warning(f"Could not parse {file_path}: {e}")
    return imports

def get_all_imports(project_root: Path) -> set:
    """
    Scans all python files in a directory and returns all unique imports.
    """
    all_imports = set()
    for py_file in project_root.glob('**/*.py'):
        if "mock" in str(py_file).lower(): # Exclude mock files
            continue
        logger.debug(f"Scanning {py_file} for imports...")
        all_imports.update(get_imported_libraries(py_file))
    return all_imports

def is_installed(package_name: str) -> bool:
    """
    Checks if a package is installed.
    """
    # A special case for opencv-python
    if package_name == "cv2":
        package_name = "opencv-python"
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ValueError, ModuleNotFoundError):
        return False


def install_package(package_name: str):
    """
    Installs a package using pip.
    """
    logger.info(f"Attempting to install {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        logger.info(f"Successfully installed {package_name}.")
    except subprocess.CalledProcessError:
        logger.error(f"Failed to install {package_name}. Please install it manually.")
        # Optionally, exit or raise an exception
        # sys.exit(1)

def check_dependencies():
    """
    The main function to check for and install missing dependencies.
    """
    logger.info("Starting dependency check...")
    project_root = Path('.')
    all_imports = get_all_imports(project_root)
    
    logger.info(f"Found imports: {', '.join(sorted(list(all_imports)))}")

    missing_packages = []
    for package in all_imports:
        if package in STANDARD_LIBS:
            continue
        if not is_installed(package):
            missing_packages.append(package)

    if not missing_packages:
        logger.info("All dependencies are satisfied.")
        return

    logger.warning(f"Missing packages detected: {', '.join(missing_packages)}")
    
    # Ask user for confirmation to install
    try:
        answer = input(f"Do you want to install them? (y/n): ").lower()
        if answer not in ('y', 'yes'):
            logger.info("Installation aborted by user.")
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        logger.info("\nInstallation aborted by user.")
        sys.exit(0)


    for package in missing_packages:
        install_package(package)

    logger.info("Dependency check finished.")

if __name__ == '__main__':
    # This is for standalone testing of the dependency manager
    from logging_manager import setup_logging
    setup_logging()
    check_dependencies()
