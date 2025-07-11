import logging
import subprocess
import sys
import importlib
import ast
from pathlib import Path

logger = logging.getLogger("neural_connector.dependency_manager")

IMPORT_TO_PACKAGE_MAP = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "mediapipe": "mediapipe",
    "pandas": "pandas",
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "aiofiles": "aiofiles",
    "openpyxl": "openpyxl",
    "autopy": "autopy",
    "pycaw": "pycaw",
    "networkx": "networkx",
    "pynmea2": "pynmea2",
    "seaborn": "seaborn",  # Added missing dependency
}

VENV_DIR = Path(".venv-neural-connector")

def setup_virtual_environment() -> str:
    """
    Checks for and sets up a virtual environment.
    Returns the path to the python executable within the virtual environment.
    """
    venv_python = VENV_DIR / "bin" / "python"
    if VENV_DIR.exists() and venv_python.exists():
        logger.info("Virtual environment found at %s", VENV_DIR)
        return str(venv_python)

    logger.warning("No virtual environment found. Creating one at %s...", VENV_DIR)
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            check=True, capture_output=True, text=True
        )
        logger.info("Successfully created virtual environment.")
        return str(venv_python)
    except subprocess.CalledProcessError as e:
        logger.critical("Failed to create virtual environment: %s\n%s", e, e.stderr)
        return sys.executable

def get_script_imports(file_path: Path) -> set:
    imports = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        logger.error("Failed to parse imports from %s: %s", file_path, e)
    return imports

def check_and_install_dependencies():
    logger.info("Starting dependency analysis and setup...")
    
    python_executable = setup_virtual_environment()

    experimental_scripts_dir = Path(__file__).parent.parent.parent
    all_imports = set()
    for py_file in experimental_scripts_dir.glob("*.py"):
        if VENV_DIR.name in py_file.parts:
            continue
        all_imports.update(get_script_imports(py_file))

    required_packages = {
        IMPORT_TO_PACKAGE_MAP[imp]
        for imp in all_imports if imp in IMPORT_TO_PACKAGE_MAP
    }
    
    if "mediapipe" in required_packages:
        logger.info("Mediapipe detected. Enforcing numpy version constraint.")
        required_packages.discard("numpy")
        required_packages.add("numpy<2.0")

    logger.info("Required packages to check/install: %s", sorted(list(required_packages)))

    installed_packages = _get_installed_packages(python_executable)
    
    packages_to_install = sorted(list(required_packages), key=lambda p: "numpy" not in p)

    for package in packages_to_install:
        package_name_for_check = package.split('<')[0].split('>')[0].split('=')[0]
        if package_name_for_check.lower() not in installed_packages:
            logger.warning("Package '%s' not found. Attempting installation...", package)
            if not _install_package(python_executable, package):
                logger.critical(
                    "Failed to install '%s'. The application may not run correctly.", package
                )
        else:
            logger.info("Package '%s' is already installed.", package_name_for_check)

def _get_installed_packages(python_executable: str) -> set:
    try:
        result = subprocess.run(
            [python_executable, "-m", "pip", "list"],
            capture_output=True, text=True, check=True
        )
        return {line.split()[0].lower() for line in result.stdout.splitlines()[2:]}
    except Exception as e:
        logger.error("Could not list installed pip packages: %s", e)
        return set()

def _install_package(python_executable: str, package_name: str) -> bool:
    try:
        command = [python_executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", package_name]
        subprocess.check_call(command)
        logger.info("Successfully installed/upgraded package: %s", package_name)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install package '%s'. Pip exited with code %d.", package_name, e.returncode)
        return False

if __name__ == '__main__':
    from logging_config import setup_logging
    setup_logging()
    check_and_install_dependencies()