from typing import List
from pathlib import Path
import numpy as np
import cv2

from utils import _log_message, _start_timer, _log_duration
from data_structures import FiberSpecifications, ZoneDefinition

def _get_user_specifications(self):
    """Prompts for fiber specs and updates internal state."""
    start_time = _start_timer()
    _log_message("Starting user specification input...")
    print("\n--- Fiber Optic Specifications ---")
    provide_specs_input = input("Provide known fiber specifications (microns)? (y/n, default: n): ").strip().lower()

    if provide_specs_input == 'y':
        _log_message("User chose to provide fiber specifications.")
        try:
            core_dia_str = input(f"Enter CORE diameter in microns (e.g., 9, 50, 62.5) (optional, press Enter to skip): ").strip()
            if core_dia_str: self.fiber_specs.core_diameter_um = float(core_dia_str)
            clad_dia_str = input(f"Enter CLADDING diameter in microns (e.g., 125) (default: {self.fiber_specs.cladding_diameter_um}): ").strip()
            if clad_dia_str: self.fiber_specs.cladding_diameter_um = float(clad_dia_str)
            ferrule_dia_str = input(f"Enter FERRULE outer diameter in microns (e.g., 250) (default: {self.fiber_specs.ferrule_diameter_um}): ").strip()
            if ferrule_dia_str: self.fiber_specs.ferrule_diameter_um = float(ferrule_dia_str)
            self.fiber_specs.fiber_type = input("Enter fiber type (e.g., single-mode, multi-mode) (optional): ").strip()

            if self.fiber_specs.cladding_diameter_um is not None:
                self.operating_mode = "MICRON_CALCULATED"
                _log_message(f"Operating mode set to MICRON_CALCULATED. Specs: Core={self.fiber_specs.core_diameter_um}, Clad={self.fiber_specs.cladding_diameter_um}, Ferrule={self.fiber_specs.ferrule_diameter_um}, Type='{self.fiber_specs.fiber_type}'.")
            else:
                self.operating_mode = "PIXEL_ONLY"
                _log_message("Cladding diameter not provided, falling back to PIXEL_ONLY mode.", level="WARNING")
        except ValueError:
            _log_message("Invalid input for diameter. Proceeding in PIXEL_ONLY mode.", level="ERROR")
            self.operating_mode = "PIXEL_ONLY"
            self.fiber_specs = FiberSpecifications()
    else:
        self.operating_mode = "PIXEL_ONLY"
        _log_message("User chose to skip fiber specifications. Operating mode set to PIXEL_ONLY.")
    _log_duration("User Specification Input", start_time)
    self._initialize_zone_parameters()

def _initialize_zone_parameters(self):
    """Initializes active_zone_definitions based on operating mode and specs."""
    _log_message("Initializing zone parameters...")
    self.active_zone_definitions = []
    if self.operating_mode == "MICRON_CALCULATED" and self.fiber_specs.cladding_diameter_um is not None:
        core_r_um = self.fiber_specs.core_diameter_um / 2.0 if self.fiber_specs.core_diameter_um else 0.0
        cladding_r_um = self.fiber_specs.cladding_diameter_um / 2.0
        ferrule_r_um = self.fiber_specs.ferrule_diameter_um / 2.0 if self.fiber_specs.ferrule_diameter_um else cladding_r_um * 2.0
        adhesive_r_um = ferrule_r_um * 1.1

        default_core = next((z for z in self.config.DEFAULT_ZONES if z.name == "core"), None)
        default_cladding = next((z for z in self.config.DEFAULT_ZONES if z.name == "cladding"), None)
        default_ferrule = next((z for z in self.config.DEFAULT_ZONES if z.name == "ferrule_contact"), None)
        default_adhesive = next((z for z in self.config.DEFAULT_ZONES if z.name == "adhesive"), None)

        self.active_zone_definitions = [
            ZoneDefinition(name="core", r_min_factor_or_um=0.0, r_max_factor_or_um=core_r_um,
                           color_bgr=default_core.color_bgr if default_core else (255,0,0),
                           max_defect_size_um=default_core.max_defect_size_um if default_core else 5.0),
            ZoneDefinition(name="cladding", r_min_factor_or_um=core_r_um, r_max_factor_or_um=cladding_r_um,
                           color_bgr=default_cladding.color_bgr if default_cladding else (0,255,0),
                           max_defect_size_um=default_cladding.max_defect_size_um if default_cladding else 10.0),
            ZoneDefinition(name="ferrule_contact", r_min_factor_or_um=cladding_r_um, r_max_factor_or_um=ferrule_r_um,
                           color_bgr=default_ferrule.color_bgr if default_ferrule else (0,0,255),
                           max_defect_size_um=default_ferrule.max_defect_size_um if default_ferrule else 25.0),
            ZoneDefinition(name="adhesive", r_min_factor_or_um=ferrule_r_um, r_max_factor_or_um=adhesive_r_um,
                           color_bgr=default_adhesive.color_bgr if default_adhesive else (0,255,255),
                           max_defect_size_um=default_adhesive.max_defect_size_um if default_adhesive else 50.0,
                           defects_allowed=default_adhesive.defects_allowed if default_adhesive else False)
        ]
        _log_message(f"Zone parameters set for MICRON_CALCULATED: Core R={core_r_um}µm, Clad R={cladding_r_um}µm.")
    else:
        self.active_zone_definitions = self.config.DEFAULT_ZONES
        _log_message(f"Zone parameters set to default relative factors for {self.operating_mode} mode.")

def _get_image_paths_from_user(self) -> List[Path]:
    """Prompts for image directory and returns list of image Paths."""
    start_time = _start_timer()
    _log_message("Starting image path collection...")
    image_paths: List[Path] = []
    while True:
        dir_path_str = input("Enter the path to the directory containing fiber images: ").strip()
        image_dir = Path(dir_path_str)
        if not image_dir.is_dir():
            _log_message(f"Error: The path '{image_dir}' is not a valid directory. Please try again.", level="ERROR")
            continue
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        for item in image_dir.iterdir():
            if item.is_file() and item.suffix.lower() in supported_extensions:
                image_paths.append(item)
        if not image_paths:
            _log_message(f"No images found in directory: {image_dir}. Please check the path or directory content.", level="WARNING")
        else:
            _log_message(f"Found {len(image_paths)} images in '{image_dir}'.")
            break
    _log_duration("Image Path Collection", start_time)
    return image_paths

def _load_single_image(self, image_path: Path) -> np.ndarray | None:
    """Loads a single image from the given path."""
    _log_message(f"Loading image: {image_path.name}")
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            _log_message(f"Failed to load image: {image_path}", level="ERROR")
            return None
        if len(image.shape) == 2:
            _log_message(f"Image '{image_path.name}' is grayscale. Will be used as is or converted if necessary by specific functions.")
        elif image.shape[2] == 4:
             _log_message(f"Image '{image_path.name}' has an alpha channel. Converting to BGR.")
             image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        _log_message(f"Successfully loaded image: {image_path.name} with shape {image.shape}")
        return image
    except Exception as e:
        _log_message(f"An error occurred while loading image {image_path}: {e}", level="ERROR")
        return None
