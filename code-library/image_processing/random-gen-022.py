import numpy as np, cv2, time, pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from core.datastore import put, get, log
from core.connector_interface import ScriptConnectorInterface, connector_exposed

SIZE = 32
OUT = pathlib.Path(__file__).resolve().parent.parent / "data"

class RandomPixelModule(ScriptConnectorInterface):
    """Random pixel generator with connector interface"""
    
    def __init__(self):
        super().__init__("random_pixel", __file__)
        
        # Register controllable parameters
        self.register_parameter("size", SIZE, self._on_size_change)
        self.register_parameter("output_dir", str(OUT), self._on_output_dir_change)
        self.register_parameter("color_mode", "grayscale")  # grayscale or rgb
        
        # Register observable variables
        self.register_variable("last_generated_image", None)
        self.register_variable("generation_count", 0)
        self.register_variable("last_generation_time", None)
        
        self._size = SIZE
        self._output_dir = OUT
        
    def _on_size_change(self, new_size):
        """Callback when size parameter changes"""
        self._size = new_size
        self.logger.info(f"Image size changed to {new_size}")
        
    def _on_output_dir_change(self, new_dir):
        """Callback when output directory changes"""
        self._output_dir = pathlib.Path(new_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory changed to {new_dir}")
        
    @connector_exposed
    def generate(self):
        """Generate a random image"""
        color_mode = self.parameters.get("color_mode", "grayscale")
        
        if color_mode == "rgb":
            img = np.random.randint(0, 256, (self._size, self._size, 3), np.uint8)
        else:
            img = np.random.randint(0, 256, (self._size, self._size), np.uint8)
            
        name = f"rand_{int(time.time()*1000)}.png"
        output_path = self._output_dir / name
        cv2.imwrite(str(output_path), img)
        put(f"rand:{name}", img)
        
        # Update variables
        self.update_variable("last_generated_image", name)
        self.update_variable("generation_count", self.variables["generation_count"] + 1)
        self.update_variable("last_generation_time", time.time())
        
        return {"image": name, "shape": img.shape, "path": str(output_path)}
        
    @connector_exposed
    def generate_guided(self):
        """Generate image based on learned distribution"""
        dist = get("dist")
        if dist is None:
            return self.generate()
            
        flat = np.random.choice(256, self._size * self._size, p=dist).astype(np.uint8)
        img = flat.reshape(self._size, self._size)
        
        name = f"guided_{int(time.time()*1000)}.png"
        output_path = self._output_dir / name
        cv2.imwrite(str(output_path), img)
        put(f"guided:{name}", img)
        
        # Update variables
        self.update_variable("last_generated_image", name)
        self.update_variable("generation_count", self.variables["generation_count"] + 1)
        self.update_variable("last_generation_time", time.time())
        
        return {"image": name, "shape": img.shape, "path": str(output_path), "type": "guided"}
        
    @connector_exposed  
    def get_statistics(self):
        """Get generation statistics"""
        return {
            "total_generated": self.variables["generation_count"],
            "last_image": self.variables["last_generated_image"],
            "last_time": self.variables["last_generation_time"],
            "current_size": self._size,
            "output_directory": str(self._output_dir)
        }
        
    def get_info(self):
        """Get module information"""
        return {
            "name": "Random Pixel Generator",
            "version": "1.0",
            "description": "Generates random pixel images with configurable size and color mode",
            "parameters": self.parameters,
            "statistics": self.get_statistics()
        }


# Module instance for connector integration
_module_instance = None

def _get_module():
    """Get or create module instance"""
    global _module_instance
    if _module_instance is None:
        _module_instance = RandomPixelModule()
    return _module_instance

# Legacy interface for backward compatibility
def gen():
    """Generate random image (legacy interface)"""
    module = _get_module()
    result = module.generate()
    
    # Return image array for compatibility
    img_path = pathlib.Path(result["path"])
    if img_path.exists():
        return cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    return np.zeros((SIZE, SIZE), np.uint8)

def guided():
    """Generate guided image (legacy interface)"""
    module = _get_module()
    result = module.generate_guided()
    
    # Return image array for compatibility
    img_path = pathlib.Path(result["path"])
    if img_path.exists():
        return cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    return np.zeros((SIZE, SIZE), np.uint8)

# Initialize module when imported
log("rand", "module ready")
_get_module()  # Create instance to register with connector