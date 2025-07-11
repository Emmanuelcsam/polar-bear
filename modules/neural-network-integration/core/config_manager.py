#!/usr/bin/env python3
"""
Interactive Configuration Management System
Provides intelligent configuration with validation, persistence, and runtime adjustment
"""

import os
import sys
import json
import yaml
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Type, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import getpass
import platform
try:
    import psutil
except ImportError:
    psutil = None
try:
    import GPUtil
except ImportError:
    GPUtil = None
import importlib

from .logger import logger, LogChannel


class ConfigType(Enum):
    """Configuration value types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    CHOICE = "choice"
    PASSWORD = "password"
    URL = "url"
    EMAIL = "email"


class ConfigLevel(Enum):
    """Configuration complexity levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ConfigOption:
    """Configuration option definition"""
    name: str
    description: str
    type: ConfigType
    default: Any
    required: bool = True
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    validation: Optional[Callable[[Any], bool]] = None
    level: ConfigLevel = ConfigLevel.INTERMEDIATE
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    affects: List[str] = field(default_factory=list)
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a configuration value"""
        # Type validation
        if self.type == ConfigType.STRING and not isinstance(value, str):
            return False, f"Expected string, got {type(value).__name__}"
        
        elif self.type == ConfigType.INTEGER:
            if not isinstance(value, int):
                return False, f"Expected integer, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be <= {self.max_value}"
        
        elif self.type == ConfigType.FLOAT:
            if not isinstance(value, (int, float)):
                return False, f"Expected number, got {type(value).__name__}"
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be >= {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be <= {self.max_value}"
        
        elif self.type == ConfigType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Expected boolean, got {type(value).__name__}"
        
        elif self.type == ConfigType.LIST:
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        
        elif self.type == ConfigType.DICT:
            if not isinstance(value, dict):
                return False, f"Expected dictionary, got {type(value).__name__}"
        
        elif self.type == ConfigType.PATH:
            if not isinstance(value, str):
                return False, "Path must be a string"
            path = Path(value)
            if self.required and not path.exists():
                return False, f"Path does not exist: {value}"
        
        elif self.type == ConfigType.CHOICE:
            if self.choices and value not in self.choices:
                return False, f"Invalid choice. Options: {', '.join(map(str, self.choices))}"
        
        elif self.type == ConfigType.URL:
            if not isinstance(value, str):
                return False, "URL must be a string"
            if not value.startswith(('http://', 'https://', 'ftp://', 'file://')):
                return False, "Invalid URL format"
        
        elif self.type == ConfigType.EMAIL:
            if not isinstance(value, str):
                return False, "Email must be a string"
            if '@' not in value or '.' not in value.split('@')[1]:
                return False, "Invalid email format"
        
        # Custom validation
        if self.validation:
            try:
                if not self.validation(value):
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {str(e)}"
        
        return True, None


@dataclass
class ConfigProfile:
    """Configuration profile for different environments"""
    name: str
    description: str
    values: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    locked: bool = False
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)


class ConfigurationManager:
    """Main configuration management system"""
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.options: Dict[str, ConfigOption] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.current_profile = "default"
        self.values: Dict[str, Any] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        
        # System detection
        self.system_info = self._detect_system()
        
        # Load configurations
        self._load_profiles()
        self._register_default_options()
        
        logger.info(LogChannel.SYSTEM, "Configuration manager initialized",
                   config_dir=str(self.config_dir))
    
    def _detect_system(self) -> Dict[str, Any]:
        """Detect system capabilities"""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count() if psutil else os.cpu_count() or 1,
            'cpu_freq': psutil.cpu_freq().current if psutil and psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total if psutil else 8 * 1024 * 1024 * 1024,  # 8GB default
            'memory_available': psutil.virtual_memory().available if psutil else 4 * 1024 * 1024 * 1024,  # 4GB default
            'disk_total': psutil.disk_usage('/').total if psutil else 100 * 1024 * 1024 * 1024,  # 100GB default
            'disk_free': psutil.disk_usage('/').free if psutil else 50 * 1024 * 1024 * 1024,  # 50GB default
            'python_version': sys.version,
            'has_gpu': False,
            'gpu_count': 0,
            'gpu_names': []
        }
        
        # Check for GPU
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    info['has_gpu'] = True
                    info['gpu_count'] = len(gpus)
                    info['gpu_names'] = [gpu.name for gpu in gpus]
            except:
                pass
        
        return info
    
    def _register_default_options(self):
        """Register default configuration options"""
        # System options
        self.register_option(ConfigOption(
            name="system.cpu_cores",
            description="Number of CPU cores to use",
            type=ConfigType.INTEGER,
            default=max(1, self.system_info['cpu_count'] - 1),
            min_value=1,
            max_value=self.system_info['cpu_count'],
            category="system",
            level=ConfigLevel.INTERMEDIATE
        ))
        
        self.register_option(ConfigOption(
            name="system.memory_limit",
            description="Maximum memory usage (MB)",
            type=ConfigType.INTEGER,
            default=int(self.system_info['memory_total'] / 1024 / 1024 * 0.8),
            min_value=512,
            max_value=int(self.system_info['memory_total'] / 1024 / 1024),
            category="system",
            level=ConfigLevel.ADVANCED
        ))
        
        self.register_option(ConfigOption(
            name="system.use_gpu",
            description="Enable GPU acceleration",
            type=ConfigType.BOOLEAN,
            default=self.system_info['has_gpu'],
            category="system",
            level=ConfigLevel.BEGINNER
        ))
        
        # Logging options
        self.register_option(ConfigOption(
            name="logging.level",
            description="Logging verbosity level",
            type=ConfigType.CHOICE,
            default="INFO",
            choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            category="logging",
            level=ConfigLevel.BEGINNER
        ))
        
        self.register_option(ConfigOption(
            name="logging.file_enabled",
            description="Enable file logging",
            type=ConfigType.BOOLEAN,
            default=True,
            category="logging",
            level=ConfigLevel.BEGINNER
        ))
        
        self.register_option(ConfigOption(
            name="logging.max_file_size",
            description="Maximum log file size (MB)",
            type=ConfigType.INTEGER,
            default=100,
            min_value=1,
            max_value=1000,
            category="logging",
            level=ConfigLevel.INTERMEDIATE
        ))
        
        # Neural network options
        self.register_option(ConfigOption(
            name="neural.batch_size",
            description="Default batch size for processing",
            type=ConfigType.INTEGER,
            default=32,
            min_value=1,
            max_value=1024,
            category="neural_network",
            level=ConfigLevel.INTERMEDIATE
        ))
        
        self.register_option(ConfigOption(
            name="neural.learning_rate",
            description="Default learning rate",
            type=ConfigType.FLOAT,
            default=0.001,
            min_value=0.00001,
            max_value=1.0,
            category="neural_network",
            level=ConfigLevel.ADVANCED
        ))
        
        self.register_option(ConfigOption(
            name="neural.auto_tune",
            description="Enable automatic parameter tuning",
            type=ConfigType.BOOLEAN,
            default=True,
            category="neural_network",
            level=ConfigLevel.BEGINNER
        ))
        
        # Performance options
        self.register_option(ConfigOption(
            name="performance.cache_enabled",
            description="Enable result caching",
            type=ConfigType.BOOLEAN,
            default=True,
            category="performance",
            level=ConfigLevel.INTERMEDIATE
        ))
        
        self.register_option(ConfigOption(
            name="performance.cache_size",
            description="Maximum cache size (MB)",
            type=ConfigType.INTEGER,
            default=1024,
            min_value=64,
            max_value=8192,
            category="performance",
            level=ConfigLevel.ADVANCED
        ))
    
    def register_option(self, option: ConfigOption):
        """Register a configuration option"""
        with self._lock:
            self.options[option.name] = option
            
            # Set default value if not already set
            if option.name not in self.values:
                self.values[option.name] = option.default
    
    def get(self, name: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.values.get(name, default)
    
    def set(self, name: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value"""
        with self._lock:
            if name not in self.options:
                logger.warning(LogChannel.SYSTEM, f"Unknown configuration option: {name}")
                return False
            
            option = self.options[name]
            
            # Validate if requested
            if validate:
                valid, error = option.validate(value)
                if not valid:
                    logger.error(LogChannel.SYSTEM, f"Invalid configuration value for {name}: {error}")
                    return False
            
            # Store old value
            old_value = self.values.get(name)
            
            # Set new value
            self.values[name] = value
            
            # Update profile
            if self.current_profile in self.profiles:
                self.profiles[self.current_profile].values[name] = value
                self.profiles[self.current_profile].modified_at = time.time()
            
            # Trigger callbacks
            if name in self.callbacks:
                for callback in self.callbacks[name]:
                    try:
                        callback(name, old_value, value)
                    except Exception as e:
                        logger.error(LogChannel.SYSTEM, f"Configuration callback error: {str(e)}")
            
            logger.info(LogChannel.SYSTEM, f"Configuration updated: {name}",
                       old_value=old_value, new_value=value)
            
            return True
    
    def register_callback(self, name: str, callback: Callable[[str, Any, Any], None]):
        """Register callback for configuration changes"""
        with self._lock:
            if name not in self.callbacks:
                self.callbacks[name] = []
            self.callbacks[name].append(callback)
    
    def interactive_setup(self, level: ConfigLevel = ConfigLevel.BEGINNER):
        """Interactive configuration wizard"""
        print("\n🔧 Neural Network Configuration Wizard 🔧")
        print("=" * 50)
        print(f"Configuration level: {level.value}")
        print("Press Ctrl+C at any time to cancel\n")
        
        # Group options by category
        categories = {}
        for name, option in self.options.items():
            if option.level.value <= level.value:
                if option.category not in categories:
                    categories[option.category] = []
                categories[option.category].append((name, option))
        
        try:
            # Configure each category
            for category, options in sorted(categories.items()):
                print(f"\n📁 {category.upper()}")
                print("-" * 40)
                
                for name, option in sorted(options):
                    current_value = self.get(name, option.default)
                    
                    # Show current value
                    print(f"\n{option.description}")
                    if option.type != ConfigType.PASSWORD:
                        print(f"Current value: {current_value}")
                    
                    # Get new value
                    new_value = self._prompt_for_value(option, current_value)
                    
                    if new_value != current_value:
                        if self.set(name, new_value):
                            print("✅ Updated successfully")
                        else:
                            print("❌ Update failed")
            
            # Save configuration
            profile_name = input("\n💾 Save as profile name (or press Enter for 'default'): ").strip()
            if not profile_name:
                profile_name = "default"
            
            self.save_profile(profile_name)
            print(f"\n✅ Configuration saved to profile: {profile_name}")
            
        except KeyboardInterrupt:
            print("\n\n❌ Configuration cancelled")
            return False
        
        return True
    
    def _prompt_for_value(self, option: ConfigOption, current_value: Any) -> Any:
        """Prompt user for configuration value"""
        # Build prompt
        prompt = f"Enter new value"
        
        if option.type == ConfigType.CHOICE and option.choices:
            prompt += f" {option.choices}"
        elif option.type == ConfigType.INTEGER or option.type == ConfigType.FLOAT:
            if option.min_value is not None and option.max_value is not None:
                prompt += f" [{option.min_value}-{option.max_value}]"
        elif option.type == ConfigType.BOOLEAN:
            prompt += " (true/false)"
        
        prompt += f" [Enter for {current_value}]: "
        
        # Get input
        if option.type == ConfigType.PASSWORD:
            value = getpass.getpass(prompt)
        else:
            value = input(prompt).strip()
        
        # Return current if empty
        if not value:
            return current_value
        
        # Parse value based on type
        try:
            if option.type == ConfigType.INTEGER:
                return int(value)
            elif option.type == ConfigType.FLOAT:
                return float(value)
            elif option.type == ConfigType.BOOLEAN:
                return value.lower() in ('true', 'yes', '1', 'on')
            elif option.type == ConfigType.LIST:
                return [v.strip() for v in value.split(',')]
            elif option.type == ConfigType.DICT:
                return json.loads(value)
            else:
                return value
        except Exception as e:
            print(f"❌ Invalid input: {str(e)}")
            return self._prompt_for_value(option, current_value)
    
    def save_profile(self, name: str, description: str = ""):
        """Save current configuration as a profile"""
        with self._lock:
            profile = ConfigProfile(
                name=name,
                description=description,
                values=self.values.copy()
            )
            
            self.profiles[name] = profile
            
            # Save to file
            profile_file = self.config_dir / f"{name}.json"
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2)
            
            logger.info(LogChannel.SYSTEM, f"Configuration profile saved: {name}")
    
    def load_profile(self, name: str) -> bool:
        """Load a configuration profile"""
        with self._lock:
            # Check if profile exists
            if name not in self.profiles:
                profile_file = self.config_dir / f"{name}.json"
                if not profile_file.exists():
                    logger.error(LogChannel.SYSTEM, f"Profile not found: {name}")
                    return False
                
                # Load from file
                try:
                    with open(profile_file, 'r') as f:
                        data = json.load(f)
                    
                    profile = ConfigProfile(**data)
                    self.profiles[name] = profile
                except Exception as e:
                    logger.error(LogChannel.SYSTEM, f"Failed to load profile: {name}", error=str(e))
                    return False
            
            # Apply profile values
            profile = self.profiles[name]
            for key, value in profile.values.items():
                self.set(key, value, validate=False)
            
            self.current_profile = name
            logger.info(LogChannel.SYSTEM, f"Configuration profile loaded: {name}")
            return True
    
    def _load_profiles(self):
        """Load all saved profiles"""
        for profile_file in self.config_dir.glob("*.json"):
            name = profile_file.stem
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                
                # Handle old format
                if isinstance(data, dict) and 'name' in data:
                    profile = ConfigProfile(**data)
                else:
                    # Old format - just values
                    profile = ConfigProfile(name=name, values=data)
                
                self.profiles[name] = profile
            except Exception as e:
                logger.warning(LogChannel.SYSTEM, f"Failed to load profile {name}: {str(e)}")
    
    def export_config(self, format: str = "json") -> str:
        """Export configuration to string"""
        data = {
            'profile': self.current_profile,
            'values': self.values,
            'system_info': self.system_info,
            'timestamp': time.time()
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        elif format == "yaml":
            return yaml.dump(data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_config(self, data: str, format: str = "json"):
        """Import configuration from string"""
        try:
            if format == "json":
                config = json.loads(data)
            elif format == "yaml":
                config = yaml.safe_load(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Apply values
            for key, value in config.get('values', {}).items():
                if key in self.options:
                    self.set(key, value)
            
            logger.info(LogChannel.SYSTEM, "Configuration imported successfully")
            
        except Exception as e:
            logger.error(LogChannel.SYSTEM, f"Failed to import configuration: {str(e)}")
            raise
    
    def auto_detect_optimal_settings(self):
        """Automatically detect and set optimal configuration"""
        logger.info(LogChannel.SYSTEM, "Auto-detecting optimal settings...")
        
        # CPU settings
        cpu_count = self.system_info['cpu_count']
        if cpu_count >= 8:
            self.set('system.cpu_cores', cpu_count - 2)
        elif cpu_count >= 4:
            self.set('system.cpu_cores', cpu_count - 1)
        else:
            self.set('system.cpu_cores', max(1, cpu_count - 1))
        
        # Memory settings
        memory_gb = self.system_info['memory_total'] / 1024 / 1024 / 1024
        if memory_gb >= 32:
            self.set('system.memory_limit', 24576)  # 24GB
            self.set('performance.cache_size', 4096)  # 4GB
        elif memory_gb >= 16:
            self.set('system.memory_limit', 12288)  # 12GB
            self.set('performance.cache_size', 2048)  # 2GB
        elif memory_gb >= 8:
            self.set('system.memory_limit', 6144)   # 6GB
            self.set('performance.cache_size', 1024)  # 1GB
        else:
            self.set('system.memory_limit', int(memory_gb * 0.7 * 1024))
            self.set('performance.cache_size', 512)
        
        # GPU settings
        if self.system_info['has_gpu']:
            self.set('system.use_gpu', True)
            
            # Check for specific GPU capabilities
            gpu_names = self.system_info['gpu_names']
            if any('RTX' in name for name in gpu_names):
                self.set('neural.batch_size', 64)
            elif any('GTX' in name for name in gpu_names):
                self.set('neural.batch_size', 32)
            else:
                self.set('neural.batch_size', 16)
        else:
            self.set('system.use_gpu', False)
            self.set('neural.batch_size', 16)
        
        logger.success(LogChannel.SYSTEM, "Optimal settings detected and applied")


# Global configuration instance
config_manager = ConfigurationManager()


if __name__ == "__main__":
    # Demo the configuration system
    print("🚀 Configuration Manager Demo")
    print("=" * 50)
    
    # Show system info
    print("\n📊 System Information:")
    for key, value in config_manager.system_info.items():
        if key not in ['python_version']:
            print(f"  {key}: {value}")
    
    # Run interactive setup
    print("\nStarting interactive configuration...")
    config_manager.interactive_setup(ConfigLevel.INTERMEDIATE)
    
    # Show final configuration
    print("\n📋 Final Configuration:")
    for name, value in sorted(config_manager.values.items()):
        print(f"  {name}: {value}")
    
    print("\n✅ Configuration demo complete!")