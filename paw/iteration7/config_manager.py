import json
import os
import time
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class ModuleConfig:
    """Configuration for a single module"""
    name: str
    enabled: bool = True
    auto_run: bool = False
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.parameters is None:
            self.parameters = {}

class ConfigManager:
    """Manages system-wide configuration"""
    
    def __init__(self):
        self.config_file = 'system_config.json'
        self.modules = {}
        self.global_settings = {
            'version': '1.0',
            'auto_save': True,
            'log_level': 'INFO',
            'output_directory': '.',
            'max_file_size_mb': 100,
            'cleanup_old_files': False,
            'cleanup_days': 7
        }
        
        self.load_config()
        self.discover_modules()
    
    def discover_modules(self):
        """Discover all available modules"""
        
        print("[CONFIG] Discovering modules...")
        
        module_patterns = {
            # Core modules
            'pixel_reader': {'category': 'core', 'description': 'Reads pixel data from images'},
            'random_generator': {'category': 'core', 'description': 'Generates random pixel values'},
            'correlator': {'category': 'core', 'description': 'Finds correlations between data streams'},
            
            # Analysis modules
            'pattern_recognizer': {'category': 'analysis', 'description': 'Detects patterns in pixel data'},
            'anomaly_detector': {'category': 'analysis', 'description': 'Identifies anomalies'},
            'intensity_analyzer': {'category': 'analysis', 'description': 'Analyzes intensity distributions'},
            'geometry_analyzer': {'category': 'analysis', 'description': 'Finds geometric patterns'},
            'trend_analyzer': {'category': 'analysis', 'description': 'Analyzes trends over time'},
            'data_calculator': {'category': 'analysis', 'description': 'Performs advanced calculations'},
            
            # AI modules
            'neural_learner': {'category': 'ai', 'description': 'Neural network learning'},
            'neural_generator': {'category': 'ai', 'description': 'Generates images using neural networks'},
            'vision_processor': {'category': 'ai', 'description': 'Computer vision processing'},
            'hybrid_analyzer': {'category': 'ai', 'description': 'Combines AI insights'},
            'ml_classifier': {'category': 'ai', 'description': 'Machine learning classification'},
            
            # HPC modules
            'gpu_accelerator': {'category': 'hpc', 'description': 'GPU-accelerated processing'},
            'gpu_image_generator': {'category': 'hpc', 'description': 'GPU-based image generation'},
            'parallel_processor': {'category': 'hpc', 'description': 'Multi-core parallel processing'},
            'distributed_analyzer': {'category': 'hpc', 'description': 'Distributed computing'},
            'hpc_optimizer': {'category': 'hpc', 'description': 'HPC optimization'},
            
            # Real-time modules
            'realtime_processor': {'category': 'realtime', 'description': 'Real-time monitoring'},
            'live_capture': {'category': 'realtime', 'description': 'Live video capture'},
            'stream_analyzer': {'category': 'realtime', 'description': 'Stream analysis'},
            
            # Utility modules
            'batch_processor': {'category': 'utility', 'description': 'Batch image processing'},
            'image_generator': {'category': 'utility', 'description': 'Basic image generation'},
            'image_categorizer': {'category': 'utility', 'description': 'Categorizes images'},
            'learning_engine': {'category': 'utility', 'description': 'Learning system'},
            'data_store': {'category': 'utility', 'description': 'Data storage management'},
            'continuous_analyzer': {'category': 'utility', 'description': 'Continuous analysis'},
            'logger': {'category': 'utility', 'description': 'Logging system'},
            'visualizer': {'category': 'utility', 'description': 'Basic visualization'},
            'advanced_visualizer': {'category': 'utility', 'description': 'Advanced visualization'},
            'network_api': {'category': 'utility', 'description': 'Network API server'},
            'data_exporter': {'category': 'utility', 'description': 'Data export/import'}
        }
        
        discovered = 0
        for module_name, info in module_patterns.items():
            module_file = f"{module_name}.py"
            if os.path.exists(module_file):
                if module_name not in self.modules:
                    self.modules[module_name] = ModuleConfig(
                        name=module_name,
                        enabled=True,
                        auto_run=False,
                        parameters={
                            'category': info['category'],
                            'description': info['description']
                        }
                    )
                discovered += 1
        
        print(f"[CONFIG] Discovered {discovered} modules")
    
    def load_config(self):
        """Load configuration from file"""
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Load global settings
                self.global_settings.update(config.get('global_settings', {}))
                
                # Load module configurations
                for module_name, module_config in config.get('modules', {}).items():
                    self.modules[module_name] = ModuleConfig(**module_config)
                
                print(f"[CONFIG] Loaded configuration from {self.config_file}")
                
            except Exception as e:
                print(f"[CONFIG] Error loading config: {e}")
    
    def save_config(self):
        """Save current configuration"""
        
        config = {
            'global_settings': self.global_settings,
            'modules': {
                name: asdict(module) 
                for name, module in self.modules.items()
            },
            'last_updated': time.time()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[CONFIG] Saved configuration to {self.config_file}")
    
    def set_module_enabled(self, module_name: str, enabled: bool):
        """Enable or disable a module"""
        
        if module_name in self.modules:
            self.modules[module_name].enabled = enabled
            print(f"[CONFIG] Module '{module_name}' {'enabled' if enabled else 'disabled'}")
            
            if self.global_settings['auto_save']:
                self.save_config()
        else:
            print(f"[CONFIG] Module '{module_name}' not found")
    
    def set_module_parameter(self, module_name: str, param: str, value: Any):
        """Set a module parameter"""
        
        if module_name in self.modules:
            self.modules[module_name].parameters[param] = value
            print(f"[CONFIG] Set {module_name}.{param} = {value}")
            
            if self.global_settings['auto_save']:
                self.save_config()
        else:
            print(f"[CONFIG] Module '{module_name}' not found")
    
    def get_enabled_modules(self, category: str = None) -> List[str]:
        """Get list of enabled modules"""
        
        enabled = []
        for name, module in self.modules.items():
            if module.enabled:
                if category is None or module.parameters.get('category') == category:
                    enabled.append(name)
        
        return enabled
    
    def create_pipeline_config(self, pipeline_name: str, modules: List[str]):
        """Create a pipeline configuration"""
        
        pipeline = {
            'name': pipeline_name,
            'modules': modules,
            'created': time.time()
        }
        
        # Save pipeline
        pipelines_file = 'pipelines.json'
        
        if os.path.exists(pipelines_file):
            with open(pipelines_file, 'r') as f:
                pipelines = json.load(f)
        else:
            pipelines = {}
        
        pipelines[pipeline_name] = pipeline
        
        with open(pipelines_file, 'w') as f:
            json.dump(pipelines, f, indent=2)
        
        print(f"[CONFIG] Created pipeline '{pipeline_name}' with {len(modules)} modules")
    
    def generate_report(self):
        """Generate configuration report"""
        
        report = {
            'timestamp': time.time(),
            'global_settings': self.global_settings,
            'module_summary': {
                'total': len(self.modules),
                'enabled': len([m for m in self.modules.values() if m.enabled]),
                'disabled': len([m for m in self.modules.values() if not m.enabled])
            },
            'categories': {}
        }
        
        # Count by category
        for module in self.modules.values():
            category = module.parameters.get('category', 'unknown')
            if category not in report['categories']:
                report['categories'][category] = {
                    'total': 0,
                    'enabled': 0,
                    'modules': []
                }
            
            report['categories'][category]['total'] += 1
            if module.enabled:
                report['categories'][category]['enabled'] += 1
            report['categories'][category]['modules'].append(module.name)
        
        # Save report
        with open('config_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_status(self):
        """Print current configuration status"""
        
        print("\n=== SYSTEM CONFIGURATION ===")
        print(f"\nGlobal Settings:")
        for key, value in self.global_settings.items():
            print(f"  {key}: {value}")
        
        print(f"\nModules ({len(self.modules)} total):")
        
        # Group by category
        categories = {}
        for name, module in self.modules.items():
            category = module.parameters.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append((name, module))
        
        for category, modules in sorted(categories.items()):
            print(f"\n  {category.upper()}:")
            for name, module in sorted(modules):
                status = "✓" if module.enabled else "✗"
                auto = " [AUTO]" if module.auto_run else ""
                print(f"    {status} {name}{auto}")

def create_preset_configurations():
    """Create preset configuration files"""
    
    print("\n[CONFIG] Creating preset configurations...")
    
    # Basic configuration
    basic_config = {
        'name': 'basic',
        'description': 'Basic pixel analysis',
        'modules': [
            'pixel_reader',
            'pattern_recognizer',
            'anomaly_detector',
            'visualizer'
        ]
    }
    
    # AI configuration
    ai_config = {
        'name': 'ai_powered',
        'description': 'AI and machine learning analysis',
        'modules': [
            'pixel_reader',
            'vision_processor',
            'neural_learner',
            'neural_generator',
            'ml_classifier',
            'hybrid_analyzer'
        ]
    }
    
    # HPC configuration
    hpc_config = {
        'name': 'high_performance',
        'description': 'High performance computing',
        'modules': [
            'pixel_reader',
            'gpu_accelerator',
            'parallel_processor',
            'distributed_analyzer',
            'hpc_optimizer'
        ]
    }
    
    # Real-time configuration
    realtime_config = {
        'name': 'real_time',
        'description': 'Real-time processing',
        'modules': [
            'live_capture',
            'realtime_processor',
            'stream_analyzer'
        ]
    }
    
    # Full configuration
    full_config = {
        'name': 'full_system',
        'description': 'All modules enabled',
        'modules': 'all'
    }
    
    # Save presets
    presets = {
        'basic': basic_config,
        'ai_powered': ai_config,
        'high_performance': hpc_config,
        'real_time': realtime_config,
        'full_system': full_config
    }
    
    with open('config_presets.json', 'w') as f:
        json.dump(presets, f, indent=2)
    
    print("[CONFIG] Created configuration presets")
    
    # Also create YAML version for readability
    with open('config_presets.yaml', 'w') as f:
        yaml.dump(presets, f, default_flow_style=False)
    
    return presets

def apply_preset(preset_name: str):
    """Apply a preset configuration"""
    
    if not os.path.exists('config_presets.json'):
        create_preset_configurations()
    
    with open('config_presets.json', 'r') as f:
        presets = json.load(f)
    
    if preset_name not in presets:
        print(f"[CONFIG] Preset '{preset_name}' not found")
        print(f"[CONFIG] Available presets: {', '.join(presets.keys())}")
        return
    
    preset = presets[preset_name]
    config = ConfigManager()
    
    # Disable all modules first
    for module_name in config.modules:
        config.set_module_enabled(module_name, False)
    
    # Enable preset modules
    if preset['modules'] == 'all':
        for module_name in config.modules:
            config.set_module_enabled(module_name, True)
    else:
        for module_name in preset['modules']:
            config.set_module_enabled(module_name, True)
    
    print(f"[CONFIG] Applied preset '{preset_name}'")
    config.save_config()

def interactive_config():
    """Interactive configuration mode"""
    
    print("\n=== INTERACTIVE CONFIGURATION ===")
    
    config = ConfigManager()
    
    while True:
        print("\nOptions:")
        print("1. View current configuration")
        print("2. Enable/disable module")
        print("3. Apply preset")
        print("4. Create pipeline")
        print("5. Generate report")
        print("6. Save and exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            config.print_status()
            
        elif choice == '2':
            module_name = input("Module name: ").strip()
            if module_name in config.modules:
                current = config.modules[module_name].enabled
                new_state = not current
                config.set_module_enabled(module_name, new_state)
            else:
                print(f"Module '{module_name}' not found")
                
        elif choice == '3':
            print("\nAvailable presets:")
            if os.path.exists('config_presets.json'):
                with open('config_presets.json', 'r') as f:
                    presets = json.load(f)
                for name, preset in presets.items():
                    print(f"  - {name}: {preset['description']}")
            
            preset_name = input("\nPreset name: ").strip()
            apply_preset(preset_name)
            config.load_config()  # Reload after applying preset
            
        elif choice == '4':
            pipeline_name = input("Pipeline name: ").strip()
            modules_str = input("Modules (comma-separated): ").strip()
            modules = [m.strip() for m in modules_str.split(',')]
            config.create_pipeline_config(pipeline_name, modules)
            
        elif choice == '5':
            report = config.generate_report()
            print(f"\n[CONFIG] Report generated: config_report.json")
            print(f"[CONFIG] Total modules: {report['module_summary']['total']}")
            print(f"[CONFIG] Enabled: {report['module_summary']['enabled']}")
            
        elif choice == '6':
            config.save_config()
            print("[CONFIG] Configuration saved")
            break
        
        else:
            print("Invalid option")

def main():
    """Main configuration management"""
    
    print("=== CONFIGURATION MANAGER ===\n")
    
    # Initialize configuration
    config = ConfigManager()
    
    # Show current status
    config.print_status()
    
    # Create presets if they don't exist
    if not os.path.exists('config_presets.json'):
        create_preset_configurations()
    
    # Generate report
    report = config.generate_report()
    
    print(f"\n[CONFIG] Configuration summary:")
    print(f"  Total modules: {report['module_summary']['total']}")
    print(f"  Enabled: {report['module_summary']['enabled']}")
    print(f"  Disabled: {report['module_summary']['disabled']}")
    
    print("\n[CONFIG] To configure interactively, run:")
    print("  python config_manager.py interactive")
    
    print("\n[CONFIG] To apply a preset, run:")
    print("  python config_manager.py preset <name>")
    
    print("\n[CONFIG] Available presets:")
    print("  - basic: Basic pixel analysis")
    print("  - ai_powered: AI and ML analysis")
    print("  - high_performance: HPC with GPU")
    print("  - real_time: Real-time processing")
    print("  - full_system: All modules enabled")
    
    # Save configuration
    config.save_config()
    
    print("\n[CONFIG] Configuration files created:")
    print("  - system_config.json")
    print("  - config_presets.json")
    print("  - config_report.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            interactive_config()
        elif sys.argv[1] == 'preset' and len(sys.argv) > 2:
            apply_preset(sys.argv[2])
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        main()