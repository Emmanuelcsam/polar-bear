import json
import os
from connector_interface import setup_connector, send_hivemind_status

print("Configuration Wizard")
print("="*40)

# Setup hivemind connector
connector = setup_connector('config-wizard.py')

config = {}

if os.path.exists('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)
    print("✓ Loaded existing configuration")
    
    modify = input("\nModify existing config? (y/n): ")
    if modify != 'y':
        print("Keeping existing configuration")
        exit()

print("\nLet's configure your system...")

# Send status update
send_hivemind_status({'status': 'configuring', 'step': 'reference_dir'}, connector)

config['reference_dir'] = input("\nPath to reference images directory: ")
while not os.path.exists(config['reference_dir']):
    print("⚠ Directory not found!")
    config['reference_dir'] = input("Path to reference images directory: ")

config['pixels_per_image'] = int(input("\nPixels to sample per reference image (default 100): ") or "100")

config['batch_size'] = int(input("\nBatch processing size (default 10): ") or "10")

config['confidence_threshold'] = float(input("\nMinimum confidence threshold (0-1, default 0.5): ") or "0.5")

config['learning_rate'] = float(input("\nLearning rate for weight updates (default 0.1): ") or "0.1")

config['comparison_samples'] = int(input("\nComparisons per category during analysis (default 100): ") or "100")

print("\nAdvanced Settings:")
advanced = input("Configure advanced settings? (y/n): ")

if advanced == 'y':
    config['enable_logging'] = input("Enable detailed logging? (y/n): ") == 'y'
    config['auto_optimize'] = input("Auto-optimize after batch processing? (y/n): ") == 'y'
    config['save_snapshots'] = input("Save analysis snapshots? (y/n): ") == 'y'
    config['prune_threshold'] = float(input("Weight threshold for pruning (default 0.3): ") or "0.3")
else:
    config['enable_logging'] = True
    config['auto_optimize'] = False
    config['save_snapshots'] = False
    config['prune_threshold'] = 0.3

# Save configuration
with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✓ Configuration saved to config.json")
print("\nYour configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Send completion status
send_hivemind_status({
    'status': 'completed',
    'config': config,
    'file': 'config.json'
}, connector)

print("\nYou can now run main.py to start the system!")