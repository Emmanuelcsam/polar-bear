import os
import subprocess
import sys

print("="*60)
print("INTELLIGENT IMAGE CATEGORIZER")
print("Random Pixel Correlation System")
print("="*60)

# Auto-install dependencies
print("\nChecking dependencies...")
exec(open('auto_installer.py').read())

print("\n" + "="*60)
print("MAIN MENU")
print("="*60)

while True:
    print("\n1. Build pixel database from references")
    print("2. Analyze single image")
    print("3. Batch process directory")
    print("4. Review categorizations")
    print("5. Optimize learning")
    print("6. Live monitoring mode")
    print("7. Exit")
    
    choice = input("\nSelect option: ")
    
    if choice == '1':
        print("\n" + "-"*40)
        print("BUILDING PIXEL DATABASE")
        print("-"*40)
        subprocess.run([sys.executable, 'pixel_sampler.py'])
        
    elif choice == '2':
        print("\n" + "-"*40)
        print("SINGLE IMAGE ANALYSIS")
        print("-"*40)
        if not os.path.exists('pixel_db.pkl'):
            print("⚠ No pixel database found! Build it first (option 1)")
            continue
        subprocess.run([sys.executable, 'correlation_analyzer.py'])
        
    elif choice == '3':
        print("\n" + "-"*40)
        print("BATCH PROCESSING")
        print("-"*40)
        if not os.path.exists('pixel_db.pkl'):
            print("⚠ No pixel database found! Build it first (option 1)")
            continue
        subprocess.run([sys.executable, 'batch_processor.py'])
        
    elif choice == '4':
        print("\n" + "-"*40)
        print("REVIEWING CATEGORIZATIONS")
        print("-"*40)
        subprocess.run([sys.executable, 'self_reviewer.py'])
        
    elif choice == '5':
        print("\n" + "-"*40)
        print("LEARNING OPTIMIZATION")
        print("-"*40)
        subprocess.run([sys.executable, 'learning_optimizer.py'])
        
    elif choice == '6':
        print("\n" + "-"*40)
        print("LIVE MONITORING")
        print("-"*40)
        if not os.path.exists('pixel_db.pkl'):
            print("⚠ No pixel database found! Build it first (option 1)")
            continue
        subprocess.run([sys.executable, 'live_monitor.py'])
        
    elif choice == '7':
        print("\nShutting down...")
        break
    
    else:
        print("Invalid option!")

print("\n✓ System shutdown complete")