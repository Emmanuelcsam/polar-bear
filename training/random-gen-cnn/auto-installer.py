import subprocess
import sys

libs = ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'scikit-learn']
for lib in libs:
    try:
        __import__(lib.replace('-', '_'))
        print(f"✓ {lib} already installed")
    except:
        print(f"Installing {lib}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', lib])
        print(f"✓ {lib} installed successfully")