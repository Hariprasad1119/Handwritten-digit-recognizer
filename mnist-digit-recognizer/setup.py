#!/usr/bin/env python3
"""
Setup script for MNIST Digit Recognizer
Automates environment setup and dependency installation
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"✗ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")
    dirs = ['models', 'results']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies...")
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages"
    )

def verify_installation():
    """Verify all dependencies are installed correctly"""
    print("\nVerifying installation...")
    
    packages = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow'
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {name} installed successfully")
        except ImportError:
            print(f"✗ {name} not found")
            all_ok = False
    
    return all_ok

def main():
    """Main setup function"""
    print("="*70)
    print(" "*15 + "MNIST DIGIT RECOGNIZER - SETUP")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher.")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    print("\nThis will install: TensorFlow, NumPy, Matplotlib, and Pillow")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    if not install_dependencies():
        print("\n✗ Installation failed. Try manually: pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Some packages failed to install correctly.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ Setup completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Train the model:  python train_model.py")
    print("  2. Make predictions: python predict.py")
    print("  3. Run examples:     python examples.py")
    print("\nFor more information, see README.md")
    print("="*70)

if __name__ == "__main__":
    main()
