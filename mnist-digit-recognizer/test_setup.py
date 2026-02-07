"""
Test script to verify project setup and dependencies
Run this to check if everything is installed correctly
"""

import sys

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not found. Install with: pip install tensorflow")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found. Install with: pip install numpy")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib not found. Install with: pip install matplotlib")
        return False
    
    try:
        from PIL import Image
        print(f"✓ Pillow (PIL)")
    except ImportError:
        print("✗ Pillow not found. Install with: pip install Pillow")
        return False
    
    return True

def test_scripts():
    """Test if project scripts can be imported"""
    print("\nTesting project scripts...")
    
    try:
        import train_model
        print("✓ train_model.py can be imported")
    except Exception as e:
        print(f"✗ Error importing train_model.py: {e}")
        return False
    
    try:
        import predict
        print("✓ predict.py can be imported")
    except Exception as e:
        print(f"✗ Error importing predict.py: {e}")
        return False
    
    return True

def main():
    print("="*60)
    print("MNIST Digit Recognizer - Environment Test")
    print("="*60)
    print()
    
    imports_ok = test_imports()
    scripts_ok = test_scripts()
    
    print()
    print("="*60)
    if imports_ok and scripts_ok:
        print("✓ All tests passed! You're ready to train the model.")
        print("\nRun: python train_model.py")
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("\n  pip install -r requirements.txt")
    print("="*60)

if __name__ == "__main__":
    main()
