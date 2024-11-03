import sys
print("Python version:", sys.version)
print("Starting import tests...\n")

def test_import(module_name):
    try:
        if module_name == "tensorflow":
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} imported successfully")
        elif module_name == "numpy":
            import numpy as np
            print(f"✓ NumPy {np.__version__} imported successfully")
        elif module_name == "src.utils":
            from src.utils import DataLoader, AdversarialVisualization
            print("✓ Project utils imported successfully")
        elif module_name == "src.attacks":
            from src.attacks import EvasionAttacks, PoisoningAttacks
            print("✓ Project attacks imported successfully")
        elif module_name == "src.defenses":
            from src.defenses import AdversarialDefense
            print("✓ Project defenses imported successfully")
    except Exception as e:
        print(f"✗ Error importing {module_name}: {str(e)}")
        print(f"  Error type: {type(e).__name__}")

# Test core packages first
print("Testing core packages:")
test_import("tensorflow")
test_import("numpy")

print("\nTesting project modules:")
test_import("src.utils")
test_import("src.attacks")
test_import("src.defenses")

print("\nTest completed!")

input("Press Enter to exit...")  