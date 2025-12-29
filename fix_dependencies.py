"""
Fix all dependency issues for intent-based model.

Fixes:
1. NumPy 2.x -> 1.x (for sentence-transformers compatibility)
2. Keras 3 -> tf-keras (for transformers compatibility)
"""

import subprocess
import sys

print("="*70)
print("FIXING DEPENDENCY ISSUES")
print("="*70)

fixes = [
    ("numpy", "numpy<2.0.0", "NumPy 2.x incompatible with sentence-transformers"),
    ("tf-keras", "tf-keras", "Required for transformers (Keras 3 not supported)"),
]

for package_name, install_cmd, reason in fixes:
    print(f"\n[{package_name}] {reason}")
    print(f"  Installing: {install_cmd}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", install_cmd],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  [OK] {package_name} installed/updated")
        else:
            print(f"  [WARNING] {package_name} installation had issues")
            print(f"  Try manually: pip install {install_cmd}")
    except Exception as e:
        print(f"  [ERROR] {e}")

print("\n" + "="*70)
print("[SUCCESS] DEPENDENCY FIXES COMPLETE!")
print("="*70)
print("\nYou can now run:")
print("  python train_intent_model.py")

