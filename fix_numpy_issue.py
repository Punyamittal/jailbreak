"""
Fix NumPy compatibility issue.

sentence-transformers requires NumPy 1.x, but NumPy 2.x is installed.
This script downgrades NumPy to fix the compatibility issue.
"""

import subprocess
import sys

print("="*70)
print("FIXING NUMPY COMPATIBILITY ISSUE")
print("="*70)

print("\nThe issue:")
print("  - NumPy 2.3.5 is installed")
print("  - sentence-transformers requires NumPy 1.x")
print("  - This causes import errors")

print("\nSolution: Downgrade NumPy to 1.x")
print("\nDowngrading NumPy...")

try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy<2.0.0", "--force-reinstall"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("[OK] NumPy downgraded successfully")
        print("\nYou can now run:")
        print("  python train_intent_model.py")
    else:
        print(f"[ERROR] Failed to downgrade NumPy")
        print(result.stderr)
        print("\nTry manually:")
        print("  pip install 'numpy<2' --force-reinstall")
except Exception as e:
    print(f"[ERROR] {e}")
    print("\nTry manually:")
    print("  pip install 'numpy<2' --force-reinstall")

