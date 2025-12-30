"""
Restore Model from Frozen Version

Usage: python restore_model.py
This will restore the model files from this frozen version to models/security/
"""
import shutil
from pathlib import Path

def restore():
    """Restore model from frozen version."""
    frozen_dir = Path(__file__).parent
    target_dir = Path("models/security")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_restore = [
        "security_model.pkl",
        "security_vectorizer.pkl",
        "security_encoder.pkl",
        "security_threshold.txt"
    ]
    
    print("="*70)
    print("RESTORING MODEL FROM FROZEN VERSION")
    print("="*70)
    print(f"\nSource: {frozen_dir}")
    print(f"Target: {target_dir}\n")
    
    restored = []
    for file_name in files_to_restore:
        src = frozen_dir / file_name
        if src.exists():
            dst = target_dir / file_name
            shutil.copy2(src, dst)
            restored.append(file_name)
            print(f"  [OK] Restored {file_name}")
        else:
            print(f"  [WARNING] {file_name} not found in frozen version")
    
    if restored:
        print(f"\n[SUCCESS] Restored {len(restored)} model files")
        print(f"Model is now restored to version: {frozen_dir.name}")
    else:
        print(f"\n[ERROR] No files restored!")

if __name__ == "__main__":
    restore()
