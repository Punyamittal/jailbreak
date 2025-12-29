"""
Test if training setup is correct before running full training.
"""

import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

print("="*70)
print("TESTING TRAINING SETUP")
print("="*70)

# Test 1: Check relabeled dataset
print("\n[TEST 1] Checking relabeled dataset...")
relabeled_file = Path("datasets/relabeled/all_relabeled_combined.jsonl")
if not relabeled_file.exists():
    print("  [FAIL] Relabeled dataset not found!")
    print("  Run: python relabel_dataset.py")
    exit(1)

count = 0
labels = []
with open(relabeled_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            label = entry.get('label', '')
            if label in ['benign', 'jailbreak_attempt']:
                labels.append(label)
                count += 1
            if count >= 1000:  # Sample first 1000
                break

print(f"  [OK] Dataset exists with {count} examples sampled")
label_counts = {l: labels.count(l) for l in set(labels)}
print(f"  Label distribution: {label_counts}")

# Test 2: Test LabelEncoder
print("\n[TEST 2] Testing LabelEncoder...")
encoder = LabelEncoder()
encoded = encoder.fit_transform(labels)
print(f"  Classes: {encoder.classes_}")
print(f"  Class indices: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

# Test 3: Test class weights
print("\n[TEST 3] Testing class weights...")
class_to_idx = {name: idx for idx, name in enumerate(encoder.classes_)}
benign_idx = class_to_idx.get('benign', 0)
jailbreak_idx = class_to_idx.get('jailbreak_attempt', 1)

custom_weights = {
    benign_idx: 1,
    jailbreak_idx: 8
}
print(f"  Class weights: {custom_weights}")
print(f"    benign (idx {benign_idx}): weight 1")
print(f"    jailbreak_attempt (idx {jailbreak_idx}): weight 8")

# Test 4: Check imports
print("\n[TEST 4] Checking imports...")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("  [OK] All required imports available")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    exit(1)

print("\n" + "="*70)
print("[SUCCESS] Training setup looks good!")
print("="*70)
print("\nYou can now run:")
print("  python train_security_model.py")

