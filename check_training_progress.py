"""
Check training progress and dataset statistics.
"""

import json
from pathlib import Path
from collections import Counter

print("="*70)
print("TRAINING PROGRESS CHECK")
print("="*70)

# Check combined dataset
combined_file = Path("datasets/all_datasets_combined.jsonl")
if combined_file.exists():
    print("\n[OK] Combined dataset exists")
    count = sum(1 for _ in open(combined_file, 'r', encoding='utf-8'))
    print(f"  Total examples: {count:,}")
    
    # Sample and check labels
    labels = []
    with open(combined_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:  # Sample first 10K
                break
            entry = json.loads(line)
            labels.append(entry.get('label', 'unknown'))
    
    label_counts = Counter(labels)
    print(f"  Label distribution (sample of {len(labels):,}):")
    for label, count in label_counts.items():
        print(f"    {label}: {count:,} ({count/len(labels)*100:.1f}%)")
else:
    print("\n[WAITING] Combined dataset not yet created")

# Check model files
print("\n[CHECKING] Model files...")
model_files = [
    "models/balanced_model.pkl",
    "models/balanced_ensemble.pkl",
    "models/balanced_vectorizer.pkl",
    "models/balanced_encoder.pkl"
]

all_exist = True
for model_file in model_files:
    path = Path(model_file)
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {path.name} ({size_mb:.2f} MB)")
    else:
        print(f"  [WAITING] {path.name}")
        all_exist = False

if all_exist:
    print("\n[SUCCESS] All model files exist! Training appears complete.")
else:
    print("\n[IN PROGRESS] Training is still running or not started.")

print("\n" + "="*70)

