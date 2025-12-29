"""
Train model with ALL available datasets - simplified version.
"""

import json
import csv
from pathlib import Path
from collections import Counter
import sys

# Ensure output is flushed
sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("COMBINING ALL DATASETS AND TRAINING")
print("="*70)
sys.stdout.flush()

all_data = []

# Load JSONL files
print("\n[1] Loading JSONL datasets...")
jsonl_paths = [
    "datasets/combined_with_custom.jsonl",
    "datasets/combined_with_hf.jsonl", 
    "datasets/combined_training_dataset.jsonl",
    "datasets/synthetic_processed.jsonl",
    "datasets/raw_processed.jsonl",
    "datasets/formatted_processed.jsonl",
    "datasets/malignant_labeled.jsonl",
    "datasets/attacks_20251228_210936.jsonl",
    "datasets/benign_20251228_210936.jsonl",
    "datasets/attacks/attacks_20251228.jsonl",
    "datasets/encoded/encoded_20251228.jsonl",
    "datasets/indirect_injection/indirect_20251228.jsonl",
    "datasets/benign/benign_20251228.jsonl",
]

for path_str in jsonl_paths:
    path = Path(path_str)
    if path.exists():
        try:
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        prompt = entry.get('prompt') or entry.get('text', '').strip()
                        if prompt:
                            label = entry.get('label', 'benign')
                            if label not in ['benign', 'jailbreak']:
                                # Convert
                                is_jb = entry.get('jailbroken', False) or entry.get('violates', False)
                                if isinstance(is_jb, str):
                                    is_jb = is_jb.lower() in ['true', '1', 'yes']
                                label = 'jailbreak' if is_jb else 'benign'
                            
                            all_data.append({'prompt': prompt, 'label': label})
                            count += 1
            if count > 0:
                print(f"  [OK] {path.name}: {count:,} examples")
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

# Load JSON files
print("\n[2] Loading JSON datasets...")
json_paths = ["jailbreak_test_dataset.json"]

for path_str in json_paths:
    path = Path(path_str)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            for item in data:
                prompt = item.get('prompt', '').strip()
                if prompt:
                    is_jb = item.get('jailbroken', False)
                    all_data.append({
                        'prompt': prompt,
                        'label': 'jailbreak' if is_jb else 'benign'
                    })
                    count += 1
            if count > 0:
                print(f"  [OK] {path.name}: {count:,} examples")
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

# Load CSV files
print("\n[3] Loading CSV datasets...")
csv_paths = ["custom_test_data.csv", "malignant.csv"]

for path_str in csv_paths:
    path = Path(path_str)
    if path.exists():
        try:
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt = row.get('prompt') or row.get('text', '').strip()
                    if prompt:
                        label_str = row.get('label') or row.get('violates', 'false')
                        is_jb = str(label_str).lower() in ['true', '1', 'yes']
                        all_data.append({
                            'prompt': prompt,
                            'label': 'jailbreak' if is_jb else 'benign'
                        })
                        count += 1
            if count > 0:
                print(f"  [OK] {path.name}: {count:,} examples")
        except Exception as e:
            print(f"  [WARN] {path.name}: {e}")

# Deduplicate
print("\n[4] Deduplicating...")
seen = set()
unique_data = []
for entry in all_data:
    prompt_lower = entry['prompt'].lower().strip()
    if prompt_lower and prompt_lower not in seen:
        seen.add(prompt_lower)
        unique_data.append(entry)

print(f"  Before: {len(all_data):,}")
print(f"  After:  {len(unique_data):,}")

# Label distribution
label_counts = Counter(e['label'] for e in unique_data)
print(f"\n  Label distribution:")
for label, count in label_counts.items():
    print(f"    {label}: {count:,} ({count/len(unique_data)*100:.1f}%)")

# Save
output_file = Path("datasets/all_datasets_combined.jsonl")
output_file.parent.mkdir(exist_ok=True)

print(f"\n[5] Saving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in unique_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"  [OK] Saved {len(unique_data):,} examples")

# Train
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

try:
    from train_balanced_model import train_balanced_model
    
    model = train_balanced_model(
        dataset_path=str(output_file),
        balance=True,
        clean=True,
        use_ensemble=True
    )
    
    print("\n" + "="*70)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*70)
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

