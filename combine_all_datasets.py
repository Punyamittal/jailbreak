"""
Combine ALL available datasets and create a comprehensive training dataset.
"""

import json
import csv
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    if not file_path.exists():
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"  [WARNING] Error loading {file_path.name}: {e}")
    
    return data

def load_json(file_path: Path) -> List[Dict]:
    """Load JSON file."""
    data = []
    if not file_path.exists():
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except Exception as e:
        print(f"  [WARNING] Error loading {file_path.name}: {e}")
    
    return data

def load_csv(file_path: Path) -> List[Dict]:
    """Load CSV file and convert to standard format."""
    data = []
    if not file_path.exists():
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try different column name variations
                prompt = row.get('prompt') or row.get('text') or row.get('input') or row.get('question')
                label = row.get('label') or row.get('jailbroken') or row.get('violates') or row.get('is_jailbreak')
                
                if prompt:
                    # Convert label to standard format
                    is_jailbreak = False
                    if label:
                        if isinstance(label, str):
                            label_lower = label.lower()
                            is_jailbreak = label_lower in ['true', '1', 'yes', 'jailbreak', 'violates']
                        else:
                            is_jailbreak = bool(label)
                    
                    data.append({
                        'prompt': prompt.strip(),
                        'label': 'jailbreak' if is_jailbreak else 'benign'
                    })
    except Exception as e:
        print(f"  [WARNING] Error loading {file_path.name}: {e}")
    
    return data

def normalize_entry(entry: Dict) -> Dict:
    """Normalize entry to standard format."""
    # Handle different formats
    prompt = entry.get('prompt') or entry.get('text') or entry.get('input') or entry.get('question') or ''
    prompt = str(prompt).strip()
    
    if not prompt:
        return None
    
    # Determine label
    label = entry.get('label')
    if not label:
        # Try other fields
        is_jailbreak = entry.get('jailbroken', False) or entry.get('violates', False) or entry.get('is_jailbreak', False)
        if isinstance(is_jailbreak, str):
            is_jailbreak = is_jailbreak.lower() in ['true', '1', 'yes']
        label = 'jailbreak' if is_jailbreak else 'benign'
    
    # Ensure label is standard
    if label not in ['benign', 'jailbreak']:
        label_lower = str(label).lower()
        if 'jailbreak' in label_lower or 'violate' in label_lower or label_lower in ['true', '1', 'yes']:
            label = 'jailbreak'
        else:
            label = 'benign'
    
    return {
        'prompt': prompt,
        'label': label
    }

def deduplicate(data: List[Dict]) -> List[Dict]:
    """Remove duplicate prompts."""
    seen: Set[str] = set()
    unique_data = []
    
    for entry in data:
        prompt = entry['prompt'].lower().strip()
        if prompt not in seen:
            seen.add(prompt)
            unique_data.append(entry)
    
    return unique_data

def main():
    print("="*70)
    print("COMBINING ALL AVAILABLE DATASETS")
    print("="*70)
    import sys
    sys.stdout.flush()
    
    all_data = []
    
    # 1. Load all JSONL files from datasets/
    print("\n[1/4] Loading JSONL datasets...")
    jsonl_files = [
        Path("datasets/combined_with_custom.jsonl"),
        Path("datasets/combined_with_hf.jsonl"),
        Path("datasets/combined_training_dataset.jsonl"),
        Path("datasets/synthetic_processed.jsonl"),
        Path("datasets/raw_processed.jsonl"),
        Path("datasets/formatted_processed.jsonl"),
        Path("datasets/malignant_labeled.jsonl"),
        Path("datasets/attacks_20251228_210936.jsonl"),
        Path("datasets/benign_20251228_210936.jsonl"),
        Path("datasets/attacks/attacks_20251228.jsonl"),
        Path("datasets/encoded/encoded_20251228.jsonl"),
        Path("datasets/indirect_injection/indirect_20251228.jsonl"),
        Path("datasets/benign/benign_20251228.jsonl"),
    ]
    
    for file_path in jsonl_files:
        if file_path.exists():
            data = load_jsonl(file_path)
            normalized = [normalize_entry(e) for e in data]
            normalized = [e for e in normalized if e is not None]
            all_data.extend(normalized)
            print(f"  [OK] {file_path.name}: {len(normalized)} examples")
    
    # 2. Load JSON files
    print("\n[2/4] Loading JSON datasets...")
    json_files = [
        Path("jailbreak_test_dataset.json"),
    ]
    
    for file_path in json_files:
        if file_path.exists():
            data = load_json(file_path)
            # Convert jailbreak_test_dataset.json format
            if file_path.name == "jailbreak_test_dataset.json":
                normalized = []
                for item in data:
                    prompt = item.get('prompt', '').strip()
                    if prompt:
                        is_jailbreak = item.get('jailbroken', False)
                        normalized.append({
                            'prompt': prompt,
                            'label': 'jailbreak' if is_jailbreak else 'benign'
                        })
            else:
                normalized = [normalize_entry(e) for e in data]
                normalized = [e for e in normalized if e is not None]
            
            all_data.extend(normalized)
            print(f"  [OK] {file_path.name}: {len(normalized)} examples")
    
    # 3. Load CSV files (if not already processed)
    print("\n[3/4] Loading CSV datasets...")
    csv_files = [
        Path("custom_test_data.csv"),
        Path("malignant.csv"),
    ]
    
    for file_path in csv_files:
        if file_path.exists():
            data = load_csv(file_path)
            all_data.extend(data)
            print(f"  [OK] {file_path.name}: {len(data)} examples")
    
    # 4. Deduplicate and save
    print("\n[4/4] Processing combined dataset...")
    print(f"  Total before deduplication: {len(all_data):,}")
    
    # Remove None entries
    all_data = [e for e in all_data if e is not None and e.get('prompt')]
    
    # Deduplicate
    all_data = deduplicate(all_data)
    
    print(f"  Total after deduplication: {len(all_data):,}")
    
    # Count labels
    label_counts = Counter(e['label'] for e in all_data)
    print(f"\n  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count:,} ({count/len(all_data)*100:.1f}%)")
    
    # Save combined dataset
    output_file = Path("datasets/all_datasets_combined.jsonl")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n[OK] Combined dataset saved to: {output_file}")
    print(f"  Total examples: {len(all_data):,}")
    
    print("\n" + "="*70)
    print("[SUCCESS] ALL DATASETS COMBINED!")
    print("="*70)
    print(f"\nNext step: Train the model with:")
    print(f"  python train_balanced_model.py")
    print(f"\nOr update train_balanced_model.py to use:")
    print(f"  datasets/all_datasets_combined.jsonl")

if __name__ == "__main__":
    main()

