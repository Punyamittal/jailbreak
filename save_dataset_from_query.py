"""
Save the jailbreak dataset provided in the user query to a JSON file.
Run this first to save the dataset, then run test_on_jailbreak_dataset.py
"""

import json

# Paste the full JSON array here or load from a file
# For now, we'll create a script that you can modify

print("="*70)
print("SAVING JAILBREAK DATASET")
print("="*70)

# The dataset should be provided as a JSON file or pasted here
# Since it's very large, we'll create a script that reads from stdin

import sys

if len(sys.argv) > 1:
    input_file = sys.argv[1]
    print(f"\nReading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
else:
    print("\n[INFO] Please provide the dataset JSON file as an argument")
    print("Usage: python save_dataset_from_query.py <input.json>")
    print("\nOr paste the JSON data directly into this script.")
    sys.exit(1)

output_file = "jailbreak_test_dataset.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Saved {len(dataset)} examples to {output_file}")

# Show statistics
jailbreak_count = sum(1 for item in dataset if item.get('jailbroken', False))
benign_count = len(dataset) - jailbreak_count

print(f"\nStatistics:")
print(f"  Total examples: {len(dataset)}")
print(f"  Jailbreaks: {jailbreak_count} ({jailbreak_count/len(dataset)*100:.1f}%)")
print(f"  Benign: {benign_count} ({benign_count/len(dataset)*100:.1f}%)")

print(f"\n✅ Ready to test! Run: python test_on_jailbreak_dataset.py")

