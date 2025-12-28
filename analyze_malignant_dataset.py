"""
Analyze malignant.csv dataset and convert it for use with our anti-jailbreak system.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from dataset_collection_script import DatasetCollector
from pipeline import AntiJailbreakPipeline
from security_types import Capability


def analyze_malignant_dataset(csv_path: str = "malignant.csv"):
    """Analyze the malignant.csv dataset."""
    print("=" * 70)
    print("ANALYZING MALIGNANT.CSV DATASET")
    print("=" * 70)
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nTotal rows: {len(rows)}")
    
    # Analyze categories
    categories = Counter(row['category'] for row in rows)
    print(f"\nCategories distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/len(rows)*100:.1f}%)")
    
    # Analyze base classes
    base_classes = Counter(row['base_class'] for row in rows)
    print(f"\nBase classes distribution:")
    for bc, count in base_classes.most_common():
        print(f"  {bc}: {count} ({count/len(rows)*100:.1f}%)")
    
    # Sample texts
    print(f"\nSample texts (first 10):")
    for i, row in enumerate(rows[:10]):
        text = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        print(f"  {i+1}. [{row['category']}] {text}")
    
    # Check for jailbreak-like patterns
    print(f"\nChecking for jailbreak patterns...")
    jailbreak_keywords = [
        'ignore', 'pretend', 'roleplay', 'override', 'bypass',
        'unrestricted', 'jailbreak', 'developer', 'admin', 'system'
    ]
    
    potential_jailbreaks = []
    for row in rows:
        text_lower = row['text'].lower()
        if any(kw in text_lower for kw in jailbreak_keywords):
            potential_jailbreaks.append(row)
    
    print(f"  Found {len(potential_jailbreaks)} rows with jailbreak keywords")
    
    if potential_jailbreaks:
        print(f"\nSample potential jailbreaks:")
        for i, row in enumerate(potential_jailbreaks[:5]):
            text = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
            print(f"  {i+1}. {text}")
    
    return rows, categories, base_classes, potential_jailbreaks


def convert_to_training_format(csv_path: str = "malignant.csv", output_dir: str = "datasets"):
    """
    Convert malignant.csv to our training dataset format.
    
    This will:
    1. Run each prompt through our pipeline
    2. Get risk scores and attack classifications
    3. Save in structured format
    """
    print("\n" + "=" * 70)
    print("CONVERTING TO TRAINING FORMAT")
    print("=" * 70)
    
    # Initialize pipeline and collector
    pipeline = AntiJailbreakPipeline(default_capabilities=[Capability.READ])
    collector = DatasetCollector(output_dir=output_dir)
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nProcessing {len(rows)} prompts through security pipeline...")
    
    processed = 0
    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(rows)}...")
        
        prompt = row['text']
        
        # Process through pipeline
        try:
            result = pipeline.process(
                prompt_text=prompt,
                user_id="dataset_import",
                session_id=f"session_{i}"
            )
            
            # Convert to dict format
            result_dict = {
                'decision': result.decision.value,
                'risk_score': result.context.risk_score.score,
                'attack_classes': [ac.value for ac in result.context.risk_score.attack_classes],
                'block_reason': result.block_reason,
                'indicators': result.context.risk_score.indicators
            }
            
            # Determine label based on category and pipeline result
            # "malignant" category likely means jailbreak
            if row['category'] == 'malignant' or 'malignant' in row.get('base_class', '').lower():
                label = "jailbreak"
            elif result.decision.value == "block":
                label = "jailbreak"
            elif result.decision.value == "allow" and result.context.risk_score.score < 0.2:
                label = "benign"
            else:
                label = "borderline"
            
            # Collect with metadata
            metadata = {
                "source": "malignant.csv",
                "original_category": row['category'],
                "original_base_class": row['base_class'],
                "has_embedding": bool(row.get('embedding')),
                "row_index": i
            }
            
            collector.collect_from_pipeline(prompt, result_dict, label=label, metadata=metadata)
            processed += 1
            
        except Exception as e:
            print(f"  Error processing row {i}: {e}")
            continue
    
    print(f"\n✓ Processed {processed}/{len(rows)} prompts")
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"\nDataset Statistics:")
    for dataset_type, count in stats.items():
        if dataset_type != 'total':
            print(f"  {dataset_type}: {count}")
    print(f"  Total: {stats['total']}")
    
    return collector


def create_labeled_dataset(csv_path: str = "malignant.csv", output_path: str = "malignant_labeled.jsonl"):
    """
    Create a labeled dataset from malignant.csv.
    
    Uses the category/base_class to infer labels:
    - 'malignant' -> 'jailbreak'
    - Others -> 'benign' or based on pipeline analysis
    """
    print("\n" + "=" * 70)
    print("CREATING LABELED DATASET")
    print("=" * 70)
    
    pipeline = AntiJailbreakPipeline(default_capabilities=[Capability.READ])
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    labeled_data = []
    
    for i, row in enumerate(rows):
        prompt = row['text']
        category = row['category']
        base_class = row['base_class']
        
        # Infer label from category
        if 'malignant' in category.lower() or 'malignant' in base_class.lower():
            inferred_label = "jailbreak"
        else:
            inferred_label = "benign"
        
        # Process through pipeline for validation
        try:
            result = pipeline.process(
                prompt_text=prompt,
                user_id="dataset_labeling",
                session_id=f"labeling_{i}"
            )
            
            # Create labeled data point
            data_point = {
                "prompt": prompt,
                "label": inferred_label,
                "category": category,
                "base_class": base_class,
                "pipeline_risk_score": result.context.risk_score.score,
                "pipeline_decision": result.decision.value,
                "pipeline_attack_classes": [ac.value for ac in result.context.risk_score.attack_classes],
                "pipeline_label": "jailbreak" if result.decision.value == "block" else "benign",
                "has_embedding": bool(row.get('embedding')),
                "metadata": {
                    "source": "malignant.csv",
                    "row_index": i
                }
            }
            
            labeled_data.append(data_point)
            
        except Exception as e:
            print(f"  Error processing row {i}: {e}")
            continue
    
    # Save to JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Created labeled dataset: {output_path}")
    print(f"  Total labeled examples: {len(labeled_data)}")
    
    # Statistics
    labels = Counter(item['label'] for item in labeled_data)
    print(f"\nLabel distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count} ({count/len(labeled_data)*100:.1f}%)")
    
    # Agreement between inferred label and pipeline
    agreement = sum(1 for item in labeled_data 
                   if item['label'] == item['pipeline_label'])
    agreement_pct = agreement / len(labeled_data) * 100
    print(f"\nLabel agreement (inferred vs pipeline): {agreement_pct:.1f}%")
    
    return labeled_data


def main():
    """Main analysis and conversion workflow."""
    # Step 1: Analyze the dataset
    rows, categories, base_classes, potential_jailbreaks = analyze_malignant_dataset()
    
    # Step 2: Convert to training format
    print("\n" + "=" * 70)
    response = input("Convert to training format? (y/n): ").strip().lower()
    if response == 'y':
        collector = convert_to_training_format()
        print("\n✓ Conversion complete! Check the 'datasets' directory.")
    
    # Step 3: Create labeled dataset
    print("\n" + "=" * 70)
    response = input("Create labeled dataset? (y/n): ").strip().lower()
    if response == 'y':
        labeled_data = create_labeled_dataset()
        print("\n✓ Labeled dataset created!")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

