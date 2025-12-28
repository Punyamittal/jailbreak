"""
Process malignant.csv dataset for ML training.

This script automatically converts malignant.csv to our training format.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from dataset_collection_script import DatasetCollector
from pipeline import AntiJailbreakPipeline
from security_types import Capability


def process_malignant_dataset(
    csv_path: str = "malignant.csv",
    output_dir: str = "datasets",
    create_labeled: bool = True
):
    """
    Process malignant.csv and convert to training format.
    
    Args:
        csv_path: Path to malignant.csv
        output_dir: Output directory for datasets
        create_labeled: Whether to create labeled dataset file
    """
    print("=" * 70)
    print("PROCESSING MALIGNANT.CSV DATASET")
    print("=" * 70)
    
    # Load CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\nTotal rows: {len(rows)}")
    
    # Analyze distribution
    categories = Counter(row['category'] for row in rows)
    base_classes = Counter(row['base_class'] for row in rows)
    
    print(f"\nCategories:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/len(rows)*100:.1f}%)")
    
    print(f"\nBase classes:")
    for bc, count in base_classes.most_common():
        print(f"  {bc}: {count} ({count/len(rows)*100:.1f}%)")
    
    # Initialize pipeline and collector
    print(f"\nInitializing security pipeline...")
    pipeline = AntiJailbreakPipeline(default_capabilities=[Capability.READ])
    collector = DatasetCollector(output_dir=output_dir)
    
    # Process each row
    print(f"\nProcessing {len(rows)} prompts through security pipeline...")
    labeled_data = []
    
    for i, row in enumerate(rows):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(rows)}...")
        
        prompt = row['text']
        category = row['category']
        base_class = row['base_class']
        
        # Infer label from category/base_class
        if category == 'jailbreak' or 'jailbreak' in base_class.lower():
            inferred_label = "jailbreak"
        elif category == 'act_as' or base_class == 'role_play':
            inferred_label = "jailbreak"  # Role-play is a jailbreak technique
        elif base_class in ['privilege_escalation', 'output_constraint']:
            inferred_label = "jailbreak"
        elif category == 'conversation' and base_class == 'conversation':
            inferred_label = "benign"
        else:
            inferred_label = "borderline"  # Will be determined by pipeline
        
        # Process through pipeline
        try:
            result = pipeline.process(
                prompt_text=prompt,
                user_id="malignant_dataset",
                session_id=f"malignant_{i}"
            )
            
            # Create result dict
            result_dict = {
                'decision': result.decision.value,
                'risk_score': result.context.risk_score.score,
                'attack_classes': [ac.value for ac in result.context.risk_score.attack_classes],
                'block_reason': result.block_reason,
                'indicators': result.context.risk_score.indicators
            }
            
            # Determine final label
            if inferred_label == "jailbreak" or result.decision.value == "block":
                final_label = "jailbreak"
            elif result.decision.value == "allow" and result.context.risk_score.score < 0.2:
                final_label = "benign"
            else:
                final_label = "borderline"
            
            # Collect for training
            metadata = {
                "source": "malignant.csv",
                "original_category": category,
                "original_base_class": base_class,
                "has_embedding": bool(row.get('embedding')),
                "row_index": i,
                "inferred_label": inferred_label
            }
            
            collector.collect_from_pipeline(
                prompt, 
                result_dict, 
                label=final_label, 
                metadata=metadata
            )
            
            # Also create labeled data point
            labeled_data.append({
                "prompt": prompt,
                "label": final_label,
                "original_category": category,
                "original_base_class": base_class,
                "pipeline_risk_score": result.context.risk_score.score,
                "pipeline_decision": result.decision.value,
                "pipeline_attack_classes": [ac.value for ac in result.context.risk_score.attack_classes],
                "inferred_label": inferred_label,
                "has_embedding": bool(row.get('embedding')),
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"  Error processing row {i}: {e}")
            continue
    
    print(f"\n[OK] Processed {len(labeled_data)}/{len(rows)} prompts")
    
    # Get statistics
    stats = collector.get_statistics()
    print(f"\nDataset Statistics:")
    for dataset_type, count in stats.items():
        if dataset_type != 'total':
            print(f"  {dataset_type}: {count}")
    print(f"  Total: {stats['total']}")
    
    # Label distribution
    labels = Counter(item['label'] for item in labeled_data)
    print(f"\nLabel Distribution:")
    for label, count in labels.items():
        print(f"  {label}: {count} ({count/len(labeled_data)*100:.1f}%)")
    
    # Create labeled dataset file
    if create_labeled:
        output_path = Path(output_dir) / "malignant_labeled.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in labeled_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n[OK] Created labeled dataset: {output_path}")
        print(f"  Total examples: {len(labeled_data)}")
    
    # Agreement analysis
    agreement = sum(1 for item in labeled_data 
                   if item['label'] == item['inferred_label'])
    agreement_pct = agreement / len(labeled_data) * 100 if labeled_data else 0
    print(f"\nLabel Agreement (inferred vs pipeline): {agreement_pct:.1f}%")
    
    # Export datasets
    print(f"\nExporting datasets...")
    for dataset_type in ['attacks', 'benign', 'false_positives']:
        if stats.get(dataset_type, 0) > 0:
            output_file = collector.export_dataset(dataset_type, "jsonl")
            if output_file:
                print(f"  [OK] Exported {dataset_type}: {output_file}")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review labeled dataset: {output_dir}/malignant_labeled.jsonl")
    print(f"  2. Check exported datasets in: {output_dir}/")
    print(f"  3. Use for ML training (see ML_DATASET_REQUIREMENTS.md)")
    
    return labeled_data, collector


if __name__ == "__main__":
    process_malignant_dataset()

