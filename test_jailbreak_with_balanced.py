"""
Test Balanced Model on Jailbreak Dataset

Uses the already-trained balanced model (TF-IDF based) to test on the jailbreak dataset.
This is faster than waiting for the intent model to finish training.
"""

import json
import sys
from pathlib import Path
from collections import Counter
import pickle

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def load_jailbreak_dataset(json_data):
    """Extract prompts and labels from dataset."""
    prompts = []
    labels = []
    
    for item in json_data:
        prompt = item.get('prompt', '').strip()
        if not prompt:
            continue
        is_jailbreak = item.get('jailbroken', False)
        label = 'jailbreak' if is_jailbreak else 'benign'
        prompts.append(prompt)
        labels.append(label)
    
    return prompts, labels


def test_balanced_model(model, prompts, labels):
    """Test the balanced model."""
    print("\n" + "="*70)
    print("TESTING BALANCED MODEL ON JAILBREAK DATASET")
    print("="*70)
    
    print(f"\nTesting {len(prompts)} prompts...")
    
    predictions = []
    confidences = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(prompts)}...")
        
        try:
            result = model.predict(prompt)
            predicted_label = result['label']
            confidence = result['confidence']
            
            predictions.append(predicted_label)
            confidences.append(confidence)
        except Exception as e:
            print(f"  [ERROR] Prompt {i+1}: {e}")
            predictions.append('benign')
            confidences.append(0.0)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    recall = recall_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    f1 = f1_score(labels, predictions, pos_label='jailbreak', zero_division=0)
    cm = confusion_matrix(labels, predictions, labels=['benign', 'jailbreak'])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1-Score:  {f1:.2%}")
    
    print(f"\n🎯 Security-Critical Metrics:")
    print(f"  Jailbreak Recall: {recall:.2%} (target: >80%)")
    print(f"  False Negatives:  {fn} (missed jailbreaks)")
    print(f"  False Negative Rate: {fnr:.2%} (target: <20%)")
    print(f"  False Positives:  {fp} (benign flagged)")
    print(f"  False Positive Rate: {fpr:.2%}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Jailbreak")
    print(f"  Actual Benign   {tn:4d}    {fp:4d}")
    print(f"  Actual Jailbreak {fn:4d}    {tp:4d}")
    
    # Per-class accuracy
    total_benign = tn + fp
    total_jailbreak = fn + tp
    benign_acc = tn / total_benign if total_benign > 0 else 0.0
    jailbreak_acc = tp / total_jailbreak if total_jailbreak > 0 else 0.0
    
    print(f"\n📋 Per-Class Accuracy:")
    print(f"  Benign:     {benign_acc:.2%} ({tn}/{total_benign})")
    print(f"  Jailbreak:  {jailbreak_acc:.2%} ({tp}/{total_jailbreak})")
    
    # Show examples
    fn_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) 
                  if p == 'benign' and l == 'jailbreak']
    if fn_indices:
        print(f"\n❌ False Negatives (Missed {len(fn_indices)} jailbreaks):")
        for idx in fn_indices[:5]:
            print(f"  [{idx+1}] Confidence: {confidences[idx]:.2%}")
            print(f"      {prompts[idx][:100]}...")
    
    fp_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) 
                  if p == 'jailbreak' and l == 'benign']
    if fp_indices:
        print(f"\n⚠️  False Positives (Benign Flagged - {len(fp_indices)}):")
        for idx in fp_indices[:5]:
            print(f"  [{idx+1}] Confidence: {confidences[idx]:.2%}")
            print(f"      {prompts[idx][:100]}...")
    
    print("\n" + "="*70)
    if recall >= 0.80 and fnr < 0.20:
        print("✅ MODEL MEETS SECURITY REQUIREMENTS")
    else:
        print("⚠️  MODEL NEEDS IMPROVEMENT")
        if recall < 0.80:
            print(f"   - Jailbreak recall ({recall:.2%}) below target (80%)")
        if fnr >= 0.20:
            print(f"   - False negative rate ({fnr:.2%}) above target (20%)")
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_negatives': fn,
        'false_positives': fp,
        'false_negative_rate': fnr,
        'false_positive_rate': fpr
    }


def main():
    """Main function."""
    print("="*70)
    print("TESTING BALANCED MODEL ON JAILBREAK DATASET")
    print("="*70)
    
    # Load dataset
    dataset_file = Path("jailbreak_test_dataset.json")
    if not dataset_file.exists():
        print(f"\n[ERROR] Dataset not found: {dataset_file}")
        print("Please ensure jailbreak_test_dataset.json exists")
        return
    
    print(f"\n[1/3] Loading dataset from {dataset_file}...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    prompts, labels = load_jailbreak_dataset(json_data)
    print(f"  Loaded {len(prompts)} examples")
    
    label_counts = Counter(labels)
    print(f"\n  Label distribution:")
    for label, count in label_counts.items():
        print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Load balanced model
    print(f"\n[2/3] Loading balanced model...")
    try:
        from train_balanced_model import BalancedAntiJailbreakModel
        model = BalancedAntiJailbreakModel()
        
        # Load the trained components
        # Try ensemble first, then regular model
        if Path("models/balanced_ensemble.pkl").exists():
            with open("models/balanced_ensemble.pkl", 'rb') as f:
                model.ensemble_model = pickle.load(f)
                model.model = model.ensemble_model  # Use ensemble as main model
            print("  [OK] Loaded ensemble model")
        else:
            with open("models/balanced_model.pkl", 'rb') as f:
                model.model = pickle.load(f)
            print("  [OK] Loaded regular model")
        
        with open("models/balanced_vectorizer.pkl", 'rb') as f:
            model.vectorizer = pickle.load(f)
        with open("models/balanced_encoder.pkl", 'rb') as f:
            model.label_encoder = pickle.load(f)
        model.is_trained = True
        
        print("  [OK] Balanced model loaded successfully")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test
    print(f"\n[3/3] Testing model...")
    results = test_balanced_model(model, prompts, labels)
    
    # Save results
    results_file = Path("test_results_balanced_jailbreak.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'false_negatives': int(results['false_negatives']),
            'false_positives': int(results['false_positives']),
            'false_negative_rate': float(results['false_negative_rate']),
            'false_positive_rate': float(results['false_positive_rate']),
        }, f, indent=2)
    
    print(f"\n[OK] Results saved to {results_file}")


if __name__ == "__main__":
    main()

