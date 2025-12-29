"""
Test intent-based model on custom dataset.
"""

import csv
from pathlib import Path
from intent_based_detector import IntentBasedDetector
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter

def load_custom_dataset(csv_path: str = "custom_test_data.csv"):
    """Load custom test dataset."""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip('"')
            violates = row['violates'].strip().lower() == 'true'
            data.append({
                'text': text,
                'label': 'jailbreak' if violates else 'benign'
            })
    return data


def test_intent_model():
    """Test intent model on custom dataset."""
    print("="*70)
    print("TESTING INTENT MODEL ON CUSTOM DATASET")
    print("="*70)
    
    # Load model
    print("\n[1/3] Loading intent model...")
    try:
        with open("models/intent_detector.pkl", 'rb') as f:
            detector = pickle.load(f)
        print("[OK] Model loaded")
    except FileNotFoundError:
        print("[ERROR] Model not found. Train first: python train_intent_model.py")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Load test data
    print("\n[2/3] Loading custom dataset...")
    test_data = load_custom_dataset()
    print(f"[OK] Loaded {len(test_data)} examples")
    
    # Test
    print("\n[3/3] Testing model...")
    predictions = []
    actuals = []
    decisions = []
    confidences = []
    
    for i, item in enumerate(test_data):
        text = item['text']
        actual = item['label']
        
        result = detector.detect(text)
        decision = result['decision']
        
        # Map decision to label
        if decision == 'block':
            predicted = 'jailbreak'
        elif decision == 'warn':
            predicted = 'jailbreak'  # Conservative
        else:
            predicted = 'benign'
        
        predictions.append(predicted)
        actuals.append(actual)
        decisions.append(decision)
        confidences.append(result['confidence'])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_data)}...")
    
    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Accuracy
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    total = len(predictions)
    accuracy = correct / total
    
    print(f"\n📊 Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Per-class
    benign_correct = sum(1 for p, a in zip(predictions, actuals) if p == 'benign' and a == 'benign')
    benign_total = sum(1 for a in actuals if a == 'benign')
    benign_acc = benign_correct / benign_total if benign_total > 0 else 0
    
    jailbreak_correct = sum(1 for p, a in zip(predictions, actuals) if p == 'jailbreak' and a == 'jailbreak')
    jailbreak_total = sum(1 for a in actuals if a == 'jailbreak')
    jailbreak_acc = jailbreak_correct / jailbreak_total if jailbreak_total > 0 else 0
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Benign: {benign_acc:.2%} ({benign_correct}/{benign_total})")
    print(f"  Jailbreak: {jailbreak_acc:.2%} ({jailbreak_correct}/{jailbreak_total})")
    
    # Confusion matrix
    cm = confusion_matrix(actuals, predictions, labels=['benign', 'jailbreak'])
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Jailbreak")
    print(f"Actual Benign    {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"Jailbreak        {cm[1,0]:5d}      {cm[1,1]:5d}")
    
    # Error rates
    fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    print(f"\nError Rates:")
    print(f"  False Positive Rate: {fp_rate:.2%}")
    print(f"  False Negative Rate: {fn_rate:.2%}")
    
    # Detailed report
    print(f"\nDetailed Metrics:")
    print(classification_report(actuals, predictions, target_names=['benign', 'jailbreak']))
    
    # Decision distribution
    decision_counts = Counter(decisions)
    print(f"\nDecision Distribution:")
    for decision, count in decision_counts.items():
        print(f"  {decision}: {count} ({count/total*100:.1f}%)")
    
    print(f"\n[SUCCESS] Test complete!")
    print(f"\nKey Metrics:")
    print(f"  Jailbreak Recall: {jailbreak_acc:.2%} (target: >80%)")
    print(f"  False Negative Rate: {fn_rate:.2%} (target: <20%)")
    
    if jailbreak_acc >= 0.80 and fn_rate <= 0.20:
        print(f"\n✅ MEETS TARGET METRICS!")
    else:
        print(f"\n⚠️  Does not meet target metrics yet.")


if __name__ == "__main__":
    test_intent_model()

