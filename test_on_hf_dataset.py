"""
Test the trained model on HuggingFace AI Jailbreak dataset.

This script loads the HF dataset and tests our model's performance.
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter
import numpy as np
import sys

# Try to import models
try:
    from train_balanced_model import BalancedAntiJailbreakModel
    BALANCED_MODEL_AVAILABLE = True
except ImportError:
    BALANCED_MODEL_AVAILABLE = False
    print("[WARNING] Balanced model not available")

try:
    from train_ml_model import AntiJailbreakMLModel
    REGULAR_MODEL_AVAILABLE = True
except ImportError:
    REGULAR_MODEL_AVAILABLE = False
    print("[WARNING] Regular model not available")

def load_hf_dataset():
    """Load the HuggingFace dataset."""
    print("="*70)
    print("LOADING HUGGINGFACE DATASET")
    print("="*70)
    
    # Try method 1: Direct parquet URL
    try:
        print("\nMethod 1: Loading from HuggingFace parquet URL...")
        df = pd.read_parquet("hf://datasets/SpawnedShoyo/ai-jailbreak/data/train-00000-of-00001.parquet")
        
        print(f"\n[SUCCESS] Dataset loaded!")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        
        # Show sample
        print(f"\n  Sample data (first 3 rows):")
        for idx, row in df.head(3).iterrows():
            print(f"    Row {idx}: {dict(row)}")
        
        return df
    
    except Exception as e:
        print(f"\n[WARNING] Method 1 failed: {e}")
        print("\nTrying Method 2: datasets library...")
        
        # Try method 2: datasets library
        try:
            from datasets import load_dataset
            print("Loading with datasets library...")
            dataset = load_dataset("SpawnedShoyo/ai-jailbreak", split="train")
            df = dataset.to_pandas()
            
            print(f"\n[SUCCESS] Dataset loaded!")
            print(f"  Total rows: {len(df):,}")
            print(f"  Columns: {list(df.columns)}")
            
            return df
        except Exception as e2:
            print(f"\n[WARNING] Method 2 failed: {e2}")
            print("\nTrying Method 3: Direct download...")
            
            # Try method 3: Direct download URL
            try:
                import requests
                url = "https://huggingface.co/datasets/SpawnedShoyo/ai-jailbreak/resolve/main/data/train-00000-of-00001.parquet"
                print(f"Downloading from: {url}")
                response = requests.get(url, stream=True)
                with open("temp_dataset.parquet", "wb") as f:
                    f.write(response.content)
                df = pd.read_parquet("temp_dataset.parquet")
                Path("temp_dataset.parquet").unlink()  # Clean up
                
                print(f"\n[SUCCESS] Dataset loaded!")
                print(f"  Total rows: {len(df):,}")
                print(f"  Columns: {list(df.columns)}")
                
                return df
            except Exception as e3:
                print(f"\n[ERROR] All methods failed!")
                print(f"  Method 1 error: {e}")
                print(f"  Method 2 error: {e2}")
                print(f"  Method 3 error: {e3}")
                print("\nPlease install required packages:")
                print("  pip install pandas datasets pyarrow requests")
                return None


def prepare_test_data(df):
    """Prepare test data from HF dataset."""
    print("\n" + "="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    
    # Check what columns we have
    print(f"\nAvailable columns: {list(df.columns)}")
    
    # Try to find prompt and label columns
    prompt_col = None
    label_col = None
    
    # Common column names
    prompt_candidates = ['prompt', 'text', 'input', 'question', 'content', 'message']
    label_candidates = ['label', 'is_jailbreak', 'jailbreak', 'type', 'category', 'class', 'target']
    
    for col in df.columns:
        col_lower = col.lower()
        if any(candidate in col_lower for candidate in prompt_candidates):
            prompt_col = col
        if any(candidate in col_lower for candidate in label_candidates):
            label_col = col
    
    if not prompt_col:
        # Use first text-like column
        for col in df.columns:
            if df[col].dtype == 'object' and len(str(df[col].iloc[0])) > 10:
                prompt_col = col
                break
    
    print(f"\nUsing columns:")
    print(f"  Prompt: {prompt_col}")
    print(f"  Label: {label_col}")
    
    if not prompt_col:
        print("[ERROR] Could not find prompt column!")
        return None, None, None
    
    # Extract prompts
    prompts = df[prompt_col].astype(str).tolist()
    
    # Extract labels if available
    labels = None
    label_col_name = None
    if label_col:
        labels = df[label_col].tolist()
        label_col_name = label_col
        # Check label distribution
        label_counts = Counter(labels)
        print(f"\nLabel distribution:")
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count:,}")
    else:
        print("\n[WARNING] No label column found. Will test predictions only.")
    
    print(f"\nPrepared {len(prompts):,} test examples")
    
    return prompts, labels, label_col_name


def normalize_label(label):
    """Normalize label to 'benign' or 'jailbreak'."""
    if isinstance(label, (int, float)):
        return 'jailbreak' if label > 0.5 else 'benign'
    elif isinstance(label, str):
        label_lower = label.lower()
        if 'jailbreak' in label_lower or label_lower in ['1', 'true', 'yes', 'malicious', 'attack']:
            return 'jailbreak'
        elif label_lower in ['0', 'false', 'no', 'benign', 'safe', 'normal']:
            return 'benign'
        else:
            return 'unknown'
    else:
        return 'unknown'


def test_model_on_hf_dataset(model, prompts, labels=None, max_samples=1000):
    """Test model on HF dataset."""
    print("\n" + "="*70)
    print("TESTING MODEL ON HF DATASET")
    print("="*70)
    
    # Limit samples for faster testing
    original_count = len(prompts)
    if len(prompts) > max_samples:
        print(f"\nLimiting to {max_samples:,} samples for faster testing...")
        import random
        random.seed(42)  # For reproducibility
        indices = random.sample(range(len(prompts)), max_samples)
        prompts = [prompts[i] for i in indices]
        if labels:
            labels = [labels[i] for i in indices]
    
    print(f"\nTesting on {len(prompts):,} examples (from {original_count:,} total)...")
    
    predictions = []
    confidences = []
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(prompts)}...")
        
        try:
            result = model.predict(prompt)
            predictions.append(result['label'])
            confidences.append(result['confidence'])
        except Exception as e:
            print(f"  Error on prompt {i}: {e}")
            predictions.append('unknown')
            confidences.append(0.0)
    
    # Analyze results
    print(f"\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    pred_counts = Counter(predictions)
    print(f"\nPredictions distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count:,} ({count/len(predictions)*100:.1f}%)")
    
    print(f"\nConfidence statistics:")
    print(f"  Mean: {np.mean(confidences):.2%}")
    print(f"  Median: {np.median(confidences):.2%}")
    print(f"  Min: {np.min(confidences):.2%}")
    print(f"  Max: {np.max(confidences):.2%}")
    
    # Calculate accuracy if labels available
    if labels:
        print(f"\n" + "="*70)
        print("ACCURACY ANALYSIS")
        print("="*70)
        
        # Normalize labels
        normalized_labels = [normalize_label(l) for l in labels]
        
        # Filter out unknown labels
        valid_indices = [i for i, l in enumerate(normalized_labels) if l != 'unknown']
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_labels = [normalized_labels[i] for i in valid_indices]
        
        if len(valid_labels) > 0:
            # Calculate accuracy
            correct = sum(1 for p, l in zip(valid_predictions, valid_labels) if p == l)
            total = len(valid_labels)
            accuracy = correct / total
            
            print(f"\n📊 ACCURACY METRICS")
            print(f"  Overall Accuracy: {accuracy:.2%}")
            print(f"  Correct: {correct:,} / {total:,}")
            print(f"  Incorrect: {total - correct:,}")
            
            # Per-class accuracy
            from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
            
            # Calculate per-class metrics
            benign_correct = sum(1 for p, l in zip(valid_predictions, valid_labels) if p == 'benign' and l == 'benign')
            benign_total = sum(1 for l in valid_labels if l == 'benign')
            benign_accuracy = benign_correct / benign_total if benign_total > 0 else 0
            
            jailbreak_correct = sum(1 for p, l in zip(valid_predictions, valid_labels) if p == 'jailbreak' and l == 'jailbreak')
            jailbreak_total = sum(1 for l in valid_labels if l == 'jailbreak')
            jailbreak_accuracy = jailbreak_correct / jailbreak_total if jailbreak_total > 0 else 0
            
            print(f"\n  Per-Class Accuracy:")
            print(f"    Benign: {benign_accuracy:.2%} ({benign_correct}/{benign_total})")
            print(f"    Jailbreak: {jailbreak_accuracy:.2%} ({jailbreak_correct}/{jailbreak_total})")
            
            # Confusion matrix
            cm = confusion_matrix(valid_labels, valid_predictions, labels=['benign', 'jailbreak'])
            print(f"\n  Confusion Matrix:")
            print(f"                Predicted")
            print(f"              Benign  Jailbreak")
            print(f"Actual Benign    {cm[0,0]:5d}      {cm[0,1]:5d}  (False Positives)")
            print(f"Jailbreak        {cm[1,0]:5d}      {cm[1,1]:5d}  (False Negatives)")
            
            # Calculate rates
            fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
            
            print(f"\n  Error Rates:")
            print(f"    False Positive Rate: {fp_rate:.2%} (benign classified as jailbreak)")
            print(f"    False Negative Rate: {fn_rate:.2%} (jailbreak missed)")
            
            # Detailed classification report
            print(f"\n  Detailed Metrics:")
            print(classification_report(
                valid_labels, valid_predictions,
                target_names=['benign', 'jailbreak'],
                digits=4
            ))
            
            # Precision, Recall, F1
            precision = precision_score(valid_labels, valid_predictions, average='weighted', labels=['benign', 'jailbreak'], zero_division=0)
            recall = recall_score(valid_labels, valid_predictions, average='weighted', labels=['benign', 'jailbreak'], zero_division=0)
            f1 = f1_score(valid_labels, valid_predictions, average='weighted', labels=['benign', 'jailbreak'], zero_division=0)
            
            print(f"\n  Summary Metrics:")
            print(f"    Precision (weighted): {precision:.4f}")
            print(f"    Recall (weighted): {recall:.4f}")
            print(f"    F1-Score (weighted): {f1:.4f}")
        else:
            print("\n[WARNING] No valid labels found for accuracy calculation")
    else:
        print("\n[INFO] No labels available - showing predictions only")
    
    # Show some examples
    print(f"\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    # Show high confidence predictions
    if len(confidences) > 0:
        high_conf_indices = np.argsort(confidences)[-5:][::-1]
        print(f"\nTop 5 highest confidence predictions:")
        for idx in high_conf_indices:
            print(f"\n  Confidence: {confidences[idx]:.2%}")
            print(f"  Prediction: {predictions[idx]}")
            print(f"  Prompt: {prompts[idx][:100]}...")
            if labels:
                actual = normalize_label(labels[idx]) if idx < len(labels) else 'N/A'
                match = "✓" if predictions[idx] == actual else "✗"
                print(f"  Actual: {actual} {match}")
    
    return predictions, confidences


def main():
    """Main testing function."""
    # Load dataset
    df = load_hf_dataset()
    if df is None:
        return
    
    # Prepare test data
    prompts, labels, label_col_name = prepare_test_data(df)
    if prompts is None:
        return
    
    # Load trained model
    print("\n" + "="*70)
    print("LOADING TRAINED MODEL")
    print("="*70)
    
    model = None
    
    # Try balanced model first
    if BALANCED_MODEL_AVAILABLE:
        model_path = Path("models/balanced_model.pkl")
        if model_path.exists():
            try:
                print("Loading balanced model...")
                import pickle
                model = BalancedAntiJailbreakModel()
                with open("models/balanced_model.pkl", 'rb') as f:
                    model.model = pickle.load(f)
                with open("models/balanced_vectorizer.pkl", 'rb') as f:
                    model.vectorizer = pickle.load(f)
                with open("models/balanced_encoder.pkl", 'rb') as f:
                    model.label_encoder = pickle.load(f)
                model.is_trained = True
                print("[OK] Balanced model loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load balanced model: {e}")
    
    # Fall back to regular model
    if model is None and REGULAR_MODEL_AVAILABLE:
        try:
            print("Loading regular model...")
            model = AntiJailbreakMLModel.load()
            print("[OK] Regular model loaded")
        except Exception as e:
            print(f"[WARNING] Failed to load regular model: {e}")
    
    if model is None:
        print(f"\n[ERROR] No trained model found!")
        print("Please train a model first:")
        print("  python train_balanced_model.py")
        print("  OR")
        print("  python train_ml_model.py")
        return
    
    # Test model
    predictions, confidences = test_model_on_hf_dataset(
        model, prompts, labels, max_samples=1000
    )
    
    print("\n" + "="*70)
    print("[SUCCESS] TESTING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

