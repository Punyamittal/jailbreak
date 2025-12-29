"""
Train model with adjustable confidence threshold.

Allows fine-tuning of jailbreak detection sensitivity.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')


class ThresholdAntiJailbreakModel:
    """Model with adjustable confidence threshold."""
    
    def __init__(self, jailbreak_threshold: float = 0.5):
        """
        Initialize model with threshold.
        
        Args:
            jailbreak_threshold: Confidence threshold for jailbreak (lower = more sensitive)
        """
        self.jailbreak_threshold = jailbreak_threshold
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 4),
            stop_words='english',
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        
        self.lr_model = LogisticRegression(
            max_iter=2000,
            class_weight={0: 2.0, 1: 1.0},  # Penalize false positives
            random_state=42,
            C=0.5,
            penalty='l2',
            solver='liblinear'
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight={0: 2.0, 1: 1.0},
            random_state=42,
            n_jobs=-1
        )
        
        self.ensemble_model = None
        self.model = None
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def balance_dataset(self, prompts: List[str], labels: List[str], target_ratio: float = 0.5):
        """Balance dataset."""
        benign_prompts = [p for p, l in zip(prompts, labels) if l == 'benign']
        jailbreak_prompts = [p for p, l in zip(prompts, labels) if l == 'jailbreak']
        
        print(f"\n  Original: Benign={len(benign_prompts):,}, Jailbreak={len(jailbreak_prompts):,}")
        
        # Undersample majority
        if len(jailbreak_prompts) > len(benign_prompts):
            target_jailbreak = int(len(benign_prompts) * target_ratio / (1 - target_ratio))
            if target_jailbreak < len(jailbreak_prompts):
                jailbreak_prompts = random.sample(jailbreak_prompts, target_jailbreak)
        else:
            target_benign = int(len(jailbreak_prompts) * (1 - target_ratio) / target_ratio)
            if target_benign < len(benign_prompts):
                benign_prompts = random.sample(benign_prompts, target_benign)
        
        balanced_prompts = benign_prompts + jailbreak_prompts
        balanced_labels = ['benign'] * len(benign_prompts) + ['jailbreak'] * len(jailbreak_prompts)
        
        combined = list(zip(balanced_prompts, balanced_labels))
        random.shuffle(combined)
        balanced_prompts, balanced_labels = zip(*combined)
        
        print(f"  Balanced: Benign={len(benign_prompts):,}, Jailbreak={len(jailbreak_prompts):,}")
        
        return list(balanced_prompts), list(balanced_labels)
    
    def load_dataset(self, jsonl_path: str):
        """Load dataset."""
        prompts = []
        labels = []
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    prompts.append(item['prompt'])
                    labels.append(item['label'])
                except:
                    continue
        
        print(f"  Loaded {len(prompts)} examples")
        return prompts, labels
    
    def train(self, prompts: List[str], labels: List[str], test_size: float = 0.2, use_ensemble: bool = True):
        """Train model."""
        print(f"\nTraining Model (threshold={self.jailbreak_threshold})...")
        
        # Balance
        prompts, labels = self.balance_dataset(prompts, labels)
        
        # Encode
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Vectorize
        X = self.vectorizer.fit_transform(prompts)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train
        print("\n  Training models...")
        self.lr_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        
        # Ensemble
        if use_ensemble:
            self.ensemble_model = VotingClassifier(
                estimators=[('lr', self.lr_model), ('rf', self.rf_model)],
                voting='soft',
                weights=[2.0, 1.0]
            )
            self.ensemble_model.fit(X_train, y_train)
            self.model = self.ensemble_model
        else:
            self.model = self.lr_model
        
        self.is_trained = True
        
        # Evaluate with threshold
        print(f"\n  Evaluating with threshold={self.jailbreak_threshold}...")
        y_pred = self._predict_with_threshold(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n  Test Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"    {cm}")
        
        return accuracy
    
    def _predict_with_threshold(self, X):
        """Predict with threshold adjustment."""
        probabilities = self.model.predict_proba(X)
        # Get jailbreak probability (class 1)
        jailbreak_probs = probabilities[:, 1]
        # Apply threshold
        predictions = (jailbreak_probs >= self.jailbreak_threshold).astype(int)
        return predictions
    
    def predict(self, prompt: str) -> Dict:
        """Predict with threshold."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        X = self.vectorizer.transform([prompt])
        probabilities = self.model.predict_proba(X)[0]
        
        jailbreak_prob = probabilities[1]
        benign_prob = probabilities[0]
        
        # Apply threshold
        if jailbreak_prob >= self.jailbreak_threshold:
            label = 'jailbreak'
            confidence = jailbreak_prob
        else:
            label = 'benign'
            confidence = benign_prob
        
        return {
            "label": label,
            "confidence": float(confidence),
            "jailbreak_probability": float(jailbreak_prob),
            "benign_probability": float(benign_prob),
            "threshold": self.jailbreak_threshold,
            "probabilities": {
                'benign': float(benign_prob),
                'jailbreak': float(jailbreak_prob)
            }
        }
    
    def set_threshold(self, threshold: float):
        """Adjust threshold (lower = more sensitive to jailbreaks)."""
        if 0.0 <= threshold <= 1.0:
            self.jailbreak_threshold = threshold
            print(f"Threshold set to {threshold:.2f}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def save(self, model_dir: str = "models"):
        """Save model."""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        with open(model_dir / "threshold_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        with open(model_dir / "threshold_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(model_dir / "threshold_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        with open(model_dir / "threshold_value.txt", 'w') as f:
            f.write(str(self.jailbreak_threshold))
        
        print(f"\n[OK] Model saved (threshold={self.jailbreak_threshold})")


def find_optimal_threshold(model, X_test, y_test, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Find optimal threshold by testing different values."""
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*70)
    
    probabilities = model.model.predict_proba(X_test)
    jailbreak_probs = probabilities[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    
    print(f"\nTesting thresholds:")
    for threshold in thresholds:
        predictions = (jailbreak_probs >= threshold).astype(int)
        f1 = f1_score(y_test, predictions, average='macro')
        accuracy = accuracy_score(y_test, predictions)
        
        cm = confusion_matrix(y_test, predictions)
        fn_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        fp_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        
        print(f"\n  Threshold: {threshold:.2f}")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    False Negative Rate: {fn_rate:.2%}")
        print(f"    False Positive Rate: {fp_rate:.2%}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n[BEST] Threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def main():
    """Main training function."""
    print("="*70)
    print("TRAINING MODEL WITH ADJUSTABLE THRESHOLD")
    print("="*70)
    
    # Try to load dataset with custom data, then HF, then default
    dataset_paths = [
        "datasets/combined_with_custom.jsonl",
        "datasets/combined_with_hf.jsonl",
        "datasets/combined_training_dataset.jsonl"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("[ERROR] No dataset found!")
        return
    
    print(f"\nUsing dataset: {dataset_path}")
    
    # Initialize model with lower threshold (more sensitive)
    model = ThresholdAntiJailbreakModel(jailbreak_threshold=0.4)  # Lower = catch more
    
    # Load and train
    prompts, labels = model.load_dataset(dataset_path)
    accuracy = model.train(prompts, labels)
    
    # Find optimal threshold
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    y_encoded = model.label_encoder.transform(labels)
    X = model.vectorizer.transform(prompts)
    _, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    optimal_threshold = find_optimal_threshold(model, X_test, y_test)
    model.set_threshold(optimal_threshold)
    
    # Save
    model.save()
    
    print("\n" + "="*70)
    print("[SUCCESS] TRAINING COMPLETE!")
    print("="*70)
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Use this threshold for better jailbreak detection!")


if __name__ == "__main__":
    main()

