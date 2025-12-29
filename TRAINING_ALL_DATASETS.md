# Training with All Available Datasets

## What's Happening

The script `train_all_data.py` is currently running and will:

1. **Combine ALL available datasets:**
   - All JSONL files from `datasets/` directory
   - `jailbreak_test_dataset.json` (your new dataset)
   - `custom_test_data.csv`
   - `malignant.csv`
   - And any other available datasets

2. **Deduplicate** the combined data (remove duplicate prompts)

3. **Train the balanced model** on the combined dataset with:
   - Dataset balancing (50/50 split)
   - Data cleaning (remove mislabeled examples)
   - Ensemble model (best performance)

## Expected Output

The script will create:
- `datasets/all_datasets_combined.jsonl` - Combined dataset
- `models/balanced_model.pkl` - Trained model
- `models/balanced_ensemble.pkl` - Ensemble model (best)
- `models/balanced_vectorizer.pkl` - TF-IDF vectorizer
- `models/balanced_encoder.pkl` - Label encoder

## Datasets Being Combined

### JSONL Files:
- `datasets/combined_with_custom.jsonl`
- `datasets/combined_with_hf.jsonl`
- `datasets/combined_training_dataset.jsonl`
- `datasets/synthetic_processed.jsonl`
- `datasets/raw_processed.jsonl`
- `datasets/formatted_processed.jsonl`
- `datasets/malignant_labeled.jsonl`
- `datasets/attacks_20251228_210936.jsonl`
- `datasets/benign_20251228_210936.jsonl`
- `datasets/attacks/attacks_20251228.jsonl`
- `datasets/encoded/encoded_20251228.jsonl`
- `datasets/indirect_injection/indirect_20251228.jsonl`
- `datasets/benign/benign_20251228.jsonl`

### JSON Files:
- `jailbreak_test_dataset.json` (100 examples)

### CSV Files:
- `custom_test_data.csv`
- `malignant.csv`

## Training Process

1. **Data Loading**: Loads all datasets (~5-10 minutes depending on size)
2. **Deduplication**: Removes duplicate prompts
3. **Balancing**: Creates 50/50 split between benign and jailbreak
4. **Cleaning**: Removes obvious mislabeled examples
5. **Vectorization**: TF-IDF with 10,000 features, n-grams (1-4)
6. **Training**: 
   - Logistic Regression
   - Random Forest
   - Ensemble (Voting Classifier)
7. **Evaluation**: Cross-validation and test set evaluation

## Expected Training Time

- Small dataset (<100K examples): ~5-10 minutes
- Medium dataset (100K-500K examples): ~15-30 minutes
- Large dataset (>500K examples): ~30-60 minutes

## After Training

Once training completes, you can:

1. **Test on jailbreak dataset:**
   ```bash
   python test_jailbreak_dataset.py
   ```

2. **Use the model in your pipeline:**
   ```python
   from train_balanced_model import BalancedAntiJailbreakModel
   import pickle
   
   model = BalancedAntiJailbreakModel()
   with open("models/balanced_ensemble.pkl", 'rb') as f:
       model.model = pickle.load(f)
   with open("models/balanced_vectorizer.pkl", 'rb') as f:
       model.vectorizer = pickle.load(f)
   with open("models/balanced_encoder.pkl", 'rb') as f:
       model.label_encoder = pickle.load(f)
   model.is_trained = True
   
   result = model.predict("Your prompt here")
   ```

## Check Progress

To check if training is complete:
```bash
# Check if combined dataset exists
ls datasets/all_datasets_combined.jsonl

# Check if models are saved
ls models/balanced_*.pkl
```

## If Training Fails

If you encounter errors:

1. **Memory issues**: The dataset might be too large. Consider:
   - Using a subset of datasets
   - Increasing system memory
   - Using data sampling

2. **File not found**: Some datasets might not exist. The script will skip missing files.

3. **Encoding errors**: Check that all files are UTF-8 encoded.

## Next Steps

After training completes:
1. Evaluate on test datasets
2. Compare performance with previous models
3. Fine-tune if needed
4. Deploy to production

