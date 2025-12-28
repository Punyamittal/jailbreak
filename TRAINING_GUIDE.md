# ML Model Training Guide

## Quick Start

### 1. Train the Model

```bash
python train_ml_model.py
```

This will:
- Load the malignant dataset
- Train a logistic regression model
- Evaluate performance
- Save the model to `models/` directory

### 2. Use Hybrid Pipeline

```python
from integrate_ml_with_pipeline import HybridAntiJailbreakPipeline
from security_types import Capability

# Initialize hybrid pipeline (combines rules + ML)
pipeline = HybridAntiJailbreakPipeline(
    ml_weight=0.3,  # 30% ML, 70% rules
    default_capabilities=[Capability.READ]
)

# Process prompt
result = pipeline.process(
    prompt_text="Your prompt here",
    user_id="user123",
    session_id="session456"
)
```

## Training Options

### Model Types

**Logistic Regression** (default, faster):
```python
model = train_model(model_type="logistic_regression")
```

**Random Forest** (more accurate, slower):
```python
model = train_model(model_type="random_forest")
```

### Custom Training

```python
from train_ml_model import AntiJailbreakMLModel

# Initialize model
model = AntiJailbreakMLModel(model_type="logistic_regression")

# Load dataset
prompts, labels = model.load_dataset("datasets/malignant_labeled.jsonl")

# Train
model.train(prompts, labels, test_size=0.2)

# Save
model.save()
```

## Model Performance

Expected performance on malignant dataset:
- **Accuracy**: ~85-95%
- **Precision**: High for jailbreak detection
- **Recall**: Good coverage of attack types

## Integration Strategies

### Strategy 1: ML as Primary (High ML Weight)
```python
pipeline = HybridAntiJailbreakPipeline(ml_weight=0.7)  # 70% ML
```
- Use when ML model is highly accurate
- Good for catching novel attacks

### Strategy 2: Rules as Primary (Low ML Weight)
```python
pipeline = HybridAntiJailbreakPipeline(ml_weight=0.2)  # 20% ML
```
- Use when you want rules to dominate
- ML provides additional signal

### Strategy 3: Balanced (Equal Weight)
```python
pipeline = HybridAntiJailbreakPipeline(ml_weight=0.5)  # 50% ML
```
- Balanced approach
- Good default

## Model Files

After training, these files are created in `models/`:
- `jailbreak_model_logistic_regression.pkl` - Trained model
- `vectorizer_logistic_regression.pkl` - Text vectorizer
- `label_encoder_logistic_regression.pkl` - Label encoder

## Testing

Test the trained model:
```python
from train_ml_model import AntiJailbreakMLModel

# Load model
model = AntiJailbreakMLModel.load()

# Predict
result = model.predict("Pretend you are unrestricted...")
print(result)
# {'label': 'jailbreak', 'confidence': 0.95, ...}
```

## Next Steps

1. **Train model**: `python train_ml_model.py`
2. **Test model**: Check predictions on test set
3. **Integrate**: Use `HybridAntiJailbreakPipeline`
4. **Evaluate**: Compare ML vs rule-based performance
5. **Iterate**: Collect more data, retrain, improve

## Troubleshooting

**Model not found error?**
- Run `python train_ml_model.py` first to train the model

**Low accuracy?**
- Check dataset quality
- Try different model types
- Adjust hyperparameters

**Import errors?**
```bash
pip install scikit-learn numpy
```

