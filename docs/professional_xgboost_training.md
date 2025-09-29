# Professional XGBoost Training Configuration

This document describes the implementation of professional XGBoost model training with unlimited historical data and full feature set support.

## Key Features Implemented

### 1. Unlimited Historical Data Loading (`ml/trainer.py`)

**Configuration:**
- `DATA_CONFIG.use_all_history = True` - Enables unlimited data loading
- `DATA_CONFIG.max_4h_training_candles = 0` - 0 means unlimited (removes SQL LIMIT)

**Implementation:**
- When `use_all_history=True`, SQL queries fetch ALL rows for each symbol without LIMIT
- Data is ordered ASC for chronological processing
- 4h alignment filter is still applied but no truncation to training limits
- Comprehensive logging shows per-symbol and total dataset statistics

**Logging Examples:**
```
[BTCUSDT] total_raw=25000 aligned_4h=6250 used_for_training=6250 oldest_ts=1640995200 newest_ts=1704067200 span_days=730.0
FULL DATASET SUMMARY: rows=25000 symbols=4 total_raw=100000 aligned_4h=25000 earliest=1640995200 latest=1704067200 span_days=730.0
```

### 2. Feature Selection Bypass (`ml/trainer.py`)

**Configuration:**
- `FEATURE_SELECTION_CONFIG.enabled = False` - Completely disables dynamic feature selection

**Implementation:**
- When disabled, uses ALL feature columns (~197 indicators)
- Skips correlation pruning, RFE, and iterative selection entirely
- Still excludes base columns (timestamp, symbol, OHLCV)
- Clear logging indicates bypass status

**Logging Examples:**
```
Feature selection disabled via config; using all available features
Using all 197 features (feature selection disabled)
FULL 4h TRAINING ROWS=25000 FEATURES(before selection)=197 FEATURE_SELECTION_ENABLED=False FEATURES_FINAL=197 XGB_TREES=8000
```

### 3. Professional XGBoost Configuration (`ml/model.py`)

**Configuration (`config/settings.py`):**
```python
XGB_PRO_CONFIG = {
    'n_estimators': 8000,  # Large number of trees for professional model
    'max_depth': 12,  # Deep trees for complex pattern recognition
    'learning_rate': 0.01,  # Low learning rate for better generalization
    'early_stopping_rounds': 300,  # Higher patience (set to 0 to disable)
    'subsample': 0.8,  # Subsampling for regularization
    'colsample_bytree': 0.8,  # Feature subsampling
    'colsample_bylevel': 0.8,  # Additional feature subsampling
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'min_child_weight': 3,  # Minimum weight in child nodes
    'gamma': 0.1,  # Minimum split loss
    'tree_method': 'hist',  # Efficient tree construction method
}
```

**Implementation:**
- XGBoost model initialized with configurable parameters
- Early stopping can be disabled by setting `early_stopping_rounds = 0`
- Comprehensive logging of model configuration
- Model file size logging after save

**Logging Examples:**
```
XGBoost n_estimators=8000, max_depth=12, learning_rate=0.01
Training professional XGBoost model with 8000 trees...
Model file size: 42.3 MB
```

### 4. Memory Management and Performance

**Optimizations:**
- Memory usage monitoring with `psutil` (optional)
- Strategic deletion of intermediate DataFrames
- Avoids unnecessary `.copy()` operations where safe
- Logs memory usage at key processing stages

**Logging Examples:**
```
Memory usage after data loading: 1250.5 MB
Memory usage after indicator calculation: 2100.8 MB
```

## Configuration Examples

### For Maximum Model Size (Professional Training)
```python
# config/settings.py
DATA_CONFIG = {
    'use_all_history': True,
    'max_4h_training_candles': 0,  # Unlimited
    'max_4h_selection_candles': 800,  # Still used for RFE when enabled
}

FEATURE_SELECTION_CONFIG = {
    'enabled': False,  # Use all ~197 features
}

XGB_PRO_CONFIG = {
    'n_estimators': 10000,  # Even more trees for larger model
    'early_stopping_rounds': 0,  # Disabled - train all trees
    'max_depth': 15,  # Deeper trees
}
```

### For Development/Testing (Faster Training)
```python
DATA_CONFIG = {
    'use_all_history': False,
    'max_4h_training_candles': 2000,  # Limited
}

FEATURE_SELECTION_CONFIG = {
    'enabled': True,  # Use feature selection
}

XGB_PRO_CONFIG = {
    'n_estimators': 1000,  # Fewer trees
    'early_stopping_rounds': 50,  # Faster early stopping
}
```

## Usage

To train a professional model with all features and unlimited data:

```python
from ml.trainer import ModelTrainer

trainer = ModelTrainer()
result = trainer.train_with_rfe()

print(f"Model version: {result['model_version']}")
print(f"Features used: {len(result['selected_features'])}")
print(f"Training time: {result['training_time']:.1f} seconds")
print(f"Model path: {result['model_path']}")
```

## Expected Results

With the professional configuration:
- **Training samples**: 10,000+ per symbol (vs previous ~1,200)
- **Total features**: ~197 indicators (vs previous 50 selected)
- **Model trees**: 8,000+ (vs previous 5,000)
- **Model size**: 50+ MB (vs previous ~20MB)
- **Training time**: Significantly longer due to more data/features/trees

## Backward Compatibility

All changes maintain backward compatibility:
- If `use_all_history=False`, uses existing limit-based logic
- If `FEATURE_SELECTION_CONFIG.enabled=True`, uses existing dynamic selection
- XGBoost configuration falls back to defaults if config not found

## Troubleshooting

### Out of Memory Issues
- Reduce `n_estimators` in `XGB_PRO_CONFIG`
- Enable `early_stopping_rounds` for automatic termination
- Set `use_all_history=False` to limit data size

### Training Takes Too Long
- Reduce `n_estimators` or increase `learning_rate`
- Enable feature selection to reduce feature count
- Use smaller `max_4h_training_candles` value

### Model Too Small
- Increase `n_estimators` in `XGB_PRO_CONFIG`
- Set `early_stopping_rounds=0` to disable early stopping
- Ensure `FEATURE_SELECTION_CONFIG.enabled=False` to use all features