# Combined Training Pipeline Implementation

This document describes the implementation of the combined training pipeline that integrates dynamic feature selection with adaptive label thresholding for the professional XGBoost model.

## Overview

The implementation fulfills the following key requirements:

1. **Dynamic Feature Selection**: Performed ONLY on the most recent N (configurable 4h) candles per symbol
2. **Full Historical Training**: XGBoost professional model trains on ALL historical aligned 4h dataset using selected features
3. **Adaptive Label Thresholding**: Calibrates thresholds to achieve target class distribution ratios

## Configuration Changes

### A. FEATURE_SELECTION_CONFIG Enhancement

```python
FEATURE_SELECTION_CONFIG = {
    'enabled': True,  # Enable dynamic feature selection on recent window
    'mode': 'dynamic',  # Dynamic selection mode
    'selection_window_4h': 800,  # Use most recent 800 4h candles for feature selection
    'min_features': 20,  # Minimum features to retain
    'method': 'dynamic_iterative_pruning',  # Method to use when enabled
    'correlation_threshold': 0.95,  # Correlation threshold for pruning
    'tolerance': 0.003,  # Tolerance for improvement in feature selection
    'max_iterations': 50,  # Maximum iterations for dynamic selection
    'max_features': 50,  # Maximum features when selection is enabled
}
```

**Key Changes:**
- `enabled: True` - Enables the dynamic feature selection pipeline
- `mode: 'dynamic'` - Specifies dynamic mode operation
- `selection_window_4h: 800` - Configures the selection window (range: 500-1000)
- `min_features: 20` - Minimum feature count to retain
- `tolerance: 0.003` - Improvement tolerance for convergence
- `max_iterations: 50` - Maximum iterations for selection process

### B. New LABELING_CONFIG

```python
LABELING_CONFIG = {
    'target_distribution': {'SELL': 0.40, 'BUY': 0.40, 'HOLD': 0.20},  # Target class distribution
    'initial_up_pct': 2.0,  # Initial up threshold percentage
    'initial_down_pct': -2.0,  # Initial down threshold percentage 
    'search_up_range': [0.4, 3.0, 0.1],  # [start, end, step] for up threshold search
    'search_down_range': [-3.0, -0.4, 0.1],  # [start, end, step] for down threshold search
    'optimization_metric': 'kl_divergence',  # Distance metric for distribution fit
    'max_search_iterations': 100,  # Maximum iterations for threshold search
    'convergence_tolerance': 0.01,  # Tolerance for distribution convergence
}
```

**Key Features:**
- `target_distribution` - Specifies SELL≈40%, BUY≈40%, HOLD≈20% ratios
- `search_up_range` - Grid search range for up thresholds (0.4% to 3.0% in 0.1% steps)
- `search_down_range` - Grid search range for down thresholds (-3.0% to -0.4% in 0.1% steps)
- `optimization_metric: 'kl_divergence'` - Uses KL divergence for distribution optimization

### C. DATA_CONFIG Confirmation

```python
DATA_CONFIG = {
    # ... existing config ...
    'max_4h_training_candles': 0,  # 0 means unlimited (use all historical data)
    'use_all_history': True,  # When True, fetch ALL historical data without limits
}
```

## Implementation Details

### 1. Dynamic Feature Selection Window

**Location**: `ml/trainer.py:prepare_training_data()`

```python
# Use selection_window_4h from FEATURE_SELECTION_CONFIG when available
selection_limit = FEATURE_SELECTION_CONFIG.get('selection_window_4h', 
                                             DATA_CONFIG.get('max_4h_selection_candles', 800))
```

**Behavior:**
- Feature selection operates on the most recent N candles per symbol (default: 800)
- Selection window is configurable within 500-1000 range
- Full training dataset remains unlimited when `use_all_history=True`

### 2. Adaptive Label Threshold Calibration

**Location**: `ml/trainer.py:_calibrate_label_thresholds()`

**Algorithm:**
1. **Price Change Calculation**: Computes 4-period forward price changes across all symbols
2. **Grid Search**: Evaluates all combinations of up/down thresholds from search ranges
3. **Distribution Matching**: Calculates actual vs target distribution for each threshold pair
4. **KL Divergence Optimization**: Selects thresholds minimizing KL divergence
5. **Constraint Validation**: Ensures up > 0 and down < 0 constraints

```python
def _calibrate_label_thresholds(self, df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calibrate label thresholds to achieve target distribution using grid search
    
    Returns:
        Tuple of (up_threshold, down_threshold) that best matches target distribution
    """
```

**Features:**
- **KL Divergence**: `D_KL(target || actual) = Σ target[i] * log(target[i] / actual[i])`
- **Early Stopping**: Terminates when convergence tolerance is reached
- **Iteration Limiting**: Maximum search iterations to prevent excessive computation
- **Fallback Handling**: Graceful degradation to static thresholds if calibration fails

### 3. Enhanced Label Generation

**Location**: `ml/trainer.py:_generate_labels()`

**Process:**
1. **Threshold Calibration**: Calls `_calibrate_label_thresholds()` to determine optimal thresholds
2. **Label Assignment**: Applies calibrated thresholds to price change calculations
3. **Distribution Logging**: Reports final achieved distribution ratios

```python
# Use calibrated thresholds for signals
if price_change_pct > up_threshold:
    label = 1  # BUY
elif price_change_pct < down_threshold:
    label = 0  # SELL
else:
    label = 2  # HOLD
```

### 4. Updated Feature Selection Integration

**Location**: `ml/trainer.py:train_with_rfe()`

**Changes:**
- Uses FEATURE_SELECTION_CONFIG parameters directly instead of config_loader
- Applies selection to recent window (`selection_X`, `selection_y`)
- Trains final model on full dataset (`X_full`, `y_full`) with selected features

```python
# Get feature selection configuration from FEATURE_SELECTION_CONFIG
min_features = FEATURE_SELECTION_CONFIG.get('min_features', 20)
max_iterations = FEATURE_SELECTION_CONFIG.get('max_iterations', 50)
tolerance = FEATURE_SELECTION_CONFIG.get('tolerance', 0.003)
```

## Pipeline Flow

1. **Data Loading**: Load ALL historical data (unlimited) for each symbol
2. **Selection Window**: Extract recent N candles per symbol for feature selection
3. **Feature Selection**: Perform dynamic selection on recent window only
4. **Label Calibration**: Calibrate thresholds on full dataset for target distribution
5. **Full Training**: Train XGBoost on ALL historical data with selected features

## Logging and Monitoring

### Feature Selection Logging
```
Dynamic selection completed: 45 features, macro_f1 improved from 0.742 to 0.758
Selected 45 features using dynamic method
FULL 4h TRAINING ROWS=25000 FEATURES(before selection)=197 FEATURE_SELECTION_ENABLED=True FEATURES_FINAL=45 XGB_TREES=8000
```

### Label Calibration Logging
```
Calibrating label thresholds for target distribution: {'SELL': 0.4, 'BUY': 0.4, 'HOLD': 0.2}
Threshold calibration completed: up=1.80%, down=-1.90%, KL_div=0.0234, iterations=42
Final label distribution: BUY=0.398, SELL=0.402, HOLD=0.200
```

## Testing

Comprehensive tests validate:
- ✅ Configuration parameter presence and values
- ✅ Target distribution ratios (SELL≈40%, BUY≈40%, HOLD≈20%)
- ✅ Selection window configuration (800 4h candles)
- ✅ Dynamic mode enablement
- ✅ Unlimited training history settings
- ✅ Integration readiness

## Backward Compatibility

The implementation maintains full backward compatibility:
- Existing configurations continue to work
- Graceful fallback to static thresholds if LABELING_CONFIG is missing
- Feature selection can be disabled via `enabled: False`
- Default values provided for all new parameters

## Performance Considerations

- **Selection Window**: Limits feature selection computation to recent data
- **Grid Search**: Bounded iteration count prevents excessive computation
- **Memory Management**: Efficient handling of full historical datasets
- **Early Stopping**: KL convergence tolerance prevents unnecessary iterations

## Integration Points

The implementation integrates seamlessly with existing systems:
- **Professional XGBoost**: Preserved XGB_PRO_CONFIG settings
- **Feature Selection**: Enhanced existing dynamic selection framework
- **Data Pipeline**: Maintained unlimited history loading capability
- **Model Training**: Preserved existing training pipeline structure