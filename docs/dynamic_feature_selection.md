# Dynamic Feature Selection

## Overview

The dynamic feature selection system implements an adaptive, performance-driven approach to feature selection that removes the hard-coded 50 feature limit. Instead of using a fixed number of features, the system dynamically determines the optimal feature set based on macro F1 score optimization and iterative pruning.

## Key Benefits

- **Performance-Driven**: Optimizes for macro F1 score rather than arbitrary feature counts
- **Adaptive**: Automatically finds the optimal number of features for the dataset
- **Correlation-Aware**: Removes highly correlated features before iterative selection
- **Robust**: Includes tolerance-based early stopping to prevent over-optimization
- **Transparent**: Provides detailed selection history for analysis and debugging

## How It Works

### Stage 1: Correlation Pruning
Before iterative selection begins, the system removes highly correlated features:
- Calculates correlation matrix for all candidate features (excluding must-include)
- Removes features with correlation > `corr_threshold` (default: 0.95)
- Preserves must-include features (base OHLCV, etc.)

### Stage 2: Baseline Evaluation
- Establishes baseline performance using all remaining features
- Uses stratified k-fold cross-validation for robust scoring
- Records baseline macro F1 score for comparison

### Stage 3: Iterative Pruning
The system iteratively removes the least important features while monitoring performance:

1. **Feature Importance Ranking**: Uses RandomForest to rank feature importance
2. **Selective Removal**: Drops lowest-importance features (excluding must-include)
3. **Performance Evaluation**: Tests new feature set with cross-validation
4. **Improvement Check**: Compares against best score with tolerance threshold
5. **Early Stopping**: Stops if no improvement for multiple iterations

### Stage 4: Final Selection
- Returns the feature set that achieved the best macro F1 score
- Includes detailed metadata about the selection process

## Configuration

### Config Keys

```yaml
feature_selection:
  mode: "dynamic"  # Enable dynamic selection
  dynamic:
    min_features: 20          # Minimum features to retain
    drop_fraction: 0.05       # Fraction of features to drop per iteration  
    corr_threshold: 0.95      # Correlation threshold for pruning
    tolerance: 0.003          # Minimum improvement to continue
    metric: 'macro_f1'        # Optimization metric
    cv_splits: 3              # Cross-validation folds
    max_iterations: 50        # Maximum pruning iterations
```

### Parameter Details

- **min_features**: Hard minimum on feature count (including must-include features)
- **drop_fraction**: Controls pruning speed; lower values = more conservative
- **corr_threshold**: Higher values = more aggressive correlation removal
- **tolerance**: Lower values = more sensitive to improvements (risk of overfitting)
- **metric**: Currently supports 'macro_f1' and 'balanced_accuracy'
- **cv_splits**: More splits = more robust but slower evaluation
- **max_iterations**: Safety limit to prevent infinite loops

## Usage

### Enabling Dynamic Selection

Set the mode in your configuration:

```yaml
feature_selection:
  mode: "dynamic"
```

### Must-Include Features

The system automatically preserves essential features:
- Base OHLCV data (open, high, low, close, volume)
- Required indicators from indicator definitions
- Any features specified in must_include parameter

### Selection History

The system logs detailed information about each iteration:

```python
selection_info = {
    'method': 'dynamic_iterative_pruning',
    'baseline_score': 0.742,
    'final_score': 0.758,
    'improvement': 0.016,
    'iterations': 12,
    'correlation_removed': 8,
    'history': [
        {
            'iteration': 0,
            'features_count': 87,
            'score': 0.742,
            'action': 'baseline'
        },
        {
            'iteration': 1,
            'features_count': 82,
            'score': 0.745,
            'action': 'dropped_5_features'
        },
        # ... more iterations
    ]
}
```

## Rationale

### Why Not Brute Force?

With N features, there are 2^N possible combinations. For 100 features, this means ~10^30 combinations - computationally impossible. The dynamic approach provides a practical solution that:

- Uses feature importance to guide selection
- Employs greedy optimization for efficiency
- Validates with cross-validation for robustness
- Includes early stopping to prevent overfitting

### Macro F1 vs Accuracy

The system optimizes for macro F1 score because:

- **Class Balance**: Macro F1 treats all classes equally, important for trading signals
- **Multi-Class**: Better suited for UP/DOWN/NEUTRAL classification
- **Robust**: Less sensitive to class imbalance than accuracy
- **Early Stopping**: XGBoost still uses mlogloss for model training early stopping

### Performance Considerations

Dynamic selection is more computationally expensive than fixed methods:

- Default `cv_splits=3` balances robustness with speed
- `drop_fraction=0.05` provides conservative pruning
- Early stopping prevents excessive iterations
- Recent sample window reduces computational load

## Backwards Compatibility

The system maintains full backwards compatibility:

- Setting `mode: "rfe"` uses original RFE selection
- Setting `mode: "shap"` uses SHAP-based selection
- Setting `mode: "hybrid"` uses hybrid RFE+statistical selection
- Existing `n_features_to_select` parameter ignored in dynamic mode

## Monitoring and Debugging

### Log Output

Dynamic selection provides extensive logging:

```
INFO: Starting dynamic feature selection with 134 features
INFO: Correlation pruning: removed 12 features, 97 candidates remain  
INFO: Baseline macro_f1: 0.7420 with 122 features
INFO: Iteration 1: macro_f1 = 0.7453 with 116 features
INFO: Improvement found: 0.7453 > 0.7450
...
INFO: Dynamic selection completed: 89 features, macro_f1 improved from 0.7420 to 0.7580
```

### Selection Metadata

Training results include comprehensive selection metadata for analysis and model serialization.

## Future Enhancements

- Support for additional metrics (precision, recall, custom)
- Multi-objective optimization (speed vs accuracy tradeoffs)
- Advanced pruning strategies (forward selection, bidirectional)
- Ensemble feature selection (multiple algorithms voting)