# Adaptive Feature Selection Documentation

## Overview

The Enhanced Feature Selection system now includes an advanced adaptive feature selection pipeline that replaces the fixed-size (50) RFE-only selection with a sophisticated multi-stage approach.

## Key Improvements

### 1. Base Feature Preservation
- **Always includes foundational OHLCV features**: `open`, `high`, `low`, `close`, `volume`, `OHLC4`
- These features are automatically identified and preserved throughout the selection process
- Candidate pool consists of technical indicators (≈196) minus base features

### 2. Multi-Stage Reduction Pipeline

#### Stage A: Initial Importance Ranking
- Uses RandomForest, SHAP, or permutation importance to rank all candidate features
- Selects top K0 features (configurable, default: 80) by global importance
- Reduces noise from irrelevant indicators early in the process

#### Stage B: Correlation Pruning  
- Removes highly correlated features (|ρ| > 0.9 by default)
- Preserves the most important feature from each correlated pair
- Reduces redundancy and multicollinearity issues

#### Stage C: Sequential Forward Floating Selection (SFFS)
- Applies SFFS with stratified k-fold cross-validation (default: 5 folds)
- Optimizes for balanced_accuracy or macro F1 score
- Uses intelligent feature subset search for optimal combinations

#### Stage D: Early Stopping
- Stops if best score improvement < epsilon (default: 0.002) over patience rounds (default: 3)
- Prevents overfitting and reduces computational overhead
- Iteratively adds features with performance validation

#### Stage E: Dynamic Subset Size Selection
- Tests different subset sizes within allowed range (default: 5-50 total features)
- Selects the best performing subset size based on cross-validation
- Final feature count is determined by performance, not fixed limits

### 3. Robust Error Handling
- Graceful fallback to simpler methods when SFFS fails
- Adaptive cross-validation fold selection based on sample size
- Handles edge cases with small datasets or insufficient samples
- NaN score filtering and validation

## Configuration

Add the following to your `config.yaml`:

```yaml
feature_selection:
  method: "shap"  # Options: rfe, shap, permutation
  target_features: 50  # Used for traditional mode
  correlation_threshold: 0.9
  importance_threshold: 0.001
  
  # Advanced Adaptive Feature Selection
  adaptive:
    enabled: true  # Set to true to enable adaptive selection
    initial_top_k: 80  # Stage A: Keep top K candidates by importance
    min_features: 5    # Minimum total features (including base)
    max_features: 50   # Maximum total features (including base) 
    cv_folds: 5        # Cross-validation folds for SFFS
    early_stopping_patience: 3     # Stop if no improvement for N rounds
    early_stopping_epsilon: 0.002  # Minimum improvement threshold
    scoring_metric: "balanced_accuracy"  # Options: balanced_accuracy, f1_macro
    base_features: ["open", "high", "low", "close", "volume", "OHLC4"]  # Always included
```

## Usage Example

```python
from ml.enhanced_feature_selection import EnhancedFeatureSelector
from config.config_loader import load_config

# Load configuration
config = load_config()

# Enable adaptive selection
config['feature_selection']['adaptive']['enabled'] = True

# Initialize selector
selector = EnhancedFeatureSelector(config)

# Select features (X should include OHLCV + technical indicators)
selected_features, selection_info = selector.select_features(X, y)

print(f"Selected {len(selected_features)} features")
print(f"Method: {selection_info['method']}")
print(f"Final CV score: {selection_info.get('final_score', 'N/A')}")
```

## Performance Characteristics

### Advantages
- **Dynamic feature count**: Automatically determines optimal number of features (5-50 range)
- **Better generalization**: Cross-validation and early stopping prevent overfitting
- **Preserved domain knowledge**: Base OHLCV features always included
- **Reduced redundancy**: Correlation pruning eliminates similar features
- **Robust selection**: Multi-stage approach with fallbacks

### Performance Considerations
- **Computational cost**: Higher than simple RFE due to cross-validation
- **Time complexity**: O(n²) for correlation analysis, O(n·k·CV) for SFFS
- **Memory usage**: Moderate - processes features in stages
- **Recommended for**: Datasets with >500 samples and >20 features

### Fallback Behavior
1. If adaptive selection fails → Falls back to SHAP-based selection
2. If SHAP fails → Falls back to permutation importance  
3. If permutation fails → Falls back to tree-based importance
4. If all fail → Falls back to RFE
5. Ultimate fallback → Returns all features up to target limit

## Monitoring and Validation

The selection process provides detailed metadata:

```python
selection_info = {
    'method': 'adaptive_multi_stage',
    'total_features': 200,
    'selected_count': 23,
    'base_features_count': 6,
    'candidate_features_count': 17,
    'stages': {
        'stage_a_initial_count': 80,
        'stage_b_correlation_count': 45,
        'stage_c_sffs_count': 25,
        'stage_c_score': 0.742,
        'stage_d_early_stopping_count': 20,
        'stage_d_score': 0.751,
        'stage_e_dynamic_count': 17,
        'stage_e_score': 0.748
    },
    'final_score': 0.748,
    'parameters': {...}
}
```

## Best Practices

1. **Enable for production datasets**: Use adaptive selection for real trading data
2. **Disable for testing**: Use traditional methods for quick prototyping
3. **Monitor performance**: Track final_score and stage scores for validation
4. **Adjust parameters**: Tune based on your dataset size and computational budget
5. **Validate results**: Compare against traditional methods to ensure improvement

## Compatibility

- **Backward compatible**: Traditional feature selection still available when `adaptive.enabled = false`
- **Existing code**: No changes required to existing EnhancedFeatureSelector usage
- **Dependencies**: Requires scikit-learn >= 1.0 for SequentialFeatureSelector
- **Optional**: SHAP library for enhanced importance calculation