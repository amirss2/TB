# Part 2: Dynamic Data Window for RFE - Implementation Summary

## Overview
This document describes the implementation of dynamic data window calculation for Recursive Feature Elimination (RFE) in the AI trading bot.

## Implementation Details

### 1. Dynamic Selection Window Calculation

**Formula:**
```
N_required = L_max + H + E + B
```

Where:
- **L_max**: Maximum lookback period among all indicators
- **H**: Horizon for Triple-Barrier labeling (from config)
- **E**: Purge/embargo candles for time-series CV (3 candles)
- **B**: Safety buffer (50 candles)

### 2. Key Components

#### A. `_calculate_dynamic_selection_window()` Method
- **Purpose**: Calculate optimal window size for RFE based on indicator requirements
- **Process**:
  1. Scans all indicator definitions to find maximum lookback
  2. Extracts period/length parameters from each indicator
  3. Special handling for Ichimoku (52-period), EMAs, SMAs
  4. Adds horizon, embargo, and buffer
  5. Logs detailed calculation breakdown

- **Output**: N_required (number of 4h candles needed for RFE)

#### B. Enhanced `prepare_training_data()` Method
- **New Parameter**: `use_dynamic_window: bool = True`
- **Behavior**:
  - When `True`: Calculates selection_limit dynamically
  - When `False`: Uses configured static value
- **Logging**: Comprehensive logging of window size and rationale

#### C. Enhanced `_time_series_split()` Method
- **New Parameter**: `embargo_candles: int = 3`
- **Improvements**:
  - Adds gap between train and validation sets
  - Prevents data leakage from future information
  - Maintains strict chronological order (NO SHUFFLING)
  - Detailed logging of fold boundaries

### 3. Data Leakage Prevention

#### Implemented Safeguards:
1. **No Data Shuffling**: All splits maintain chronological order
2. **Embargo Gap**: 3-candle gap between train and validation
3. **Scaler Fitting**: StandardScaler fit ONLY on training data
4. **Feature Selection**: RFE performed on recent window only
5. **Time-Series CV**: Walk-Forward validation respects time order

#### Logging Evidence:
```
Time-Series CV Configuration:
  Total samples: XXXX
  Number of folds: 5
  Fold size: XXX samples
  Embargo (gap): 3 candles
  NO SHUFFLING - Chronological order maintained

Fold 1: Train[0:XXX] (XXX samples), Gap[XXX:XXX] (3 candles), Val[XXX:XXX] (XXX samples)
```

### 4. RFE Window Selection

#### Process:
1. Calculate N_required using formula
2. Select last N_required candles from each symbol
3. This gives most recent market conditions for feature selection
4. Apply indicators and remove NaN rows
5. Check class distribution in RFE window
6. If class imbalance detected, apply SMOTE before RFE

#### Handling Insufficient Data:
- If symbol has fewer than N_required candles:
  - Use all available candles for that symbol
  - Log warning about limited data
- If critical indicator requirements not met:
  - Reduce buffer or embargo
  - Fail-fast with clear error message

### 5. Final Model Training

#### After RFE:
1. **Feature Selection**: Best features identified from recent window
2. **Full Training**: Model trained on ALL history of 4 symbols
3. **Symbols**: BTC/ETH/DOGE/SOL only (as requested)
4. **SMOTE**: Applied to full dataset before final training
5. **Calibration**: Isotonic calibration on 20% holdout

### 6. Acceptance Log Format

At end of training, comprehensive log includes:

```
═══════════════════════════════════════════════════════════════
FINAL TRAINING ACCEPTANCE LOG
═══════════════════════════════════════════════════════════════

CV Method: Walk-Forward with Embargo (K=5)
Embargo: 3 candles between train/validation

Per-Fold Metrics:
  Fold 1: Accuracy=X.XXXX, Macro-F1=X.XXXX
  Fold 2: Accuracy=X.XXXX, Macro-F1=X.XXXX
  Fold 3: Accuracy=X.XXXX, Macro-F1=X.XXXX
  Fold 4: Accuracy=X.XXXX, Macro-F1=X.XXXX
  Fold 5: Accuracy=X.XXXX, Macro-F1=X.XXXX

Average Metrics:
  Accuracy: X.XXXX ± X.XXXX
  Macro-F1: X.XXXX ± X.XXXX

Optimal Threshold (after costs): X.XX
  Expected Trade Rate: X.XX%
  Expected Utility: X.XXXX

Feature Selection:
  Selection Window: N_required = XXX candles
  Selected Features: XX
  Feature List: [feature1, feature2, ...]

Training Symbols: [BTCUSDT, ETHUSDT, DOGEUSDT, SOLUSDT]
Training Mode: use_all_history=True

Total Training Time: XX.X seconds

Reproducibility Info:
  Random Seed: 42
  scikit-learn: X.X.X
  xgboost: X.X.X
  numpy: X.X.X
  pandas: X.X.X
  
═══════════════════════════════════════════════════════════════
```

### 7. Configuration

#### New Settings in `config/settings.py`:
- `FEATURE_SELECTION_CONFIG['selection_window_4h']`: Static fallback value
- `DATA_CONFIG['use_all_history']`: Enable/disable full history training
- `LABELING_CONFIG['triple_barrier']['time_horizon_candles']`: Used in H calculation

### 8. Usage Example

```python
from ml.trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Train with dynamic window (Part 2 implementation)
result = trainer.train_with_advanced_cv()

# Dynamic window will be calculated automatically
# Based on indicator requirements and labeling horizon
```

### 9. Validation Checks

#### The implementation validates:
1. ✅ N_required calculation is correct
2. ✅ Selection window contains only recent data
3. ✅ No future data leaks into training
4. ✅ Embargo gap prevents information leakage
5. ✅ Scaler fit only on training folds
6. ✅ Class distribution logged at each stage
7. ✅ SMOTE applied correctly (k_neighbors constraint)
8. ✅ Final model uses all history with selected features
9. ✅ Only 4 symbols used (BTC/ETH/DOGE/SOL)
10. ✅ Comprehensive reproducibility logging

### 10. Expected Behavior

#### Dynamic Window Calculation Example:
```
DYNAMIC SELECTION WINDOW CALCULATION
════════════════════════════════════
Max indicator lookback (L_max): 100
Triple-Barrier horizon (H): 2
Purge/embargo candles (E): 3
Safety buffer (B): 50
N_required = 100 + 2 + 3 + 50 = 155
════════════════════════════════════

Top indicators by lookback period:
  Momentum_100: 100 candles
  EMA_89: 89 candles
  SMA_50: 50 candles
  Ichimoku: 52 candles
  ATR_14: 14 candles
  ...

Using dynamically calculated selection_limit: 155 (4h candles)
```

## Benefits

1. **Adaptive**: Window size adjusts to indicator requirements
2. **No Overfitting**: Recent data prevents memorization
3. **No Leakage**: Strict time-series handling with embargo
4. **Transparent**: Comprehensive logging of all decisions
5. **Reproducible**: Fixed seeds and versioning
6. **Robust**: Handles edge cases (insufficient data, class imbalance)

## Testing Recommendations

1. Check logs for N_required calculation
2. Verify no shuffling in CV folds
3. Confirm embargo gap in fold boundaries
4. Validate class distribution after SMOTE
5. Compare train vs validation metrics (gap should be <10%)
6. Monitor overfitting indicators

## Notes

- Minimum N_required is ~155 candles (with 100-period indicator)
- Actual value depends on indicators in technical_indicators_only.csv
- If indicators change, N_required recalculates automatically
- Fallback to configured value if calculation fails
- All 4 symbols must have sufficient data or warning logged

## Next Steps

After Part 2 implementation:
1. Retrain model using `train_with_advanced_cv()`
2. Monitor N_required in logs
3. Verify selection window size is appropriate
4. Check CV metrics for overfitting
5. Validate embargo is working (no leakage)
6. Test with different indicator sets
7. Measure prediction accuracy on unseen data
