# Enhanced Trading System - Model Pipeline Documentation

## Overview

This document describes the comprehensive ML model pipeline implemented in the enhanced trading bot system. The pipeline addresses all the critical issues identified in the original system and provides a robust, production-ready trading solution.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Enhanced Trading System                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Data Layer    │  │   ML Pipeline   │  │  Signal Layer   │  │
│  │                 │  │                 │  │                 │  │
│  │ • Multi-source  │  │ • Enhanced      │  │ • Adaptive      │  │
│  │   API client    │  │   indicators    │  │   thresholds    │  │
│  │ • Failover      │  │ • SHAP-based    │  │ • WebSocket     │  │
│  │   CoinEx→Binance│  │   selection     │  │   broadcast     │  │
│  │ • Health        │  │ • CatBoost +    │  │ • Persistent    │  │
│  │   monitoring    │  │   XGBoost       │  │   storage       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Ingestion
- **Primary Source**: CoinEx API with robust error handling
- **Secondary Source**: Binance API for automatic failover
- **Health Monitoring**: Real-time status tracking with consecutive failure detection
- **Retry Logic**: Exponential backoff with configurable limits

### 2. Enhanced Indicator Calculation
The system calculates 196+ technical indicators, including previously missing ones:

#### Volume-Based Indicators
- **On Balance Volume (OBV)**: Cumulative volume flow indicator
- **Chaikin Money Flow (CMF)**: Volume-weighted price momentum (10/20/50 periods)
- **Accumulation/Distribution Line**: Price-volume relationship
- **Force Index**: Price change and volume combination (2-period)

#### Volatility Indicators
- **Donchian Channels**: Breakout levels (20 and 55 periods)
- **Chaikin Volatility**: High-low spread volatility (14-period)

#### Momentum Indicators
- **QStick**: Open-close momentum (20-period)
- **Know Sure Thing (KST)**: Multi-timeframe momentum oscillator
- **True Strength Index (TSI)**: Double-smoothed momentum (25,13 and 13,7)

#### Price Transformation
- **Heikin Ashi**: Smoothed candlestick representation (OHLC)

#### Additional Features
- **Returns**: 1-period and 5-period price returns
- **Volatility**: 10-period realized volatility
- **Volume Ratios**: Current vs 20-period average volume
- **Price Z-Scores**: 20-period standardized prices

## Target Labeling Strategy

### Enhanced Labeling Logic
```python
# Configurable thresholds
UP: future_return >= +2.0%     # Bullish signal
DOWN: future_return <= -2.0%   # Bearish signal
NEUTRAL: -2.0% < return < +2.0% # No clear direction
```

### Class Balancing
- **Method Options**: class_weight, oversample, undersample, downsample_neutral
- **Max Class Ratio**: Prevent any single class from exceeding 70% of data
- **Validation**: Automatic detection of extreme imbalances

## Feature Selection Pipeline

### Traditional SHAP-Based Selection
1. **Importance Calculation**: SHAP values for feature importance
2. **Correlation Analysis**: Remove features with correlation > 0.9
3. **Importance Filtering**: Keep features with importance > 0.001
4. **Target Selection**: Top N features based on importance

### Advanced Adaptive Selection (NEW)
When `adaptive.enabled = true`, the system uses a sophisticated multi-stage pipeline:

1. **Base Feature Preservation**: Always includes OHLCV/foundational features
2. **Stage A - Initial Importance**: RandomForest/SHAP ranking → top K0 candidates (e.g., 80)
3. **Stage B - Correlation Pruning**: Remove highly correlated features (|ρ| > 0.9)
4. **Stage C - SFFS with CV**: Sequential Forward Floating Selection with stratified k-fold CV optimizing F1 macro score
5. **Stage D - Early Stopping**: Stop if F1 macro improvement < epsilon over patience rounds
6. **Stage E - Dynamic Subset**: Find optimal subset size (5-50 range) via F1 macro performance testing

**Key Benefits:**
- Dynamic feature count (not fixed at 50)
- Better generalization through cross-validation with F1 macro optimization
- Preserved domain knowledge (base features always included)
- Early stopping prevents overfitting (adaptive for F1 metric variance)
- Robust fallback mechanisms

### Fallback Methods
- **Permutation Importance**: When SHAP is unavailable
- **Tree-Based Importance**: Final fallback using RandomForest
- **Recursive Feature Elimination**: Legacy method with sklearn

## Model Training Pipeline

### Multi-Algorithm Support

#### XGBoost Configuration
```yaml
xgboost:
  n_estimators: 1500
  max_depth: 6
  learning_rate: 0.05
  early_stopping_rounds: 100
  subsample: 0.8
  colsample_bytree: 0.8
```

#### CatBoost Configuration
```yaml
catboost:
  iterations: 1500
  depth: 6
  learning_rate: 0.05
  early_stopping_rounds: 100
  loss_function: "MultiClass"
  auto_class_weights: "Balanced"
```

### Training Enhancements
1. **Cross-Validation**: 5-fold StratifiedKFold for robust performance estimation
2. **Class Weighting**: Automatic inverse frequency weighting
3. **Probability Calibration**: Isotonic regression for trustworthy probabilities
4. **Early Stopping**: Prevent overfitting with validation monitoring

### Model Metadata
Each trained model saves comprehensive metadata:
```json
{
  "model_type": "catboost",
  "model_version": "20240101_120000",
  "val_accuracy": 0.742,
  "train_accuracy": 0.756,
  "precision": 0.731,
  "recall": 0.742,
  "f1_score": 0.736,
  "roc_auc": 0.812,
  "feature_count": 78,
  "training_samples": 45230,
  "class_distribution": {"UP": 15234, "DOWN": 14890, "NEUTRAL": 15106},
  "cv_mean_accuracy": 0.738,
  "cv_std_accuracy": 0.012,
  "training_time_seconds": 342.5,
  "git_commit_hash": "b269128...",
  "feature_importances": {...}
}
```

## Prediction and Signal Generation

### Prediction Scheduler
- **Interval**: Configurable prediction frequency (default: 120 seconds)
- **Intra-Candle**: Predictions between timeframe boundaries
- **Data Requirements**: Minimum data points validation
- **Error Handling**: Graceful failures with retry logic

### Adaptive Threshold Management
Traditional fixed thresholds (0.7) often suppress signals, especially for minority classes. The adaptive system:

1. **Target Rate**: Maintain 3-8 signals per 24h per symbol
2. **Dynamic Adjustment**: Increase threshold if too many signals, decrease if too few
3. **Range Constraints**: Keep thresholds within [0.5, 0.85]
4. **Evaluation Window**: Rolling 24-hour analysis window

```python
# Pseudo-code for adaptive threshold
if signal_rate_24h > target_rate + tolerance:
    threshold = min(max_threshold, current_threshold + adjustment_rate)
elif signal_rate_24h < target_rate - tolerance:
    threshold = max(min_threshold, current_threshold - adjustment_rate)
```

### Signal Persistence and Broadcasting
1. **Database Storage**: Persistent signal history with status tracking
2. **WebSocket Broadcasting**: Real-time updates to connected clients
3. **Signal Expiry**: Automatic cleanup of old signals
4. **Status Management**: Track signal lifecycle (active, executed, expired)

## API Endpoints

### Enhanced Model Status
```
GET /api/model/status
```
Returns comprehensive model information:
```json
{
  "success": true,
  "model_type": "catboost",
  "model_version": "20240101_120000",
  "val_accuracy": 0.742,
  "feature_count": 78,
  "training_samples": 45230,
  "last_trained_at": "2024-01-01T12:00:00",
  "class_distribution": {...},
  "confusion_matrix": [[...]]
}
```

### Signal Management
```
GET /api/signals/recent?limit=50&symbol=BTCUSDT&hours=24
PUT /api/signals/{signal_id}/status
```

### Health Monitoring
```
GET /api/health
```
Returns system-wide health status including data sources, model status, and service availability.

### Manual Prediction
```
GET /api/prediction/force?symbol=BTCUSDT
```
Triggers immediate prediction for debugging purposes.

## Configuration Management

### YAML-Based Configuration
```yaml
# config/config.yaml
trading:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
  timeframe: "4h"
  confidence_threshold: 0.7  # Base threshold

adaptive_threshold:
  enabled: true
  target_signals_per_24h: 5
  min_threshold: 0.5
  max_threshold: 0.85

ml:
  model_type: "catboost"  # or "xgboost"
  labeling:
    up_threshold: 0.02
    down_threshold: -0.02

feature_selection:
  adaptive:
    enabled: true
    scoring_metric: "f1_macro"  # Default optimization metric
    early_stopping_patience: 5  # Adjusted for F1 variance
    early_stopping_epsilon: 0.001  # Adjusted for F1 variance
    down_threshold: -0.02
```

### Environment Variable Overrides
- `MODEL_TYPE`: Override model type
- `CONFIDENCE_THRESHOLD`: Override base threshold
- `COINEX_API_KEY`: API credentials
- Database connection strings

## Logging and Observability

### UTF-8 Safe Logging
- **Windows Compatibility**: Automatic emoji stripping for cp1252 encoding
- **Structured Events**: ML and trading event logging with metadata
- **Log Levels**: Configurable verbosity with external library noise reduction

### Health Monitoring
- **Data Source Health**: Primary/secondary API status
- **Model Performance**: Real-time accuracy tracking
- **Signal Rate Monitoring**: Adaptive threshold effectiveness
- **System Resources**: Database connection, memory usage

## Performance Optimizations

### Training Optimizations
1. **Efficient Feature Selection**: Sample-based SHAP calculation
2. **Memory Management**: Chunked data processing for large datasets
3. **Parallel Processing**: Multi-threaded indicator calculation
4. **Incremental Learning**: Model update strategies

### Prediction Optimizations
1. **Feature Caching**: Reuse calculated indicators
2. **Batch Predictions**: Group predictions for efficiency
3. **Lazy Loading**: On-demand data fetching

## Deployment Considerations

### Database Setup
```sql
CREATE DATABASE TB;
-- Tables created automatically by SQLAlchemy
```

### Environment Requirements
- Python 3.8+
- MySQL/MariaDB for persistence
- Optional: Redis for caching
- WebSocket support for real-time updates

### Scaling Options
1. **Horizontal Scaling**: Multiple prediction instances
2. **Model Serving**: Dedicated model servers
3. **Cache Layer**: Redis for feature caching
4. **Load Balancing**: Multiple web interface instances

## Troubleshooting

### Common Issues
1. **No Signals Generated**: Check adaptive threshold settings and model performance
2. **Data Source Failures**: Verify API credentials and network connectivity
3. **Model Training Failures**: Check data quality and class balance
4. **WebSocket Disconnections**: Review network stability and client implementation

### Debug Endpoints
- `/api/prediction/force`: Manual prediction trigger
- `/api/health`: Comprehensive system status
- `/api/adaptive-threshold/status`: Threshold adjustment tracking

## Future Enhancements

### Planned Features
1. **Multi-Timeframe Models**: 1h, 4h, 1d combined analysis
2. **Ensemble Methods**: Multiple model voting
3. **Risk Management**: Position sizing and portfolio optimization
4. **Backtesting Framework**: Historical performance validation
5. **A/B Testing**: Model comparison framework

### Monitoring Improvements
1. **Grafana Integration**: Real-time dashboards
2. **Alerting System**: Slack/email notifications
3. **Performance Tracking**: Model drift detection
4. **Automated Retraining**: Schedule-based model updates

This enhanced pipeline addresses all the critical issues identified in the original system while providing a robust, scalable foundation for algorithmic trading.