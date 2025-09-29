# Migration Guide: Upgrading to Enhanced Trading System

## Overview

This guide helps you migrate from the original trading bot to the enhanced version with comprehensive improvements and new features.

## What's New in the Enhanced Version

### 🚀 Major Improvements
- **Multi-Algorithm ML Models**: CatBoost + XGBoost with probability calibration
- **Adaptive Confidence Thresholds**: Dynamic adjustment for optimal signal rates
- **Enhanced Technical Indicators**: 196+ properly implemented indicators
- **Multi-Source Data Feed**: Automatic failover from CoinEx to Binance
- **Real-time Signal Broadcasting**: WebSocket updates and persistent storage
- **Comprehensive Health Monitoring**: System-wide status tracking
- **UTF-8 Safe Logging**: Windows compatibility with emoji stripping
- **SHAP-based Feature Selection**: Intelligent feature importance analysis

### 🔧 Technical Enhancements
- **Class Balancing**: Automatic handling of imbalanced datasets
- **Cross-validation**: 5-fold stratified validation for robust metrics
- **Configuration Management**: YAML-based config with environment overrides
- **Enhanced API Endpoints**: Comprehensive model and signal management
- **Improved Error Handling**: Graceful failures with retry logic

---

## Pre-Migration Checklist

### 1. Backup Your Current System
```bash
# Backup database
mysqldump TB > tb_backup_$(date +%Y%m%d).sql

# Backup model files
cp -r models models_backup_$(date +%Y%m%d)

# Backup configuration
cp config/settings.py config_backup_$(date +%Y%m%d).py
```

### 2. System Requirements
- Python 3.8+
- MySQL/MariaDB 5.7+
- Additional Python packages (see requirements below)

### 3. Install Additional Dependencies
```bash
pip install pyyaml catboost shap websockets
```

---

## Migration Steps

### Step 1: Configuration Migration

#### Create new YAML configuration
Create `config/config.yaml` based on your current `config/settings.py`:

```yaml
# config/config.yaml
trading:
  timeframe: "4h"  # From TRADING_CONFIG['timeframe']
  demo_balance: 100.0  # From TRADING_CONFIG['demo_balance'] 
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]  # From TRADING_CONFIG['symbols']
  confidence_threshold: 0.7  # From TRADING_CONFIG['confidence_threshold']
  max_positions: 4
  risk_per_trade: 0.02

adaptive_threshold:
  enabled: true  # NEW FEATURE
  target_signals_per_24h: 5
  min_threshold: 0.5
  max_threshold: 0.85
  adjustment_rate: 0.05

ml:
  model_type: "xgboost"  # or "catboost" for new model
  training_data_size: 58000
  test_size: 0.2
  random_state: 42
  
  # NEW: Enhanced labeling
  labeling:
    up_threshold: 0.02    # 2% return for UP
    down_threshold: -0.02 # -2% return for DOWN
  
  # NEW: Class balancing
  class_balance:
    method: "class_weight"
    max_class_ratio: 0.7

# NEW: Enhanced feature selection
feature_selection:
  method: "shap"  # or "rfe" for legacy behavior
  target_features: 50
  correlation_threshold: 0.9

# Enhanced logging
logging:
  level: "INFO"
  unicode_safe: true
  emoji_strip: true  # For Windows compatibility

# Database (migrate from DATABASE_CONFIG)
database:
  host: "localhost"  # From DATABASE_CONFIG['host']
  port: 3306
  user: "root"
  password: ""  # Set via environment variable
  database: "TB"
  charset: "utf8mb4"

# API Configuration (migrate from COINEX_CONFIG)
coinex:
  api_key: ""  # Set via environment variable  
  secret_key: ""  # Set via environment variable
  base_url: "https://api.coinex.com/v1/"
```

#### Environment Variables
Create `.env` file for sensitive data:
```bash
# .env
DB_PASSWORD=your_database_password
COINEX_API_KEY=your_coinex_api_key
COINEX_SECRET_KEY=your_coinex_secret_key
MODEL_TYPE=catboost  # Optional: override config
```

### Step 2: Update Application Entry Point

Replace your current main.py usage with the enhanced version:

#### Option A: Use Enhanced Main (Recommended)
```bash
python enhanced_main.py
```

#### Option B: Update Existing main.py
Add enhanced engine import:
```python
# At top of main.py
from trading.enhanced_engine import EnhancedTradingEngine
from config.config_loader import load_config

# Replace TradingEngine with EnhancedTradingEngine
config = load_config()
trading_engine = EnhancedTradingEngine(demo_mode=True)
```

### Step 3: Database Schema Updates

The enhanced system uses the same database schema but adds new tables:

```sql
-- New tables will be created automatically
-- No manual migration required
-- Your existing data is preserved
```

### Step 4: Model Migration

#### Automatic Model Retraining
The enhanced system will automatically retrain your model with new features:
- Previous model files in `models/` are preserved
- New models use enhanced indicators and better validation
- Model metadata is now saved with comprehensive metrics

#### Manual Model File Migration (Optional)
```bash
# Create symlink for backward compatibility
ln -s models/model_$(date +%Y%m%d)_*.joblib models/latest_model.joblib
```

---

## Testing Your Migration

### 1. Configuration Test
```bash
python test_integration.py
```

### 2. Start Enhanced System
```bash
python enhanced_main.py
```

### 3. Verify Web Interface
Open http://localhost:5000 and check:
- ✅ Dashboard loads without errors
- ✅ Model status shows training progress
- ✅ Health endpoint returns green status
- ✅ Signal history is accessible

### 4. API Endpoint Tests
```bash
# Test new endpoints
curl http://localhost:5000/api/health
curl http://localhost:5000/api/model/status  
curl http://localhost:5000/api/signals/recent
```

---

## Feature Comparison

| Feature | Original | Enhanced | Notes |
|---------|----------|----------|--------|
| ML Models | XGBoost only | XGBoost + CatBoost | Better handling of categorical features |
| Confidence Threshold | Fixed 0.7 | Adaptive 0.5-0.85 | Maintains optimal signal rates |
| Technical Indicators | ~78 (many placeholder) | 196+ (fully implemented) | All indicators properly calculated |
| Data Sources | CoinEx only | CoinEx + Binance failover | Improved reliability |
| Feature Selection | Basic RFE | SHAP-based + correlation | More intelligent selection |
| Class Balancing | None | Multiple methods | Handles imbalanced data |
| Signal Persistence | Database only | Database + WebSocket | Real-time updates |
| Health Monitoring | Basic | Comprehensive | Multi-layer health checks |
| Configuration | Python file | YAML + env overrides | More flexible and secure |
| Logging | Basic | UTF-8 safe + structured | Windows compatible |
| Model Metadata | Basic | Comprehensive | Full training metrics |
| Cross-validation | None | 5-fold stratified | Better performance estimates |

---

## Rollback Plan

If you need to rollback to the original system:

### 1. Stop Enhanced System
```bash
# Stop if running
pkill -f enhanced_main.py
```

### 2. Restore Original Configuration
```bash
# Your original config/settings.py is unchanged
# Simply use the original main.py
python main.py
```

### 3. Restore Database (if needed)
```bash
mysql TB < tb_backup_YYYYMMDD.sql
```

---

## Common Migration Issues

### Issue: "No module named 'catboost'"
**Solution**: Install missing dependencies
```bash
pip install catboost
```

### Issue: "YAML configuration not found"
**Solution**: Ensure config.yaml exists or set path
```bash
export CONFIG_PATH=/path/to/your/config.yaml
```

### Issue: "Database connection failed"
**Solution**: Check environment variables
```bash
export DB_PASSWORD=your_password
export DB_HOST=localhost
```

### Issue: "No signals generated"
**Solution**: Check adaptive threshold settings
- Verify `adaptive_threshold.enabled: true`
- Check model training logs
- Use `/api/prediction/force` to test predictions

### Issue: "WebSocket connection failed"
**Solution**: Check port availability
- Default WebSocket port: 8765
- Ensure no firewall blocking
- Check `signals.websocket_broadcast: true`

---

## Performance Considerations

### Enhanced Features Impact
- **Startup Time**: +30-60s (additional indicator calculations)
- **Memory Usage**: +20-30% (more features and caching)
- **Training Time**: +50-100% (cross-validation and better validation)
- **Prediction Latency**: Similar (optimized feature calculation)

### Optimization Tips
1. **Reduce Training Data**: Lower `training_data_size` if needed
2. **Disable Cross-validation**: Set `cross_validation.enabled: false`
3. **Use XGBoost**: Slightly faster than CatBoost for some datasets
4. **Adjust Feature Count**: Lower `target_features` for faster training

---

## Getting Help

### Debug Mode
Enable detailed logging:
```yaml
logging:
  level: "DEBUG"
  structured_json: true
```

### Health Check
Monitor system health:
```bash
curl http://localhost:5000/api/health | jq
```

### Log Analysis
Check structured logs:
```bash
tail -f logs/trading_bot.log | grep "ML_EVENT\|TRADING_EVENT"
```

### Community Support
- Create GitHub issue with logs and configuration
- Include output from `/api/health` endpoint
- Mention migration step where issue occurred

---

## Next Steps After Migration

1. **Monitor Performance**: Check model accuracy and signal quality for 24-48 hours
2. **Adjust Thresholds**: Fine-tune adaptive threshold parameters if needed
3. **Explore CatBoost**: Try CatBoost model for potentially better performance
4. **Set Up Monitoring**: Consider Grafana/Prometheus integration
5. **Review Signals**: Analyze signal history and adjust parameters

## Success Metrics

Your migration is successful when:
- ✅ System starts without errors
- ✅ Model trains with validation accuracy > 60%
- ✅ Signals are generated within 24 hours
- ✅ Adaptive thresholds are adjusting appropriately
- ✅ WebSocket updates work in browser
- ✅ All API endpoints return valid responses