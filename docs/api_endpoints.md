# API Endpoints Documentation

## Overview
This document describes the enhanced API endpoints available in the trading bot system.

## Base URL
```
http://localhost:5000
```

## Authentication
Most endpoints are publicly accessible in demo mode. Production deployments should implement proper authentication.

---

## Health Monitoring

### GET /api/health
**Description**: Comprehensive system health check

**Response**:
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "status": "healthy",
  "services": {
    "database": {
      "status": "healthy",
      "last_check": "2024-01-01T12:00:00Z"
    },
    "trading_engine": {
      "status": "healthy",
      "details": {
        "is_running": true,
        "model_trained": true
      }
    },
    "prediction_scheduler": {
      "status": "healthy",
      "details": {
        "is_running": true,
        "statistics": {...}
      }
    }
  },
  "data_sources": {
    "primary_source": {
      "name": "coinex",
      "status": "healthy",
      "consecutive_failures": 0
    },
    "secondary_source": {
      "name": "binance", 
      "status": "healthy"
    }
  }
}
```

---

## Model Management

### GET /api/model/status
**Description**: Get comprehensive model information

**Response**:
```json
{
  "success": true,
  "model_type": "catboost",
  "model_version": "20240101_120000",
  "is_trained": true,
  "feature_count": 78,
  "training_samples": 45230,
  "val_accuracy": 0.742,
  "train_accuracy": 0.756,
  "precision": 0.731,
  "recall": 0.742,
  "f1_score": 0.736,
  "roc_auc": 0.812,
  "last_trained_at": "2024-01-01T12:00:00",
  "training_time_seconds": 342.5,
  "class_distribution": {
    "UP": 15234,
    "DOWN": 14890, 
    "NEUTRAL": 15106
  },
  "confusion_matrix": [[5023, 234, 123], [...]],
  "cv_mean_accuracy": 0.738,
  "cv_std_accuracy": 0.012
}
```

---

## Signal Management

### GET /api/signals/recent
**Description**: Get recent trading signals

**Parameters**:
- `limit` (optional): Number of signals to return (default: 50, max: 100)
- `symbol` (optional): Filter by specific symbol
- `hours` (optional): Number of hours to look back (default: 24)

**Example**: `/api/signals/recent?limit=20&symbol=BTCUSDT&hours=6`

**Response**:
```json
{
  "success": true,
  "signals": [
    {
      "id": "BTCUSDT_1704110400",
      "timestamp": "2024-01-01T12:00:00Z",
      "symbol": "BTCUSDT",
      "direction": "BUY",
      "confidence": 0.756,
      "price": 43250.50,
      "threshold_used": 0.650,
      "model_version": "20240101_120000",
      "status": "active"
    }
  ],
  "count": 1,
  "filters": {
    "symbol": "BTCUSDT",
    "hours_back": 6,
    "limit": 20
  }
}
```

### PUT /api/signals/{signal_id}/status
**Description**: Update signal status

**Request Body**:
```json
{
  "status": "executed"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Signal BTCUSDT_1704110400 status updated to executed"
}
```

---

## Prediction System

### GET /api/prediction/force
**Description**: Force immediate prediction for debugging

**Parameters**:
- `symbol` (optional): Specific symbol to predict (if not provided, all symbols)

**Example**: `/api/prediction/force?symbol=BTCUSDT`

**Response**:
```json
{
  "status": "completed",
  "results": {
    "BTCUSDT": "success",
    "ETHUSDT": "success"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## System Status

### GET /api/system/status
**Description**: Get comprehensive system status

**Response**:
```json
{
  "success": true,
  "system_status": {
    "timestamp": "2024-01-01T12:00:00Z",
    "services": {
      "trading_engine": {
        "is_running": true,
        "demo_mode": true,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "components": {...}
      },
      "prediction_scheduler": {
        "is_running": true,
        "interval_seconds": 120,
        "statistics": {
          "total_predictions": 1250,
          "signals_generated": 23,
          "last_run": "2024-01-01T11:58:00Z",
          "errors": 0
        }
      }
    },
    "statistics": {
      "signals": {
        "persistence": {...},
        "websocket": {...}
      }
    }
  }
}
```

---

## Position Management

### GET /api/positions
**Description**: Get current positions

**Response**:
```json
{
  "success": true,
  "positions": [
    {
      "id": 1,
      "symbol": "BTCUSDT",
      "side": "BUY",
      "size": 0.1,
      "entry_price": 43000.00,
      "current_price": 43250.50,
      "pnl": 25.05,
      "pnl_percent": 0.58,
      "status": "open"
    }
  ]
}
```

---

## Market Data

### GET /api/market/overview
**Description**: Get market overview for all symbols

**Response**:
```json
{
  "success": true,
  "market_data": {
    "BTCUSDT": {
      "price": 43250.50,
      "change_24h": 2.34,
      "volume": 1250000000,
      "high_24h": 44000.00,
      "low_24h": 42500.00,
      "last_update": "2024-01-01T12:00:00Z",
      "health_status": "healthy"
    },
    "ETHUSDT": {
      "price": 2580.75,
      "change_24h": -0.45,
      "volume": 850000000,
      "high_24h": 2620.00,
      "low_24h": 2550.00,
      "last_update": "2024-01-01T12:00:00Z",
      "health_status": "healthy"
    }
  }
}
```

---

## Logging

### GET /api/logs
**Description**: Get recent system logs

**Parameters**:
- `limit` (optional): Number of log entries (default: 50, max: 200)
- `level` (optional): Log level filter (DEBUG, INFO, WARNING, ERROR, ALL)

**Example**: `/api/logs?limit=10&level=ERROR`

**Response**:
```json
{
  "success": true,
  "logs": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "level": "INFO",
      "module": "trading.enhanced_engine",
      "message": "Model training completed successfully",
      "details": null
    }
  ]
}
```

---

## Configuration

### GET /api/config
**Description**: Get current system configuration (sensitive data masked)

**Response**:
```json
{
  "success": true,
  "config": {
    "trading": {
      "symbols": ["BTCUSDT", "ETHUSDT"],
      "timeframe": "4h",
      "confidence_threshold": 0.7
    },
    "ml": {
      "model_type": "catboost"
    },
    "adaptive_threshold": {
      "enabled": true,
      "target_signals_per_24h": 5
    },
    "coinex": {
      "api_key": "***",
      "secret_key": "***",
      "base_url": "https://api.coinex.com/v1/"
    }
  }
}
```

---

## Adaptive Thresholds

### GET /api/adaptive-threshold/status
**Description**: Get adaptive threshold status for all symbols

**Response**:
```json
{
  "success": true,
  "adaptive_threshold": {
    "enabled": true,
    "target_signals_per_24h": 5,
    "evaluation_window_hours": 24,
    "symbols": {
      "BTCUSDT": {
        "current_threshold": 0.675,
        "recent_signals": 7,
        "recent_predictions": 145,
        "signal_rate_24h": 7.0,
        "threshold_range": [0.5, 0.85]
      },
      "ETHUSDT": {
        "current_threshold": 0.720,
        "recent_signals": 3,
        "recent_predictions": 142,
        "signal_rate_24h": 3.0,
        "threshold_range": [0.5, 0.85]
      }
    }
  }
}
```

---

## WebSocket Events

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8765');
```

### Event Types

#### new_signal
```json
{
  "type": "new_signal",
  "data": {
    "id": "BTCUSDT_1704110400",
    "timestamp": "2024-01-01T12:00:00Z",
    "symbol": "BTCUSDT",
    "direction": "BUY",
    "confidence": 0.756,
    "price": 43250.50
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### prediction_update
```json
{
  "type": "prediction_update", 
  "data": {
    "symbol": "BTCUSDT",
    "prediction": "UP",
    "confidence": 0.723,
    "threshold": 0.675
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### status_update
```json
{
  "type": "status_update",
  "data": {
    "system_status": "healthy",
    "active_signals": 5,
    "model_accuracy": 0.742
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## Error Responses

### Standard Error Format
```json
{
  "success": false,
  "error": "Error description",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (endpoint or resource)
- `500`: Internal Server Error
- `503`: Service Unavailable (system maintenance)

---

## Rate Limiting
- General endpoints: 100 requests/minute
- Health endpoints: 300 requests/minute  
- Force prediction: 10 requests/minute

## WebSocket Limits
- Maximum 50 concurrent connections
- Message rate: 10 messages/second per connection