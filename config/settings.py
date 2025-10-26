# Trading Bot Configuration
import os

# Load environment variables from .env file manually if dotenv is not available
def load_env_file():
    """Load .env file manually if python-dotenv is not available"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback to manual loading if dotenv is not available
    load_env_file()

# Database Configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'TB'),
    'charset': 'utf8mb4'
}

# Trading Configuration
TRADING_CONFIG = {
    'timeframe': '4h',  # 4-hour timeframe as specified
    'demo_balance': 1000.0,  # Starting demo balance in USD
    'confidence_threshold': 0.7,  # Reduced from 0.9 to 0.7 to allow more signals
    'training_symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],  # Symbols for model training (kept as requested)
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],  # Keep backward compatibility
    'analysis_symbols': [],  # Will be populated with CoinMarketCap top symbols available on CoinEx
    'coinmarketcap_limit': 1000,  # Number of top symbols to fetch from CoinMarketCap
    'max_positions': 5,  # Maximum concurrent positions - STRICTLY ENFORCED
    'risk_per_trade': 100.0,  # Fixed $100 per trade as requested by user
    'min_order_value': 5.0,  # Minimum order value in USD to prevent dust orders
    'use_coinmarketcap_symbols': True,  # Enable CoinMarketCap-based symbol selection
    'health_check_interval_minutes': 15,  # Health check every X minutes
    
    # Performance Enhancement Configuration
    'use_websocket': True,
    'ws_channels': ['ticker', 'kline_1m'],
    'ws_max_subscriptions_tier1': 250,
    'rest_concurrency': 80,
    'rest_timeout_sec': 10,
    'fetch_batch_size': 200,
    'scan_tier1_size': 200,
    'scan_tier1_interval_sec': 60,
    'scan_tier2_interval_sec': 240,
    'process_pool_workers': max(2, os.cpu_count() - 1),
    'ring_buffer_size': 1000,
    'analysis_on_candle_close_only': True,
    'backoff_base_sec': 1,
    'backoff_max_sec': 60,
}

# Take Profit / Stop Loss Configuration
TP_SL_CONFIG = {
    'tp1_percent': 3.0,  # First take profit at +3% from entry
    'tp2_percent': 6.0,  # Second take profit at +6% from entry (when TP1 hit)
    'tp3_percent': 10.0,  # Third take profit at +10% from entry (when TP2 hit)
    'initial_sl_percent': 3.0,  # Initial stop loss at -3% from entry price
    'trailing_enabled': True,
}

# CoinEx API Configuration
COINEX_CONFIG = {
    'api_key': os.getenv('COINEX_API_KEY', ''),
    'secret_key': os.getenv('COINEX_SECRET_KEY', ''),
    'sandbox_mode': os.getenv('COINEX_SANDBOX', 'false').lower() == 'true',  # Default to spot trading API
    'base_url': 'https://api.coinex.com/v1/',  # Spot trading API
    'sandbox_url': 'https://api.coinex.com/v1/',  # Use same spot API for better compatibility
}

# Trading Fees Configuration (for accurate profit calculation)
FEE_CONFIG = {
    'spot_trading': {
        'maker_fee': 0.0016,  # 0.16% maker fee (CoinEx default)
        'taker_fee': 0.0026,  # 0.26% taker fee (CoinEx default)
    },
    'spread': {
        'estimate_pct': 0.001,  # Estimated 0.1% spread between bid/ask
    },
    'slippage': {
        'estimate_pct': 0.0005,  # Estimated 0.05% slippage on market orders
    }
}

# CoinMarketCap API Configuration
COINMARKETCAP_CONFIG = {
    'api_key': 'b63aec19-7b5c-4da3-8fdb-b10c441bd4c4',
    'base_url': 'https://pro-api.coinmarketcap.com/v1/',
    'listings_endpoint': 'cryptocurrency/listings/latest',
    'limit': 1000,  # Get top 1000 cryptocurrencies
}

# Machine Learning Configuration
ML_CONFIG = {
    'training_data_size': 58000,  # Use 58k historical records as specified
    'rfe_sample_size': 1000,  # Use last 1000 samples for RFE selection
    'selected_features': 50,  # Select 50 best indicators via RFE
    'test_size': 0.2,  # 20% for testing
    'random_state': 42,
    'model_retrain_interval': 24,  # Retrain every 24 hours
}

# Web Dashboard Configuration
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': os.getenv('DEBUG', 'false').lower() == 'true',
    'secret_key': os.getenv('SECRET_KEY', 'your-secret-key-change-this'),
}

# Data Update Configuration
DATA_CONFIG = {
    'update_interval': 30,  # Update every 30 seconds to avoid API rate limits
    'batch_size': 3,  # Fetch last 3 candles for real-time updates
    'max_retries': 3,
    'timeout': 30,
    'min_4h_candles': 800,  # Minimum aligned 4h candles required for training
    'max_4h_selection_candles': 800,  # Maximum candles for feature selection subset
    'max_4h_training_candles': 0,  # Maximum candles for full training (0 or None means use all)
    'use_all_history': True,  # When True, fetch ALL historical data without limits
    'real_time_fetch_limit': 3,  # Number of latest candles to fetch for real-time updates
    'real_time_min_interval': 10,  # Minimum 10 seconds between updates per symbol to avoid rate limits
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/trading_bot.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Feature Selection Configuration
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

# Professional XGBoost Configuration
XGB_PRO_CONFIG = {
    'n_estimators': 2000,  # Reduced from 8000 to prevent overfitting
    'max_depth': 6,  # Reduced from 12 for simpler trees
    'learning_rate': 0.05,  # Increased from 0.01 for faster learning with fewer trees
    'early_stopping_rounds': 100,  # Reduced from 300 for quicker stopping
    'subsample': 0.7,  # Reduced from 0.8 for more regularization
    'colsample_bytree': 0.7,  # Reduced from 0.8 for more regularization
    'colsample_bylevel': 0.7,  # Reduced from 0.8 for more regularization
    'reg_alpha': 1.0,  # Increased from 0.1 for stronger L1 regularization
    'reg_lambda': 5.0,  # Increased from 1.0 for stronger L2 regularization
    'min_child_weight': 10,  # Increased from 3 to prevent overfitting
    'gamma': 0.5,  # Increased from 0.1 for higher split threshold
    'tree_method': 'hist',  # Efficient tree construction method
    'scale_pos_weight': 1.0,  # Will be adjusted dynamically for class imbalance
}

# Triple-Barrier Labeling Configuration
LABELING_CONFIG = {
    'method': 'triple_barrier',  # 'triple_barrier' or 'simple_threshold'
    
    # Triple-Barrier Method Settings
    'triple_barrier': {
        'enabled': True,
        'profit_target_atr_multiplier': 0.5,  # TP = current_price + (0.5 × ATR)
        'stop_loss_atr_multiplier': 0.5,      # SL = current_price - (0.5 × ATR)
        'time_horizon_candles': 2,            # Look ahead 1-2 candles (4-8 hours for 4h timeframe)
        'max_hold_candles': 4,                # Maximum holding period (16 hours for 4h)
        'use_adaptive_atr': True,             # Use adaptive ATR calculation
        'min_profit_target_pct': 0.3,         # Minimum profit target of 0.3%
        'min_stop_loss_pct': 0.3,             # Minimum stop loss of 0.3%
    },
    
    # Fallback: Simple Threshold Method (when triple-barrier is disabled)
    'simple_threshold': {
        'target_distribution': {'SELL': 0.30, 'BUY': 0.40, 'HOLD': 0.30},  # Favor BUY signals
        'initial_up_pct': 1.5,  # Reduced from 2.0 for more BUY signals
        'initial_down_pct': -2.0,  # Keep stricter for SELL
        'search_up_range': [0.4, 3.0, 0.1],
        'search_down_range': [-3.0, -0.4, 0.1],
        'optimization_metric': 'kl_divergence',
        'max_search_iterations': 100,
        'convergence_tolerance': 0.01,
    }
}
