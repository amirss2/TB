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
    'timeframe': '1m',  # 4-hour timeframe as specified
    'demo_balance': 100.0,  # Starting demo balance in USD
    'confidence_threshold': 0.6,  # 60% confidence minimum as requested by user
    'training_symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],  # Symbols for model training (kept as requested)
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],  # Keep backward compatibility
    'analysis_symbols': [],  # Will be populated with CoinMarketCap top symbols available on CoinEx
    'coinmarketcap_limit': 1000,  # Number of top symbols to fetch from CoinMarketCap
    'max_positions': 4,  # Maximum concurrent positions
    'risk_per_trade': 0.5,  # 50% of portfolio per trade as requested by user
    'use_coinmarketcap_symbols': True,  # Enable CoinMarketCap-based symbol selection
    
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
    'update_interval': 60,  # Update every 60 seconds (1 minute) to collect data sequentially
    'batch_size': 20,  # Increased to 20 candles for better analysis
    'max_retries': 3,
    'timeout': 30,
    'min_1m_candles': 1440,  # Minimum aligned 4h candles required for training
    'max_1m_selection_candles': 1440,  # Maximum candles for feature selection subset
    'max_1m_training_candles': 0,  # Maximum candles for full training (0 or None means use all)
    'use_all_history': True,  # When True, fetch ALL historical data without limits
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
    'selection_window_1m': 1440,  # Use most recent 800 4h candles for feature selection
    'min_features': 20,  # Minimum features to retain
    'method': 'dynamic_iterative_pruning',  # Method to use when enabled
    'correlation_threshold': 0.95,  # Correlation threshold for pruning
    'tolerance': 0.003,  # Tolerance for improvement in feature selection
    'max_iterations': 50,  # Maximum iterations for dynamic selection
    'max_features': 50,  # Maximum features when selection is enabled
}

# Professional XGBoost Configuration
XGB_PRO_CONFIG = {
    'n_estimators': 8000,  # Large number of trees for professional model (can be increased for larger models)
    'max_depth': 12,  # Deep trees for complex pattern recognition
    'learning_rate': 0.01,  # Low learning rate for better generalization with many trees
    'early_stopping_rounds': 300,  # Higher patience for early stopping (set to 0 to disable)
    'subsample': 0.8,  # Subsampling for regularization
    'colsample_bytree': 0.8,  # Feature subsampling
    'colsample_bylevel': 0.8,  # Additional feature subsampling
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'min_child_weight': 3,  # Minimum weight in child nodes
    'gamma': 0.1,  # Minimum split loss
    'tree_method': 'hist',  # Efficient tree construction method
}

# Adaptive Labeling Configuration
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
