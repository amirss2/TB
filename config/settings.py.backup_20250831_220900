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
    'demo_balance': 100.0,  # Starting demo balance in USD
    'confidence_threshold': 0.7,  # 70% confidence minimum
    'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
    'max_positions': 4,  # Maximum concurrent positions
    'risk_per_trade': 0.02,  # 2% risk per trade
}

# Take Profit / Stop Loss Configuration
TP_SL_CONFIG = {
    'tp1_percent': 3.0,  # First take profit at 3%
    'tp2_percent': 5.0,  # Second take profit at 5%
    'tp3_percent': 8.0,  # Third take profit at 8%
    'initial_sl_percent': 2.0,  # Initial stop loss at 2%
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
    'update_interval': 300,  # Update every 300 seconds for 4h timeframe
    'batch_size': 100,
    'max_retries': 3,
    'timeout': 30,
    'min_4h_candles': 800,  # Minimum aligned 4h candles required for training
    'max_4h_selection_candles': 800,  # Maximum candles for feature selection subset
    'max_4h_training_candles': 0,  # Maximum candles for full training (0 or None means use all)
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
