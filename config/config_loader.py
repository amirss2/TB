"""
Configuration loader with YAML support and environment variable overrides
"""

import os
import yaml
import json
from typing import Dict, Any, Optional

class ConfigLoader:
    """Loads and manages configuration from YAML files with environment overrides"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config', 
                'config.yaml'
            )
        
        self.config_path = config_path
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self._config is not None:
            return self._config
            
        try:
            # Load YAML config
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                # Fallback to default config
                self._config = self._get_default_config()
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            return self._config
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # Database
            'DB_HOST': ['database', 'host'],
            'DB_PORT': ['database', 'port'],
            'DB_USER': ['database', 'user'],
            'DB_PASSWORD': ['database', 'password'],
            'DB_NAME': ['database', 'database'],
            
            # Trading
            'COINEX_API_KEY': ['coinex', 'api_key'],
            'COINEX_SECRET_KEY': ['coinex', 'secret_key'],
            'COINEX_SANDBOX': ['coinex', 'sandbox_mode'],
            
            # ML
            'MODEL_TYPE': ['ml', 'model_type'],
            'CONFIDENCE_THRESHOLD': ['trading', 'confidence_threshold'],
            
            # Web
            'WEB_HOST': ['web', 'host'],
            'WEB_PORT': ['web', 'port'],
            'DEBUG': ['web', 'debug'],
            'SECRET_KEY': ['web', 'secret_key'],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(self._config, config_path, value)
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: str):
        """Set nested dictionary value from path"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value types
        final_key = path[-1]
        if value.lower() in ('true', 'false'):
            current[final_key] = value.lower() == 'true'
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration as fallback"""
        return {
            'trading': {
                'timeframe': '4h',
                'demo_balance': 100.0,
                'confidence_threshold': 0.9,  # Fixed: Changed to 0.9 (90% as per latest user requirements)
                'symbols': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT'],
                'max_positions': 4,  # STRICTLY ENFORCED
                'risk_per_trade': 0.5,  # Fixed: Changed from 0.02 to 0.5 (50% as per user requirements)
                'min_order_value': 5.0  # Minimum order value in USD to prevent dust orders
            },
            'adaptive_threshold': {
                'enabled': True,
                'target_signals_per_24h': 5,
                'min_threshold': 0.5,
                'max_threshold': 0.85,
                'adjustment_rate': 0.05,
                'evaluation_window': 24
            },
            'data_sources': {
                'primary': 'coinex',
                'secondary': 'binance',
                'failover_after_failures': 3,
                'timeout_seconds': 30,
                'retry_count': 3,
                'retry_backoff': 2.0
            },
            'ml': {
                'model_type': 'xgboost',
                'training_data_size': 58000,
                'test_size': 0.2,
                'random_state': 42,
                'model_retrain_interval': 24,
                'labeling': {
                    'up_threshold': 0.02,
                    'down_threshold': -0.02
                },
                'class_balance': {
                    'method': 'class_weight',
                    'max_class_ratio': 0.7
                },
                'cross_validation': {
                    'enabled': True,
                    'folds': 5,
                    'strategy': 'stratified'
                },
                'calibration': {
                    'enabled': True,
                    'method': 'isotonic'
                }
            },
            'xgboost': {
                'n_estimators': 1500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'early_stopping_rounds': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_weight': 3,
                'gamma': 0.1
            },
            'catboost': {
                'iterations': 1500,
                'depth': 6,
                'learning_rate': 0.05,
                'early_stopping_rounds': 100,
                'loss_function': 'MultiClass',
                'auto_class_weights': 'Balanced'
            },
            'feature_selection': {
                'mode': 'dynamic',
                'method': 'shap',
                'target_features': 50,
                'correlation_threshold': 0.9,
                'importance_threshold': 0.001,
                'dynamic': {
                    'min_features': 20,
                    'drop_fraction': 0.05,
                    'corr_threshold': 0.95,
                    'tolerance': 0.003,
                    'metric': 'macro_f1',
                    'cv_splits': 3,
                    'max_iterations': 50
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'logs/trading_bot.log',
                'max_size_mb': 10,
                'backup_count': 5,
                'unicode_safe': True,
                'emoji_strip': True,
                'structured_json': False
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'websocket_enabled': True
            },
            'prediction': {
                'scheduler_interval': 120,
                'intracandle_predictions': True,
                'min_data_points': 200
            },
            'signals': {
                'persistence_enabled': True,
                'websocket_broadcast': True,
                'max_signals_history': 1000,
                'signal_expiry_hours': 24
            },
            'health': {
                'check_interval': 300,
                'endpoints': ['/api/health', '/api/model/status', '/api/signals/recent']
            },
            'database': {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '',
                'database': 'TB',
                'charset': 'utf8mb4'
            },
            'coinex': {
                'api_key': '',
                'secret_key': '',
                'sandbox_mode': False,
                'base_url': 'https://api.coinex.com/v1/',
                'sandbox_url': 'https://api.coinex.com/v1/'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot notation path"""
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save_config(self, config_path: str = None):
        """Save current configuration to YAML file"""
        if config_path is None:
            config_path = self.config_path
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)

# Global config instance
_config_loader = None

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration using global loader"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader.load_config()

def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get configuration value by dot notation"""
    global _config_loader
    if _config_loader is None:
        load_config()
    return _config_loader.get(key_path, default)