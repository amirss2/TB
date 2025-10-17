"""
Symbol Cache Management
Handles caching of CoinMarketCap symbols to persist through internet disconnections
"""
import json
import os
import logging
import time
from typing import List, Optional
from datetime import datetime, timedelta


class SymbolCache:
    """Manages caching of symbols to persist through network issues"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "coinmarketcap_symbols.json")
        self.cache_duration = 24 * 3600  # 24 hours in seconds
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
    def save_symbols(self, symbols: List[str], metadata: dict = None) -> bool:
        """Save symbols to cache with timestamp"""
        try:
            cache_data = {
                'symbols': symbols,
                'timestamp': time.time(),
                'count': len(symbols),
                'metadata': metadata or {}
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info(f"Cached {len(symbols)} symbols to {self.cache_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save symbols cache: {e}")
            return False
    
    def load_symbols(self, max_age_hours: int = 24) -> Optional[List[str]]:
        """Load symbols from cache if not expired"""
        try:
            if not os.path.exists(self.cache_file):
                self.logger.info("No symbols cache found")
                return None
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_timestamp = cache_data.get('timestamp', 0)
            current_time = time.time()
            age_hours = (current_time - cache_timestamp) / 3600
            
            if age_hours > max_age_hours:
                self.logger.warning(f"Symbols cache expired ({age_hours:.1f}h old), max age is {max_age_hours}h")
                return None
            
            symbols = cache_data.get('symbols', [])
            count = cache_data.get('count', 0)
            
            self.logger.info(f"Loaded {count} symbols from cache (age: {age_hours:.1f}h)")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to load symbols cache: {e}")
            return None
    
    def is_cache_valid(self, max_age_hours: int = 24) -> bool:
        """Check if cache exists and is valid"""
        try:
            if not os.path.exists(self.cache_file):
                return False
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_timestamp = cache_data.get('timestamp', 0)
            age_hours = (time.time() - cache_timestamp) / 3600
            
            return age_hours <= max_age_hours
            
        except Exception:
            return False
    
    def clear_cache(self) -> bool:
        """Clear the symbols cache"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                self.logger.info("Symbols cache cleared")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache"""
        try:
            if not os.path.exists(self.cache_file):
                return {'exists': False}
            
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_timestamp = cache_data.get('timestamp', 0)
            age_hours = (time.time() - cache_timestamp) / 3600
            
            return {
                'exists': True,
                'symbol_count': cache_data.get('count', 0),
                'age_hours': age_hours,
                'timestamp': cache_timestamp,
                'cache_file': self.cache_file,
                'metadata': cache_data.get('metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return {'exists': False, 'error': str(e)}