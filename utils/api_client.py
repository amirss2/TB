"""
Robust API Client with failover support for multiple exchanges
"""

import time
import hashlib
import hmac
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from enum import Enum

class DataSourceStatus(Enum):
    """Data source health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class APIError(Exception):
    """Base API error"""
    pass

class RateLimitError(APIError):
    """Rate limit exceeded"""
    pass

class AuthenticationError(APIError):
    """Authentication failed"""
    pass

class NetworkError(APIError):
    """Network/connection error"""
    pass

class RobustAPIClient:
    """Robust API client with retry logic and failover support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Retry configuration
        self.max_retries = config.get('retry_count', 3)
        self.backoff_multiplier = config.get('retry_backoff', 2.0)
        self.timeout = config.get('timeout_seconds', 30)
        
        # Health tracking
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        self.status = DataSourceStatus.HEALTHY
        self.failure_count = 0
        
        # Request statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
    
    def _make_request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                self.stats['total_requests'] += 1
                
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # Update response time stats
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                
                # Check for rate limiting
                if response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                
                # Check for authentication errors
                if response.status_code == 401:
                    raise AuthenticationError("Authentication failed")
                
                # Check for other client/server errors
                response.raise_for_status()
                
                # Success - update health status
                self._record_success()
                return response
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = NetworkError(f"Network error: {str(e)}")
                self.logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    last_exception = RateLimitError("Rate limit exceeded")
                elif e.response.status_code == 401:
                    last_exception = AuthenticationError("Authentication failed")
                else:
                    last_exception = APIError(f"HTTP error: {e}")
                self.logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_exception = APIError(f"Unexpected error: {str(e)}")
                self.logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                wait_time = self.backoff_multiplier ** attempt
                self.logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # All retries failed
        self._record_failure()
        raise last_exception
    
    def _record_success(self):
        """Record successful request"""
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        self.status = DataSourceStatus.HEALTHY
        self.stats['successful_requests'] += 1
        
        self.logger.debug("Request successful - health status: HEALTHY")
    
    def _record_failure(self):
        """Record failed request"""
        self.consecutive_failures += 1
        self.failure_count += 1
        self.stats['failed_requests'] += 1
        
        # Update health status based on consecutive failures
        if self.consecutive_failures >= 3:
            self.status = DataSourceStatus.FAILED
        elif self.consecutive_failures >= 1:
            self.status = DataSourceStatus.DEGRADED
        
        self.logger.warning(f"Request failed - consecutive failures: {self.consecutive_failures}, status: {self.status}")
    
    def _update_response_time(self, response_time: float):
        """Update average response time statistics"""
        current_avg = self.stats['avg_response_time']
        total_requests = self.stats['total_requests']
        
        # Calculate new average
        self.stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status and statistics"""
        time_since_success = (datetime.now() - self.last_success).total_seconds()
        
        return {
            'status': self.status.value,
            'consecutive_failures': self.consecutive_failures,
            'total_failures': self.failure_count,
            'time_since_last_success': time_since_success,
            'last_success': self.last_success.isoformat(),
            'statistics': self.stats.copy()
        }
    
    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        return self.status == DataSourceStatus.HEALTHY

class CoinExAPIClient(RobustAPIClient):
    """CoinEx API client with robust error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config.get('data_sources', {}))
        
        # CoinEx specific configuration
        self.api_key = config.get('coinex', {}).get('api_key', '')
        self.secret_key = config.get('coinex', {}).get('secret_key', '')
        self.base_url = config.get('coinex', {}).get('base_url', 'https://api.coinex.com/v1/')
        
        # Ensure base_url ends with /
        if not self.base_url.endswith('/'):
            self.base_url += '/'
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate signature for CoinEx API authentication"""
        if not self.secret_key:
            return ""
            
        # Sort parameters by key
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Create signature
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest().upper()
        
        return signature
    
    def _make_coinex_request(self, endpoint: str, params: Dict[str, Any] = None, 
                           auth_required: bool = False) -> Dict[str, Any]:
        """Make authenticated CoinEx API request"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Add authentication if required
        headers = {}
        if auth_required and self.api_key and self.secret_key:
            params['access_id'] = self.api_key
            params['tonce'] = int(time.time() * 1000)
            
            signature = self._generate_signature(params)
            headers['authorization'] = signature
        
        try:
            response = self._make_request_with_retry('GET', url, params=params, headers=headers)
            result = response.json()
            
            # Check CoinEx specific error format
            if result.get('code') != 0:
                error_msg = result.get('message', 'Unknown CoinEx API error')
                self.logger.error(f"CoinEx API error: {error_msg}")
                raise APIError(f"CoinEx API error: {error_msg}")
            
            return result.get('data', {})
            
        except APIError:
            raise
        except Exception as e:
            self.logger.error(f"CoinEx request failed: {e}")
            raise APIError(f"CoinEx request failed: {str(e)}")
    
    def get_market_data(self, symbol: str, timeframe: str = '4h', limit: int = 1000) -> List[Dict[str, Any]]:
        """Get market data (OHLCV) for symbol"""
        # Convert timeframe to CoinEx format
        timeframe_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1hour', '4h': '4hour', '1d': '1day'
        }
        
        coinex_timeframe = timeframe_map.get(timeframe, '4hour')
        
        try:
            data = self._make_coinex_request(
                f'market/kline',
                params={
                    'market': symbol,
                    'type': coinex_timeframe,
                    'limit': limit
                }
            )
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information for symbol"""
        try:
            data = self._make_coinex_request(f'market/ticker', params={'market': symbol})
            return data.get('ticker', {})
        except Exception as e:
            self.logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}
    
    def get_all_tickers(self) -> Dict[str, Any]:
        """Get all ticker information"""
        try:
            data = self._make_coinex_request('market/ticker/all')
            return data.get('ticker', {})
        except Exception as e:
            self.logger.error(f"Failed to get all tickers: {e}")
            return {}

class BinanceAPIClient(RobustAPIClient):
    """Binance API client as secondary data source"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config.get('data_sources', {}))
        self.base_url = 'https://api.binance.com/api/v3/'
    
    def _make_binance_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make Binance API request"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self._make_request_with_retry('GET', url, params=params)
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Binance request failed: {e}")
            raise APIError(f"Binance request failed: {str(e)}")
    
    def get_market_data(self, symbol: str, timeframe: str = '4h', limit: int = 1000) -> List[List]:
        """Get market data from Binance"""
        try:
            data = self._make_binance_request(
                'klines',
                params={
                    'symbol': symbol,
                    'interval': timeframe,
                    'limit': limit
                }
            )
            
            return data if isinstance(data, list) else []
            
        except Exception as e:
            self.logger.error(f"Failed to get Binance market data for {symbol}: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker from Binance"""
        try:
            data = self._make_binance_request('ticker/24hr', params={'symbol': symbol})
            return data
        except Exception as e:
            self.logger.error(f"Failed to get Binance ticker for {symbol}: {e}")
            return {}

class MultiSourceAPIClient:
    """API client with automatic failover between data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.primary_client = CoinExAPIClient(config)
        self.secondary_client = BinanceAPIClient(config)
        
        # Failover configuration
        self.failover_threshold = config.get('data_sources', {}).get('failover_after_failures', 3)
        self.current_client = self.primary_client
        self.using_primary = True
        
    def _should_failover(self) -> bool:
        """Check if we should failover to secondary client"""
        if self.using_primary:
            return self.primary_client.consecutive_failures >= self.failover_threshold
        else:
            # Check if primary has recovered
            return self.primary_client.consecutive_failures == 0
    
    def _perform_failover(self):
        """Perform failover between clients"""
        if self.using_primary:
            self.logger.warning("Failing over from CoinEx to Binance")
            self.current_client = self.secondary_client
            self.using_primary = False
        else:
            self.logger.info("Failing back to CoinEx (primary)")
            self.current_client = self.primary_client
            self.using_primary = True
    
    def get_market_data(self, symbol: str, timeframe: str = '4h', limit: int = 1000) -> List[Any]:
        """Get market data with automatic failover"""
        # Check if we should failover
        if self._should_failover():
            self._perform_failover()
        
        try:
            data = self.current_client.get_market_data(symbol, timeframe, limit)
            
            if not data:
                # Try other client if current returns empty
                other_client = self.secondary_client if self.using_primary else self.primary_client
                self.logger.warning(f"No data from current source, trying alternative for {symbol}")
                data = other_client.get_market_data(symbol, timeframe, limit)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Market data request failed: {e}")
            # Try other client as fallback
            try:
                other_client = self.secondary_client if self.using_primary else self.primary_client
                self.logger.info("Trying alternative data source...")
                return other_client.get_market_data(symbol, timeframe, limit)
            except Exception as e2:
                self.logger.error(f"Both data sources failed: {e2}")
                return []
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker with automatic failover"""
        if self._should_failover():
            self._perform_failover()
        
        try:
            return self.current_client.get_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Ticker request failed: {e}")
            try:
                other_client = self.secondary_client if self.using_primary else self.primary_client
                return other_client.get_ticker(symbol)
            except Exception as e2:
                self.logger.error(f"Both ticker sources failed: {e2}")
                return {}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all data sources"""
        return {
            'primary_source': {
                'name': 'coinex',
                'active': self.using_primary,
                **self.primary_client.get_health_status()
            },
            'secondary_source': {
                'name': 'binance',
                'active': not self.using_primary,
                **self.secondary_client.get_health_status()
            },
            'failover_threshold': self.failover_threshold
        }