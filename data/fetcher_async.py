import asyncio
import aiohttp
import time
import logging
from typing import Dict, List, Optional, Any
from config.settings import TRADING_CONFIG, COINEX_CONFIG

class AsyncRESTFetcher:
    """
    Async REST client for CoinEx API with concurrency control and batching
    Used as fallback when WebSocket is unavailable or for Tier-2 scanning
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = COINEX_CONFIG['base_url']
        
        # Configuration
        self.concurrency = TRADING_CONFIG['rest_concurrency']
        self.timeout_sec = TRADING_CONFIG['rest_timeout_sec']
        self.batch_size = TRADING_CONFIG['fetch_batch_size']
        self.backoff_base = TRADING_CONFIG['backoff_base_sec']
        self.backoff_max = TRADING_CONFIG['backoff_max_sec']
        
        # Session management
        self.session = None
        self.semaphore = None
        
        # Rate limiting and backoff
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_delay = 0.05  # 50ms between requests
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
    
    async def start(self):
        """Initialize the async HTTP session and semaphore"""
        if self.session is None:
            # Configure connection with optimizations
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            connector = aiohttp.TCPConnector(
                limit=self.concurrency,
                ttl_dns_cache=300,
                use_dns_cache=True,
                limit_per_host=self.concurrency // 2,
                enable_cleanup_closed=True
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'TradingBot-AsyncFetcher/1.0',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                }
            )
            
            self.semaphore = asyncio.Semaphore(self.concurrency)
            self.logger.info(f"AsyncRESTFetcher started with concurrency: {self.concurrency}")
    
    async def stop(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.semaphore = None
            self.logger.info("AsyncRESTFetcher stopped")
    
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ticker for a single symbol"""
        endpoint = f"market/ticker"
        params = {'market': symbol}
        
        try:
            response = await self._make_request(endpoint, params)
            if response and 'ticker' in response:
                return response['ticker']
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
        
        return None
    
    async def fetch_tickers_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch tickers for multiple symbols concurrently"""
        if not symbols:
            return {}
        
        # Split into batches to avoid overwhelming the API
        batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]
        results = {}
        
        for batch in batches:
            batch_tasks = []
            for symbol in batch:
                task = asyncio.create_task(self.fetch_ticker(symbol))
                batch_tasks.append((symbol, task))
            
            # Wait for batch completion
            for symbol, task in batch_tasks:
                try:
                    ticker_data = await task
                    if ticker_data:
                        results[symbol] = ticker_data
                except Exception as e:
                    self.logger.error(f"Error in batch fetch for {symbol}: {e}")
            
            # Small delay between batches to be respectful to API
            if len(batches) > 1:
                await asyncio.sleep(0.1)
        
        self.logger.debug(f"Fetched tickers for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def fetch_klines(self, symbol: str, interval: str = "1m", limit: int = 200) -> Optional[List[Dict[str, Any]]]:
        """Fetch kline/candlestick data for a symbol"""
        endpoint = "market/kline"
        params = {
            'market': symbol,
            'type': self._convert_interval(interval),
            'limit': limit
        }
        
        try:
            response = await self._make_request(endpoint, params)
            if response and 'data' in response:
                # Convert to standardized format
                return self._convert_klines(response['data'])
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {e}")
        
        return None
    
    async def fetch_klines_batch(self, symbols: List[str], interval: str = "1m", limit: int = 200) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch klines for multiple symbols concurrently"""
        if not symbols:
            return {}
        
        # Create tasks for concurrent fetching
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(self.fetch_klines(symbol, interval, limit))
            tasks.append((symbol, task))
        
        results = {}
        
        # Process results as they complete
        for symbol, task in tasks:
            try:
                klines = await task
                if klines:
                    results[symbol] = klines
            except Exception as e:
                self.logger.error(f"Error in batch klines fetch for {symbol}: {e}")
        
        self.logger.debug(f"Fetched klines for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def fetch_all_tickers(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all tickers at once (more efficient than individual calls)"""
        endpoint = "market/ticker/all"
        
        try:
            response = await self._make_request(endpoint)
            if response and 'ticker' in response:
                return response['ticker']
        except Exception as e:
            self.logger.error(f"Error fetching all tickers: {e}")
        
        return {}
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make HTTP request with rate limiting and error handling"""
        if not self.session:
            await self.start()
        
        url = f"{self.base_url}{endpoint}"
        
        async with self.semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - time_since_last)
            
            self.last_request_time = time.time()
            self.request_count += 1
            
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == 0:
                            return data
                        else:
                            self.logger.warning(f"API error: {data.get('message', 'Unknown error')}")
                    
                    elif response.status == 429:
                        # Rate limited - apply exponential backoff
                        retry_after = int(response.headers.get('Retry-After', 1))
                        delay = min(retry_after, self.backoff_max)
                        self.logger.warning(f"Rate limited, waiting {delay}s")
                        await asyncio.sleep(delay)
                        
                        # Retry once
                        return await self._make_request(endpoint, params)
                    
                    else:
                        self.logger.error(f"HTTP error {response.status} for {endpoint}")
            
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout for {endpoint}")
            except Exception as e:
                self.logger.error(f"Request error for {endpoint}: {e}")
        
        return None
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to CoinEx format"""
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '4h': '4hour',
            '1d': '1day'
        }
        return interval_map.get(interval, '1min')
    
    def _convert_klines(self, raw_klines: List[List[Any]]) -> List[Dict[str, Any]]:
        """Convert raw kline data to standardized format"""
        converted = []
        
        for kline in raw_klines:
            if len(kline) >= 6:
                converted.append({
                    'timestamp': kline[0],
                    'open': float(kline[1]),
                    'close': float(kline[2]),
                    'high': float(kline[3]),
                    'low': float(kline[4]),
                    'volume': float(kline[5]),
                })
        
        return converted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        return {
            'request_count': self.request_count,
            'concurrency_limit': self.concurrency,
            'is_active': self.session is not None,
            'last_request_time': self.last_request_time
        }


# Global instance for reuse
_fetcher_instance = None

async def get_async_fetcher() -> AsyncRESTFetcher:
    """Get or create global async fetcher instance"""
    global _fetcher_instance
    
    if _fetcher_instance is None:
        _fetcher_instance = AsyncRESTFetcher()
        await _fetcher_instance.start()
    
    return _fetcher_instance

async def cleanup_async_fetcher():
    """Clean up global async fetcher instance"""
    global _fetcher_instance
    
    if _fetcher_instance:
        await _fetcher_instance.stop()
        _fetcher_instance = None