import asyncio
import logging
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import multiprocessing
from config.settings import TRADING_CONFIG

class IndicatorCompute:
    """
    CPU-parallel indicator computation with ProcessPoolExecutor
    Handles incremental updates and ring buffer management for efficiency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_workers = TRADING_CONFIG.get('process_pool_workers', max(2, multiprocessing.cpu_count() - 1))
        self.ring_buffer_size = TRADING_CONFIG.get('ring_buffer_size', 1000)
        self.computation_timeout = 30  # seconds
        
        # ProcessPoolExecutor
        self.executor = None
        self.is_running = False
        
        # Ring buffers for OHLCV data per symbol
        self.symbol_buffers = {}  # symbol -> deque of OHLCV data
        self.last_computed = {}   # symbol -> timestamp of last computation
        self.computation_cache = {}  # symbol -> cached indicator results
        
        # Performance tracking
        self.stats = {
            'total_computations': 0,
            'successful_computations': 0,
            'failed_computations': 0,
            'timeout_computations': 0,
            'avg_computation_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    async def start(self):
        """Start the indicator computation system"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info(f"Starting indicator compute with {self.max_workers} workers...")
        
        try:
            # Initialize ProcessPoolExecutor
            self.executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=multiprocessing.get_context('spawn')  # More stable for complex computations
            )
            
            self.logger.info("Indicator compute system started")
            
        except Exception as e:
            self.logger.error(f"Error starting indicator compute: {e}")
            raise
    
    async def stop(self):
        """Stop the indicator computation system"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info("Stopping indicator compute...")
        
        if self.executor:
            self.executor.shutdown(wait=True, timeout=10)
            self.executor = None
        
        # Clear buffers
        self.symbol_buffers.clear()
        self.last_computed.clear()
        self.computation_cache.clear()
        
        self.logger.info("Indicator compute stopped")
    
    async def update_ohlcv(self, symbol: str, ohlcv_data: List[Dict[str, Any]]):
        """
        Update OHLCV ring buffer for a symbol
        
        Args:
            symbol: Trading symbol
            ohlcv_data: List of OHLCV candles (newest first or last)
        """
        try:
            if symbol not in self.symbol_buffers:
                self.symbol_buffers[symbol] = deque(maxlen=self.ring_buffer_size)
            
            buffer = self.symbol_buffers[symbol]
            
            # Add new candles to buffer
            for candle in ohlcv_data:
                # Convert to standardized format
                standardized_candle = {
                    'timestamp': candle.get('timestamp', time.time()),
                    'open': float(candle.get('open', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': float(candle.get('volume', 0))
                }
                
                # Only add if we have valid data
                if all(standardized_candle[key] > 0 for key in ['open', 'high', 'low', 'close']):
                    buffer.append(standardized_candle)
            
            self.logger.debug(f"Updated OHLCV buffer for {symbol}: {len(buffer)} candles")
            
        except Exception as e:
            self.logger.error(f"Error updating OHLCV for {symbol}: {e}")
    
    async def compute_indicators(self, symbol: str, force_full_compute: bool = False) -> Optional[Dict[str, Any]]:
        """
        Compute technical indicators for a symbol
        Uses incremental computation when possible
        
        Args:
            symbol: Trading symbol
            force_full_compute: Force full computation instead of incremental
            
        Returns:
            Dictionary with computed indicators or None if failed
        """
        if not self.is_running or not self.executor:
            return None
        
        try:
            start_time = time.time()
            
            # Get OHLCV data from buffer
            if symbol not in self.symbol_buffers or len(self.symbol_buffers[symbol]) < 20:
                self.logger.debug(f"Insufficient data for {symbol}: {len(self.symbol_buffers.get(symbol, []))} candles")
                return None
            
            buffer = self.symbol_buffers[symbol]
            ohlcv_data = list(buffer)  # Convert deque to list
            
            # Check if we can use cached results (incremental update)
            use_incremental = (
                not force_full_compute and 
                symbol in self.computation_cache and
                len(ohlcv_data) > 0
            )
            
            if use_incremental:
                # Get only the latest candles for incremental computation
                last_computed_time = self.last_computed.get(symbol, 0)
                new_candles = [c for c in ohlcv_data if c['timestamp'] > last_computed_time]
                
                if len(new_candles) == 0:
                    # No new data - return cached results
                    self.stats['cache_hits'] += 1
                    return self.computation_cache[symbol]
                
                # Use recent data for incremental computation (last 200 candles)
                compute_data = ohlcv_data[-200:] if len(ohlcv_data) > 200 else ohlcv_data
            else:
                # Full computation
                compute_data = ohlcv_data
                self.stats['cache_misses'] += 1
            
            # Submit computation to process pool
            loop = asyncio.get_event_loop()
            future = self.executor.submit(
                compute_technical_indicators_worker,
                symbol,
                compute_data,
                use_incremental
            )
            
            # Wait for result with timeout
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, future.result, self.computation_timeout),
                    timeout=self.computation_timeout + 5
                )
                
                if result:
                    # Cache the result
                    self.computation_cache[symbol] = result
                    self.last_computed[symbol] = ohlcv_data[-1]['timestamp'] if ohlcv_data else time.time()
                    
                    # Update stats
                    computation_time = (time.time() - start_time) * 1000  # ms
                    self._update_computation_stats(computation_time, success=True)
                    
                    self.logger.debug(f"Computed indicators for {symbol}: {computation_time:.2f}ms")
                    return result
                else:
                    self.stats['failed_computations'] += 1
                    return None
                    
            except (asyncio.TimeoutError, FutureTimeoutError):
                self.logger.warning(f"Indicator computation timeout for {symbol}")
                self.stats['timeout_computations'] += 1
                future.cancel()
                return None
            
        except Exception as e:
            self.logger.error(f"Error computing indicators for {symbol}: {e}")
            self.stats['failed_computations'] += 1
            return None
    
    async def compute_indicators_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compute indicators for multiple symbols concurrently
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary mapping symbol to computed indicators
        """
        if not self.is_running or not symbols:
            return {}
        
        try:
            # Create computation tasks
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self.compute_indicators(symbol))
                tasks.append((symbol, task))
            
            # Wait for all computations to complete
            results = {}
            for symbol, task in tasks:
                try:
                    result = await task
                    if result:
                        results[symbol] = result
                except Exception as e:
                    self.logger.error(f"Error in batch computation for {symbol}: {e}")
            
            self.logger.debug(f"Batch computation completed: {len(results)}/{len(symbols)} successful")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch computation: {e}")
            return {}
    
    def get_cached_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached indicator results for a symbol"""
        return self.computation_cache.get(symbol)
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear computation cache for a symbol or all symbols"""
        if symbol:
            self.computation_cache.pop(symbol, None)
            self.last_computed.pop(symbol, None)
        else:
            self.computation_cache.clear()
            self.last_computed.clear()
    
    def _update_computation_stats(self, computation_time_ms: float, success: bool):
        """Update computation statistics"""
        self.stats['total_computations'] += 1
        
        if success:
            self.stats['successful_computations'] += 1
            
            # Update rolling average computation time
            current_avg = self.stats['avg_computation_time_ms']
            total_successful = self.stats['successful_computations']
            
            if total_successful == 1:
                self.stats['avg_computation_time_ms'] = computation_time_ms
            else:
                # Weighted average favoring recent computations
                self.stats['avg_computation_time_ms'] = (current_avg * 0.9) + (computation_time_ms * 0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics"""
        return {
            **self.stats.copy(),
            'active_symbols': len(self.symbol_buffers),
            'cached_symbols': len(self.computation_cache),
            'max_workers': self.max_workers,
            'is_running': self.is_running
        }
    
    def get_buffer_status(self, symbol: str) -> Dict[str, Any]:
        """Get buffer status for a specific symbol"""
        buffer = self.symbol_buffers.get(symbol, deque())
        
        return {
            'symbol': symbol,
            'buffer_size': len(buffer),
            'max_buffer_size': self.ring_buffer_size,
            'last_computed': self.last_computed.get(symbol, 0),
            'has_cache': symbol in self.computation_cache,
            'latest_timestamp': buffer[-1]['timestamp'] if buffer else 0
        }


def compute_technical_indicators_worker(symbol: str, ohlcv_data: List[Dict[str, Any]], incremental: bool = False) -> Optional[Dict[str, Any]]:
    """
    Worker function that runs in separate process for CPU-intensive indicator computation
    This function cannot access class instance variables due to process isolation
    """
    try:
        if len(ohlcv_data) < 20:
            return None
        
        # Convert to pandas DataFrame for easier computation
        df = pd.DataFrame(ohlcv_data)
        df = df.sort_values('timestamp')  # Ensure chronological order
        
        # Extract price arrays
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_prices = df['open'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_20'] = _compute_sma(close, 20)
        indicators['sma_50'] = _compute_sma(close, 50) if len(close) >= 50 else None
        indicators['ema_12'] = _compute_ema(close, 12)
        indicators['ema_26'] = _compute_ema(close, 26)
        
        # MACD
        if indicators['ema_12'] is not None and indicators['ema_26'] is not None:
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = _compute_ema(indicators['macd'], 9)
            if indicators['macd_signal'] is not None:
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI
        indicators['rsi_14'] = _compute_rsi(close, 14)
        
        # Bollinger Bands
        bb_result = _compute_bollinger_bands(close, 20, 2)
        if bb_result:
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = bb_result
        
        # ATR (Average True Range)
        indicators['atr_14'] = _compute_atr(high, low, close, 14)
        
        # Stochastic
        stoch_result = _compute_stochastic(high, low, close, 14, 3)
        if stoch_result:
            indicators['stoch_k'], indicators['stoch_d'] = stoch_result
        
        # Volume indicators
        indicators['volume_sma_20'] = _compute_sma(volume, 20)
        
        # Price action indicators
        indicators['price_change_pct'] = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) >= 2 else 0
        indicators['high_low_pct'] = ((high[-1] - low[-1]) / close[-1] * 100) if close[-1] > 0 else 0
        
        # Latest values (for immediate use)
        result = {
            'symbol': symbol,
            'timestamp': df.iloc[-1]['timestamp'],
            'computation_time': time.time(),
            'incremental': incremental,
            'data_points': len(ohlcv_data),
            'latest_price': close[-1],
            'indicators': {}
        }
        
        # Extract latest values from computed indicators
        for key, values in indicators.items():
            if values is not None:
                if isinstance(values, np.ndarray):
                    result['indicators'][key] = float(values[-1]) if len(values) > 0 else None
                else:
                    result['indicators'][key] = float(values) if values is not None else None
            else:
                result['indicators'][key] = None
        
        return result
        
    except Exception as e:
        # Can't use self.logger in worker process
        print(f"Error in worker computation for {symbol}: {e}")
        return None


# Technical indicator computation functions
def _compute_sma(prices: np.ndarray, period: int) -> Optional[np.ndarray]:
    """Simple Moving Average"""
    if len(prices) < period:
        return None
    return pd.Series(prices).rolling(window=period).mean().values

def _compute_ema(prices: np.ndarray, period: int) -> Optional[np.ndarray]:
    """Exponential Moving Average"""
    if len(prices) < period:
        return None
    return pd.Series(prices).ewm(span=period).mean().values

def _compute_rsi(prices: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return None
    
    price_series = pd.Series(prices)
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def _compute_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Bollinger Bands"""
    if len(prices) < period:
        return None
    
    price_series = pd.Series(prices)
    middle = price_series.rolling(window=period).mean()
    std = price_series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper.values, middle.values, lower.values

def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Optional[np.ndarray]:
    """Average True Range"""
    if len(high) < period + 1:
        return None
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr.values

def _compute_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Stochastic Oscillator"""
    if len(close) < k_period:
        return None
    
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    
    k_percent = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent.values, d_percent.values


# Global instance
_compute_instance = None

async def get_indicator_compute() -> IndicatorCompute:
    """Get or create global indicator compute instance"""
    global _compute_instance
    
    if _compute_instance is None:
        _compute_instance = IndicatorCompute()
        await _compute_instance.start()
    
    return _compute_instance

async def cleanup_indicator_compute():
    """Clean up global indicator compute instance"""
    global _compute_instance
    
    if _compute_instance:
        await _compute_instance.stop()
        _compute_instance = None