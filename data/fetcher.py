import pandas as pd
import logging
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from trading.coinex_api import CoinExAPI
from indicators.calculator import IndicatorCalculator
from database.connection import db_connection
from database.models import Candle
from config.settings import TRADING_CONFIG, DATA_CONFIG
from config.config_loader import get_config_value

class DataFetcher:
    """
    Real-time data fetching and management system
    """
    
    def __init__(self, api: CoinExAPI):
        self.api = api
        self.logger = logging.getLogger(__name__)
        self.calculator = IndicatorCalculator()
        
        # Configuration
        self.training_symbols = TRADING_CONFIG['training_symbols']  # For model training only
        self.symbols = TRADING_CONFIG['symbols']  # Backward compatibility
        self.analysis_symbols = []  # Will be populated with top symbols
        self.timeframe = TRADING_CONFIG['timeframe']
        self.update_interval = DATA_CONFIG['update_interval']
        self.use_coinmarketcap_symbols = TRADING_CONFIG.get('use_coinmarketcap_symbols', True)
        self.coinmarketcap_limit = TRADING_CONFIG.get('coinmarketcap_limit', 1000)
        
        # Threading
        self.update_thread = None
        self.continuous_updater_thread = None
        self.position_monitor_thread = None
        self.stop_updates = False
        
        # Data cache
        self.latest_prices = {}
        self.latest_data_cache = {}
        
        # Throttling for data fetch spam prevention - more conservative for API stability
        self.last_fetch_times = {}  # symbol -> timestamp
        self.min_fetch_interval = get_config_value('data.real_time_min_interval', 10)  # 10 seconds for API stability
        
        # Real-time update configuration
        self.real_time_fetch_limit = DATA_CONFIG.get('real_time_fetch_limit', 3)
        
        # Continuous update configuration
        self.symbols_per_second = 10  # Update 10 symbols per second
        self.symbols_queue = []  # Rolling queue for continuous updates
        
        # Initialize analysis symbols (using cache)
        self.logger.info("🚀 Initializing symbol management system...")
        self._update_analysis_symbols()
        
        self.logger.info(f"📊 Symbol configuration summary:")
        self.logger.info(f"   🎓 Training symbols (for ML): {len(self.training_symbols)} symbols")
        self.logger.info(f"   📈 Analysis symbols (for trading): {len(self.analysis_symbols)} symbols")
        
        # Log details for debugging
        if len(self.analysis_symbols) > len(self.training_symbols):
            self.logger.info(f"✅ Successfully expanded symbol list for trading and analysis")
            self.logger.info(f"   Training: {self.training_symbols}")
            self.logger.info(f"   Additional analysis symbols: {len(self.analysis_symbols) - len(self.training_symbols)}")
        else:
            self.logger.warning(f"⚠️  Using minimal symbol set (training symbols only)")
            self.logger.warning(f"   This may indicate CoinMarketCap/CoinEx integration issues")
    
    def _update_analysis_symbols(self):
        """Update the list of analysis symbols from CoinMarketCap top cryptocurrencies available on CoinEx
        Always tries fresh fetch on startup, falls back to cache only on failure"""
        try:
            if self.use_coinmarketcap_symbols:
                self.logger.info(f"🔄 Starting symbol fetching process...")
                self.logger.info(f"   📊 Target: Top {self.coinmarketcap_limit} from CoinMarketCap")
                self.logger.info(f"   🎯 Training symbols (always included): {self.training_symbols}")
                self.logger.info(f"   🆕 FRESH FETCH: Always attempting new symbol list on startup")
                
                # Always try to get fresh symbols from CoinMarketCap → CoinEx → Cache
                cmc_symbols = self.api.get_coinmarketcap_available_symbols(limit=self.coinmarketcap_limit)
                
                if cmc_symbols and len(cmc_symbols) > len(self.training_symbols):
                    self.analysis_symbols = cmc_symbols
                    self.logger.info(f"✅ Symbol fetching successful:")
                    self.logger.info(f"   📈 Analysis symbols: {len(self.analysis_symbols)}")
                    self.logger.info(f"   🎓 Training symbols: {len(self.training_symbols)} (subset for ML training)")
                    self.logger.info(f"   💾 Fresh symbol list cached for future use")
                    
                    # Update the global config for other components
                    TRADING_CONFIG['analysis_symbols'] = self.analysis_symbols
                else:
                    # Fallback to training symbols
                    self.analysis_symbols = self.training_symbols.copy()
                    self.logger.warning(f"⚠️  Symbol fetching failed or insufficient symbols")
                    self.logger.warning(f"   Got: {len(cmc_symbols) if cmc_symbols else 0} symbols")
                    self.logger.warning(f"   Expected: > {len(self.training_symbols)} symbols")
                    self.logger.warning(f"   Falling back to training symbols only")
            else:
                # Use training symbols for analysis if CoinMarketCap integration disabled
                self.analysis_symbols = self.training_symbols.copy()
                self.logger.info("CoinMarketCap integration disabled, using training symbols for analysis")
                
        except Exception as e:
            self.logger.error(f"Error updating analysis symbols: {e}")
            # Try to load from symbol cache as emergency fallback
            try:
                from utils.symbol_cache import SymbolCache
                cache = SymbolCache()
                cached_symbols = cache.load_symbols(max_age_hours=168)  # Accept old cache in emergency
                if cached_symbols and len(cached_symbols) > len(self.training_symbols):
                    self.analysis_symbols = cached_symbols
                    self.logger.info(f"🆘 Emergency cache recovery: using {len(cached_symbols)} cached symbols")
                else:
                    self.analysis_symbols = self.training_symbols.copy()
                    self.logger.warning("No usable cached symbols, using training symbols")
            except Exception as cache_error:
                self.logger.error(f"Cache emergency fallback failed: {cache_error}")
                self.analysis_symbols = self.training_symbols.copy()
    
    def get_active_symbols(self) -> List[str]:
        """
        Get symbols that should be actively monitored for trading/analysis
        This includes the expanded list from CoinMarketCap (if available)
        """
        return self.analysis_symbols if self.use_coinmarketcap_symbols else self.training_symbols
    
    def get_training_symbols(self) -> List[str]:
        """
        Get symbols that should be used for model training ONLY
        Always returns the 4 core training symbols: BTC, ETH, SOL, DOGE
        """
        return self.training_symbols
    
    def start_real_time_updates(self):
        """Start all real-time data update systems"""
        # Start legacy update thread (background refresh)
        if not self.update_thread or not self.update_thread.is_alive():
            self.stop_updates = False
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            self.logger.info("📡 Legacy update thread started (background refresh)")
        
        # Start continuous updater (10 symbols/second rolling updates)
        self.start_continuous_updates()
        
        # Start position monitor (1-second updates for open positions)
        self.start_position_monitor()
        
        self.logger.info("✅ All real-time data systems started:")
        self.logger.info("   • Legacy updater: Background refresh")
        self.logger.info("   • Continuous updater: 10 symbols/sec")
        self.logger.info("   • Position monitor: Every 1 second")
    
    def stop_real_time_updates(self):
        """Stop all real-time data update systems"""
        self.stop_updates = True
        
        # Stop all threads
        threads_to_stop = [
            ("Legacy updater", self.update_thread),
            ("Continuous updater", self.continuous_updater_thread),
            ("Position monitor", self.position_monitor_thread)
        ]
        
        for name, thread in threads_to_stop:
            if thread and thread.is_alive():
                thread.join(timeout=10)
                self.logger.info(f"✓ {name} stopped")
        
        self.logger.info("All real-time data systems stopped")
    
    def _should_fetch_data(self, symbol: str) -> bool:
        """Check if we should fetch data for symbol based on throttling rules"""
        current_time = time.time()
        last_fetch = self.last_fetch_times.get(symbol, 0)
        
        time_since_last = current_time - last_fetch
        
        # For real-time updates, use minimal interval (1 second)
        if time_since_last < self.min_fetch_interval:
            return False
            
        return True
    
    def _record_fetch_time(self, symbol: str):
        """Record the fetch time for a symbol"""
        self.last_fetch_times[symbol] = time.time()

    def _update_loop(self):
        """Main update loop for real-time data with fast concurrent updates"""
        import concurrent.futures
        import threading
        
        symbols_refresh_interval = 24 * 3600  # Refresh CoinMarketCap symbols once per day
        last_symbols_refresh = 0
        
        while not self.stop_updates:
            try:
                # Periodically refresh analysis symbols (less frequent for CoinMarketCap)
                current_time = time.time()
                if current_time - last_symbols_refresh > symbols_refresh_interval:
                    self.logger.info("Refreshing CoinMarketCap analysis symbols...")
                    self._update_analysis_symbols()
                    last_symbols_refresh = current_time
                
                # Get active symbols for monitoring (using cached symbols)
                active_symbols = self.get_active_symbols()
                
                if not active_symbols:
                    self.logger.warning("No active symbols available, waiting...")
                    time.sleep(10)
                    continue
                
                # Log symbol count for debugging
                if len(active_symbols) != len(self.training_symbols):
                    self.logger.info(f"Monitoring {len(active_symbols)} symbols (expanded from {len(self.training_symbols)} training symbols)")
                
                # Update all symbols concurrently for speed
                self._update_symbols_concurrent(active_symbols)
                
                # Sleep for next iteration (30 seconds to respect API limits)
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in data update loop: {e}")
                time.sleep(5)
    
    def _update_symbols_concurrent(self, symbols: List[str]):
        """Update multiple symbols concurrently for faster processing"""
        import concurrent.futures
        
        # Limit concurrent connections to avoid overwhelming the API
        max_workers = min(5, len(symbols))  # Reduced from 20 to 5 to avoid rate limits
        
        # Process symbols in smaller batches to avoid rate limiting
        batch_size = 10  # Process 10 symbols at a time
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit batch of symbol updates
                future_to_symbol = {}
                
                for symbol in batch:
                    if self._should_fetch_data(symbol):
                        future = executor.submit(self._update_single_symbol_complete, symbol)
                        future_to_symbol[future] = symbol
                
                # Collect results
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_symbol, timeout=30):
                    symbol = future_to_symbol[future]
                    try:
                        future.result()
                        self._record_fetch_time(symbol)
                        completed_count += 1
                    except Exception as e:
                        self.logger.error(f"Error updating {symbol}: {e}")
                
                if completed_count > 0:
                    self.logger.debug(f"Updated {completed_count}/{len(batch)} symbols in batch")
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(symbols):
                import time
                time.sleep(2)  # 2 second delay between batches
    
    def _update_single_symbol_complete(self, symbol: str):
        """Complete update for a single symbol - both price and recent candles"""
        try:
            # 1. Update current price (ticker data)
            self._update_symbol_price(symbol)
            
            # 2. Update last 3 candles for real-time data
            self._update_recent_candles(symbol)
            
        except Exception as e:
            self.logger.error(f"Error in complete update for {symbol}: {e}")
    
    def _update_recent_candles(self, symbol: str):
        """Update the last 3 candles for real-time trading data"""
        try:
            # Fetch last 3 candles in 4h timeframe
            kline_data = self.api.get_kline_data(symbol, self.timeframe, limit=self.real_time_fetch_limit)
            
            if not kline_data:
                return
            
            # Store in database
            session = db_connection.get_session()
            
            for candle_data in kline_data:
                try:
                    timestamp = candle_data['timestamp']
                    
                    # For 4h timeframe, only process aligned candles
                    if self.timeframe == '4h' and not self._is_aligned_4h(timestamp):
                        continue
                    
                    # Check if candle already exists
                    existing_candle = session.query(Candle).filter(
                        Candle.symbol == symbol,
                        Candle.timestamp == timestamp
                    ).first()
                    
                    if not existing_candle:
                        # Create new candle
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                        session.add(candle)
                    else:
                        # Update existing candle (most recent one might be incomplete)
                        existing_candle.high = max(existing_candle.high, candle_data['high'])
                        existing_candle.low = min(existing_candle.low, candle_data['low'])
                        existing_candle.close = candle_data['close']
                        existing_candle.volume = candle_data['volume']
                        
                except Exception as candle_error:
                    self.logger.error(f"Error processing candle for {symbol}: {candle_error}")
                    continue
            
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating recent candles for {symbol}: {e}")
    
    def _update_symbol_price(self, symbol: str):
        """Update current price for a symbol"""
        try:
            ticker = self.api.get_ticker(symbol)
            current_price = float(ticker.get('last', 0))
            
            if current_price > 0:
                self.latest_prices[symbol] = {
                    'price': current_price,
                    'timestamp': datetime.now(),
                    'bid': float(ticker.get('buy', current_price)),
                    'ask': float(ticker.get('sell', current_price)),
                    'volume': float(ticker.get('vol', 0))
                }
        
        except Exception as e:
            self.logger.error(f"Error updating price for {symbol}: {e}")
    
    def _should_update_historical(self, symbol: str) -> bool:
        """Check if historical data should be updated"""
        try:
            # For 4-hour timeframe, update at the beginning of each 4-hour period
            current_time = datetime.now()
            
            # Check if we're at a 4-hour boundary
            if current_time.hour % 4 == 0 and current_time.minute < 5:
                return True
            
            # Also update if we don't have recent data in cache
            if symbol not in self.latest_data_cache:
                return True
            
            last_update = self.latest_data_cache.get(f"{symbol}_last_update")
            if not last_update or (current_time - last_update).total_seconds() > 3600:  # 1 hour
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking update requirement for {symbol}: {e}")
            return False
    
    def _update_historical_data(self, symbol: str):
        """Update historical candlestick data"""
        try:
            self.logger.info(f"Updating historical data for {symbol}")
            
            # Get recent kline data from API (20 candles for better analysis)
            kline_data = self.api.get_kline_data(symbol, self.timeframe, limit=20)
            
            if not kline_data:
                self.logger.warning(f"No kline data received for {symbol}")
                return
            
            # Store in database
            session = db_connection.get_session()
            
            for candle_data in kline_data:
                try:
                    timestamp = candle_data['timestamp']
                    
                    # For 4h timeframe, only process aligned candles
                    if self.timeframe == '4h' and not self._is_aligned_4h(timestamp):
                        continue
                    
                    # Check if candle already exists
                    existing_candle = session.query(Candle).filter(
                        Candle.symbol == symbol,
                        Candle.timestamp == timestamp
                    ).first()
                    
                    if not existing_candle:
                        # Create new candle
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                        session.add(candle)
                    else:
                        # Update existing candle (in case it's the current incomplete candle)
                        existing_candle.high = max(existing_candle.high, candle_data['high'])
                        existing_candle.low = min(existing_candle.low, candle_data['low'])
                        existing_candle.close = candle_data['close']
                        existing_candle.volume = candle_data['volume']
                
                except Exception as e:
                    self.logger.error(f"Error processing candle data for {symbol}: {e}")
            
            session.commit()
            session.close()
            
            # Update cache timestamp
            self.latest_data_cache[f"{symbol}_last_update"] = datetime.now()
            
            self.logger.info(f"Historical data updated for {symbol}: {len(kline_data)} candles")
            
        except Exception as e:
            self.logger.error(f"Error updating historical data for {symbol}: {e}")
    
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest price for symbol"""
        return self.latest_prices.get(symbol)
    
    def get_historical_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get historical data from database"""
        try:
            session = db_connection.get_session()
            
            # Query candles for symbol, ordered by timestamp descending
            query = session.query(Candle).filter(
                Candle.symbol == symbol
            ).order_by(Candle.timestamp.desc()).limit(limit)
            
            candles = query.all()
            session.close()
            
            if not candles:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            } for candle in candles])
            
            # Sort by timestamp (oldest first)
            data = data.sort_values('timestamp').reset_index(drop=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_data_with_indicators(self, symbol: str, lookback: int = 200) -> Optional[pd.DataFrame]:
        """
        Get latest data with all technical indicators calculated
        
        Args:
            symbol: Trading symbol
            lookback: Number of periods to include for indicator calculation
            
        Returns:
            DataFrame with OHLCV data and all indicators
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_with_indicators"
            cache_time_key = f"{symbol}_indicators_time"
            
            # Use cache if recent (within 1 minute)
            if (cache_key in self.latest_data_cache and 
                cache_time_key in self.latest_data_cache and
                (datetime.now() - self.latest_data_cache[cache_time_key]).total_seconds() < 60):
                
                return self.latest_data_cache[cache_key]
            
            # Get historical data
            historical_data = self.get_historical_data(symbol, lookback)
            
            if historical_data.empty:
                return None
            
            # For 4h timeframe, filter to only aligned 4h candles
            if self.timeframe == '4h':
                aligned_mask = historical_data['timestamp'].apply(self._is_aligned_4h)
                historical_data = historical_data[aligned_mask].reset_index(drop=True)
                
                if historical_data.empty:
                    self.logger.warning(f"No aligned 4h candles found for {symbol}")
                    return None
            
            # For 4h timeframe, do NOT add synthetic current price row
            # Only add current price for shorter timeframes
            if self.timeframe != '4h':
                current_price_info = self.get_latest_price(symbol)
                if current_price_info:
                    current_time = int(current_price_info['timestamp'].timestamp())
                    current_price = current_price_info['price']
                    
                    # Check if we need to add current data point
                    last_timestamp = historical_data['timestamp'].iloc[-1]
                    
                    # If current time is significantly different from last timestamp, add current point
                    if current_time - last_timestamp > 3600:  # More than 1 hour difference
                        current_row = {
                            'timestamp': current_time,
                            'open': current_price,
                            'high': current_price,
                            'low': current_price,
                            'close': current_price,
                            'volume': current_price_info.get('volume', 0)
                        }
                        historical_data = pd.concat([historical_data, pd.DataFrame([current_row])], ignore_index=True)
            
            # Calculate all technical indicators
            data_with_indicators = self.calculator.calculate_all_indicators(historical_data)
            
            # Cache the result
            self.latest_data_cache[cache_key] = data_with_indicators
            self.latest_data_cache[cache_time_key] = datetime.now()
            
            return data_with_indicators
            
        except Exception as e:
            self.logger.error(f"Error getting data with indicators for {symbol}: {e}")
            return None
    
    def get_recent_performance_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get recent performance data for analysis"""
        try:
            # Calculate timestamp for N days ago
            days_ago = datetime.now() - timedelta(days=days)
            timestamp_ago = int(days_ago.timestamp())
            
            session = db_connection.get_session()
            
            query = session.query(Candle).filter(
                Candle.symbol == symbol,
                Candle.timestamp >= timestamp_ago
            ).order_by(Candle.timestamp.asc())
            
            candles = query.all()
            session.close()
            
            if not candles:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume,
                'date': datetime.fromtimestamp(candle.timestamp)
            } for candle in candles])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting recent performance data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _is_aligned_4h(self, timestamp: int) -> bool:
        """Check if timestamp is aligned to 4-hour boundary"""
        try:
            dt = datetime.fromtimestamp(timestamp)
            # 4h aligned means hour is divisible by 4 and minute/second are 0
            return dt.hour % 4 == 0 and dt.minute == 0 and dt.second == 0
        except Exception:
            return False
    
    def count_4h_candles(self, symbol: str) -> int:
        """Count number of properly aligned 4h candles for symbol"""
        try:
            session = db_connection.get_session()
            
            candles = session.query(Candle).filter(
                Candle.symbol == symbol
            ).order_by(Candle.timestamp.asc()).all()
            
            session.close()
            
            aligned_count = 0
            for candle in candles:
                if self._is_aligned_4h(candle.timestamp):
                    aligned_count += 1
            
            return aligned_count
            
        except Exception as e:
            self.logger.error(f"Error counting 4h candles for {symbol}: {e}")
            return 0
    
    def backfill_4h(self, symbol: str, min_candles: int = None) -> bool:
        """Backfill 4h aligned candles to ensure minimum count"""
        try:
            if min_candles is None:
                min_candles = DATA_CONFIG.get('min_4h_candles', 800)
            
            current_count = self.count_4h_candles(symbol)
            self.logger.info(f"Current 4h candles for {symbol}: {current_count}")
            
            if current_count >= min_candles:
                self.logger.info(f"Sufficient 4h candles for {symbol} ({current_count} >= {min_candles})")
                return True
            
            # Calculate how many more candles we need
            needed_candles = min_candles - current_count
            
            # For 4h timeframe, we need to go back needed_candles * 4 hours
            hours_back = needed_candles * 4
            
            self.logger.info(f"Backfilling {needed_candles} 4h candles for {symbol} (going back {hours_back} hours)")
            
            # Try to get historical data from API
            # Note: This is a simplified implementation - in production you might need
            # multiple API calls due to limits
            try:
                kline_data = self.api.get_kline_data(symbol, '4h', limit=min(1000, needed_candles))
                
                if not kline_data:
                    self.logger.warning(f"No backfill data received from API for {symbol}")
                    return False
                
                session = db_connection.get_session()
                added_count = 0
                
                for candle_data in kline_data:
                    timestamp = candle_data['timestamp']
                    
                    # Only add if it's 4h aligned
                    if not self._is_aligned_4h(timestamp):
                        continue
                    
                    # Check if candle already exists
                    existing_candle = session.query(Candle).filter(
                        Candle.symbol == symbol,
                        Candle.timestamp == timestamp
                    ).first()
                    
                    if not existing_candle:
                        candle = Candle(
                            symbol=symbol,
                            timestamp=timestamp,
                            open=candle_data['open'],
                            high=candle_data['high'],
                            low=candle_data['low'],
                            close=candle_data['close'],
                            volume=candle_data['volume']
                        )
                        session.add(candle)
                        added_count += 1
                
                session.commit()
                session.close()
                
                self.logger.info(f"Backfilled {added_count} 4h candles for {symbol}")
                
                # Check if we now have enough
                final_count = self.count_4h_candles(symbol)
                if final_count >= min_candles:
                    return True
                else:
                    self.logger.warning(f"Still insufficient 4h candles for {symbol} after backfill: {final_count} < {min_candles}")
                    return False
                    
            except Exception as api_error:
                self.logger.error(f"API error during backfill for {symbol}: {api_error}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error backfilling 4h data for {symbol}: {e}")
            return False
    
    def validate_data_quality(self, symbol: str) -> Dict[str, Any]:
        """Validate data quality for a symbol"""
        try:
            data = self.get_historical_data(symbol, limit=1000)
            
            if data.empty:
                return {
                    'valid': False,
                    'reason': 'No data available'
                }
            
            # Check for gaps in data
            timestamps = data['timestamp'].sort_values()
            time_diffs = timestamps.diff().dropna()
            
            # For 4-hour timeframe, expect ~14400 seconds (4 hours) between candles
            expected_interval = 14400
            large_gaps = time_diffs[time_diffs > expected_interval * 2]
            
            # Check for price anomalies
            price_changes = data['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]  # More than 50% change
            
            # Check for volume anomalies
            volume_zeros = (data['volume'] == 0).sum()
            
            quality_report = {
                'valid': True,
                'total_records': len(data),
                'date_range': {
                    'start': datetime.fromtimestamp(data['timestamp'].min()),
                    'end': datetime.fromtimestamp(data['timestamp'].max())
                },
                'large_gaps': len(large_gaps),
                'extreme_price_changes': len(extreme_changes),
                'zero_volume_candles': volume_zeros,
                'data_completeness': (len(data) - len(large_gaps)) / len(data) * 100
            }
            
            # Mark as invalid if data quality is poor
            if (quality_report['large_gaps'] > len(data) * 0.1 or  # More than 10% gaps
                quality_report['extreme_price_changes'] > len(data) * 0.05):  # More than 5% extreme changes
                quality_report['valid'] = False
                quality_report['reason'] = 'Poor data quality detected'
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error validating data quality for {symbol}: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}'
            }
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview for all symbols"""
        try:
            overview = {}
            
            for symbol in self.get_active_symbols():
                price_info = self.get_latest_price(symbol)
                if price_info:
                    # Get 24h change
                    recent_data = self.get_recent_performance_data(symbol, days=1)
                    price_change_24h = 0.0
                    
                    if not recent_data.empty and len(recent_data) > 1:
                        price_24h_ago = recent_data['close'].iloc[0]
                        current_price = price_info['price']
                        price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                    
                    overview[symbol] = {
                        'price': price_info['price'],
                        'change_24h': price_change_24h,
                        'volume': price_info.get('volume', 0),
                        'last_update': price_info['timestamp']
                    }
            
            return overview
            
        except Exception as e:
            self.logger.error(f"Error getting market overview: {e}")
            return {}
    
    def force_data_refresh(self, symbol: str = None):
        """Force refresh of data for symbol or all symbols"""
        try:
            symbols_to_refresh = [symbol] if symbol else self.get_active_symbols()
            
            for sym in symbols_to_refresh:
                # Clear cache
                cache_keys_to_clear = [key for key in self.latest_data_cache.keys() if sym in key]
                for key in cache_keys_to_clear:
                    del self.latest_data_cache[key]
                
                # Force update
                self._update_symbol_price(sym)
                self._update_historical_data(sym)
            
            self.logger.info(f"Data refreshed for: {symbols_to_refresh}")
            
        except Exception as e:
            self.logger.error(f"Error forcing data refresh: {e}")
    
    def start_continuous_updates(self):
        """Start continuous data updates - 10 symbols per second"""
        if self.continuous_updater_thread and self.continuous_updater_thread.is_alive():
            self.logger.warning("Continuous updates already running")
            return
        
        # Initialize symbols queue
        self.symbols_queue = list(self.get_active_symbols())
        self.logger.info(f"🔄 Starting continuous data updater: {len(self.symbols_queue)} symbols, {self.symbols_per_second}/sec")
        
        self.continuous_updater_thread = threading.Thread(target=self._continuous_update_loop, daemon=True)
        self.continuous_updater_thread.start()
    
    def start_position_monitor(self):
        """Start active position monitoring - updates every second"""
        if self.position_monitor_thread and self.position_monitor_thread.is_alive():
            self.logger.warning("Position monitor already running")
            return
        
        self.logger.info("👁️  Starting active position monitor: Updates every 1 second")
        self.position_monitor_thread = threading.Thread(target=self._position_monitor_loop, daemon=True)
        self.position_monitor_thread.start()
    
    def _continuous_update_loop(self):
        """Continuous update loop - processes 10 symbols per second in rolling queue"""
        from collections import deque
        import concurrent.futures
        
        # Convert to deque for efficient rotation
        symbols_deque = deque(self.symbols_queue)
        
        self.logger.info(f"📊 Continuous updater initialized: {len(symbols_deque)} symbols in queue")
        
        while not self.stop_updates:
            try:
                if not symbols_deque:
                    # Re-populate queue if empty
                    symbols_deque = deque(self.get_active_symbols())
                    self.logger.info(f"🔄 Queue refreshed: {len(symbols_deque)} symbols")
                
                # Get next batch (10 symbols)
                batch = []
                for _ in range(min(self.symbols_per_second, len(symbols_deque))):
                    if symbols_deque:
                        symbol = symbols_deque.popleft()
                        batch.append(symbol)
                        symbols_deque.append(symbol)  # Add back to end of queue
                
                if not batch:
                    time.sleep(1)
                    continue
                
                # Update batch in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(self._update_single_symbol_complete, symbol): symbol
                        for symbol in batch
                    }
                    
                    for future in concurrent.futures.as_completed(futures, timeout=10):
                        symbol = futures[future]
                        try:
                            future.result()
                            self._record_fetch_time(symbol)
                        except Exception as e:
                            self.logger.debug(f"Error updating {symbol} in continuous loop: {e}")
                
                # Sleep for 1 second before next batch
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in continuous update loop: {e}")
                time.sleep(1)
    
    def _position_monitor_loop(self):
        """Monitor active positions - update every second for precise SL/TP management"""
        import concurrent.futures
        from database.models import Position
        
        self.logger.info("🎯 Position monitor initialized")
        
        while not self.stop_updates:
            try:
                # Get open positions from database
                session = db_connection.get_session()
                open_positions = session.query(Position).filter(
                    Position.status == 'OPEN'
                ).all()
                session.close()
                
                if not open_positions:
                    # No open positions, sleep and continue
                    time.sleep(1)
                    continue
                
                # Get symbols of open positions
                position_symbols = list(set([pos.symbol for pos in open_positions]))
                
                self.logger.debug(f"📍 Monitoring {len(position_symbols)} symbols with open positions")
                
                # Update all position symbols in parallel (high priority)
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = {
                        executor.submit(self._update_single_symbol_complete, symbol): symbol
                        for symbol in position_symbols
                    }
                    
                    for future in concurrent.futures.as_completed(futures, timeout=5):
                        symbol = futures[future]
                        try:
                            future.result()
                            # Don't record fetch time for position updates (higher priority)
                        except Exception as e:
                            self.logger.debug(f"Error updating position symbol {symbol}: {e}")
                
                # Sleep for 1 second before next update
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in position monitor loop: {e}")
                time.sleep(1)
            self.logger.error(f"Error forcing data refresh: {e}")