import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import statistics

from config.settings import TRADING_CONFIG
from data.fetcher_async import get_async_fetcher
from trading.coinex_ws import CoinExWebSocket
from utils.coinmarketcap_api import CoinMarketCapAPI

class TieredScheduler:
    """
    Intelligent two-tier scanning system:
    - Tier 1: Top symbols by volume/volatility, scanned every 60s via WebSocket
    - Tier 2: Remaining symbols, scanned every 240s via REST batch
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.tier1_size = TRADING_CONFIG['scan_tier1_size']
        self.tier1_interval = TRADING_CONFIG['scan_tier1_interval_sec']
        self.tier2_interval = TRADING_CONFIG['scan_tier2_interval_sec']
        self.analysis_on_close_only = TRADING_CONFIG['analysis_on_candle_close_only']
        
        # Symbol management
        self.all_symbols = []
        self.tier1_symbols = []
        self.tier2_symbols = []
        self.symbol_metrics = {}  # symbol -> metrics for ranking
        
        # WebSocket and REST clients
        self.websocket = None
        self.rest_fetcher = None
        
        # Task management
        self.is_running = False
        self.scheduler_tasks = []
        
        # Event queues
        self.signal_queue = asyncio.Queue(maxsize=1000)  # For sending signals to execution engine
        self.price_queue = asyncio.Queue(maxsize=1000)   # For real-time prices
        
        # Performance tracking
        self.scan_stats = {
            'tier1_scans': 0,
            'tier2_scans': 0,
            'tier1_avg_duration': 0,
            'tier2_avg_duration': 0,
            'last_tier1_scan': 0,
            'last_tier2_scan': 0,
            'websocket_events': 0,
            'rest_requests': 0
        }
        
        # Candle close detection
        self.last_candle_times = defaultdict(int)  # symbol -> last candle timestamp
        
    async def start(self):
        """Start the tiered scheduler"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info("Starting tiered scheduler...")
        
        # Initialize clients
        await self._initialize_clients()
        
        # Load and rank symbols
        await self._initialize_symbols()
        
        # Start scheduling tasks
        self.scheduler_tasks = [
            asyncio.create_task(self._tier1_scheduler()),
            asyncio.create_task(self._tier2_scheduler()),
            asyncio.create_task(self._websocket_handler()),
            asyncio.create_task(self._symbol_reranking_scheduler()),
            asyncio.create_task(self._performance_monitor())
        ]
        
        self.logger.info("Tiered scheduler started")
    
    async def stop(self):
        """Stop the tiered scheduler"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info("Stopping tiered scheduler...")
        
        # Cancel all tasks
        for task in self.scheduler_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.scheduler_tasks:
            await asyncio.gather(*self.scheduler_tasks, return_exceptions=True)
        
        # Cleanup clients
        if self.websocket:
            await self.websocket.stop()
        
        if self.rest_fetcher:
            await self.rest_fetcher.stop()
        
        self.scheduler_tasks.clear()
        self.logger.info("Tiered scheduler stopped")
    
    async def _initialize_clients(self):
        """Initialize WebSocket and REST clients"""
        try:
            # Initialize WebSocket
            if TRADING_CONFIG.get('use_websocket', True):
                self.websocket = CoinExWebSocket()
                self.websocket.set_ticker_handler(self._on_websocket_ticker)
                self.websocket.set_kline_handler(self._on_websocket_kline)
                
                # Start WebSocket in background
                asyncio.create_task(self.websocket.start())
            
            # Initialize async REST fetcher
            self.rest_fetcher = await get_async_fetcher()
            
            self.logger.info("Clients initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing clients: {e}")
            raise
    
    async def _initialize_symbols(self):
        """Load and rank all available symbols"""
        try:
            # Get CoinMarketCap symbols (as implemented in previous commits)
            from trading.coinex_api import CoinExAPI
            
            coinex_api = CoinExAPI()
            self.all_symbols = coinex_api.get_coinmarketcap_available_symbols(1000)
            
            if not self.all_symbols:
                # Fallback to basic symbols
                self.all_symbols = TRADING_CONFIG['training_symbols'].copy()
            
            self.logger.info(f"Loaded {len(self.all_symbols)} symbols for scanning")
            
            # Get initial metrics for ranking
            await self._update_symbol_metrics()
            
            # Perform initial ranking
            await self._rank_symbols()
            
        except Exception as e:
            self.logger.error(f"Error initializing symbols: {e}")
            self.all_symbols = TRADING_CONFIG['training_symbols'].copy()
            await self._rank_symbols()
    
    async def _update_symbol_metrics(self):
        """Update metrics for all symbols to determine tier ranking"""
        try:
            # Get all tickers at once for efficiency
            all_tickers = await self.rest_fetcher.fetch_all_tickers()
            
            for symbol in self.all_symbols:
                if symbol in all_tickers:
                    ticker = all_tickers[symbol]
                    
                    try:
                        price = float(ticker.get('last', 0))
                        volume = float(ticker.get('vol', 0))
                        high = float(ticker.get('high', price))
                        low = float(ticker.get('low', price))
                        
                        # Calculate volatility (simple high-low range)
                        volatility = ((high - low) / price) * 100 if price > 0 else 0
                        
                        # Calculate volume in USDT
                        volume_usdt = volume * price
                        
                        self.symbol_metrics[symbol] = {
                            'volume_24h': volume_usdt,
                            'volatility': volatility,
                            'price': price,
                            'updated_at': time.time()
                        }
                        
                    except (ValueError, TypeError, ZeroDivisionError):
                        # Use default metrics for symbols with invalid data
                        self.symbol_metrics[symbol] = {
                            'volume_24h': 0,
                            'volatility': 0,
                            'price': 0,
                            'updated_at': time.time()
                        }
            
            self.logger.info(f"Updated metrics for {len(self.symbol_metrics)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating symbol metrics: {e}")
    
    async def _rank_symbols(self):
        """Rank symbols and assign to tiers based on volume and volatility"""
        try:
            # Create ranking score: volume weight 70%, volatility weight 30%
            ranked_symbols = []
            
            for symbol, metrics in self.symbol_metrics.items():
                volume_score = metrics.get('volume_24h', 0)
                volatility_score = metrics.get('volatility', 0)
                
                # Normalize scores (simple approach)
                composite_score = (volume_score * 0.7) + (volatility_score * 100 * 0.3)
                
                ranked_symbols.append((symbol, composite_score))
            
            # Sort by composite score descending
            ranked_symbols.sort(key=lambda x: x[1], reverse=True)
            
            # Assign to tiers
            self.tier1_symbols = [symbol for symbol, _ in ranked_symbols[:self.tier1_size]]
            self.tier2_symbols = [symbol for symbol, _ in ranked_symbols[self.tier1_size:]]
            
            # Ensure training symbols are in tier 1 (high priority)
            training_symbols = TRADING_CONFIG['training_symbols']
            for symbol in training_symbols:
                if symbol not in self.tier1_symbols and symbol in self.all_symbols:
                    # Replace lowest priority tier1 symbol with training symbol
                    if len(self.tier1_symbols) >= self.tier1_size:
                        moved_symbol = self.tier1_symbols.pop()
                        self.tier2_symbols.insert(0, moved_symbol)
                    self.tier1_symbols.insert(0, symbol)
            
            self.logger.info(f"Ranked symbols - Tier1: {len(self.tier1_symbols)}, Tier2: {len(self.tier2_symbols)}")
            self.logger.debug(f"Tier1 symbols: {self.tier1_symbols[:10]}...")  # Show first 10
            
            # Update WebSocket subscriptions
            if self.websocket and self.websocket.is_connected:
                await self._update_websocket_subscriptions()
                
        except Exception as e:
            self.logger.error(f"Error ranking symbols: {e}")
    
    async def _update_websocket_subscriptions(self):
        """Update WebSocket subscriptions for Tier1 symbols"""
        try:
            if not self.websocket:
                return
            
            # Unsubscribe from all current subscriptions
            await self.websocket.unsubscribe_all()
            
            # Subscribe to tier1 symbols
            if self.tier1_symbols:
                await self.websocket.subscribe_ticker(self.tier1_symbols)
                
                # Subscribe to klines for tier1 (if enabled)
                if 'kline_1m' in TRADING_CONFIG['ws_channels']:
                    await self.websocket.subscribe_kline(self.tier1_symbols, '1m')
            
            self.logger.info(f"Updated WebSocket subscriptions for {len(self.tier1_symbols)} Tier1 symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating WebSocket subscriptions: {e}")
    
    async def _tier1_scheduler(self):
        """Scheduler for Tier1 symbols (high frequency)"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Tier1 scanning is primarily event-driven via WebSocket
                # This scheduler handles fallback and maintenance
                
                if not self.websocket or not self.websocket.is_connected:
                    # WebSocket unavailable - use REST fallback
                    await self._scan_tier1_rest_fallback()
                
                # Update stats
                duration = time.time() - start_time
                self._update_scan_stats('tier1', duration)
                
                # Sleep until next scheduled scan
                await asyncio.sleep(self.tier1_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Tier1 scheduler: {e}")
                await asyncio.sleep(10)
    
    async def _tier2_scheduler(self):
        """Scheduler for Tier2 symbols (lower frequency, REST-based)"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Scan Tier2 symbols via REST batch
                await self._scan_tier2_batch()
                
                # Update stats
                duration = time.time() - start_time
                self._update_scan_stats('tier2', duration)
                
                # Sleep until next scheduled scan
                await asyncio.sleep(self.tier2_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Tier2 scheduler: {e}")
                await asyncio.sleep(30)
    
    async def _scan_tier1_rest_fallback(self):
        """Scan Tier1 symbols using REST when WebSocket is unavailable"""
        try:
            if not self.tier1_symbols:
                return
            
            # Fetch tickers for tier1 symbols
            tickers = await self.rest_fetcher.fetch_tickers_batch(self.tier1_symbols)
            
            # Process each ticker
            for symbol, ticker_data in tickers.items():
                try:
                    price = float(ticker_data.get('last', 0))
                    if price > 0:
                        await self._process_price_update(symbol, price, time.time())
                except (ValueError, TypeError):
                    continue
            
            self.scan_stats['rest_requests'] += 1
            self.logger.debug(f"Tier1 REST fallback: processed {len(tickers)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in Tier1 REST fallback: {e}")
    
    async def _scan_tier2_batch(self):
        """Scan Tier2 symbols using REST batch requests"""
        try:
            if not self.tier2_symbols:
                return
            
            # Split tier2 symbols into smaller batches
            batch_size = min(100, len(self.tier2_symbols))
            batches = [self.tier2_symbols[i:i + batch_size] 
                      for i in range(0, len(self.tier2_symbols), batch_size)]
            
            total_processed = 0
            
            for batch in batches:
                # Fetch klines for analysis (more comprehensive than just tickers)
                klines_data = await self.rest_fetcher.fetch_klines_batch(batch, '1m', 50)
                
                # Process each symbol's klines
                for symbol, klines in klines_data.items():
                    if klines and len(klines) > 0:
                        # Get latest price from most recent candle
                        latest_candle = klines[-1]
                        price = latest_candle.get('close', 0)
                        timestamp = latest_candle.get('timestamp', time.time())
                        
                        if price > 0:
                            await self._process_price_update(symbol, price, timestamp)
                            
                            # Check if this is a new candle close
                            if self.analysis_on_close_only:
                                await self._check_candle_close(symbol, timestamp, klines)
                        
                        total_processed += 1
                
                # Small delay between batches
                if len(batches) > 1:
                    await asyncio.sleep(0.5)
            
            self.scan_stats['rest_requests'] += len(batches)
            self.logger.debug(f"Tier2 batch scan: processed {total_processed} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in Tier2 batch scan: {e}")
    
    async def _websocket_handler(self):
        """Handle WebSocket events and messages"""
        while self.is_running:
            try:
                if not self.websocket:
                    await asyncio.sleep(1)
                    continue
                
                # Get message from WebSocket
                message = await self.websocket.wait_for_message(1.0)
                
                if message:
                    self.scan_stats['websocket_events'] += 1
                    await self._process_websocket_message(message)
                
            except Exception as e:
                self.logger.error(f"Error in WebSocket handler: {e}")
                await asyncio.sleep(1)
    
    async def _process_websocket_message(self, message: Dict[str, Any]):
        """Process WebSocket message"""
        try:
            method = message.get('method', '')
            data = message.get('data', {})
            
            if method == 'state.update':
                # Ticker update
                params = data.get('params', [])
                if len(params) >= 2:
                    symbol = params[0]
                    ticker_data = params[1]
                    
                    price = float(ticker_data.get('last', 0))
                    if price > 0:
                        await self._process_price_update(symbol, price, time.time())
            
            elif method == 'deals.update':
                # Kline/trade update - check for candle closes
                params = data.get('params', [])
                if len(params) >= 2:
                    symbol = params[0]
                    deals_data = params[1]
                    
                    if self.analysis_on_close_only:
                        # Check if this represents a candle close
                        await self._check_websocket_candle_close(symbol, deals_data)
        
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    async def _on_websocket_ticker(self, symbol: str, ticker_data: Dict[str, Any]):
        """Handle WebSocket ticker update"""
        try:
            price = float(ticker_data.get('last', 0))
            if price > 0:
                await self._process_price_update(symbol, price, time.time())
        except Exception as e:
            self.logger.error(f"Error handling WebSocket ticker for {symbol}: {e}")
    
    async def _on_websocket_kline(self, symbol: str, kline_data: Dict[str, Any]):
        """Handle WebSocket kline update"""
        try:
            if self.analysis_on_close_only:
                await self._check_websocket_candle_close(symbol, kline_data)
        except Exception as e:
            self.logger.error(f"Error handling WebSocket kline for {symbol}: {e}")
    
    async def _process_price_update(self, symbol: str, price: float, timestamp: float):
        """Process a price update and send to execution engine"""
        try:
            # Send to price queue for execution engine
            price_update = {
                'symbol': symbol,
                'price': price,
                'timestamp': timestamp,
                'source': 'websocket' if self.websocket and self.websocket.is_connected else 'rest'
            }
            
            try:
                self.price_queue.put_nowait(price_update)
            except asyncio.QueueFull:
                # Drop oldest price if queue is full
                try:
                    self.price_queue.get_nowait()
                    self.price_queue.put_nowait(price_update)
                except asyncio.QueueEmpty:
                    pass
            
        except Exception as e:
            self.logger.error(f"Error processing price update for {symbol}: {e}")
    
    async def _check_candle_close(self, symbol: str, timestamp: float, klines: List[Dict[str, Any]]):
        """Check if we have a new candle close and trigger analysis"""
        try:
            if not klines:
                return
            
            # Get the timestamp of the latest complete candle
            latest_candle = klines[-1]
            candle_timestamp = latest_candle.get('timestamp', 0)
            
            # Check if this is a new candle compared to last seen
            last_seen = self.last_candle_times[symbol]
            
            if candle_timestamp > last_seen:
                self.last_candle_times[symbol] = candle_timestamp
                
                # Trigger analysis for this symbol
                await self._trigger_analysis(symbol, klines)
        
        except Exception as e:
            self.logger.error(f"Error checking candle close for {symbol}: {e}")
    
    async def _check_websocket_candle_close(self, symbol: str, kline_data: Dict[str, Any]):
        """Check WebSocket kline data for candle close"""
        try:
            # This depends on CoinEx WebSocket format - implementation may vary
            # For now, we'll assume any kline update could be a close
            current_time = time.time()
            current_minute = int(current_time // 60) * 60
            
            last_minute = self.last_candle_times.get(symbol, 0)
            
            if current_minute > last_minute:
                self.last_candle_times[symbol] = current_minute
                
                # Fetch recent klines for analysis
                klines = await self.rest_fetcher.fetch_klines(symbol, '1m', 50)
                if klines:
                    await self._trigger_analysis(symbol, klines)
        
        except Exception as e:
            self.logger.error(f"Error checking WebSocket candle close for {symbol}: {e}")
    
    async def _trigger_analysis(self, symbol: str, klines: List[Dict[str, Any]]):
        """Trigger technical analysis for a symbol and generate signals"""
        try:
            # This would integrate with the indicators computation
            # For now, we'll create a placeholder signal
            
            signal = {
                'symbol': symbol,
                'timestamp': time.time(),
                'action': 'analyze',
                'data': {
                    'klines_count': len(klines),
                    'latest_price': klines[-1].get('close', 0) if klines else 0
                }
            }
            
            # Send to signal queue
            try:
                self.signal_queue.put_nowait(signal)
            except asyncio.QueueFull:
                # Drop oldest signal if queue is full
                try:
                    self.signal_queue.get_nowait()
                    self.signal_queue.put_nowait(signal)
                except asyncio.QueueEmpty:
                    pass
            
            self.logger.debug(f"Triggered analysis for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error triggering analysis for {symbol}: {e}")
    
    async def _symbol_reranking_scheduler(self):
        """Periodically rerank symbols and update tiers"""
        while self.is_running:
            try:
                # Rerank every 30 minutes
                await asyncio.sleep(1800)
                
                if self.is_running:
                    self.logger.info("Reranking symbols...")
                    await self._update_symbol_metrics()
                    await self._rank_symbols()
                
            except Exception as e:
                self.logger.error(f"Error in symbol reranking: {e}")
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                stats = self.scan_stats.copy()
                current_time = time.time()
                
                self.logger.info(
                    f"Scanner Performance - T1 scans: {stats['tier1_scans']}, "
                    f"T2 scans: {stats['tier2_scans']}, WS events: {stats['websocket_events']}, "
                    f"REST requests: {stats['rest_requests']}, "
                    f"Price queue: {self.price_queue.qsize()}, Signal queue: {self.signal_queue.qsize()}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
    
    def _update_scan_stats(self, tier: str, duration: float):
        """Update scanning statistics"""
        key_scans = f"{tier}_scans"
        key_duration = f"{tier}_avg_duration"
        key_last = f"last_{tier}_scan"
        
        self.scan_stats[key_scans] += 1
        self.scan_stats[key_last] = time.time()
        
        # Update rolling average duration
        current_avg = self.scan_stats[key_duration]
        scan_count = self.scan_stats[key_scans]
        
        if scan_count == 1:
            self.scan_stats[key_duration] = duration
        else:
            # Simple moving average
            self.scan_stats[key_duration] = (current_avg * 0.9) + (duration * 0.1)
    
    async def get_price_update(self) -> Optional[Dict[str, Any]]:
        """Get next price update (for execution engine)"""
        try:
            return self.price_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def get_signal(self) -> Optional[Dict[str, Any]]:
        """Get next trading signal"""
        try:
            return self.signal_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        return {
            'is_running': self.is_running,
            'tier1_symbols': len(self.tier1_symbols),
            'tier2_symbols': len(self.tier2_symbols),
            'total_symbols': len(self.all_symbols),
            'websocket_connected': self.websocket.is_connected if self.websocket else False,
            'price_queue_size': self.price_queue.qsize(),
            'signal_queue_size': self.signal_queue.qsize(),
            'scan_stats': self.scan_stats.copy()
        }