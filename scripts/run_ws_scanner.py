#!/usr/bin/env python3
"""
High-Performance WebSocket Scanner Runner
Demonstrates the complete performance enhancement system with all components
"""
import asyncio
import logging
import signal
import sys
import time
from typing import Optional

# Add project root to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TRADING_CONFIG
from scanner.scheduler import TieredScheduler
from trading.execution_engine import ExecutionEngine
from trading.position_manager import PositionManager
from trading.coinex_api import CoinExAPI
from indicators.compute import get_indicator_compute
from data.fetcher_async import cleanup_async_fetcher

class PerformanceTestRunner:
    """
    Test runner for the complete performance enhancement system
    """
    
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('performance_test.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.scheduler = None
        self.execution_engine = None
        self.position_manager = None
        self.indicator_compute = None
        
        # Runtime control
        self.is_running = False
        self.main_task = None
        
        # Performance tracking
        self.start_time = time.time()
        self.stats = {
            'price_updates_processed': 0,
            'signals_generated': 0,
            'indicators_computed': 0,
            'tp_sl_checks': 0,
            'websocket_events': 0,
            'rest_requests': 0
        }
    
    async def start(self):
        """Start the performance test system"""
        try:
            self.logger.info("Starting Performance Enhancement Test System")
            self.logger.info("=" * 60)
            
            # Initialize components
            await self._initialize_components()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start main processing loop
            self.is_running = True
            self.main_task = asyncio.create_task(self._main_loop())
            
            self.logger.info("Performance test system started successfully")
            self.logger.info(f"Configuration: Tier1={TRADING_CONFIG['scan_tier1_size']} symbols, "
                           f"Tier2 interval={TRADING_CONFIG['scan_tier2_interval_sec']}s, "
                           f"Workers={TRADING_CONFIG['process_pool_workers']}")
            
            # Wait for main task
            await self.main_task
            
        except Exception as e:
            self.logger.error(f"Error starting performance test: {e}")
            raise
    
    async def stop(self):
        """Stop the performance test system"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.logger.info("Stopping performance test system...")
        
        # Cancel main task
        if self.main_task and not self.main_task.done():
            self.main_task.cancel()
            
        # Stop components in reverse order
        if self.execution_engine:
            await self.execution_engine.stop()
            
        if self.scheduler:
            await self.scheduler.stop()
            
        if self.indicator_compute:
            await self.indicator_compute.stop()
            
        # Cleanup global instances
        await cleanup_async_fetcher()
        
        # Final stats
        await self._print_final_stats()
        
        self.logger.info("Performance test system stopped")
    
    async def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize CoinEx API and position manager
            coinex_api = CoinExAPI()
            self.position_manager = PositionManager(coinex_api)
            
            # Initialize execution engine
            self.execution_engine = ExecutionEngine(self.position_manager)
            await self.execution_engine.start()
            self.logger.info("✅ Execution engine started")
            
            # Initialize indicator compute
            self.indicator_compute = await get_indicator_compute()
            self.logger.info("✅ Indicator compute system started")
            
            # Initialize tiered scheduler
            self.scheduler = TieredScheduler()
            await self.scheduler.start()
            self.logger.info("✅ Tiered scheduler started")
            
            # Add some demo positions for TP/SL testing
            await self._setup_demo_positions()
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def _setup_demo_positions(self):
        """Setup demo positions for TP/SL testing"""
        try:
            demo_positions = [
                {
                    'symbol': 'BTCUSDT',
                    'side': 'buy',
                    'entry_price': 45000.0,
                    'quantity': 0.001,
                    'timestamp': time.time()
                },
                {
                    'symbol': 'ETHUSDT',
                    'side': 'buy', 
                    'entry_price': 2500.0,
                    'quantity': 0.01,
                    'timestamp': time.time()
                }
            ]
            
            for position in demo_positions:
                await self.execution_engine.add_position(position['symbol'], position)
                
            self.logger.info(f"Added {len(demo_positions)} demo positions for TP/SL testing")
            
        except Exception as e:
            self.logger.error(f"Error setting up demo positions: {e}")
    
    async def _main_loop(self):
        """Main processing loop that coordinates all components"""
        self.logger.info("Starting main processing loop...")
        
        last_stats_report = time.time()
        stats_interval = 60  # Report stats every minute
        
        while self.is_running:
            try:
                # Process price updates from scheduler
                await self._process_price_updates()
                
                # Process trading signals
                await self._process_trading_signals()
                
                # Periodic stats reporting
                current_time = time.time()
                if current_time - last_stats_report > stats_interval:
                    await self._report_performance_stats()
                    last_stats_report = current_time
                
                # Short sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_price_updates(self):
        """Process price updates from scheduler and send to execution engine"""
        try:
            # Get price updates from scheduler
            while True:
                price_update = await self.scheduler.get_price_update()
                if not price_update:
                    break
                
                # Forward to execution engine for TP/SL monitoring
                await self.execution_engine.update_price(
                    price_update['symbol'],
                    price_update['price'],
                    price_update.get('timestamp')
                )
                
                self.stats['price_updates_processed'] += 1
                
        except Exception as e:
            self.logger.error(f"Error processing price updates: {e}")
    
    async def _process_trading_signals(self):
        """Process trading signals from scanner"""
        try:
            while True:
                signal = await self.scheduler.get_signal()
                if not signal:
                    break
                
                # Process the signal (trigger indicator computation)
                await self._handle_trading_signal(signal)
                
                self.stats['signals_generated'] += 1
                
        except Exception as e:
            self.logger.error(f"Error processing trading signals: {e}")
    
    async def _handle_trading_signal(self, signal: Dict[str, Any]):
        """Handle a trading signal by computing indicators and generating trade decisions"""
        try:
            symbol = signal.get('symbol')
            if not symbol:
                return
            
            # Get OHLCV data from signal (would be from kline data in real implementation)
            # For demo, we'll create some sample data
            sample_ohlcv = [
                {
                    'timestamp': time.time() - i * 60,
                    'open': 45000 + (i * 10),
                    'high': 45100 + (i * 10),
                    'low': 44900 + (i * 10), 
                    'close': 45050 + (i * 10),
                    'volume': 100 + (i * 5)
                }
                for i in range(50)  # 50 candles
            ]
            
            # Update OHLCV buffer
            await self.indicator_compute.update_ohlcv(symbol, sample_ohlcv)
            
            # Compute indicators
            indicators = await self.indicator_compute.compute_indicators(symbol)
            
            if indicators:
                self.stats['indicators_computed'] += 1
                self.logger.debug(f"Computed indicators for {symbol}: "
                                f"{len(indicators.get('indicators', {}))} indicators")
                
                # In a real implementation, this would feed into the ML model
                # For demo, just log the computation
                
        except Exception as e:
            self.logger.error(f"Error handling trading signal for {signal.get('symbol')}: {e}")
    
    async def _report_performance_stats(self):
        """Report performance statistics"""
        try:
            current_time = time.time()
            runtime_minutes = (current_time - self.start_time) / 60
            
            # Get component stats
            scheduler_status = self.scheduler.get_status()
            execution_status = self.execution_engine.get_status()
            compute_stats = self.indicator_compute.get_stats()
            
            # Calculate rates
            price_updates_per_min = self.stats['price_updates_processed'] / max(runtime_minutes, 1)
            signals_per_min = self.stats['signals_generated'] / max(runtime_minutes, 1)
            
            self.logger.info("=" * 60)
            self.logger.info("PERFORMANCE REPORT")
            self.logger.info(f"Runtime: {runtime_minutes:.1f} minutes")
            self.logger.info(f"Price updates: {self.stats['price_updates_processed']} ({price_updates_per_min:.1f}/min)")
            self.logger.info(f"Trading signals: {self.stats['signals_generated']} ({signals_per_min:.1f}/min)")
            self.logger.info(f"Indicators computed: {self.stats['indicators_computed']}")
            
            # Scheduler stats
            self.logger.info(f"Scanner - Tier1: {scheduler_status['tier1_symbols']} symbols, "
                           f"Tier2: {scheduler_status['tier2_symbols']} symbols")
            self.logger.info(f"Scanner - WebSocket: {'✅' if scheduler_status['websocket_connected'] else '❌'}, "
                           f"Queues: P:{scheduler_status['price_queue_size']} S:{scheduler_status['signal_queue_size']}")
            
            # Execution engine stats
            latency_stats = execution_status['latency_stats']
            self.logger.info(f"Execution - Open positions: {execution_status['open_positions']}, "
                           f"Avg latency: {latency_stats['avg_latency_ms']:.2f}ms, "
                           f"Max latency: {latency_stats['max_latency_ms']:.2f}ms")
            
            # Compute stats
            self.logger.info(f"Compute - Active symbols: {compute_stats['active_symbols']}, "
                           f"Cached: {compute_stats['cached_symbols']}, "
                           f"Avg time: {compute_stats['avg_computation_time_ms']:.2f}ms")
            self.logger.info(f"Compute - Success rate: {compute_stats['successful_computations']}/{compute_stats['total_computations']}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error reporting performance stats: {e}")
    
    async def _print_final_stats(self):
        """Print final statistics on shutdown"""
        try:
            runtime_seconds = time.time() - self.start_time
            
            self.logger.info("=" * 60)
            self.logger.info("FINAL PERFORMANCE SUMMARY")
            self.logger.info(f"Total runtime: {runtime_seconds:.1f} seconds")
            self.logger.info(f"Price updates processed: {self.stats['price_updates_processed']}")
            self.logger.info(f"Trading signals generated: {self.stats['signals_generated']}")  
            self.logger.info(f"Indicators computed: {self.stats['indicators_computed']}")
            
            if self.execution_engine:
                execution_stats = self.execution_engine.get_status()
                latency_stats = execution_stats['latency_stats']
                self.logger.info(f"Execution latency - Avg: {latency_stats['avg_latency_ms']:.2f}ms, "
                               f"Max: {latency_stats['max_latency_ms']:.2f}ms")
                
                # Check if we met the sub-300ms target
                if latency_stats['avg_latency_ms'] < 300:
                    self.logger.info("✅ LATENCY TARGET MET: Sub-300ms execution achieved")
                else:
                    self.logger.warning(f"❌ LATENCY TARGET MISSED: {latency_stats['avg_latency_ms']:.2f}ms > 300ms")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error printing final stats: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            self.logger.info(f"Received signal {sig}, shutting down...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    runner = PerformanceTestRunner()
    
    try:
        await runner.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Error in performance test: {e}")
    finally:
        await runner.stop()

if __name__ == "__main__":
    # Use uvloop if available (faster event loop on Linux)
    try:
        import uvloop
        uvloop.install()
        print("Using uvloop for enhanced performance")
    except ImportError:
        print("Using default asyncio event loop")
    
    asyncio.run(main())