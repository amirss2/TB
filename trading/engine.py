import logging
import threading
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from ml.model import TradingModel
from ml.trainer import ModelTrainer
from trading.position_manager import PositionManager
from trading.coinex_api import CoinExAPI
from data.fetcher import DataFetcher
from database.connection import db_connection
from database.models import TradingSignal, Position, TradingMetrics
from config.settings import TRADING_CONFIG, ML_CONFIG
from utils.network_utils import network_checker

class TradingEngine:
    """
    Main trading engine that coordinates all components
    """
    
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.api = CoinExAPI()
        self.position_manager = PositionManager(self.api)
        self.data_fetcher = DataFetcher(self.api)
        self.model = None
        self.trainer = ModelTrainer()
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.model_trained = False
        self.demo_balance = TRADING_CONFIG['demo_balance']
        self.used_balance = 0.0
        
        # Configuration
        self.training_symbols = TRADING_CONFIG['training_symbols']
        self.symbols = TRADING_CONFIG['symbols']  # Keep for backward compatibility
        self.confidence_threshold = TRADING_CONFIG['confidence_threshold']
        self.timeframe = TRADING_CONFIG['timeframe']
        
        # Thread management
        self._stop_trading = False   # renamed to avoid conflict with method name
        
        self.logger.info(f"Trading engine initialized (Demo: {demo_mode})")
        self.logger.info(f"[CONFIG] timeframe={self.timeframe} threshold={self.confidence_threshold}")

    def start_system(self):
        """Start the complete trading system"""
        try:
            self.logger.info("Starting trading system...")
            
            # 1. Test connections (but don't fail if API is down in demo mode)
            api_available = self._test_connections()
            if not api_available:
                if self.demo_mode:
                    self.logger.warning("API connection failed, running in offline demo mode")
                else:
                    raise Exception("Connection tests failed and not in demo mode")
            else:
                self.logger.info("All connection tests passed")
            
            # 2. Initialize data fetching
            self.data_fetcher.start_real_time_updates()
            
            # 3. Train model if needed (with fallback for demo mode)
            if not self.model_trained:
                self.logger.info("Training AI model...")
                try:
                    self.train_model()
                except Exception as model_error:
                    if self.demo_mode:
                        self.logger.warning(f"Model training failed in demo mode, using fallback: {model_error}")
                        self._use_fallback_model()
                    else:
                        raise model_error
            
            # 4. Start trading if model is available
            if self.model_trained:
                self.logger.info("🚀 Starting trading system with trained AI model...")
                self.start_trading()
                self.logger.info("✅ Trading system started successfully - Bot is now actively monitoring markets!")
                self.logger.info(f"📊 Model: {self.model.model_type if hasattr(self.model, 'model_type') else 'Fallback'}")
                self.logger.info(f"🎯 Confidence Threshold: {self.confidence_threshold * 100:.1f}%")
                self.logger.info(f"💰 Demo Balance: ${self.demo_balance:.2f}")
                self.logger.info(f"📈 Monitoring Symbols: {', '.join(self.symbols)}")
            else:
                self.logger.warning("Trading system started in limited demo mode (no AI model)")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading system: {e}")
            if not self.demo_mode:
                raise
            else:
                self.logger.warning("Continuing in demo mode despite errors")
    
    def _use_fallback_model(self):
        """Use fallback model when training fails in demo mode"""
        self.logger.info("Setting up fallback model for demo mode...")
        # Create a mock model that always returns low confidence predictions
        class FallbackModel:
            def __init__(self):
                self.model_version = "fallback_demo_v1.0"
            
            def predict(self, data):
                # Always return neutral prediction with low confidence
                return {'action': 'hold', 'confidence': 0.5, 'predicted_return': 0.0}
        
        self.model = FallbackModel()
        self.model_trained = True
        self.logger.info("Fallback model configured for demo mode")
    
    def stop_system(self):
        """Stop the complete trading system"""
        try:
            self.logger.info("Stopping trading system...")
            # Stop trading loop
            self._stop_trading = True
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            # Stop position monitoring
            self.position_manager.stop_position_monitoring()
            # Stop data fetching
            self.data_fetcher.stop_real_time_updates()
            self.is_running = False
            self.logger.info("Trading system stopped")
        except Exception as e:
            self.logger.error(f"Error stopping trading system: {e}")
    
    def train_model(self, retrain: bool = False) -> Dict[str, Any]:
        """Train or retrain the AI model"""
        try:
            self.logger.info(f"{'Retraining' if retrain else 'Training'} AI model...")
            
            # For 4h timeframe, ensure we have sufficient aligned candles before training
            if self.timeframe == '4h':
                self.logger.info("Checking and backfilling 4h candles before training...")
                # Use training symbols for backfill (not analysis symbols)
                for symbol in self.training_symbols:
                    success = self.data_fetcher.backfill_4h(symbol)
                    if not success:
                        self.logger.warning(f"Could not ensure sufficient 4h candles for {symbol}")
            
            # Train model with RFE
            training_result = self.trainer.train_with_rfe(retrain=retrain)
            
            # Load trained model
            self.model = self.trainer.model
            self.model_trained = True
            
            self.logger.info(f"Model training completed: {training_result['model_version']}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def start_trading(self):
        """Start the trading loop"""
        if self.is_running:
            self.logger.warning("Trading is already running")
            return
        
        if not self.model_trained or not self.model:
            raise ValueError("Model must be trained before starting trading")
        
        self.is_running = True
        self._stop_trading = False
        
        # Start position monitoring
        self.position_manager.start_position_monitoring()
        
        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("🔄 Trading loop started - Bot will now generate signals and manage positions")
        self.logger.info(f"⏰ Signal generation: Every 4h + enhanced frequency for active trading")
        self.logger.info(f"🎯 Confidence threshold: {self.confidence_threshold * 100:.1f}%")
    
    def stop_trading_loop(self):
        """Stop trading loop (internal)"""
        self._stop_trading = True
        self.is_running = False
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)
        self.logger.info("Trading loop stopped")
    
    def _trading_loop(self):
        """Main trading loop with network connectivity pause/resume - 60 second cycle with parallel analysis"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        self.logger.info("🔄 Trading loop initialized:")
        self.logger.info("   ⏰ Analysis cycle: Every 60 seconds")
        self.logger.info("   🧵 Processing: Parallel analysis of all symbols")
        self.logger.info("   📊 Timeframe: 4h-based calculations")
        
        while not self._stop_trading:
            try:
                cycle_start = time.time()
                
                # Check network connectivity before processing
                if not network_checker.is_connected():
                    self.logger.warning("⏸️  Trading paused: No network connectivity")
                    self.logger.info("Waiting for network connection to resume trading...")
                    
                    # Wait for connection to be restored
                    if network_checker.wait_for_connection(timeout=300, check_interval=30):
                        self.logger.info("🔄 Network restored, resuming trading operations")
                        # Re-test all connections after network restoration
                        if not self._test_connections():
                            self.logger.warning("Connection tests failed after network restoration, continuing to wait...")
                            time.sleep(60)
                            continue
                    else:
                        self.logger.warning("Network timeout, continuing to wait...")
                        time.sleep(60)
                        continue
                
                # Get active symbols for trading/analysis
                active_symbols = self.data_fetcher.get_active_symbols()
                self.logger.info(f"🔍 Analyzing {len(active_symbols)} symbols in parallel...")
                
                # Process all symbols in parallel for fast analysis
                processed = 0
                errors = 0
                
                # Reduced workers from 50 to 10 to prevent connection pool exhaustion
                with ThreadPoolExecutor(max_workers=10) as executor:
                    # Submit all symbol processing tasks
                    future_to_symbol = {
                        executor.submit(self._process_symbol, symbol): symbol 
                        for symbol in active_symbols
                    }
                    
                    # Process completed tasks
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            # Check if network disconnected during processing
                            if not network_checker.is_connected():
                                self.logger.warning(f"Network disconnected during analysis, pausing...")
                                # Cancel remaining futures
                                for f in future_to_symbol:
                                    f.cancel()
                                break
                            
                            future.result()  # Get result or raise exception
                            processed += 1
                        except Exception as e:
                            self.logger.debug(f"Error processing {symbol}: {e}")
                            errors += 1
                
                # Update trading metrics
                self._update_trading_metrics()
                
                # Calculate elapsed time and sleep remainder
                elapsed = time.time() - cycle_start
                self.logger.info(f"✅ Analysis cycle completed: {processed} symbols processed in {elapsed:.2f}s ({errors} errors)")
                
                # Sleep for remainder of 60 seconds (1 minute cycle)
                sleep_time = max(0, 60 - elapsed)
                if sleep_time > 0:
                    self.logger.debug(f"💤 Sleeping {sleep_time:.1f}s until next cycle...")
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"⚠️  Cycle took {elapsed:.2f}s (> 60s target)")
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)
    
    def _process_symbol(self, symbol: str):
        """Process trading signals for a specific symbol"""
        try:
            # Get latest market data with indicators
            latest_data = self.data_fetcher.get_latest_data_with_indicators(symbol)
            
            if latest_data is None or latest_data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return
            
            # Check if we're at a new timeframe boundary (enhanced for more frequent trading)
            if not self._is_new_timeframe(symbol):
                # Log less frequently to avoid spam, but show we're checking
                if datetime.now().minute % 30 == 0:
                    self.logger.debug(f"Waiting for timeframe signal for {symbol}")
                return
            
            self.logger.info(f"Processing timeframe signal for {symbol} - generating prediction")
            
            # Generate prediction
            prediction_result = self._generate_signal(latest_data, symbol)
            
            if prediction_result is None:
                self.logger.warning(f"Failed to generate prediction for {symbol}")
                return
            
            if prediction_result and prediction_result['meets_threshold']:
                signal = prediction_result['signal']
                confidence = prediction_result['confidence']
                
                self.logger.info(f"🎯 TRADING SIGNAL for {symbol}: {signal} (confidence: {confidence:.3f})")
                self.logger.info(f"   Probabilities: BUY={prediction_result['probabilities']['BUY']:.3f}, "
                               f"SELL={prediction_result['probabilities']['SELL']:.3f}, "
                               f"HOLD={prediction_result['probabilities']['HOLD']:.3f}")
                
                # Record signal in database
                self._record_signal(symbol, signal, confidence, latest_data['close'].iloc[-1])
                
                # Process signal
                if signal == 'BUY':
                    self.logger.info(f"📈 Processing BUY signal for {symbol}")
                    self._process_buy_signal(symbol, confidence, latest_data['close'].iloc[-1])
                elif signal == 'SELL':
                    self.logger.info(f"📉 Processing SELL signal for {symbol}")
                    self._process_sell_signal(symbol, confidence, latest_data['close'].iloc[-1])
                else:
                    self.logger.info(f"⏸️  HOLD signal for {symbol} - no action taken")
            else:
                # Show when signals don't meet threshold
                if prediction_result:
                    signal = prediction_result['signal'] 
                    confidence = prediction_result['confidence']
                    self.logger.info(f"Signal for {symbol}: {signal} (confidence: {confidence:.3f}) - Below threshold ({self.confidence_threshold:.3f})")
                else:
                    self.logger.warning(f"No prediction result for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {e}")
    
    def _generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal using AI model"""
        try:
            # Get features from the latest row
            feature_columns = self.model.feature_names
            latest_features = {}
            
            for feature in feature_columns:
                if feature in data.columns:
                    latest_features[feature] = data[feature].iloc[-1]
                else:
                    self.logger.warning(f"Feature {feature} not found for {symbol}")
                    latest_features[feature] = 0.0
            
            # Generate prediction
            prediction_result = self.model.predict_single(latest_features)
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _process_buy_signal(self, symbol: str, confidence: float, current_price: float):
        """Process BUY signal"""
        try:
            # Check if we already have a position for this symbol
            existing_position = self._get_open_position(symbol)
            if existing_position:
                self.logger.info(f"Already have open position for {symbol}, skipping BUY signal")
                return
            
            # CRITICAL: Enforce max_positions limit
            open_positions_count = self._get_open_positions_count()
            max_positions = TRADING_CONFIG['max_positions']
            if open_positions_count >= max_positions:
                self.logger.warning(f"Max positions limit reached ({open_positions_count}/{max_positions}). "
                                  f"Skipping BUY signal for {symbol}. Close existing positions first.")
                return
            
            # Validate price is not zero or invalid
            if current_price is None or current_price <= 0:
                self.logger.error(f"Invalid price for {symbol}: {current_price}. Cannot open position.")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, current_price)
            
            if position_size <= 0:
                self.logger.warning(f"Insufficient balance for {symbol} position")
                return
            
            # CRITICAL: Enforce minimum order value to prevent dust orders
            order_value = position_size * current_price
            min_order_value = TRADING_CONFIG.get('min_order_value', 5.0)
            if order_value < min_order_value:
                self.logger.warning(f"Order value ${order_value:.2f} below minimum ${min_order_value:.2f} for {symbol}. "
                                  f"Skipping to prevent dust order.")
                return
            
            if self.demo_mode:
                # Demo trading
                position_id = self.position_manager.open_position(
                    symbol, 'LONG', position_size, current_price, confidence
                )
                
                if position_id:
                    # Update used balance
                    position_value = position_size * current_price
                    self.used_balance += position_value
                    self.logger.info(f"Demo BUY order placed for {symbol}: {position_size} @ {current_price}")
            else:
                # Live trading
                try:
                    # Place market buy order
                    order_result = self.api.place_order(
                        symbol, 'buy', position_size, order_type='market'
                    )
                    
                    if order_result:
                        position_id = self.position_manager.open_position(
                            symbol, 'LONG', position_size, current_price, confidence
                        )
                        self.logger.info(f"Live BUY order placed for {symbol}: {position_size} @ {current_price}")
                
                except Exception as e:
                    self.logger.error(f"Failed to place live BUY order for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error processing BUY signal for {symbol}: {e}")
    
    def _process_sell_signal(self, symbol: str, confidence: float, current_price: float):
        """Process SELL signal"""
        try:
            # Check for existing position
            existing_position = self._get_open_position(symbol)
            
            if existing_position:
                # Emergency close if high confidence opposite signal
                if confidence >= self.confidence_threshold:
                    self.position_manager.emergency_close_position(existing_position['id'], confidence)
                    self.logger.info(f"Emergency close triggered for {symbol} position")
            else:
                # No existing position for SELL signal - just log
                self.logger.info(f"SELL signal for {symbol} but no open position - waiting for BUY signal")
            
        except Exception as e:
            self.logger.error(f"Error processing SELL signal for {symbol}: {e}")
    
    def _calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on user's requirement of 50% portfolio allocation"""
        try:
            available_balance = self.demo_balance - self.used_balance if self.demo_mode else self._get_real_balance()
            
            # Use 50% of available balance as requested by user (both demo and live)
            allocation_amount = available_balance * TRADING_CONFIG['risk_per_trade']  # Now set to 0.5 (50%)
            position_size = allocation_amount / price
            
            self.logger.info(f"Position sizing for {symbol}: Available=${available_balance:.2f}, "
                           f"Allocation(50%)=${allocation_amount:.2f}, Price=${price:.6f}, Size={position_size:.6f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0
    
    def _get_real_balance(self) -> float:
        """Get real account balance"""
        try:
            balance_info = self.api.get_balance()
            # Extract USDT balance
            return float(balance_info.get('USDT', {}).get('available', 0.0))
        except Exception as e:
            self.logger.error(f"Error getting real balance: {e}")
            return 0.0
    
    def _get_open_positions_count(self) -> int:
        """Get count of all open positions across all symbols"""
        try:
            session = db_connection.get_session()
            count = session.query(Position).filter(
                Position.status == 'OPEN'
            ).count()
            session.close()
            return count
        except Exception as e:
            self.logger.error(f"Error getting open positions count: {e}")
            return 0
    
    def _get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get open position for symbol"""
        try:
            session = db_connection.get_session()
            position = session.query(Position).filter(
                Position.symbol == symbol,
                Position.status == 'OPEN'
            ).first()
            session.close()
            
            if position:
                return {
                    'id': position.id,
                    'symbol': position.symbol,
                    'side': position.side,
                    'entry_price': position.entry_price,
                    'quantity': position.quantity
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting open position for {symbol}: {e}")
            return None
    
    def _record_signal(self, symbol: str, signal: str, confidence: float, price: float):
        """Record trading signal in database"""
        try:
            session = db_connection.get_session()
            
            trading_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal,
                confidence=confidence,
                price=price,
                model_version=self.model.model_version if self.model else 'unknown'
            )
            
            session.add(trading_signal)
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error recording signal: {e}")
    
    def _is_new_timeframe(self, symbol: str) -> bool:
        """
        ALWAYS return True for continuous trading every minute.
        
        User requirement: Analyze every minute with 4h-based calculations.
        The 4h timeframe is used for indicator calculations (RSI, MACD, etc.),
        NOT for limiting trading frequency.
        
        This allows the bot to:
        - Analyze all symbols every 60 seconds
        - Use 4h candles for technical indicator calculations
        - Generate trading signals based on latest 4h indicator values
        - Trade immediately when high-confidence signals appear
        
        Instead of waiting for 4-hour boundaries (0:00, 4:00, 8:00, etc.),
        the bot now trades continuously based on real-time 4h indicator analysis.
        """
        return True  # Always trade - indicators are 4h-based, not trading frequency
    
    def _update_trading_metrics(self):
        """Update daily trading metrics"""
        try:
            # Get today's metrics
            today = datetime.now().date()
            
            session = db_connection.get_session()
            
            # Get existing or create new metric record
            metric = session.query(TradingMetrics).filter(
                TradingMetrics.date == today
            ).first()
            
            if not metric:
                metric = TradingMetrics(
                    date=today,
                    portfolio_value=self.demo_balance if self.demo_mode else self._get_real_balance(),
                    available_balance=self.demo_balance - self.used_balance if self.demo_mode else self._get_real_balance()
                )
                session.add(metric)
            
            # Update with current data
            positions = self.position_manager.get_active_positions()
            total_pnl = sum(pos['pnl'] for pos in positions)
            
            metric.daily_pnl = total_pnl
            metric.daily_pnl_percentage = (total_pnl / self.demo_balance) * 100 if self.demo_balance > 0 else 0
            
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error updating trading metrics: {e}")
    
    def _test_connections(self) -> bool:
        """Test all system connections including network connectivity"""
        try:
            # Test database
            if not db_connection.test_connection():
                self.logger.error("Database connection failed")
                return False
            
            # Test network connectivity first
            if not network_checker.is_connected(force_check=True):
                self.logger.warning("Network connectivity check failed")
                if self.demo_mode:
                    self.logger.warning("Demo mode: continuing without network connectivity")
                    return False  # Return False to trigger pause mode even in demo
                else:
                    self.logger.error("Live mode: network connectivity required")
                    return False
            
            # Test API (only if network is available)
            api_ok = self.api.test_connection()
            if not api_ok:
                if self.demo_mode:
                    self.logger.warning("API connection failed but continuing in demo mode")
                    return True  # Don't fail in demo mode if network is available
                else:
                    self.logger.error("API connection failed in live mode")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'demo_mode': self.demo_mode,
            'model_trained': self.model_trained,
            'model_version': self.model.model_version if self.model else None,
            'demo_balance': self.demo_balance,
            'used_balance': self.used_balance,
            'available_balance': self.demo_balance - self.used_balance,
            'active_positions': len(self.position_manager.get_active_positions()),
            'symbols': self.symbols,
            'timeframe': self.timeframe,
            'confidence_threshold': self.confidence_threshold
        }
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        try:
            positions = self.position_manager.get_active_positions()
            total_pnl = sum(pos['pnl'] for pos in positions)
            
            return {
                'total_positions': len(positions),
                'total_unrealized_pnl': total_pnl,
                'portfolio_value': self.demo_balance + total_pnl,
                'positions': positions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading summary: {e}")
            return {}
