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
from database.models import TradingSignal, Position, TradingMetrics, Wallet, WalletTransaction
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
        self.position_manager = PositionManager(self.api, trading_engine=self)  # Pass reference to self
        self.data_fetcher = DataFetcher(self.api)
        self.model = None
        self.trainer = ModelTrainer()
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.model_trained = False
        self.demo_balance = TRADING_CONFIG['demo_balance']
        self.used_balance = 0.0
        
        # Network state management
        self.network_connected = True
        self.trading_paused_by_network = False
        
        # Configuration
        self.training_symbols = TRADING_CONFIG['training_symbols']
        self.symbols = TRADING_CONFIG['symbols']  # Keep for backward compatibility
        self.confidence_threshold = TRADING_CONFIG['confidence_threshold']
        self.timeframe = TRADING_CONFIG['timeframe']
        
        # Thread management
        self._stop_trading = False   # renamed to avoid conflict with method name
        
        # Initialize wallet from database
        self._initialize_wallet()
        
        self.logger.info(f"Trading engine initialized (Demo: {demo_mode})")
        self.logger.info(f"[CONFIG] timeframe={self.timeframe} threshold={self.confidence_threshold}")
    
    def _initialize_wallet(self):
        """Initialize or restore wallet from database with idempotent state recovery"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INITIALIZING/RESTORING WALLET STATE")
            self.logger.info("=" * 80)
            
            session = db_connection.get_session()
            account_type = 'demo' if self.demo_mode else 'live'
            
            # Try to get existing wallet
            wallet = session.query(Wallet).filter_by(account_type=account_type).first()
            
            if wallet:
                # Restore from database (idempotent recovery)
                self.demo_balance = wallet.total_balance
                self.logger.info(f"✓ Restored wallet from database: {account_type} balance = ${wallet.total_balance:.2f}")
                
                # Restore used_balance from open positions (idempotent)
                open_positions = session.query(Position).filter_by(status='OPEN').all()
                self.used_balance = sum(pos.entry_price * pos.quantity for pos in open_positions)
                self.logger.info(f"✓ Restored used_balance from {len(open_positions)} open positions: ${self.used_balance:.2f}")
                
                # Log details of open positions
                if open_positions:
                    self.logger.info("   Open positions found:")
                    for pos in open_positions:
                        position_value = pos.entry_price * pos.quantity
                        self.logger.info(f"   - {pos.symbol}: {pos.side}, Entry=${pos.entry_price:.6f}, "
                                       f"Qty={pos.quantity}, Value=${position_value:.2f}")
                
                # Update wallet locked balance (reconciliation)
                wallet.locked_balance = self.used_balance
                wallet.available_balance = wallet.total_balance - self.used_balance
                
                # Ensure available_balance is never negative
                if wallet.available_balance < 0:
                    self.logger.warning(f"⚠️  Available balance would be negative ({wallet.available_balance:.2f}), clamping to 0")
                    wallet.available_balance = 0
                
                session.commit()
                
                self.logger.info(f"✓ Wallet reconciled: Total=${wallet.total_balance:.2f}, "
                               f"Available=${wallet.available_balance:.2f}, Locked=${wallet.locked_balance:.2f}")
                
                # Run health check to verify state
                session.close()
                session = db_connection.get_session()
                wallet = session.query(Wallet).filter_by(account_type=account_type).first()
                transactions = session.query(WalletTransaction).filter_by(account_type=account_type).all()
                
                # Quick reconciliation check
                initial_balance = TRADING_CONFIG['demo_balance'] if account_type == 'demo' else 0.0
                transaction_sum = sum(t.amount for t in transactions)
                expected_balance = initial_balance + transaction_sum
                balance_match = abs(expected_balance - wallet.total_balance) < 0.01
                
                if balance_match:
                    self.logger.info(f"✅ RECONCILIATION: PASSED - Balance matches transaction history")
                else:
                    self.logger.warning(f"⚠️  RECONCILIATION: Difference of ${wallet.total_balance - expected_balance:.2f} detected")
                
            else:
                # Create new wallet (first-time initialization)
                wallet = Wallet(
                    account_type=account_type,
                    total_balance=self.demo_balance,
                    available_balance=self.demo_balance,
                    locked_balance=0.0
                )
                session.add(wallet)
                session.commit()
                self.logger.info(f"✓ Created new {account_type} wallet with ${self.demo_balance:.2f}")
            
            session.close()
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error initializing wallet: {e}", exc_info=True)
            # Continue with in-memory values
    
    def _update_wallet_balance(self, amount: float, transaction_type: str, position_id: int = None, description: str = None):
        """Update wallet balance in database"""
        try:
            session = db_connection.get_session()
            account_type = 'demo' if self.demo_mode else 'live'
            
            wallet = session.query(Wallet).filter_by(account_type=account_type).first()
            if not wallet:
                self.logger.error(f"Wallet not found for {account_type}")
                session.close()
                return
            
            balance_before = wallet.total_balance
            
            # Update balances
            if transaction_type in ['position_open']:
                # Locking funds
                wallet.locked_balance += abs(amount)
                wallet.available_balance -= abs(amount)
            elif transaction_type in ['position_close']:
                # Unlocking funds and adding/subtracting PnL
                wallet.locked_balance -= abs(amount)
                wallet.available_balance += amount  # amount includes PnL
            
            balance_after = wallet.total_balance
            
            # Record transaction
            transaction = WalletTransaction(
                account_type=account_type,
                transaction_type=transaction_type,
                amount=amount,
                balance_before=balance_before,
                balance_after=balance_after,
                position_id=position_id,
                description=description
            )
            session.add(transaction)
            session.commit()
            session.close()
            
            self.logger.debug(f"Wallet updated: {transaction_type} ${amount:.2f}, Available: ${wallet.available_balance:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating wallet: {e}")
    
    def _get_real_coinex_balance(self) -> float:
        """Get real account balance from CoinEx API with validation"""
        try:
            if self.demo_mode:
                return self.demo_balance - self.used_balance
            
            balance_info = self.api.get_balance()
            if not balance_info:
                self.logger.error("Failed to get balance from CoinEx API")
                return 0.0
            
            # Extract USDT balance
            usdt_balance = float(balance_info.get('USDT', {}).get('available', 0.0))
            self.logger.info(f"Real CoinEx balance: ${usdt_balance:.2f} USDT")
            return usdt_balance
            
        except Exception as e:
            self.logger.error(f"Error getting real CoinEx balance: {e}")
            return 0.0


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
        """Stop the complete trading system with graceful shutdown"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INITIATING GRACEFUL SHUTDOWN")
            self.logger.info("=" * 80)
            
            # 1. Save current state before shutdown
            self.logger.info("Saving system state...")
            self._save_shutdown_state()
            
            # 2. Stop trading loop
            self.logger.info("Stopping trading loop...")
            self._stop_trading = True
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # 3. Stop position monitoring
            self.logger.info("Stopping position monitoring...")
            self.position_manager.stop_position_monitoring()
            
            # 4. Stop data fetching
            self.logger.info("Stopping data fetching...")
            self.data_fetcher.stop_real_time_updates()
            
            # 5. Final health check before shutdown
            self.logger.info("Running final health check...")
            health_report = self.comprehensive_health_check()
            
            # 6. Update system state
            self.is_running = False
            
            self.logger.info("=" * 80)
            self.logger.info("GRACEFUL SHUTDOWN COMPLETED")
            self.logger.info(f"System stopped with {health_report.get('positions', {}).get('open_count', 0)} open positions")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}", exc_info=True)
    
    def _save_shutdown_state(self):
        """Save critical state before shutdown for resume capability"""
        try:
            session = db_connection.get_session()
            account_type = 'demo' if self.demo_mode else 'live'
            
            # Update wallet with current state
            wallet = session.query(Wallet).filter_by(account_type=account_type).first()
            if wallet:
                # Recalculate locked balance from open positions
                open_positions = session.query(Position).filter_by(status='OPEN').all()
                wallet.locked_balance = sum(pos.entry_price * pos.quantity for pos in open_positions)
                wallet.available_balance = wallet.total_balance - wallet.locked_balance
                
                self.logger.info(f"State saved: {len(open_positions)} open positions, "
                               f"${wallet.locked_balance:.2f} locked, ${wallet.available_balance:.2f} available")
                
                session.commit()
            
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error saving shutdown state: {e}", exc_info=True)
    
    
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
                
                # CRITICAL: Check network connectivity before ALL operations
                if not network_checker.is_connected():
                    if not self.trading_paused_by_network:
                        self.network_connected = False
                        self.trading_paused_by_network = True
                        self.logger.warning("⏸️  SYSTEM PAUSED: No network connectivity - ALL trading operations frozen")
                        self.logger.info("ℹ️  Open positions will NOT be checked/closed until network is restored")
                        self.logger.info("ℹ️  No new trades will be opened until network is restored")
                    
                    # Wait for connection to be restored
                    if network_checker.wait_for_connection(timeout=300, check_interval=30):
                        self.logger.info("🔄 Network restored - verifying connections before resuming...")
                        # Re-test all connections after network restoration
                        if self._test_connections():
                            self.network_connected = True
                            self.trading_paused_by_network = False
                            self.logger.info("✅ SYSTEM RESUMED: All connections verified - trading operations active")
                            # Give system a moment to stabilize prices
                            time.sleep(5)
                        else:
                            self.logger.warning("❌ Connection tests failed after network restoration, remaining paused...")
                            time.sleep(60)
                            continue
                    else:
                        self.logger.warning("⏳ Network still unavailable, remaining paused...")
                        time.sleep(60)
                        continue
                
                # Ensure we're in connected state before proceeding
                if not self.network_connected or self.trading_paused_by_network:
                    self.logger.warning("⏸️  System paused - skipping cycle")
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
        """Process BUY signal with real balance check and wallet tracking"""
        try:
            # Check if we already have a position for this symbol
            existing_position = self._get_open_position(symbol)
            if existing_position:
                self.logger.info(f"Already have open position for {symbol}, skipping BUY signal")
                return
            
            # CRITICAL: Enforce max_positions limit with detailed logging
            open_positions_count = self._get_open_positions_count()
            max_positions = TRADING_CONFIG['max_positions']
            if open_positions_count >= max_positions:
                # Get current open positions for comparison
                session = db_connection.get_session()
                current_positions = session.query(Position).filter_by(status='OPEN').all()
                session.close()
                
                # Log detailed rejection information
                self.logger.warning("=" * 80)
                self.logger.warning(f"⚠️  MAX POSITIONS LIMIT REACHED ({open_positions_count}/{max_positions})")
                self.logger.warning(f"REJECTED SIGNAL:")
                self.logger.warning(f"   Symbol: {symbol}")
                self.logger.warning(f"   Type: BUY (LONG)")
                self.logger.warning(f"   Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
                self.logger.warning(f"   Price: ${current_price:.6f}")
                self.logger.warning(f"   Reason: All position slots occupied")
                self.logger.warning("")
                self.logger.warning("CURRENT OPEN POSITIONS:")
                for idx, pos in enumerate(current_positions, 1):
                    # Get current price for each position
                    pos_current_price = self.position_manager._get_current_price(pos.symbol)
                    if pos_current_price:
                        pnl_info = self.position_manager._calculate_pnl(pos, pos_current_price)
                        self.logger.warning(f"   {idx}. {pos.symbol}: {pos.side}, Entry=${pos.entry_price:.6f}, "
                                          f"Current=${pos_current_price:.6f}, Net PnL=${pnl_info['pnl']:.4f} "
                                          f"({pnl_info['pnl_percentage']:.2f}%)")
                    else:
                        self.logger.warning(f"   {idx}. {pos.symbol}: {pos.side}, Entry=${pos.entry_price:.6f}, "
                                          f"Current=N/A")
                self.logger.warning("")
                self.logger.warning("ACTION REQUIRED: Close existing positions to free up slots for new signals")
                self.logger.warning("=" * 80)
                return
            
            # Validate price is not zero or invalid
            if current_price is None or current_price <= 0:
                self.logger.error(f"Invalid price for {symbol}: {current_price}. Cannot open position.")
                return
            
            # CRITICAL: For live trading, check real CoinEx balance before proceeding
            if not self.demo_mode:
                real_balance = self._get_real_coinex_balance()
                if real_balance <= 0:
                    self.logger.error(f"Real CoinEx balance is ${real_balance:.2f}, cannot open position for {symbol}")
                    return
                self.logger.info(f"Real CoinEx balance verified: ${real_balance:.2f} USDT available")
            
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
                    
                    # Record in wallet database
                    self._update_wallet_balance(
                        amount=position_value,
                        transaction_type='position_open',
                        position_id=position_id,
                        description=f"Opened LONG position for {symbol}"
                    )
                    
                    self.logger.info(f"Demo BUY order placed for {symbol}: {position_size} @ {current_price}, "
                                   f"Value: ${position_value:.2f}, Used: ${self.used_balance:.2f}")
            else:
                # Live trading - verify balance one more time before order
                real_balance = self._get_real_coinex_balance()
                if order_value > real_balance:
                    self.logger.error(f"Insufficient CoinEx balance: Need ${order_value:.2f}, Have ${real_balance:.2f}")
                    return
                
                try:
                    # Place market buy order
                    order_result = self.api.place_order(
                        symbol, 'buy', position_size, order_type='market'
                    )
                    
                    if order_result:
                        position_id = self.position_manager.open_position(
                            symbol, 'LONG', position_size, current_price, confidence
                        )
                        
                        if position_id:
                            position_value = position_size * current_price
                            
                            # Record in wallet database
                            self._update_wallet_balance(
                                amount=position_value,
                                transaction_type='position_open',
                                position_id=position_id,
                                description=f"Opened LONG position for {symbol}"
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
        """Calculate position size based on fixed $100 per trade"""
        try:
            available_balance = self.demo_balance - self.used_balance if self.demo_mode else self._get_real_balance()
            
            # Use fixed $100 per trade as requested by user
            allocation_amount = TRADING_CONFIG['risk_per_trade']  # Fixed $100
            
            # Check if we have enough balance
            if allocation_amount > available_balance:
                self.logger.warning(f"Insufficient balance for {symbol}: Need ${allocation_amount:.2f}, Have ${available_balance:.2f}")
                return 0.0
            
            position_size = allocation_amount / price
            
            self.logger.info(f"Position sizing for {symbol}: Available=${available_balance:.2f}, "
                           f"Fixed allocation=${allocation_amount:.2f}, Price=${price:.6f}, Size={position_size:.6f}")
            
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
    
    def comprehensive_health_check(self) -> Dict[str, Any]:
        """
        Comprehensive system health check that reports:
        - Number of OPEN positions
        - Wallet balance breakdown (total/available/locked)
        - Total unrealized PnL (after fees)
        - Wallet reconciliation status
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("SYSTEM HEALTH CHECK")
            self.logger.info("=" * 80)
            
            # 1. Get wallet information
            session = db_connection.get_session()
            account_type = 'demo' if self.demo_mode else 'live'
            wallet = session.query(Wallet).filter_by(account_type=account_type).first()
            
            # 2. Get open positions
            open_positions = session.query(Position).filter_by(status='OPEN').all()
            open_positions_count = len(open_positions)
            
            # 3. Calculate total unrealized PnL (with all fees)
            total_unrealized_pnl = 0.0
            position_details = []
            for pos in open_positions:
                current_price = self.position_manager._get_current_price(pos.symbol)
                if current_price:
                    pnl_info = self.position_manager._calculate_pnl(pos, current_price)
                    total_unrealized_pnl += pnl_info['pnl']  # Net PnL after all costs
                    position_details.append({
                        'symbol': pos.symbol,
                        'side': pos.side,
                        'entry_price': pos.entry_price,
                        'current_price': current_price,
                        'quantity': pos.quantity,
                        'net_pnl': pnl_info['pnl'],
                        'gross_pnl': pnl_info['gross_pnl'],
                        'total_costs': pnl_info['total_costs']
                    })
            
            session.close()
            
            # 4. Get wallet reconciliation
            reconciliation = self.position_manager.wallet_health_check()
            
            # 5. Prepare health check report
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'account_type': account_type,
                'positions': {
                    'open_count': open_positions_count,
                    'max_positions': TRADING_CONFIG['max_positions'],
                    'available_slots': TRADING_CONFIG['max_positions'] - open_positions_count,
                    'details': position_details
                },
                'wallet': {
                    'total_balance': wallet.total_balance if wallet else 0.0,
                    'available_balance': wallet.available_balance if wallet else 0.0,
                    'locked_balance': wallet.locked_balance if wallet else 0.0,
                    'total_pnl': wallet.total_pnl if wallet else 0.0
                },
                'pnl': {
                    'unrealized_pnl': total_unrealized_pnl,
                    'realized_pnl': wallet.total_pnl if wallet else 0.0,
                    'total_pnl': (wallet.total_pnl if wallet else 0.0) + total_unrealized_pnl
                },
                'reconciliation': reconciliation,
                'system': {
                    'is_running': self.is_running,
                    'model_trained': self.model_trained,
                    'demo_mode': self.demo_mode
                }
            }
            
            # 6. Log detailed report
            self.logger.info(f"📊 POSITIONS: {open_positions_count}/{TRADING_CONFIG['max_positions']} OPEN "
                           f"({TRADING_CONFIG['max_positions'] - open_positions_count} slots available)")
            
            if position_details:
                self.logger.info("   Position Details:")
                for pos in position_details:
                    self.logger.info(f"   - {pos['symbol']}: {pos['side']}, "
                                   f"Entry=${pos['entry_price']:.6f}, Current=${pos['current_price']:.6f}, "
                                   f"Net PnL=${pos['net_pnl']:.4f} (Gross=${pos['gross_pnl']:.4f}, Costs=${pos['total_costs']:.4f})")
            
            self.logger.info(f"💰 WALLET: Total=${wallet.total_balance if wallet else 0:.2f}, "
                           f"Available=${wallet.available_balance if wallet else 0:.2f}, "
                           f"Locked=${wallet.locked_balance if wallet else 0:.2f}")
            
            self.logger.info(f"📈 PnL: Unrealized=${total_unrealized_pnl:.2f}, "
                           f"Realized=${wallet.total_pnl if wallet else 0:.2f}, "
                           f"Total=${(wallet.total_pnl if wallet else 0.0) + total_unrealized_pnl:.2f}")
            
            if reconciliation['status'] == 'HEALTHY':
                self.logger.info(f"✅ RECONCILIATION: PASSED - Wallet balances are consistent")
            else:
                self.logger.warning(f"⚠️  RECONCILIATION: FAILED - {reconciliation.get('status', 'UNKNOWN')}")
                if 'reconciliation' in reconciliation:
                    self.logger.warning(f"   Balance difference: ${reconciliation['reconciliation'].get('difference', 0):.2f}")
            
            self.logger.info("=" * 80)
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive health check: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
