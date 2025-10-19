import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

from database.connection import db_connection
from database.models import Position, Wallet, WalletTransaction
from trading.coinex_api import CoinExAPI
from config.settings import TP_SL_CONFIG, TRADING_CONFIG, FEE_CONFIG

class PositionManager:
    """
    Advanced position management with TP/SL trailing system
    """
    
    def __init__(self, api: CoinExAPI, trading_engine=None):
        self.api = api
        self.trading_engine = trading_engine  # Reference to trading engine for balance management
        self.logger = logging.getLogger(__name__)
        self.active_positions = {}
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # TP/SL configuration
        self.tp1_percent = TP_SL_CONFIG['tp1_percent']
        self.tp2_percent = TP_SL_CONFIG['tp2_percent']
        self.tp3_percent = TP_SL_CONFIG['tp3_percent']
        self.initial_sl_percent = TP_SL_CONFIG['initial_sl_percent']
        self.trailing_enabled = TP_SL_CONFIG['trailing_enabled']
    
    def open_position(self, symbol: str, side: str, quantity: float, 
                     entry_price: float, signal_confidence: float) -> Optional[int]:
        """
        Open a new trading position with atomic wallet transaction
        
        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            quantity: Position size
            entry_price: Entry price
            signal_confidence: AI model confidence
            
        Returns:
            Position ID if successful, None otherwise
        """
        try:
            self.logger.info(f"Opening {side} position for {symbol}: {quantity} @ {entry_price}")
            
            # Calculate TP/SL levels
            tp_sl_levels = self._calculate_tp_sl_levels(entry_price, side)
            position_value = entry_price * quantity
            
            # Debug logging for TP/SL calculation
            self.logger.info(f"TP/SL levels calculated for {symbol}: "
                           f"Entry=${entry_price:.6f}, Value=${position_value:.2f}, "
                           f"SL=${tp_sl_levels['initial_sl']:.6f} (-3%), "
                           f"TP1=${tp_sl_levels['tp1']:.6f} (+3%), "
                           f"TP2=${tp_sl_levels['tp2']:.6f} (+6%), "
                           f"TP3=${tp_sl_levels['tp3']:.6f} (+10%)")
            
            # Use atomic transaction to create position and update wallet
            with db_connection.get_transaction_session() as session:
                # Create position
                position = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    current_price=entry_price,
                    initial_sl=tp_sl_levels['initial_sl'],
                    current_sl=tp_sl_levels['initial_sl'],
                    tp1_price=tp_sl_levels['tp1'],
                    tp2_price=tp_sl_levels['tp2'],
                    tp3_price=tp_sl_levels['tp3'],
                    status='OPEN'
                )
                
                session.add(position)
                session.flush()  # Get position ID before commit
                position_id = position.id
                
                # Update wallet within same transaction
                account_type = 'demo' if (self.trading_engine and self.trading_engine.demo_mode) else 'live'
                wallet = session.query(Wallet).filter_by(account_type=account_type).with_for_update().first()
                
                if wallet:
                    balance_before = wallet.available_balance
                    
                    # Lock funds for position
                    wallet.locked_balance += position_value
                    wallet.available_balance -= position_value
                    
                    # Ensure available_balance doesn't go negative
                    if wallet.available_balance < 0:
                        self.logger.error(f"Insufficient balance: need ${position_value:.2f}, have ${balance_before:.2f}")
                        raise ValueError("Insufficient available balance")
                    
                    balance_after = wallet.available_balance
                    
                    # Record wallet transaction
                    transaction = WalletTransaction(
                        account_type=account_type,
                        transaction_type='position_open',
                        amount=-position_value,  # Negative for locked funds
                        balance_before=balance_before,
                        balance_after=balance_after,
                        position_id=position_id,
                        description=f"Opened {side} position for {symbol}: {quantity} @ ${entry_price:.6f}"
                    )
                    session.add(transaction)
                    
                    self.logger.info(f"Wallet updated: Locked ${position_value:.2f}, Available: ${balance_before:.2f} → ${balance_after:.2f}")
                
                # Commit happens automatically via context manager
            
            # Add to active positions for monitoring
            self.active_positions[position_id] = {
                'symbol': symbol,
                'side': side,
                'entry_price': entry_price,
                'quantity': quantity,
                'tp_sl_levels': tp_sl_levels,
                'confidence': signal_confidence
            }
            
            # Start monitoring if not already running
            if not self.monitoring_thread or not self.monitoring_thread.is_alive():
                self.start_position_monitoring()
            
            self.logger.info(f"✓ Position opened successfully: ID {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}", exc_info=True)
            return None
    
    def close_position(self, position_id: int, reason: str = "Manual close") -> bool:
        """
        Close a position and free up allocated balance
        Uses atomic transaction to ensure data consistency
        """
        # Idempotency key to prevent double closing
        close_key = f"close_{position_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if not hasattr(self, '_closed_positions'):
            self._closed_positions = set()
        
        if position_id in self._closed_positions:
            self.logger.warning(f"Position {position_id} already closed, skipping duplicate close")
            return False
        
        try:
            # Use transaction context manager for atomic operations
            with db_connection.get_transaction_session() as session:
                # Use merge to attach position to current session
                position = session.query(Position).filter_by(id=position_id).with_for_update().first()
                
                if not position or position.status != 'OPEN':
                    self.logger.warning(f"Position {position_id} not found or not open")
                    return False
                
                # Get current price
                current_price = self._get_current_price(position.symbol)
                if current_price is None:
                    self.logger.error(f"Could not get current price for {position.symbol} - cannot close position safely")
                    return False
                
                # Calculate PnL with all costs
                pnl_info = self._calculate_pnl(position, current_price)
                
                # Calculate position value to free from used_balance
                position_value = position.entry_price * position.quantity
                
                # Update position in database (within same transaction)
                position.current_price = current_price
                position.pnl = pnl_info['pnl']
                position.pnl_percentage = pnl_info['pnl_percentage']
                position.status = 'CLOSED'
                position.closed_at = datetime.now()
                
                # Update wallet within same transaction
                account_type = 'demo' if (self.trading_engine and self.trading_engine.demo_mode) else 'live'
                wallet = session.query(Wallet).filter_by(account_type=account_type).with_for_update().first()
                
                if wallet:
                    balance_before = wallet.available_balance
                    
                    # Unlock funds and apply PnL
                    wallet.locked_balance = max(0, wallet.locked_balance - position_value)
                    wallet.available_balance = wallet.available_balance + position_value + pnl_info['pnl']
                    wallet.total_pnl += pnl_info['pnl']
                    
                    # Ensure available_balance is never negative
                    if wallet.available_balance < 0:
                        self.logger.error(f"Available balance would become negative ({wallet.available_balance:.2f}), clamping to 0")
                        wallet.available_balance = 0
                    
                    balance_after = wallet.available_balance
                    
                    # Record wallet transaction
                    transaction = WalletTransaction(
                        account_type=account_type,
                        transaction_type='position_close',
                        amount=pnl_info['pnl'],  # Record net PnL
                        balance_before=balance_before,
                        balance_after=balance_after,
                        position_id=position_id,
                        description=f"Closed {position.side} {position.symbol}: {reason}, Net PnL=${pnl_info['pnl']:.2f} (Gross=${pnl_info['gross_pnl']:.2f}, Costs=${pnl_info['total_costs']:.2f})"
                    )
                    session.add(transaction)
                    
                    self.logger.info(f"Wallet updated: Freed ${position_value:.2f}, PnL=${pnl_info['pnl']:.2f}, "
                                   f"Available: ${balance_before:.2f} → ${balance_after:.2f}")
                
                # Commit happens automatically via context manager
            
            # Update in-memory state after successful commit
            if position_id in self.active_positions:
                del self.active_positions[position_id]
            
            # Free up allocated balance in trading engine
            if self.trading_engine and hasattr(self.trading_engine, 'used_balance'):
                self.trading_engine.used_balance = max(0, self.trading_engine.used_balance - position_value)
                self.logger.info(f"Freed ${position_value:.2f} from used_balance. New used_balance: ${self.trading_engine.used_balance:.2f}")
            
            # Mark as closed for idempotency
            self._closed_positions.add(position_id)
            
            self.logger.info(f"✓ Position {position_id} closed: {reason}, Net PnL: ${pnl_info['pnl']:.4f} ({pnl_info['pnl_percentage']:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}", exc_info=True)
            return False
    
    def emergency_close_position(self, position_id: int, signal_confidence: float) -> bool:
        """
        Emergency close when opposite signal with high confidence is received
        """
        if signal_confidence >= TRADING_CONFIG['confidence_threshold']:
            return self.close_position(position_id, f"Emergency close - opposite signal {signal_confidence:.2f}")
        return False
    
    def start_position_monitoring(self):
        """Start the position monitoring thread"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_positions, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Position monitoring started")
    
    def stop_position_monitoring(self):
        """Stop position monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Position monitoring stopped")
    
    def _monitor_positions(self):
        """Monitor all active positions for TP/SL triggers"""
        while not self.stop_monitoring:
            try:
                if not self.active_positions:
                    time.sleep(5)  # Sleep longer if no positions
                    continue
                
                for position_id in list(self.active_positions.keys()):
                    try:
                        self._check_position_triggers(position_id)
                    except Exception as e:
                        self.logger.error(f"Error monitoring position {position_id}: {e}")
                
                time.sleep(1)  # Check every second as specified
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring loop: {e}")
                time.sleep(5)
    
    def _check_position_triggers(self, position_id: int):
        """Check TP/SL triggers for a specific position"""
        try:
            # CRITICAL: Skip position checks if trading engine is paused due to network issues
            if self.trading_engine and hasattr(self.trading_engine, 'trading_paused_by_network'):
                if self.trading_engine.trading_paused_by_network:
                    # System is paused, don't check positions
                    return
            
            # Use transaction context for atomic updates
            with db_connection.get_transaction_session() as session:
                # Use merge to ensure position is bound to current session
                position = session.query(Position).filter_by(id=position_id).first()
                
                if not position or position.status != 'OPEN':
                    # Remove from active monitoring
                    if position_id in self.active_positions:
                        del self.active_positions[position_id]
                    return
                
                # Get current price
                current_price = self._get_current_price(position.symbol)
                if current_price is None:
                    # CRITICAL FIX: Don't process position if we can't get a valid price
                    # This prevents closing positions with incorrect/stale prices during network issues
                    self.logger.warning(f"Cannot get valid price for {position.symbol}, skipping position check to prevent incorrect closure")
                    return
                
                # Debug logging for position status
                if datetime.now().second % 30 == 0:  # Log every 30 seconds to avoid spam
                    self.logger.info(f"Monitoring position {position_id} ({position.symbol}): "
                                    f"Entry=${position.entry_price:.6f}, "
                                    f"Current=${current_price:.6f}, "
                                    f"SL=${position.current_sl:.6f}, "
                                    f"TP1=${position.tp1_price:.6f}")
                
                # Update current price
                position.current_price = current_price
                position.updated_at = datetime.now()
                
                # Check TP/SL triggers based on position side
                if position.side == 'LONG':
                    self._check_long_position_triggers(position, current_price)
                else:
                    self._check_short_position_triggers(position, current_price)
                
                # Commit happens automatically via context manager
            
        except Exception as e:
            self.logger.error(f"Error checking triggers for position {position_id}: {e}", exc_info=True)
    
    def _check_long_position_triggers(self, position: Position, current_price: float):
        """Check triggers for LONG positions with user's specific TP/SL progression"""
        # Check Stop Loss
        if current_price <= position.current_sl:
            self.logger.info(f"SL triggered for position {position.id}: {current_price} <= {position.current_sl}")
            self.close_position(position.id, "Stop Loss triggered")
            return
        
        # Check Take Profit levels and update trailing SL according to user specifications
        if not position.tp1_hit and current_price >= position.tp1_price:
            # TP1 hit (+3%) - move SL to entry price (breakeven) and set TP2 at +6%
            position.tp1_hit = True
            position.current_sl = position.entry_price  # Move SL to breakeven
            position.tp2_price = position.entry_price * (1 + 6.0 / 100)  # TP2 at +6% from entry
            self.logger.info(f"TP1 hit for position {position.id} at +3%. SL moved to entry: {position.entry_price}, TP2 set to +6%: {position.tp2_price}")
        
        elif position.tp1_hit and not position.tp2_hit and current_price >= position.tp2_price:
            # TP2 hit (+6%) - move SL to TP1 price and set TP3 at +10%
            position.tp2_hit = True
            position.current_sl = position.tp1_price  # Move SL to TP1 (+3%)
            position.tp3_price = position.entry_price * (1 + 10.0 / 100)  # TP3 at +10% from entry
            self.logger.info(f"TP2 hit for position {position.id} at +6%. SL moved to TP1: {position.tp1_price}, TP3 set to +10%: {position.tp3_price}")
        
        elif position.tp2_hit and not position.tp3_hit and current_price >= position.tp3_price:
            # TP3 hit (+10%) - CLOSE POSITION IMMEDIATELY
            position.tp3_hit = True
            self.logger.info(f"TP3 hit for position {position.id} at +10%. Closing position immediately.")
            self.close_position(position.id, "TP3 (+10%) reached")
    
    def _check_short_position_triggers(self, position: Position, current_price: float):
        """Check triggers for SHORT positions with user's specific TP/SL progression"""
        # Check Stop Loss
        if current_price >= position.current_sl:
            self.logger.info(f"SL triggered for position {position.id}: {current_price} >= {position.current_sl}")
            self.close_position(position.id, "Stop Loss triggered")
            return
        
        # Check Take Profit levels and update trailing SL according to user specifications
        if not position.tp1_hit and current_price <= position.tp1_price:
            # TP1 hit (-3%) - move SL to entry price (breakeven) and set TP2 at -6%
            position.tp1_hit = True
            position.current_sl = position.entry_price  # Move SL to breakeven
            position.tp2_price = position.entry_price * (1 - 6.0 / 100)  # TP2 at -6% from entry
            self.logger.info(f"TP1 hit for position {position.id} at -3%. SL moved to entry: {position.entry_price}, TP2 set to -6%: {position.tp2_price}")
        
        elif position.tp1_hit and not position.tp2_hit and current_price <= position.tp2_price:
            # TP2 hit (-6%) - move SL to TP1 price and set TP3 at -10%
            position.tp2_hit = True
            position.current_sl = position.tp1_price  # Move SL to TP1 (-3%)
            position.tp3_price = position.entry_price * (1 - 10.0 / 100)  # TP3 at -10% from entry
            self.logger.info(f"TP2 hit for position {position.id} at -6%. SL moved to TP1: {position.tp1_price}, TP3 set to -10%: {position.tp3_price}")
        
        elif position.tp2_hit and not position.tp3_hit and current_price <= position.tp3_price:
            # TP3 hit (-10%) - CLOSE POSITION IMMEDIATELY
            position.tp3_hit = True
            self.logger.info(f"TP3 hit for position {position.id} at -10%. Closing position immediately.")
            self.close_position(position.id, "TP3 (-10%) reached")
    
    def _calculate_tp_sl_levels(self, entry_price: float, side: str) -> Dict[str, float]:
        """Calculate TP/SL levels based on entry price and side"""
        if side == 'LONG':
            return {
                'tp1': entry_price * (1 + self.tp1_percent / 100),  # +3% from entry
                'tp2': entry_price * (1 + self.tp2_percent / 100),  # +6% from entry  
                'tp3': entry_price * (1 + self.tp3_percent / 100),  # +10% from entry
                'initial_sl': entry_price * (1 - self.initial_sl_percent / 100)  # -3% from entry
            }
        else:  # SHORT
            return {
                'tp1': entry_price * (1 - self.tp1_percent / 100),  # -3% from entry
                'tp2': entry_price * (1 - self.tp2_percent / 100),  # -6% from entry
                'tp3': entry_price * (1 - self.tp3_percent / 100),  # -10% from entry
                'initial_sl': entry_price * (1 + self.initial_sl_percent / 100)  # +3% from entry
            }
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol with validation to prevent incorrect closures"""
        try:
            ticker = self.api.get_ticker(symbol)
            
            # Handle both API response formats (direct and nested)
            if 'ticker' in ticker and isinstance(ticker['ticker'], dict):
                # Nested format from fallback data
                price = float(ticker['ticker'].get('last', 0))
            else:
                # Direct format from live API
                price = float(ticker.get('last', 0))
            
            if price == 0:
                self.logger.warning(f"Got zero price for {symbol}, ticker data: {ticker}")
                return None
            
            # CRITICAL FIX: Additional validation to prevent stale/incorrect prices
            # Store last valid price per symbol to detect unreasonable price changes
            if not hasattr(self, '_last_valid_prices'):
                self._last_valid_prices = {}
            
            if symbol in self._last_valid_prices:
                last_price = self._last_valid_prices[symbol]
                price_change_pct = abs((price - last_price) / last_price) * 100
                
                # If price changed more than 50% in one check, it's likely incorrect
                if price_change_pct > 50:
                    self.logger.error(f"Suspicious price change for {symbol}: {last_price:.6f} -> {price:.6f} ({price_change_pct:.1f}%). "
                                    f"Rejecting price to prevent incorrect position closure.")
                    return None
            
            # Update last valid price
            self._last_valid_prices[symbol] = price
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_pnl(self, position: Position, current_price: float) -> Dict[str, float]:
        """
        Calculate PnL for position with comprehensive fee/spread/slippage calculation
        
        This includes:
        - Entry fee (taker fee when opening position)
        - Exit fee (taker fee when closing position)
        - Spread cost (bid/ask spread)
        - Slippage (market order execution difference)
        """
        # Get fee configuration
        maker_fee = FEE_CONFIG['spot_trading']['maker_fee']
        taker_fee = FEE_CONFIG['spot_trading']['taker_fee']
        spread_pct = FEE_CONFIG['spread']['estimate_pct']
        slippage_pct = FEE_CONFIG['slippage']['estimate_pct']
        
        # Calculate position value at entry and exit
        entry_value = position.entry_price * position.quantity
        exit_value = current_price * position.quantity
        
        # Calculate gross PnL (before fees)
        if position.side == 'LONG':
            gross_pnl = exit_value - entry_value
            gross_pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # SHORT
            gross_pnl = entry_value - exit_value
            gross_pnl_percentage = ((position.entry_price - current_price) / position.entry_price) * 100
        
        # Calculate all costs
        entry_fee = entry_value * taker_fee  # Fee when opening position
        exit_fee = exit_value * taker_fee    # Fee when closing position
        spread_cost = (entry_value + exit_value) / 2 * spread_pct  # Spread cost (avg of entry and exit)
        slippage_cost = (entry_value + exit_value) / 2 * slippage_pct  # Slippage cost
        
        # Total costs
        total_costs = entry_fee + exit_fee + spread_cost + slippage_cost
        
        # Net PnL (after all fees and costs)
        net_pnl = gross_pnl - total_costs
        net_pnl_percentage = (net_pnl / entry_value) * 100 if entry_value > 0 else 0
        
        # Log detailed breakdown
        self.logger.info(
            f"PnL Calculation for {position.symbol}: "
            f"Gross PnL=${gross_pnl:.4f} ({gross_pnl_percentage:.2f}%), "
            f"Entry Fee=${entry_fee:.4f}, Exit Fee=${exit_fee:.4f}, "
            f"Spread=${spread_cost:.4f}, Slippage=${slippage_cost:.4f}, "
            f"Total Costs=${total_costs:.4f}, "
            f"Net PnL=${net_pnl:.4f} ({net_pnl_percentage:.2f}%)"
        )
        
        return {
            'pnl': net_pnl,  # Net PnL after all costs
            'pnl_percentage': net_pnl_percentage,
            'gross_pnl': gross_pnl,
            'gross_pnl_percentage': gross_pnl_percentage,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'spread_cost': spread_cost,
            'slippage_cost': slippage_cost,
            'total_costs': total_costs
        }
    
    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active positions"""
        try:
            session = db_connection.get_session()
            positions = session.query(Position).filter(Position.status == 'OPEN').all()
            
            result = []
            for position in positions:
                current_price = self._get_current_price(position.symbol)
                if current_price:
                    pnl_info = self._calculate_pnl(position, current_price)
                    
                    result.append({
                        'id': position.id,
                        'symbol': position.symbol,
                        'side': position.side,
                        'entry_price': position.entry_price,
                        'current_price': current_price,
                        'quantity': position.quantity,
                        'pnl': pnl_info['pnl'],
                        'pnl_percentage': pnl_info['pnl_percentage'],
                        'current_sl': position.current_sl,
                        'tp1_price': position.tp1_price,
                        'tp2_price': position.tp2_price,
                        'tp3_price': position.tp3_price,
                        'tp1_hit': position.tp1_hit,
                        'tp2_hit': position.tp2_hit,
                        'tp3_hit': position.tp3_hit,
                        'opened_at': position.opened_at
                    })
            
            session.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions"""
        active_positions = self.get_active_positions()
        
        total_pnl = sum(pos['pnl'] for pos in active_positions)
        total_positions = len(active_positions)
        
        return {
            'total_active_positions': total_positions,
            'total_unrealized_pnl': total_pnl,
            'positions': active_positions
        }
    
    def wallet_health_check(self) -> Dict[str, Any]:
        """
        Health check to verify wallet balance reconciliation
        Ensures: sum(wallet_transactions.amount) + initial_balance == wallet.total_balance
        """
        try:
            session = db_connection.get_session()
            account_type = 'demo' if (self.trading_engine and self.trading_engine.demo_mode) else 'live'
            
            # Get wallet
            wallet = session.query(Wallet).filter_by(account_type=account_type).first()
            if not wallet:
                session.close()
                return {
                    'status': 'ERROR',
                    'message': f'Wallet not found for {account_type}'
                }
            
            # Get all transactions
            transactions = session.query(WalletTransaction).filter_by(account_type=account_type).all()
            
            # Calculate expected balance
            initial_balance = TRADING_CONFIG['demo_balance'] if account_type == 'demo' else 0.0
            transaction_sum = sum(t.amount for t in transactions)
            expected_balance = initial_balance + transaction_sum
            
            # Check if balances match
            balance_match = abs(expected_balance - wallet.total_balance) < 0.01  # Allow 1 cent difference
            
            # Calculate available_balance expectation
            open_positions = session.query(Position).filter_by(status='OPEN').all()
            locked_in_positions = sum(pos.entry_price * pos.quantity for pos in open_positions)
            expected_available = wallet.total_balance - locked_in_positions
            available_match = abs(expected_available - wallet.available_balance) < 0.01
            
            session.close()
            
            health_status = {
                'status': 'HEALTHY' if (balance_match and available_match) else 'MISMATCH',
                'account_type': account_type,
                'wallet': {
                    'total_balance': wallet.total_balance,
                    'available_balance': wallet.available_balance,
                    'locked_balance': wallet.locked_balance,
                    'total_pnl': wallet.total_pnl
                },
                'reconciliation': {
                    'initial_balance': initial_balance,
                    'transaction_count': len(transactions),
                    'transaction_sum': transaction_sum,
                    'expected_total_balance': expected_balance,
                    'balance_match': balance_match,
                    'difference': wallet.total_balance - expected_balance
                },
                'positions': {
                    'open_count': len(open_positions),
                    'locked_in_positions': locked_in_positions,
                    'expected_available': expected_available,
                    'available_match': available_match,
                    'available_difference': wallet.available_balance - expected_available
                }
            }
            
            if health_status['status'] == 'HEALTHY':
                self.logger.info(f"✓ Wallet health check PASSED: Total=${wallet.total_balance:.2f}, Available=${wallet.available_balance:.2f}")
            else:
                self.logger.warning(f"⚠ Wallet health check FAILED: {health_status}")
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in wallet health check: {e}", exc_info=True)
            return {
                'status': 'ERROR',
                'message': str(e)
            }