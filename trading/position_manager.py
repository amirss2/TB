import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

from database.connection import db_connection
from database.models import Position
from trading.coinex_api import CoinExAPI
from config.settings import TP_SL_CONFIG, TRADING_CONFIG

class PositionManager:
    """
    Advanced position management with TP/SL trailing system
    """
    
    def __init__(self, api: CoinExAPI):
        self.api = api
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
        Open a new trading position
        
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
            
            # Debug logging for TP/SL calculation
            self.logger.info(f"TP/SL levels calculated for {symbol}: "
                           f"Entry=${entry_price:.6f}, "
                           f"SL=${tp_sl_levels['initial_sl']:.6f} (-3%), "
                           f"TP1=${tp_sl_levels['tp1']:.6f} (+3%), "
                           f"TP2=${tp_sl_levels['tp2']:.6f} (+6%), "
                           f"TP3=${tp_sl_levels['tp3']:.6f} (+10%)")
            
            # Create position in database
            session = db_connection.get_session()
            
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
            session.commit()
            position_id = position.id
            session.close()
            
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
            
            self.logger.info(f"Position opened successfully: ID {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return None
    
    def close_position(self, position_id: int, reason: str = "Manual close") -> bool:
        """Close a position"""
        try:
            session = db_connection.get_session()
            position = session.query(Position).get(position_id)
            
            if not position or position.status != 'OPEN':
                self.logger.warning(f"Position {position_id} not found or not open")
                session.close()
                return False
            
            # Get current price
            current_price = self._get_current_price(position.symbol)
            if current_price is None:
                self.logger.error(f"Could not get current price for {position.symbol}")
                session.close()
                return False
            
            # Calculate PnL
            pnl_info = self._calculate_pnl(position, current_price)
            
            # Update position in database
            position.current_price = current_price
            position.pnl = pnl_info['pnl']
            position.pnl_percentage = pnl_info['pnl_percentage']
            position.status = 'CLOSED'
            position.closed_at = datetime.now()
            
            session.commit()
            session.close()
            
            # Remove from active monitoring
            if position_id in self.active_positions:
                del self.active_positions[position_id]
            
            self.logger.info(f"Position {position_id} closed: {reason}, PnL: {pnl_info['pnl']:.4f} ({pnl_info['pnl_percentage']:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position {position_id}: {e}")
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
            # Get position from database
            session = db_connection.get_session()
            position = session.query(Position).get(position_id)
            
            if not position or position.status != 'OPEN':
                # Remove from active monitoring
                if position_id in self.active_positions:
                    del self.active_positions[position_id]
                session.close()
                return
            
            # Get current price
            current_price = self._get_current_price(position.symbol)
            if current_price is None:
                session.close()
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
            
            session.commit()
            session.close()
            
        except Exception as e:
            self.logger.error(f"Error checking triggers for position {position_id}: {e}")
    
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
            # TP3 hit (+10%) - move SL to TP2 price and continue progression
            position.tp3_hit = True
            position.current_sl = position.tp2_price  # Move SL to TP2 (+6%)
            self.logger.info(f"TP3 hit for position {position.id} at +10%. SL moved to TP2: {position.tp2_price}")
    
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
            # TP3 hit (-10%) - move SL to TP2 price and continue progression
            position.tp3_hit = True
            position.current_sl = position.tp2_price  # Move SL to TP2 (-6%)
            self.logger.info(f"TP3 hit for position {position.id} at -10%. SL moved to TP2: {position.tp2_price}")
    
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
        """Get current market price for symbol"""
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
                
            return price
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _calculate_pnl(self, position: Position, current_price: float) -> Dict[str, float]:
        """Calculate PnL for position"""
        if position.side == 'LONG':
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_percentage = ((current_price - position.entry_price) / position.entry_price) * 100
        else:  # SHORT
            pnl = (position.entry_price - current_price) * position.quantity
            pnl_percentage = ((position.entry_price - current_price) / position.entry_price) * 100
        
        return {
            'pnl': pnl,
            'pnl_percentage': pnl_percentage
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