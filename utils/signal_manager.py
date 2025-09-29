"""
Signal persistence and WebSocket broadcasting for real-time updates
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import sqlite3
import os

# Try to import WebSocket libraries
try:
    import websocket
    from websockets.server import serve
    import asyncio
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

@dataclass
class StoredSignal:
    """Structure for persisted signals"""
    id: str
    timestamp: datetime
    symbol: str
    direction: str
    confidence: float
    price: float
    threshold_used: float
    features_hash: str
    model_version: str
    status: str = 'active'  # active, expired, executed
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO string
        for key in ['timestamp', 'created_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredSignal':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        for key in ['timestamp', 'created_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        
        return cls(**data)

class SignalPersistence:
    """Handles signal persistence to database"""
    
    def __init__(self, config: Dict[str, Any], db_path: str = "signals.db"):
        self.config = config
        self.signals_config = config.get('signals', {})
        
        self.enabled = self.signals_config.get('persistence_enabled', True)
        self.max_signals = self.signals_config.get('max_signals_history', 1000)
        self.signal_expiry_hours = self.signals_config.get('signal_expiry_hours', 24)
        
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self._init_database()
            self.logger.info("Signal persistence initialized")
        else:
            self.logger.info("Signal persistence disabled")
    
    def _init_database(self):
        """Initialize SQLite database for signal storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price REAL NOT NULL,
                        threshold_used REAL NOT NULL,
                        features_hash TEXT,
                        model_version TEXT,
                        status TEXT DEFAULT 'active',
                        created_at TEXT NOT NULL
                    )
                ''')
                
                # Create indexes for faster queries
                conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON signals(symbol, timestamp DESC)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON signals(status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON signals(created_at DESC)')
                
                conn.commit()
                
            self.logger.info(f"Signal database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize signal database: {e}")
            self.enabled = False
    
    def store_signal(self, signal: StoredSignal) -> bool:
        """Store signal in database"""
        if not self.enabled:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO signals 
                    (id, timestamp, symbol, direction, confidence, price, threshold_used, 
                     features_hash, model_version, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.id,
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.direction,
                    signal.confidence,
                    signal.price,
                    signal.threshold_used,
                    signal.features_hash,
                    signal.model_version,
                    signal.status,
                    signal.created_at.isoformat()
                ))
                conn.commit()
            
            # Cleanup old signals
            self._cleanup_old_signals()
            
            self.logger.debug(f"Signal stored: {signal.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store signal {signal.id}: {e}")
            return False
    
    def get_recent_signals(self, limit: int = 50, symbol: str = None, 
                          hours_back: int = None) -> List[StoredSignal]:
        """Get recent signals from database"""
        if not self.enabled:
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM signals"
                params = []
                conditions = []
                
                if symbol:
                    conditions.append("symbol = ?")
                    params.append(symbol)
                
                if hours_back:
                    cutoff_time = datetime.now() - timedelta(hours=hours_back)
                    conditions.append("timestamp > ?")
                    params.append(cutoff_time.isoformat())
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                signals = []
                for row in rows:
                    signal_data = dict(row)
                    # Convert ISO strings back to datetime
                    signal_data['timestamp'] = datetime.fromisoformat(signal_data['timestamp'])
                    signal_data['created_at'] = datetime.fromisoformat(signal_data['created_at'])
                    
                    signals.append(StoredSignal(**signal_data))
                
                return signals
                
        except Exception as e:
            self.logger.error(f"Failed to get recent signals: {e}")
            return []
    
    def update_signal_status(self, signal_id: str, status: str) -> bool:
        """Update signal status"""
        if not self.enabled:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "UPDATE signals SET status = ? WHERE id = ?",
                    (status, signal_id)
                )
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.debug(f"Signal {signal_id} status updated to {status}")
                    return True
                else:
                    self.logger.warning(f"Signal {signal_id} not found for status update")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to update signal status: {e}")
            return False
    
    def _cleanup_old_signals(self):
        """Clean up old signals to maintain database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove expired signals
                if self.signal_expiry_hours > 0:
                    expiry_time = datetime.now() - timedelta(hours=self.signal_expiry_hours)
                    conn.execute(
                        "DELETE FROM signals WHERE timestamp < ? AND status = 'active'",
                        (expiry_time.isoformat(),)
                    )
                
                # Limit total number of signals
                conn.execute('''
                    DELETE FROM signals WHERE id IN (
                        SELECT id FROM signals 
                        ORDER BY timestamp DESC 
                        LIMIT -1 OFFSET ?
                    )
                ''', (self.max_signals,))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old signals: {e}")
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal statistics"""
        if not self.enabled:
            return {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total signals
                total_cursor = conn.execute("SELECT COUNT(*) FROM signals")
                total_signals = total_cursor.fetchone()[0]
                
                # Signals by status
                status_cursor = conn.execute(
                    "SELECT status, COUNT(*) FROM signals GROUP BY status"
                )
                status_counts = dict(status_cursor.fetchall())
                
                # Signals by symbol (recent)
                symbol_cursor = conn.execute('''
                    SELECT symbol, COUNT(*) FROM signals 
                    WHERE timestamp > ? 
                    GROUP BY symbol
                ''', ((datetime.now() - timedelta(hours=24)).isoformat(),))
                symbol_counts = dict(symbol_cursor.fetchall())
                
                # Recent signal rate
                recent_cursor = conn.execute('''
                    SELECT COUNT(*) FROM signals 
                    WHERE timestamp > ?
                ''', ((datetime.now() - timedelta(hours=1)).isoformat(),))
                recent_signals = recent_cursor.fetchone()[0]
                
                return {
                    'total_signals': total_signals,
                    'status_distribution': status_counts,
                    'symbol_distribution_24h': symbol_counts,
                    'signals_last_hour': recent_signals,
                    'signal_rate_per_hour': recent_signals
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get signal stats: {e}")
            return {}

class WebSocketBroadcaster:
    """WebSocket broadcaster for real-time updates"""
    
    def __init__(self, config: Dict[str, Any], port: int = 8765):
        self.config = config
        self.signals_config = config.get('signals', {})
        
        self.enabled = self.signals_config.get('websocket_broadcast', True) and HAS_WEBSOCKETS
        self.port = port
        
        self.connected_clients = set()
        self.server = None
        self.server_task = None
        self.loop = None
        
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"WebSocket broadcaster initialized on port {port}")
        else:
            if not HAS_WEBSOCKETS:
                self.logger.warning("WebSockets not available - real-time updates disabled")
            else:
                self.logger.info("WebSocket broadcasting disabled")
    
    def start(self):
        """Start WebSocket server in background thread"""
        if not self.enabled:
            return
        
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            async def handle_client(websocket, path):
                self.connected_clients.add(websocket)
                self.logger.info(f"WebSocket client connected: {websocket.remote_address}")
                
                try:
                    # Send initial connection message
                    await websocket.send(json.dumps({
                        'type': 'connection',
                        'message': 'Connected to trading bot signals',
                        'timestamp': datetime.now().isoformat()
                    }))
                    
                    # Keep connection alive
                    async for message in websocket:
                        # Handle client messages if needed
                        pass
                        
                except Exception as e:
                    self.logger.error(f"WebSocket client error: {e}")
                finally:
                    self.connected_clients.discard(websocket)
                    self.logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
            
            async def start_server():
                self.server = await serve(handle_client, "localhost", self.port)
                self.logger.info(f"WebSocket server started on ws://localhost:{self.port}")
                await self.server.wait_closed()
            
            self.loop.run_until_complete(start_server())
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        import time
        time.sleep(1)
    
    def stop(self):
        """Stop WebSocket server"""
        if not self.enabled or not self.server:
            return
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.server.close(), self.loop)
        
        self.logger.info("WebSocket server stopped")
    
    def broadcast_signal(self, signal: StoredSignal):
        """Broadcast new signal to all connected clients"""
        if not self.enabled or not self.connected_clients:
            return
        
        message = {
            'type': 'new_signal',
            'data': signal.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        self._broadcast_message(message)
    
    def broadcast_prediction(self, prediction_data: Dict[str, Any]):
        """Broadcast prediction update to all connected clients"""
        if not self.enabled or not self.connected_clients:
            return
        
        message = {
            'type': 'prediction_update',
            'data': prediction_data,
            'timestamp': datetime.now().isoformat()
        }
        
        self._broadcast_message(message)
    
    def broadcast_status_update(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        if not self.enabled or not self.connected_clients:
            return
        
        message = {
            'type': 'status_update',
            'data': status_data,
            'timestamp': datetime.now().isoformat()
        }
        
        self._broadcast_message(message)
    
    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.loop:
            return
        
        async def send_to_all():
            if not self.connected_clients:
                return
            
            message_json = json.dumps(message)
            disconnected_clients = set()
            
            for client in self.connected_clients:
                try:
                    await client.send(message_json)
                except Exception as e:
                    self.logger.warning(f"Failed to send to client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
        
        asyncio.run_coroutine_threadsafe(send_to_all(), self.loop)
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket broadcaster status"""
        return {
            'enabled': self.enabled,
            'connected_clients': len(self.connected_clients) if self.enabled else 0,
            'port': self.port,
            'server_running': self.server is not None
        }

class SignalManager:
    """Main signal management class combining persistence and broadcasting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.persistence = SignalPersistence(config)
        self.broadcaster = WebSocketBroadcaster(config)
        
        self.logger.info("Signal manager initialized")
    
    def start(self):
        """Start signal management services"""
        self.broadcaster.start()
        self.logger.info("Signal management services started")
    
    def stop(self):
        """Stop signal management services"""
        self.broadcaster.stop()
        self.logger.info("Signal management services stopped")
    
    def handle_new_signal(self, signal_data: Dict[str, Any]):
        """Handle new signal from prediction scheduler"""
        try:
            # Create stored signal
            signal = StoredSignal(
                id=signal_data['id'],
                timestamp=signal_data['timestamp'],
                symbol=signal_data['symbol'],
                direction=signal_data['direction'],
                confidence=signal_data['confidence'],
                price=signal_data['price'],
                threshold_used=signal_data['threshold_used'],
                features_hash=signal_data['features_hash'],
                model_version=signal_data['model_version']
            )
            
            # Store signal
            if self.persistence.store_signal(signal):
                self.logger.info(f"Signal stored: {signal.symbol} {signal.direction}")
                
                # Broadcast to WebSocket clients
                self.broadcaster.broadcast_signal(signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to handle new signal: {e}")
            return False
    
    def get_recent_signals(self, limit: int = 50, symbol: str = None) -> List[Dict[str, Any]]:
        """Get recent signals for API"""
        signals = self.persistence.get_recent_signals(limit=limit, symbol=symbol)
        return [signal.to_dict() for signal in signals]
    
    def update_signal_status(self, signal_id: str, status: str) -> bool:
        """Update signal status"""
        return self.persistence.update_signal_status(signal_id, status)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive signal statistics"""
        stats = self.persistence.get_signal_stats()
        websocket_status = self.broadcaster.get_status()
        
        return {
            'persistence': stats,
            'websocket': websocket_status,
            'total_active_signals': len(self.get_recent_signals(hours_back=24))
        }
    
    def broadcast_prediction_update(self, prediction_data: Dict[str, Any]):
        """Broadcast prediction update"""
        self.broadcaster.broadcast_prediction(prediction_data)
    
    def broadcast_status_update(self, status_data: Dict[str, Any]):
        """Broadcast system status update"""
        self.broadcaster.broadcast_status_update(status_data)