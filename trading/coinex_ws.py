import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from config.settings import TRADING_CONFIG

class CoinExWebSocket:
    """
    CoinEx WebSocket client for real-time market data
    Handles reconnection, rate limiting, and subscription management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ws_url = "wss://socket.coinex.com/v2/spot/"
        self.websocket = None
        self.is_connected = False
        self.is_running = False
        
        # Configuration
        self.max_subscriptions = TRADING_CONFIG['ws_max_subscriptions_tier1']
        self.channels = TRADING_CONFIG['ws_channels']
        self.backoff_base = TRADING_CONFIG['backoff_base_sec']
        self.backoff_max = TRADING_CONFIG['backoff_max_sec']
        
        # State management
        self.subscribed_symbols = set()
        self.subscriptions = {}  # channel -> symbols mapping
        self.message_queue = asyncio.Queue()
        self.reconnect_count = 0
        self.last_ping = 0
        self.ping_interval = 30  # seconds
        
        # Event handlers
        self.on_ticker = None
        self.on_kline = None
        self.on_error = None
        
    async def connect(self):
        """Establish WebSocket connection with exponential backoff"""
        while self.is_running:
            try:
                self.logger.info("Attempting to connect to CoinEx WebSocket...")
                self.websocket = await websockets.connect(
                    self.ws_url,
                    ping_interval=self.ping_interval,
                    ping_timeout=10,
                    close_timeout=10
                )
                
                self.is_connected = True
                self.reconnect_count = 0
                self.logger.info("Connected to CoinEx WebSocket")
                
                # Start message handler
                await asyncio.gather(
                    self._message_handler(),
                    self._heartbeat_handler(),
                    return_exceptions=True
                )
                
            except Exception as e:
                self.logger.error(f"WebSocket connection failed: {e}")
                self.is_connected = False
                
                if self.is_running:
                    await self._backoff_delay()
                    continue
                else:
                    break
    
    async def disconnect(self):
        """Gracefully disconnect WebSocket"""
        self.is_running = False
        self.is_connected = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket: {e}")
        
        self.logger.info("WebSocket disconnected")
    
    async def start(self):
        """Start WebSocket connection and message handling"""
        self.is_running = True
        await self.connect()
    
    async def stop(self):
        """Stop WebSocket connection"""
        await self.disconnect()
    
    async def subscribe_ticker(self, symbols: List[str]):
        """Subscribe to ticker updates for symbols"""
        if not symbols:
            return
            
        # Limit subscriptions to avoid overwhelming
        symbols = symbols[:self.max_subscriptions]
        
        for symbol in symbols:
            if symbol not in self.subscribed_symbols and len(self.subscribed_symbols) < self.max_subscriptions:
                await self._subscribe("state.update", [symbol])
                self.subscribed_symbols.add(symbol)
                
        self.subscriptions['ticker'] = list(self.subscribed_symbols)
        self.logger.info(f"Subscribed to ticker for {len(self.subscribed_symbols)} symbols")
    
    async def subscribe_kline(self, symbols: List[str], interval: str = "1m"):
        """Subscribe to kline/candlestick updates for symbols"""
        if not symbols:
            return
            
        # Limit subscriptions for klines (more data intensive)
        max_kline_subs = min(self.max_subscriptions // 2, len(symbols))
        symbols = symbols[:max_kline_subs]
        
        for symbol in symbols:
            await self._subscribe("deals.update", [symbol])
                
        self.subscriptions['kline'] = symbols
        self.logger.info(f"Subscribed to kline for {len(symbols)} symbols")
    
    async def unsubscribe_all(self):
        """Unsubscribe from all channels"""
        for channel, symbols in self.subscriptions.items():
            for symbol in symbols:
                await self._unsubscribe("state.update" if channel == "ticker" else "deals.update", [symbol])
        
        self.subscriptions.clear()
        self.subscribed_symbols.clear()
        self.logger.info("Unsubscribed from all channels")
    
    async def _subscribe(self, method: str, params: List[str]):
        """Send subscription message"""
        if not self.is_connected:
            return
            
        message = {
            "method": method,
            "params": params,
            "id": int(time.time() * 1000)
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            self.logger.debug(f"Sent subscription: {method} for {params}")
        except Exception as e:
            self.logger.error(f"Failed to send subscription: {e}")
    
    async def _unsubscribe(self, method: str, params: List[str]):
        """Send unsubscription message"""
        if not self.is_connected:
            return
            
        message = {
            "method": method.replace("update", "unsubscribe"),
            "params": params,
            "id": int(time.time() * 1000)
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            self.logger.debug(f"Sent unsubscription: {method} for {params}")
        except Exception as e:
            self.logger.error(f"Failed to send unsubscription: {e}")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Invalid JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"Message handler error: {e}")
            self.is_connected = False
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        method = data.get("method")
        params = data.get("params", [])
        
        if method == "state.update" and len(params) >= 2:
            # Ticker update
            symbol = params[0]
            ticker_data = params[1]
            
            if self.on_ticker:
                try:
                    await self.on_ticker(symbol, ticker_data)
                except Exception as e:
                    self.logger.error(f"Error in ticker handler: {e}")
        
        elif method == "deals.update" and len(params) >= 2:
            # Trade/kline update
            symbol = params[0]
            deals_data = params[1]
            
            if self.on_kline:
                try:
                    await self.on_kline(symbol, deals_data)
                except Exception as e:
                    self.logger.error(f"Error in kline handler: {e}")
        
        # Add to queue for other consumers
        try:
            self.message_queue.put_nowait({
                'timestamp': time.time(),
                'method': method,
                'data': data
            })
        except asyncio.QueueFull:
            # Drop oldest message if queue is full
            try:
                self.message_queue.get_nowait()
                self.message_queue.put_nowait({
                    'timestamp': time.time(),
                    'method': method,
                    'data': data
                })
            except asyncio.QueueEmpty:
                pass
    
    async def _heartbeat_handler(self):
        """Send periodic heartbeat/ping"""
        while self.is_connected and self.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_ping > self.ping_interval:
                    if self.websocket and not self.websocket.closed:
                        pong_waiter = await self.websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=5)
                        self.last_ping = current_time
                        self.logger.debug("WebSocket ping/pong successful")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.TimeoutError:
                self.logger.warning("WebSocket ping timeout")
                self.is_connected = False
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                self.is_connected = False
                break
    
    async def _backoff_delay(self):
        """Calculate and apply exponential backoff delay"""
        delay = min(
            self.backoff_base * (2 ** self.reconnect_count),
            self.backoff_max
        )
        
        self.reconnect_count += 1
        self.logger.info(f"Reconnecting in {delay} seconds (attempt {self.reconnect_count})")
        await asyncio.sleep(delay)
    
    async def get_message(self) -> Optional[Dict[str, Any]]:
        """Get next message from queue (non-blocking)"""
        try:
            return self.message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def wait_for_message(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Wait for next message from queue with timeout"""
        try:
            return await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def set_ticker_handler(self, handler: Callable):
        """Set ticker message handler"""
        self.on_ticker = handler
    
    def set_kline_handler(self, handler: Callable):
        """Set kline message handler"""
        self.on_kline = handler
    
    def set_error_handler(self, handler: Callable):
        """Set error message handler"""
        self.on_error = handler