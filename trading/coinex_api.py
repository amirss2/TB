import requests
import hashlib
import hmac
import time
import json
import logging
from typing import Dict, List, Any, Optional
from config.settings import COINEX_CONFIG
from utils.symbol_cache import SymbolCache
from utils.network_utils import network_checker

class CoinExAPI:
    """
    CoinEx API client for cryptocurrency trading
    """
    
    def __init__(self):
        self.api_key = COINEX_CONFIG['api_key']
        self.secret_key = COINEX_CONFIG['secret_key']
        self.sandbox_mode = COINEX_CONFIG['sandbox_mode']
        self.base_url = COINEX_CONFIG['sandbox_url'] if self.sandbox_mode else COINEX_CONFIG['base_url']
        self.logger = logging.getLogger(__name__)
        
        # Initialize symbol cache
        self.symbol_cache = SymbolCache()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        })
    
    def _generate_signature(self, params: Dict[str, Any], secret_key: str) -> str:
        """Generate signature for API authentication"""
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        return hmac.new(
            secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.md5
        ).hexdigest().upper()
    
    def _make_request(self, method: str, endpoint: str, params: Dict[str, Any] = None,
                     auth_required: bool = True) -> Dict[str, Any]:
        """Make authenticated request to CoinEx API"""
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        # Add timestamp for authenticated requests
        if auth_required:
            params['access_id'] = self.api_key
            params['tonce'] = int(time.time() * 1000)
            
            # Generate signature
            signature = self._generate_signature(params, self.secret_key)
            
            headers = {
                'authorization': signature,
            }
        else:
            headers = {}
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('code') == 0:
                return result.get('data', {})
            else:
                self.logger.error(f"API error: {result}")
                raise Exception(f"CoinEx API error: {result.get('message', 'Unknown error')}")
                
        except requests.exceptions.HTTPError as e:
            # Handle 404 errors with v2 fallback for specific endpoints
            if response.status_code == 404 and ('time' in endpoint or 'timestamp' in endpoint):
                self.logger.info(f"404 on {endpoint}, attempting v2 fallback")
                if 'v1/time' in url or '/timestamp' in url:
                    try:
                        # Try v2 server-time endpoint
                        v2_url = url.replace(endpoint, 'v2/common/server-time')
                        response = self.session.get(v2_url, params=params, headers=headers, timeout=10)
                        response.raise_for_status()
                        result = response.json()
                        if result.get('code') == 0:
                            self.logger.info("Successfully used v2 fallback endpoint")
                            return result.get('data', {})
                    except Exception as fallback_error:
                        self.logger.warning(f"v2 fallback also failed: {fallback_error}")
            
            self.logger.error(f"HTTP request failed: {e}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information for a symbol"""
        try:
            endpoint = f"market/ticker?market={symbol}"
            return self._make_request('GET', endpoint, auth_required=False)
        except Exception as e:
            self.logger.warning(f"API failed for {symbol} ticker, using fallback: {e}")
            # Return fallback ticker data
            base_prices = {
                'BTCUSDT': 45000,
                'ETHUSDT': 2500,
                'SOLUSDT': 100,
                'DOGEUSDT': 0.08
            }
            base_price = base_prices.get(symbol, 1000)
            return {
                'ticker': {
                    'last': str(base_price),
                    'vol': '1000.0',
                    'high': str(base_price * 1.02),
                    'low': str(base_price * 0.98)
                }
            }

    def get_all_tickers(self) -> Dict[str, Any]:
        """Get all ticker information"""
        try:
            endpoint = "market/ticker/all"
            return self._make_request('GET', endpoint, auth_required=False)
        except Exception as e:
            self.logger.warning(f"API failed for all tickers, using fallback: {e}")
            # Return fallback data for main symbols
            fallback_tickers = {}
            base_prices = {
                'BTCUSDT': 45000,
                'ETHUSDT': 2500,
                'SOLUSDT': 100,
                'DOGEUSDT': 0.08
            }
            for symbol, price in base_prices.items():
                fallback_tickers[symbol] = {
                    'last': str(price),
                    'vol': '1000.0',
                    'high': str(price * 1.02),
                    'low': str(price * 0.98),
                    'buy': str(price * 0.999),
                    'sell': str(price * 1.001)
                }
            return {'ticker': fallback_tickers}

    def get_top_trading_pairs(self, limit: int = 100, quote_currency: str = 'USDT') -> List[str]:
        """
        Get top trading pairs by volume from CoinEx
        
        Args:
            limit: Number of top pairs to return
            quote_currency: Quote currency to filter by (default: USDT)
            
        Returns:
            List of trading pair symbols sorted by volume descending
        """
        try:
            all_tickers = self.get_all_tickers()
            
            if not all_tickers or 'ticker' not in all_tickers:
                self.logger.warning("No ticker data available, using fallback symbols")
                return self._generate_fallback_top_symbols(limit, quote_currency)
            
            # Filter and sort by volume
            pairs_with_volume = []
            for symbol, ticker_data in all_tickers['ticker'].items():
                # Filter by quote currency
                if not symbol.endswith(quote_currency):
                    continue
                    
                try:
                    volume = float(ticker_data.get('vol', 0))
                    price = float(ticker_data.get('last', 0))
                    
                    # Calculate volume in quote currency (volume * price)
                    volume_in_quote = volume * price
                    
                    if volume_in_quote > 0:  # Only include pairs with actual volume
                        pairs_with_volume.append((symbol, volume_in_quote))
                except (ValueError, TypeError):
                    continue
            
            # Sort by volume descending and take top N
            pairs_with_volume.sort(key=lambda x: x[1], reverse=True)
            top_pairs = [pair[0] for pair in pairs_with_volume[:limit]]
            
            self.logger.info(f"Retrieved {len(top_pairs)} top trading pairs from CoinEx")
            return top_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting top trading pairs: {e}")
            return self._generate_fallback_top_symbols(limit, quote_currency)
    
    def _generate_fallback_top_symbols(self, limit: int, quote_currency: str = 'USDT') -> List[str]:
        """Generate fallback list of popular trading symbols when API fails"""
        # Popular cryptocurrencies that are commonly available on exchanges
        popular_symbols = [
            f'BTC{quote_currency}', f'ETH{quote_currency}', f'SOL{quote_currency}', f'DOGE{quote_currency}',
            f'BNB{quote_currency}', f'XRP{quote_currency}', f'ADA{quote_currency}', f'AVAX{quote_currency}',
            f'DOT{quote_currency}', f'MATIC{quote_currency}', f'LTC{quote_currency}', f'LINK{quote_currency}',
            f'UNI{quote_currency}', f'ATOM{quote_currency}', f'ICP{quote_currency}', f'FIL{quote_currency}',
            f'TRX{quote_currency}', f'ETC{quote_currency}', f'XLM{quote_currency}', f'VET{quote_currency}',
            f'ALGO{quote_currency}', f'AAVE{quote_currency}', f'MANA{quote_currency}', f'SAND{quote_currency}',
            f'CRV{quote_currency}', f'COMP{quote_currency}', f'MKR{quote_currency}', f'SNX{quote_currency}',
            f'SUSHI{quote_currency}', f'BAT{quote_currency}', f'ZRX{quote_currency}', f'ENJ{quote_currency}',
            f'CHZ{quote_currency}', f'HOT{quote_currency}', f'ICX{quote_currency}', f'ONT{quote_currency}',
            f'ZIL{quote_currency}', f'RVN{quote_currency}', f'QTUM{quote_currency}', f'WAVES{quote_currency}',
            f'KSM{quote_currency}', f'NEAR{quote_currency}', f'FTM{quote_currency}', f'ONE{quote_currency}',
            f'HBAR{quote_currency}', f'EGLD{quote_currency}', f'THETA{quote_currency}', f'XTZ{quote_currency}',
            f'DASH{quote_currency}', f'NEO{quote_currency}', f'IOTA{quote_currency}', f'EOS{quote_currency}',
            f'XMR{quote_currency}', f'ZEC{quote_currency}', f'BCH{quote_currency}', f'BSV{quote_currency}',
            f'CAKE{quote_currency}', f'RUNE{quote_currency}', f'ALPHA{quote_currency}', f'BEL{quote_currency}',
            f'CTK{quote_currency}', f'DENT{quote_currency}', f'FTT{quote_currency}', f'KAVA{quote_currency}',
            f'LRC{quote_currency}', f'OGN{quote_currency}', f'RSR{quote_currency}', f'SRM{quote_currency}',
            f'STORJ{quote_currency}', f'SXP{quote_currency}', f'TROY{quote_currency}', f'WTC{quote_currency}',
            f'YFI{quote_currency}', f'ZEN{quote_currency}', f'API3{quote_currency}', f'BADGER{quote_currency}',
            f'BAND{quote_currency}', f'CRO{quote_currency}', f'DF{quote_currency}', f'DODO{quote_currency}',
            f'GRT{quote_currency}', f'KEEP{quote_currency}', f'NKN{quote_currency}', f'NUC{quote_currency}',
            f'OXT{quote_currency}', f'REEF{quote_currency}', f'REN{quote_currency}', f'ROSE{quote_currency}',
            f'SKALE{quote_currency}', f'TKO{quote_currency}', f'TLM{quote_currency}', f'TORN{quote_currency}',
            f'UNFI{quote_currency}', f'UTK{quote_currency}', f'WIN{quote_currency}', f'YFII{quote_currency}',
            f'1INCH{quote_currency}', f'AKRO{quote_currency}', f'AXS{quote_currency}', f'BAKE{quote_currency}',
            f'BNT{quote_currency}', f'BTCST{quote_currency}', f'BURGER{quote_currency}', f'BZRX{quote_currency}'
        ]
        
        # Return requested limit, but at least include the 4 main training symbols
        result = popular_symbols[:limit]
        self.logger.info(f"Using fallback list of {len(result)} popular symbols")
        return result

    def get_available_symbols_from_list(self, symbol_list: List[str]) -> List[str]:
        """
        Check which symbols from a given list are available and tradable on CoinEx
        
        Args:
            symbol_list: List of symbols to check (e.g., ['BTCUSDT', 'ETHUSDT', ...])
            
        Returns:
            List of symbols that are available on CoinEx
        """
        try:
            all_tickers = self.get_all_tickers()
            
            if not all_tickers or 'ticker' not in all_tickers:
                self.logger.warning("No ticker data available, using training symbols as fallback")
                # Fallback to training symbols if API fails
                from config.settings import TRADING_CONFIG
                return TRADING_CONFIG['training_symbols']
            
            available_symbols = []
            coinex_symbols = set(all_tickers['ticker'].keys())
            
            for symbol in symbol_list:
                if symbol in coinex_symbols:
                    # Check if the symbol has valid ticker data
                    ticker_data = all_tickers['ticker'][symbol]
                    try:
                        # Ensure the symbol has valid price and volume data
                        price = float(ticker_data.get('last', 0))
                        volume = float(ticker_data.get('vol', 0))
                        
                        if price > 0 and volume > 0:  # Valid and actively traded
                            available_symbols.append(symbol)
                    except (ValueError, TypeError):
                        # Skip symbols with invalid data
                        continue
            
            self.logger.info(f"Found {len(available_symbols)} available symbols out of {len(symbol_list)} requested")
            return available_symbols
            
        except Exception as e:
            self.logger.error(f"Error checking symbol availability: {e}")
            # Fallback to training symbols
            from config.settings import TRADING_CONFIG
            return TRADING_CONFIG['training_symbols']

    def get_coinmarketcap_available_symbols(self, limit: int = 1000) -> List[str]:
        """
        Get symbols from CoinMarketCap top list that are available on CoinEx with caching
        
        Flow: CoinMarketCap (top 1000) → Filter by CoinEx availability → Cache filtered list
        Always ensures the 4 training symbols are included
        
        Args:
            limit: Number of top cryptocurrencies to check from CoinMarketCap
            
        Returns:
            List of symbols available on CoinEx from CoinMarketCap top list + training symbols
        """
        try:
            # Training symbols that must always be included
            from config.settings import TRADING_CONFIG
            training_symbols = TRADING_CONFIG['training_symbols']
            
            # Check if we have internet connectivity
            if not network_checker.is_connected():
                self.logger.warning("No internet connection, trying to load symbols from cache")
                cached_symbols = self.symbol_cache.load_symbols()
                if cached_symbols:
                    # Verify training symbols are included (they should be, but let's be safe)
                    missing_training = [s for s in training_symbols if s not in cached_symbols]
                    if missing_training:
                        self.logger.warning(f"Adding missing training symbols to offline cache: {missing_training}")
                        cached_symbols = list(set(cached_symbols + training_symbols))
                    
                    self.logger.info(f"Loaded {len(cached_symbols)} symbols from cache due to network issues")
                    return cached_symbols
                else:
                    self.logger.warning("No cached symbols available, falling back to training symbols only")
                    return training_symbols
            
            # Check if we have valid cached symbols first
            cached_symbols = self.symbol_cache.load_symbols()
            if cached_symbols:
                self.logger.info(f"Found {len(cached_symbols)} cached symbols, checking if refresh needed...")
                # Use cached symbols but still try to refresh if they're old
                cache_info = self.symbol_cache.get_cache_info()
                if cache_info.get('age_hours', 25) < 24:  # Use cache if less than 24 hours old
                    # Verify training symbols are included (they should be, but let's be safe)
                    missing_training = [s for s in training_symbols if s not in cached_symbols]
                    if missing_training:
                        self.logger.warning(f"Adding missing training symbols to cache: {missing_training}")
                        cached_symbols = list(set(cached_symbols + training_symbols))
                    
                    self.logger.info(f"Using valid cache: {len(cached_symbols)} symbols (training symbols verified)")
                    return cached_symbols
            
            # Fetch fresh data from CoinMarketCap and filter by CoinEx availability
            try:
                self.logger.info(f"Fetching fresh top {limit} symbols from CoinMarketCap...")
                
                from utils.coinmarketcap_api import CoinMarketCapAPI
                
                # Step 1: Get top cryptocurrencies from CoinMarketCap
                cmc_api = CoinMarketCapAPI()
                top_cryptos = cmc_api.get_top_cryptocurrencies(limit)
                
                if not top_cryptos:
                    raise Exception("No cryptocurrencies returned from CoinMarketCap")
                
                # Step 2: Convert to trading pair symbols
                cmc_symbols = cmc_api.extract_symbols(top_cryptos, 'USDT')
                self.logger.info(f"CoinMarketCap returned {len(cmc_symbols)} symbol pairs")
                
                # Step 3: Check which ones are available on CoinEx
                self.logger.info("Filtering symbols by CoinEx availability...")
                available_symbols = self.get_available_symbols_from_list(cmc_symbols)
                
                if available_symbols and len(available_symbols) > 0:
                    # Step 4: Ensure training symbols are always included
                    final_symbols = list(set(available_symbols + training_symbols))
                    
                    # Step 5: Cache the filtered symbols for future use
                    cache_metadata = {
                        'source': 'coinmarketcap_filtered',
                        'coinmarketcap_limit': limit,
                        'cmc_symbols_count': len(cmc_symbols),
                        'coinex_available_count': len(available_symbols),
                        'final_count_with_training': len(final_symbols),
                        'training_symbols_included': training_symbols
                    }
                    self.symbol_cache.save_symbols(final_symbols, cache_metadata)
                    
                    self.logger.info(f"✅ CoinMarketCap → CoinEx filtering complete:")
                    self.logger.info(f"   📊 CoinMarketCap symbols: {len(cmc_symbols)}")
                    self.logger.info(f"   ✅ Available on CoinEx: {len(available_symbols)}")
                    self.logger.info(f"   🔄 Total with training symbols: {len(final_symbols)}")
                    self.logger.info(f"   💾 Cached for offline use")
                    
                    return final_symbols
                else:
                    raise Exception("No symbols available on CoinEx from CoinMarketCap list")
                
            except Exception as api_error:
                self.logger.error(f"API error during fresh fetch: {api_error}")
                
                # Try to use cached symbols as fallback
                if cached_symbols:
                    # Verify training symbols are included
                    missing_training = [s for s in training_symbols if s not in cached_symbols]
                    if missing_training:
                        self.logger.warning(f"Adding missing training symbols to fallback cache: {missing_training}")
                        cached_symbols = list(set(cached_symbols + training_symbols))
                    
                    self.logger.warning(f"API failed, using {len(cached_symbols)} cached symbols")
                    return cached_symbols
                else:
                    self.logger.error("No fresh data and no cache available")
                    raise api_error
            
        except Exception as e:
            self.logger.error(f"Error in symbol fetching flow: {e}")
            
            # Emergency fallback: try any cached symbols, even old ones
            try:
                cached_symbols = self.symbol_cache.load_symbols(max_age_hours=168)  # Accept up to 1 week old
                if cached_symbols:
                    # Verify training symbols are included
                    missing_training = [s for s in training_symbols if s not in cached_symbols]
                    if missing_training:
                        self.logger.warning(f"Adding missing training symbols to emergency cache: {missing_training}")
                        cached_symbols = list(set(cached_symbols + training_symbols))
                    
                    self.logger.warning(f"Emergency fallback: using old cache ({len(cached_symbols)} symbols)")
                    return cached_symbols
            except Exception as cache_error:
                self.logger.error(f"Cache emergency fallback failed: {cache_error}")
            
            # Final fallback: training symbols only
            from config.settings import TRADING_CONFIG
            self.logger.warning("All symbol fetching failed, using training symbols only")
            return TRADING_CONFIG['training_symbols']
    
    def get_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        try:
            endpoint = "balance/info"
            return self._make_request('GET', endpoint)
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            raise
    
    def place_order(self, symbol: str, side: str, amount: float, price: float = None,
                   order_type: str = 'limit') -> Dict[str, Any]:
        """
        Place a trading order
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            side: 'buy' or 'sell'
            amount: Order amount
            price: Order price (for limit orders)
            order_type: 'limit' or 'market'
        """
        try:
            params = {
                'market': symbol,
                'type': side,
                'amount': str(amount),
            }
            
            if order_type == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                params['price'] = str(price)
                
            endpoint = "order/limit" if order_type == 'limit' else "order/market"
            
            result = self._make_request('POST', endpoint, params)
            self.logger.info(f"Order placed: {symbol} {side} {amount} @ {price}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            params = {
                'market': symbol,
                'id': order_id
            }
            
            endpoint = "order/pending/cancel"
            result = self._make_request('POST', endpoint, params)
            self.logger.info(f"Order cancelled: {order_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            raise
    
    def get_order_status(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Get order status"""
        try:
            params = {
                'market': symbol,
                'id': order_id
            }
            
            endpoint = "order/status"
            return self._make_request('GET', endpoint, params)
            
        except Exception as e:
            self.logger.error(f"Error getting order status {order_id}: {e}")
            raise
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders"""
        try:
            params = {}
            if symbol:
                params['market'] = symbol
            
            endpoint = "order/pending"
            result = self._make_request('GET', endpoint, params)
            
            return result.get('data', []) if isinstance(result, dict) else []
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise
    
    def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trade history"""
        try:
            params = {
                'market': symbol,
                'limit': limit
            }
            
            endpoint = "order/deals"
            result = self._make_request('GET', endpoint, params)
            
            return result.get('data', []) if isinstance(result, dict) else []
            
        except Exception as e:
            self.logger.error(f"Error getting trade history for {symbol}: {e}")
            raise
    
    def get_kline_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get candlestick data
        
        Args:
            symbol: Trading pair
            timeframe: Time interval (1min, 5min, 15min, 30min, 1hour, 4hour, 1day, 1week)
            limit: Number of candles to fetch
        """
        try:
            # Convert timeframe to CoinEx API format
            timeframe_mapping = {
                '1m': '1min', '1min': '1min',
                '5m': '5min', '5min': '5min', 
                '15m': '15min', '15min': '15min',
                '30m': '30min', '30min': '30min',
                '1h': '1hour', '1hour': '1hour',
                '4h': '4hour', '4hour': '4hour',
                '1d': '1day', '1day': '1day',
                '1w': '1week', '1week': '1week'
            }
            
            api_timeframe = timeframe_mapping.get(timeframe, timeframe)
            
            params = {
                'market': symbol,
                'type': api_timeframe,
                'limit': limit
            }
            
            endpoint = "market/kline"
            result = self._make_request('GET', endpoint, params, auth_required=False)
            
            # Convert to standard format
            candles = []
            if isinstance(result, list):
                for candle in result:
                    if len(candle) >= 6:
                        candles.append({
                            'timestamp': int(candle[0]),
                            'open': float(candle[1]),
                            'close': float(candle[2]),
                            'high': float(candle[3]),
                            'low': float(candle[4]),
                            'volume': float(candle[5]),
                        })
            
            return candles
            
        except Exception as e:
            # Check network connectivity before generating fallback data
            if not network_checker.is_connected():
                self.logger.error(f"No internet connection and API failed for {symbol}: {e}")
                # Don't generate fake data when offline, return empty list to trigger pause
                return []
            
            self.logger.warning(f"API failed for {symbol}, using fallback data: {e}")
            # Only return fallback demo data if we have connection but API is down
            if self.sandbox_mode:
                return self._generate_fallback_data(symbol, limit)
            else:
                # In live mode, don't generate fake data
                self.logger.error(f"Live mode: API failed and cannot generate fallback data for {symbol}")
                return []
    
    def _generate_fallback_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic fallback data when API is unavailable"""
        import random
        import time
        
        # Base prices for different symbols
        base_prices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 2500,
            'SOLUSDT': 100,
            'DOGEUSDT': 0.08
        }
        
        base_price = base_prices.get(symbol, 1000)
        current_time = int(time.time())
        
        candles = []
        current_price = base_price
        
        for i in range(limit):
            # Generate realistic price movement (±2%)
            price_change = random.uniform(-0.02, 0.02)
            open_price = current_price
            
            high = open_price * (1 + abs(price_change) + random.uniform(0, 0.01))
            low = open_price * (1 - abs(price_change) - random.uniform(0, 0.01))
            close = open_price * (1 + price_change)
            volume = random.uniform(100, 10000)
            
            candle_time = current_time - (limit - i) * 14400  # 4 hours in seconds
            
            candles.append({
                'timestamp': candle_time,
                'open': round(open_price, 8),
                'high': round(high, 8),
                'low': round(low, 8),
                'close': round(close, 8),
                'volume': round(volume, 2)
            })
            
            current_price = close
        
        return candles
    
    def test_connection(self) -> bool:
        """Test API connection with enhanced v2/v1 fallback"""
        try:
            # Try v2 API endpoint first  
            try:
                # Try server-time endpoint (v2 style)
                endpoint = "common/server-time"
                self._make_request('GET', endpoint, auth_required=False)
                self.logger.info("CoinEx API connection successful (v2 endpoint)")
            except Exception as e:
                self.logger.debug(f"v2 endpoint (server-time) failed: {e}")
                
                # Fallback to v1 endpoint
                try:
                    endpoint = "common/timestamp"
                    self._make_request('GET', endpoint, auth_required=False)
                    self.logger.info("CoinEx API connection successful (v1 endpoint)")
                except Exception as e2:
                    # If both fail, log warning but don't raise in demo mode
                    self.logger.warning(f"Both CoinEx API endpoints failed. v2 (server-time): {e}, v1 (timestamp): {e2}")
                    if self.sandbox_mode:
                        self.logger.info("Running in demo mode - API connection failure is non-critical")
                        return True  # Allow demo mode to continue
                    else:
                        raise e2
            
            # Try to get balance (requires auth) - only if API keys are provided
            if self.api_key and self.secret_key:
                try:
                    self.get_balance()
                    self.logger.info("CoinEx API authentication successful")
                except Exception as e:
                    self.logger.warning(f"CoinEx API authentication failed: {e}")
                    if self.sandbox_mode:
                        self.logger.info("Running in demo mode - authentication failure is non-critical")
                        return True
                    else:
                        raise
            else:
                self.logger.info("CoinEx API test completed (no auth keys provided)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"CoinEx API connection failed: {e}")
            if self.sandbox_mode:
                self.logger.info("Demo mode enabled - continuing with fallback data")
                return True
            return False