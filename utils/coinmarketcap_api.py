import requests
import logging
from typing import Dict, List, Any, Optional
from config.settings import COINMARKETCAP_CONFIG

class CoinMarketCapAPI:
    """
    CoinMarketCap API client for getting top cryptocurrencies
    """
    
    def __init__(self):
        self.api_key = COINMARKETCAP_CONFIG['api_key']
        self.base_url = COINMARKETCAP_CONFIG['base_url']
        self.limit = COINMARKETCAP_CONFIG['limit']
        self.logger = logging.getLogger(__name__)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'X-CMC_PRO_API_KEY': self.api_key,
            'Accept': 'application/json',
            'Accept-Encoding': 'deflate, gzip'
        })
    
    def get_top_cryptocurrencies(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get top cryptocurrencies from CoinMarketCap
        
        Args:
            limit: Number of cryptocurrencies to retrieve (default: from config)
            
        Returns:
            List of cryptocurrency data with symbol, name, etc.
        """
        if limit is None:
            limit = self.limit
            
        try:
            endpoint = COINMARKETCAP_CONFIG['listings_endpoint']
            url = f"{self.base_url}{endpoint}"
            
            params = {
                'start': 1,
                'limit': limit,
                'convert': 'USD',
                'sort': 'market_cap',
                'sort_dir': 'desc'
            }
            
            self.logger.info(f"Fetching top {limit} cryptocurrencies from CoinMarketCap...")
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status', {}).get('error_code') == 0:
                    cryptocurrencies = data.get('data', [])
                    self.logger.info(f"Successfully retrieved {len(cryptocurrencies)} cryptocurrencies from CoinMarketCap")
                    return cryptocurrencies
                else:
                    error_msg = data.get('status', {}).get('error_message', 'Unknown error')
                    self.logger.error(f"CoinMarketCap API error: {error_msg}")
                    return self._get_fallback_cryptocurrencies(limit)
            else:
                self.logger.error(f"CoinMarketCap API request failed: HTTP {response.status_code}")
                return self._get_fallback_cryptocurrencies(limit)
                
        except requests.RequestException as e:
            self.logger.error(f"Network error accessing CoinMarketCap: {e}")
            return self._get_fallback_cryptocurrencies(limit)
        except Exception as e:
            self.logger.error(f"Unexpected error accessing CoinMarketCap: {e}")
            return self._get_fallback_cryptocurrencies(limit)
    
    def _get_fallback_cryptocurrencies(self, limit: int) -> List[Dict[str, Any]]:
        """
        Provide fallback cryptocurrency list when API is unavailable
        """
        self.logger.warning("Using fallback cryptocurrency list")
        
        # Top cryptocurrencies by market cap (as of common knowledge)
        fallback_cryptos = [
            {'symbol': 'BTC', 'name': 'Bitcoin'},
            {'symbol': 'ETH', 'name': 'Ethereum'},
            {'symbol': 'BNB', 'name': 'BNB'},
            {'symbol': 'XRP', 'name': 'XRP'},
            {'symbol': 'SOL', 'name': 'Solana'},
            {'symbol': 'USDC', 'name': 'USD Coin'},
            {'symbol': 'ADA', 'name': 'Cardano'},
            {'symbol': 'AVAX', 'name': 'Avalanche'},
            {'symbol': 'DOGE', 'name': 'Dogecoin'},
            {'symbol': 'TRX', 'name': 'TRON'},
            {'symbol': 'DOT', 'name': 'Polkadot'},
            {'symbol': 'MATIC', 'name': 'Polygon'},
            {'symbol': 'LTC', 'name': 'Litecoin'},
            {'symbol': 'SHIB', 'name': 'Shiba Inu'},
            {'symbol': 'WBTC', 'name': 'Wrapped Bitcoin'},
            {'symbol': 'BCH', 'name': 'Bitcoin Cash'},
            {'symbol': 'LINK', 'name': 'Chainlink'},
            {'symbol': 'UNI', 'name': 'Uniswap'},
            {'symbol': 'ATOM', 'name': 'Cosmos'},
            {'symbol': 'LEO', 'name': 'UNUS SED LEO'},
            {'symbol': 'XLM', 'name': 'Stellar'},
            {'symbol': 'XMR', 'name': 'Monero'},
            {'symbol': 'ETC', 'name': 'Ethereum Classic'},
            {'symbol': 'ICP', 'name': 'Internet Computer'},
            {'symbol': 'HBAR', 'name': 'Hedera'},
            {'symbol': 'FIL', 'name': 'Filecoin'},
            {'symbol': 'VET', 'name': 'VeChain'},
            {'symbol': 'APT', 'name': 'Aptos'},
            {'symbol': 'NEAR', 'name': 'NEAR Protocol'},
            {'symbol': 'CRO', 'name': 'Cronos'},
            {'symbol': 'GRT', 'name': 'The Graph'},
            {'symbol': 'LDO', 'name': 'Lido DAO'},
            {'symbol': 'QNT', 'name': 'Quant'},
            {'symbol': 'ARB', 'name': 'Arbitrum'},
            {'symbol': 'ALGO', 'name': 'Algorand'},
            {'symbol': 'VGX', 'name': 'Voyager Token'},
            {'symbol': 'MANA', 'name': 'Decentraland'},
            {'symbol': 'AAVE', 'name': 'Aave'},
            {'symbol': 'SAND', 'name': 'The Sandbox'},
            {'symbol': 'FTM', 'name': 'Fantom'},
            {'symbol': 'OP', 'name': 'Optimism'},
            {'symbol': 'EGLD', 'name': 'MultiversX'},
            {'symbol': 'AXS', 'name': 'Axie Infinity'},
            {'symbol': 'FLOW', 'name': 'Flow'},
            {'symbol': 'XTZ', 'name': 'Tezos'},
            {'symbol': 'CHZ', 'name': 'Chiliz'},
            {'symbol': 'THETA', 'name': 'Theta Network'},
            {'symbol': 'EOS', 'name': 'EOS'},
            {'symbol': 'KAVA', 'name': 'Kava'},
            {'symbol': 'BSV', 'name': 'Bitcoin SV'},
            {'symbol': 'ZEC', 'name': 'Zcash'},
            {'symbol': 'DASH', 'name': 'Dash'},
            {'symbol': 'NEO', 'name': 'Neo'},
            {'symbol': 'IOTA', 'name': 'IOTA'},
            {'symbol': 'KSM', 'name': 'Kusama'},
            {'symbol': 'WAVES', 'name': 'Waves'},
            {'symbol': 'ONE', 'name': 'Harmony'},
            {'symbol': 'ZIL', 'name': 'Zilliqa'},
            {'symbol': 'ENJ', 'name': 'Enjin Coin'},
            {'symbol': 'BAT', 'name': 'Basic Attention Token'},
            {'symbol': 'CRV', 'name': 'Curve DAO Token'},
        ]
        
        # Extend list with more common cryptocurrencies if needed
        if limit > len(fallback_cryptos):
            additional_cryptos = [
                {'symbol': f'TOKEN{i}', 'name': f'Token {i}'} 
                for i in range(len(fallback_cryptos) + 1, limit + 1)
            ]
            fallback_cryptos.extend(additional_cryptos)
        
        return fallback_cryptos[:limit]
    
    def extract_symbols(self, cryptocurrencies: List[Dict[str, Any]], quote_currency: str = 'USDT') -> List[str]:
        """
        Extract trading pair symbols from cryptocurrency data
        
        Args:
            cryptocurrencies: List of cryptocurrency data from CoinMarketCap
            quote_currency: Quote currency to pair with (default: USDT)
            
        Returns:
            List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        symbols = []
        for crypto in cryptocurrencies:
            symbol = crypto.get('symbol', '').upper()
            if symbol and symbol != quote_currency:  # Don't pair USDT with USDT
                pair_symbol = f"{symbol}{quote_currency}"
                symbols.append(pair_symbol)
        
        self.logger.info(f"Generated {len(symbols)} trading pairs with {quote_currency}")
        return symbols