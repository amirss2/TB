#!/usr/bin/env python3
"""
Temporary script to populate symbol cache for testing
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.symbol_cache import SymbolCache

def populate_test_cache():
    """Populate cache with test symbols"""
    print("Populating symbol cache with test data...")
    
    # Create a list of popular trading symbols for testing
    test_symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT',  # Original training symbols
        'ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT',
        'LTCUSDT', 'BCHUSDT', 'UNIUSDT', 'ATOMUSDT', 'VETUSDT', 'FILUSDT',
        'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'THETAUSDT', 'ICXUSDT', 'ZILUSDT',
        'ONTUSDT', 'QTUMUSDT', 'BATUSDT', 'ZRXUSDT', 'ENJUSDT', 'IOTAUSDT',
        'NEOUSDT', 'OMGUSDT', 'ZECUSDT', 'DASHUSDT', 'WAVESUSDT', 'LSKUSDT',
        'NKNUSDT', 'CVCUSDT', 'STXUSDT', 'KNCUSDT', 'REPUSDT', 'STORJUSDT',
        'ALGOUSDT', 'BANDUSDT', 'BALUSDT', 'CRVUSDT', 'COMPUSDT', 'SUSHIUSDT',
        'YFIUSDT', 'SNXUSDT', 'RENUSDT', 'KSMUSDT', 'LUNAUSDT', 'OCEANUSDT'
    ]
    
    # Create cache and save symbols
    cache = SymbolCache()
    
    metadata = {
        'source': 'test_population',
        'test_mode': True,
        'symbol_count': len(test_symbols),
        'note': 'Temporary test data for debugging'
    }
    
    success = cache.save_symbols(test_symbols, metadata)
    
    if success:
        print(f"✅ Successfully cached {len(test_symbols)} test symbols")
        
        # Verify cache
        loaded = cache.load_symbols()
        if loaded:
            print(f"✅ Verification: Loaded {len(loaded)} symbols from cache")
            print(f"First 10 symbols: {loaded[:10]}")
        else:
            print("❌ Verification failed: Could not load symbols from cache")
    else:
        print("❌ Failed to save symbols to cache")

if __name__ == "__main__":
    populate_test_cache()