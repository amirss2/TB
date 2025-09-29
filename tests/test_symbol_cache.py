"""
Test for symbol cache functionality
"""
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.symbol_cache import SymbolCache


class TestSymbolCache(unittest.TestCase):
    """Test cases for symbol cache"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SymbolCache(cache_dir=self.temp_dir)
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_symbols(self):
        """Test saving and loading symbols"""
        # Save symbols
        metadata = {'source': 'test', 'count': len(self.test_symbols)}
        self.assertTrue(self.cache.save_symbols(self.test_symbols, metadata))
        
        # Load symbols
        loaded_symbols = self.cache.load_symbols()
        self.assertEqual(loaded_symbols, self.test_symbols)
    
    def test_cache_expiration(self):
        """Test cache expiration logic"""
        # Save symbols
        self.cache.save_symbols(self.test_symbols)
        
        # Should be valid immediately
        self.assertTrue(self.cache.is_cache_valid(max_age_hours=1))
        
        # Should be expired with very short max age
        self.assertFalse(self.cache.is_cache_valid(max_age_hours=0))
    
    def test_empty_cache(self):
        """Test behavior with empty cache"""
        # Should return None for non-existent cache
        self.assertIsNone(self.cache.load_symbols())
        self.assertFalse(self.cache.is_cache_valid())
    
    def test_cache_info(self):
        """Test cache information retrieval"""
        # Empty cache
        info = self.cache.get_cache_info()
        self.assertFalse(info['exists'])
        
        # With cache
        self.cache.save_symbols(self.test_symbols)
        info = self.cache.get_cache_info()
        self.assertTrue(info['exists'])
        self.assertEqual(info['symbol_count'], len(self.test_symbols))


if __name__ == '__main__':
    unittest.main()