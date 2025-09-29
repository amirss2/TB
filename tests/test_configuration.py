"""
Test configuration changes and timeframe updates
"""
import os
import unittest

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TRADING_CONFIG, DATA_CONFIG, FEATURE_SELECTION_CONFIG


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration changes"""
    
    def test_timeframe_configuration(self):
        """Test that timeframe is set to 4h"""
        self.assertEqual(TRADING_CONFIG['timeframe'], '4h')
    
    def test_confidence_threshold(self):
        """Test that confidence threshold is set to 0.7 (70%)"""
        self.assertEqual(TRADING_CONFIG['confidence_threshold'], 0.7)
    
    def test_data_config_4h(self):
        """Test data configuration for 4h candles"""
        self.assertIn('min_4h_candles', DATA_CONFIG)
        self.assertIn('max_4h_selection_candles', DATA_CONFIG)
        self.assertIn('max_4h_training_candles', DATA_CONFIG)
        
        # Check that old 1m config is removed
        self.assertNotIn('min_1m_candles', DATA_CONFIG)
        self.assertNotIn('max_1m_selection_candles', DATA_CONFIG)
        self.assertNotIn('max_1m_training_candles', DATA_CONFIG)
    
    def test_feature_selection_config(self):
        """Test feature selection configuration for 4h"""
        self.assertIn('selection_window_4h', FEATURE_SELECTION_CONFIG)
        self.assertNotIn('selection_window_1m', FEATURE_SELECTION_CONFIG)
    
    def test_training_symbols(self):
        """Test that training symbols are maintained"""
        expected_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT']
        self.assertEqual(TRADING_CONFIG['training_symbols'], expected_symbols)
    
    def test_real_time_data_config(self):
        """Test real-time data fetching configuration"""
        self.assertEqual(DATA_CONFIG['update_interval'], 30)  # Updated to 30 seconds
        self.assertEqual(DATA_CONFIG['real_time_fetch_limit'], 3)
        self.assertEqual(DATA_CONFIG['real_time_min_interval'], 10)  # Updated to 10 seconds
        self.assertEqual(DATA_CONFIG['batch_size'], 3)
    
    def test_coinmarketcap_integration(self):
        """Test CoinMarketCap integration settings"""
        self.assertTrue(TRADING_CONFIG['use_coinmarketcap_symbols'])
        self.assertEqual(TRADING_CONFIG['coinmarketcap_limit'], 1000)


if __name__ == '__main__':
    unittest.main()