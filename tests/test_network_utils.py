"""
Test for network connectivity utilities
"""
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_utils import NetworkConnectivity


class TestNetworkConnectivity(unittest.TestCase):
    """Test cases for network connectivity"""
    
    def setUp(self):
        """Set up test environment"""
        self.network = NetworkConnectivity()
    
    @patch('socket.create_connection')
    def test_socket_connectivity_success(self, mock_socket):
        """Test successful socket connectivity"""
        mock_socket.return_value = MagicMock()
        self.assertTrue(self.network.check_socket_connectivity())
    
    @patch('socket.create_connection')
    def test_socket_connectivity_failure(self, mock_socket):
        """Test failed socket connectivity"""
        mock_socket.side_effect = Exception("Connection failed")
        # The method should handle exceptions and return False
        self.assertFalse(self.network.check_socket_connectivity())
    
    @patch('requests.get')
    def test_http_connectivity_success(self, mock_get):
        """Test successful HTTP connectivity"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        self.assertTrue(self.network.check_http_connectivity())
    
    @patch('requests.get')
    def test_http_connectivity_failure(self, mock_get):
        """Test failed HTTP connectivity"""
        import requests
        mock_get.side_effect = requests.RequestException("Connection failed")
        # The method should handle exceptions and return False
        self.assertFalse(self.network.check_http_connectivity())
    
    def test_connectivity_status(self):
        """Test connectivity status reporting"""
        status = self.network.get_connectivity_status()
        self.assertIn('connected', status)
        self.assertIn('socket_connectivity', status)
        self.assertIn('http_connectivity', status)
        self.assertIn('test_endpoints', status)
        self.assertIn('http_endpoints', status)


if __name__ == '__main__':
    unittest.main()