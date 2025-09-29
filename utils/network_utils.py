"""
Network Connectivity Utilities
Handles network connection checking and status monitoring
"""
import socket
import requests
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class NetworkConnectivity:
    """Manages network connectivity monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_check_time = 0
        self.last_status = True
        self.check_interval = 30  # Check every 30 seconds minimum
        
        # Test endpoints for connectivity checks
        self.test_endpoints = [
            ('8.8.8.8', 53),  # Google DNS
            ('1.1.1.1', 53),  # Cloudflare DNS
            ('api.coinex.com', 443),  # CoinEx API
        ]
        
        self.http_endpoints = [
            'https://api.coinex.com/v1/common/timestamp',
            'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?limit=1',
        ]
    
    def check_socket_connectivity(self, timeout: float = 5.0) -> bool:
        """Check basic internet connectivity using socket connections"""
        for host, port in self.test_endpoints:
            try:
                sock = socket.create_connection((host, port), timeout=timeout)
                sock.close()
                return True
            except (socket.error, socket.timeout):
                continue
        return False
    
    def check_http_connectivity(self, timeout: float = 10.0) -> bool:
        """Check HTTP connectivity to trading APIs"""
        for url in self.http_endpoints:
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code in [200, 401, 403]:  # 401/403 means server is reachable
                    return True
            except (requests.RequestException, requests.Timeout):
                continue
        return False
    
    def is_connected(self, force_check: bool = False) -> bool:
        """
        Check if we have internet connectivity
        Uses caching to avoid too frequent checks
        """
        current_time = time.time()
        
        # Use cached result if recent enough and not forced
        if not force_check and (current_time - self.last_check_time) < self.check_interval:
            return self.last_status
        
        # Perform actual connectivity check
        socket_ok = self.check_socket_connectivity()
        http_ok = self.check_http_connectivity() if socket_ok else False
        
        # Update cache
        self.last_check_time = current_time
        self.last_status = socket_ok and http_ok
        
        if not self.last_status:
            self.logger.warning("Network connectivity check failed")
        
        return self.last_status
    
    def wait_for_connection(self, timeout: int = 300, check_interval: int = 10) -> bool:
        """
        Wait for network connection to be restored
        Returns True if connection restored, False if timeout
        """
        start_time = time.time()
        self.logger.info(f"Waiting for network connection (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.is_connected(force_check=True):
                self.logger.info("Network connection restored!")
                return True
            
            self.logger.debug(f"Still no connection, waiting {check_interval}s...")
            time.sleep(check_interval)
        
        self.logger.error(f"Network connection timeout after {timeout}s")
        return False
    
    def get_connectivity_status(self) -> Dict:
        """Get detailed connectivity status"""
        socket_status = self.check_socket_connectivity()
        http_status = self.check_http_connectivity() if socket_status else False
        
        return {
            'connected': socket_status and http_status,
            'socket_connectivity': socket_status,
            'http_connectivity': http_status,
            'last_check': datetime.fromtimestamp(self.last_check_time).isoformat() if self.last_check_time else None,
            'test_endpoints': [f"{host}:{port}" for host, port in self.test_endpoints],
            'http_endpoints': self.http_endpoints
        }


# Global instance for easy access
network_checker = NetworkConnectivity()