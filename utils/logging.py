"""
Enhanced Logging System for Trading Bot
Provides UTF-8 safe logging with emoji stripping for Windows compatibility
"""

import logging
import sys
import os
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler

class IndicatorWarningFilter(logging.Filter):
    """Filter to suppress repeated indicator calculation warnings"""
    
    def __init__(self):
        super().__init__()
        self.warned_indicators = set()
    
    def filter(self, record):
        """Filter out repeated generic calculation warnings"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'Generic calculation used for' in record.msg and 'may need specific implementation' in record.msg:
                # Extract indicator name
                import re
                match = re.search(r'Generic calculation used for (\w+)', record.msg)
                if match:
                    indicator_name = match.group(1)
                    if indicator_name in self.warned_indicators:
                        return False  # Suppress this log
                    else:
                        self.warned_indicators.add(indicator_name)
                        # Allow first warning but change level to DEBUG for less noise
                        record.levelno = logging.DEBUG
                        record.levelname = 'DEBUG'
        return True

class SafeFormatter(logging.Formatter):
    """UTF-8 safe formatter that handles encoding issues on Windows"""
    
    def __init__(self, fmt=None, emoji_strip=True):
        super().__init__(fmt)
        self.emoji_strip = emoji_strip
        
        # Emoji regex pattern
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002500-\U00002BEF"  # chinese char
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # dingbats
            "\u3030"
            "]+", 
            flags=re.UNICODE
        )
    
    def format(self, record):
        """Format log record with safe encoding"""
        try:
            # Get the formatted message
            message = super().format(record)
            
            # Strip emojis if enabled
            if self.emoji_strip:
                message = self.emoji_pattern.sub('', message)
            
            # Handle encoding for Windows cp1252
            if sys.platform.startswith('win'):
                try:
                    # Try to encode as cp1252 and replace problematic characters
                    message.encode('cp1252')
                except UnicodeEncodeError:
                    # Replace problematic characters
                    message = message.encode('ascii', 'replace').decode('ascii')
            
            return message
            
        except Exception as e:
            # Fallback to safe ASCII representation
            return f"LOG_FORMAT_ERROR: {str(e)}"

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module if hasattr(record, 'module') else record.name,
            'message': record.getMessage(),
            'pathname': record.pathname,
            'lineno': record.lineno,
            'funcName': record.funcName
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
            
        return json.dumps(log_data, ensure_ascii=True)

class TradingBotLogger:
    """Enhanced logger for the trading bot system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        log_file = self.config.get('file', 'logs/trading_bot.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Setup root logger
        level = getattr(logging, self.config.get('level', 'INFO'))
        logging.getLogger().setLevel(level)
        
        # Remove existing handlers
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        
        # File handler with rotation
        max_size = self.config.get('max_size_mb', 10) * 1024 * 1024
        backup_count = self.config.get('backup_count', 5)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Setup formatters
        if self.config.get('structured_json', False):
            formatter = JSONFormatter()
        else:
            format_string = self.config.get(
                'format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            formatter = SafeFormatter(
                format_string,
                emoji_strip=self.config.get('emoji_strip', True)
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().addHandler(console_handler)
        
        # Reduce noise from external libraries
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        
        # Reduce indicator calculator noise - suppress repeated generic calculation warnings
        indicators_logger = logging.getLogger('indicators.calculator')
        if not hasattr(indicators_logger, '_warning_suppressor_added'):
            indicators_logger.addFilter(IndicatorWarningFilter())
            indicators_logger._warning_suppressor_added = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance for a module"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_ml_event(self, event_type: str, data: Dict[str, Any], logger_name: str = 'ml'):
        """Log structured ML events"""
        logger = self.get_logger(logger_name)
        
        # Create structured log entry
        log_data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        
        # Log with extra data for JSON formatter
        logger.info(f"ML_EVENT: {event_type}", extra={'extra_data': log_data})
    
    def log_trading_event(self, event_type: str, symbol: str, data: Dict[str, Any], 
                         logger_name: str = 'trading'):
        """Log structured trading events"""
        logger = self.get_logger(logger_name)
        
        log_data = {
            'event_type': event_type,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }
        
        logger.info(f"TRADING_EVENT: {event_type} for {symbol}", 
                   extra={'extra_data': log_data})

# Global logger instance
_logger_instance = None

def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger instance"""
    global _logger_instance
    if _logger_instance is None:
        # Default config if not initialized
        try:
            from config.config_loader import load_config
            config = load_config()
            _logger_instance = TradingBotLogger(config.get('logging', {}))
        except ImportError:
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(name)
    
    return _logger_instance.get_logger(name)

def initialize_logging(config: Dict[str, Any]) -> TradingBotLogger:
    """Initialize logging system with configuration"""
    global _logger_instance
    _logger_instance = TradingBotLogger(config.get('logging', {}))
    return _logger_instance