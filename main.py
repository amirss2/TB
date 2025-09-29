#!/usr/bin/env python3
"""
AI Trading Bot - Main Application Entry Point
...
"""

import os
import sys
import logging
import signal
import threading
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import components
from config.settings import WEB_CONFIG, TRADING_CONFIG, LOGGING_CONFIG
from database.connection import db_connection
from database.models import SystemLog, Base
from trading.engine import TradingEngine
from web.app import create_app

class TradingBotApplication:
    """Main application class that orchestrates all components"""
    
    def __init__(self):
        # 1) ابتدا ماژول‌های مورد نیاز برای کانفیگ و لاگ را ایمپورت کن
        from config.config_loader import load_config
        from utils.logging import initialize_logging, get_logger
        
        # 2) یک لاگر اولیه موقتی برای جلوگیری از AttributeError (حتی اگر initialize_logging بعداً ساختار را عوض کند)
        self.logger = logging.getLogger("tb.bootstrap")
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
        self.logger.setLevel(logging.INFO)
        
        # 3) لود کانفیگ و ذخیره در self.config
        try:
            self.config = load_config()            # باید dict برگرداند
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = {}
        
        # 4) استخراج بخش trading از self.config (اگر نبود از TRADING_CONFIG می‌گیریم)
        trading_cfg = {}
        if isinstance(self.config, dict):
            trading_cfg = self.config.get('trading', {}) or {}
        if not trading_cfg:
            trading_cfg = TRADING_CONFIG  # fallback
        
        # 5) ست کردن پارامترهای کاربردی قبل از لاگ
        self.confidence_threshold = trading_cfg.get('confidence_threshold', 0.7)
        self.timeframe = trading_cfg.get('timeframe', TRADING_CONFIG.get('timeframe', '4h'))
        
        # 6) حالا سیستم لاگ پیشرفته را مقداردهی کن (بعد از داشتن config)
        try:
            self.logger_system = initialize_logging(self.config)
        except Exception as e:
            self.logger.warning(f"Enhanced logging init failed, fallback basic logger used: {e}")
            self.logger_system = None
        
        # 7) لاگر اصلی را جایگزین کن
        try:
            self.logger = get_logger(__name__)
        except Exception:
            # اگر get_logger شکست خورد همان لاگر bootstrap را نگه می‌داریم
            pass
        
        # 8) حالا می‌توانیم با خیال راحت لاگ کانفیگ بزنیم
        self.logger.info(f"[CONFIG] timeframe={self.timeframe} threshold={self.confidence_threshold}")
        
        # 9) آماده‌سازی سایر اجزای اپلیکیشن
        self.trading_engine = None
        self.web_app = None
        self.flask_thread = None
        
        self.running = False
        self.shutdown_event = threading.Event()
        
        # 10) ثبت سیگنال‌های سیستم (ممکن است روی ویندوز بعضی سیگنال‌ها محدود باشد)
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGTERM, self.signal_handler)
        except Exception:
            pass
        
        self.logger.info("AI Trading Bot Application initialized")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
    
    def initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.logger.info("Initializing database connection...")
            if not db_connection.init_engine():
                raise Exception("Database engine initialization failed")
            if not db_connection.test_connection():
                raise Exception("Database connection test failed")
            self.logger.info("Creating database tables...")
            Base.metadata.create_all(db_connection.engine)
            self.logger.info("Database initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.logger.info("You may need to:")
            self.logger.info("1. Install and start MySQL server")
            self.logger.info("2. Create the database: CREATE DATABASE TB;")
            self.logger.info("3. Update credentials in .env file")
            self.logger.info("4. Or run setup_database.py for automated setup")
            return False
    
    def initialize_trading_engine(self):
        """Initialize the trading engine"""
        try:
            self.logger.info("Initializing trading engine...")
            self.trading_engine = TradingEngine(demo_mode=True)
            self.logger.info("Trading engine initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Trading engine initialization failed: {e}")
            return False
    
    def initialize_web_interface(self):
        """Initialize the web dashboard"""
        try:
            self.logger.info("Initializing web interface...")
            self.web_app = create_app(self.trading_engine)
            self.logger.info("Web interface initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Web interface initialization failed: {e}")
            return False
    
    def start_web_interface(self):
        """Start the web interface in a separate thread"""
        try:
            def run_flask():
                self.web_app.run(
                    host=WEB_CONFIG['host'],
                    port=WEB_CONFIG['port'],
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            self.logger.info(f"Web interface started on http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start web interface: {e}")
            return False
    
    def log_system_event(self, level: str, module: str, message: str, details: str = None):
        """Log system event to database and enhanced logger"""
        try:
            if hasattr(self, 'logger_system') and self.logger_system:
                if level.upper() == 'INFO':
                    self.logger_system.log_ml_event('system_event', {
                        'level': level,
                        'module': module,
                        'message': message,
                        'details': details
                    })
                else:
                    self.logger_system.log_trading_event('system_event', 'SYSTEM', {
                        'level': level,
                        'module': module,
                        'message': message,
                        'details': details
                    })
            session = db_connection.get_session()
            log_entry = SystemLog(level=level, module=module, message=message, details=details)
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception as e:
            self.logger.error(f"Failed to log system event: {e}")
    
    def start(self):
        """Start the complete trading bot system"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting AI Trading Bot System")
            self.logger.info("=" * 60)
            self.log_system_event('INFO', 'main', 'AI Trading Bot system startup initiated')
            if not self.initialize_database():
                raise Exception("Database initialization failed")
            if not self.initialize_trading_engine():
                raise Exception("Trading engine initialization failed")
            if not self.initialize_web_interface():
                raise Exception("Web interface initialization failed")
            if not self.start_web_interface():
                raise Exception("Failed to start web interface")
            time.sleep(2)
            self.logger.info("Starting complete trading system...")
            self.trading_engine.start_system()
            self.running = True
            self.log_system_event('INFO', 'main', 'AI Trading Bot system started successfully')
            self.logger.info("=" * 60)
            self.logger.info("🚀 AI Trading Bot is now running!")
            self.logger.info(f"📊 Web Dashboard: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
            self.logger.info(f"💰 Demo Mode: ${TRADING_CONFIG['demo_balance']}")
            self.logger.info(f"📈 Symbols: {', '.join(TRADING_CONFIG['symbols'])}")
            self.logger.info(f"⏱️  Timeframe: {TRADING_CONFIG['timeframe']}")
            self.logger.info(f"🎯 Confidence Threshold: {TRADING_CONFIG['confidence_threshold']*100}%")
            self.logger.info("=" * 60)
            self.main_loop()
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
            self.shutdown()
        except Exception as e:
            self.logger.error(f"Failed to start trading bot: {e}")
            self.log_system_event('ERROR', 'main', f'System startup failed: {str(e)}')
            self.shutdown()
            sys.exit(1)
    
    def main_loop(self):
        try:
            while self.running and not self.shutdown_event.is_set():
                self.check_system_health()
                self.shutdown_event.wait(timeout=60)
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.shutdown()
    
    def check_system_health(self):
        try:
            if not db_connection.test_connection():
                self.logger.error("Database connection lost")
                self.log_system_event('ERROR', 'health', 'Database connection lost')
            if self.trading_engine:
                status = self.trading_engine.get_system_status()
                if not status.get('is_running', False) and self.running:
                    self.logger.warning("Trading engine not running")
            if datetime.now().minute % 15 == 0:
                self.logger.debug("System health check completed")
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    def shutdown(self):
        try:
            self.logger.info("Initiating graceful shutdown...")
            self.running = False
            self.shutdown_event.set()
            if self.trading_engine:
                self.logger.info("Stopping trading system...")
                self.trading_engine.stop_system()
            self.log_system_event('INFO', 'main', 'AI Trading Bot system shutdown completed')
            self.logger.info("Graceful shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          AI TRADING BOT SYSTEM                              ║
║                                                                              ║
║  🤖 Advanced AI-powered cryptocurrency trading bot                          ║
║  📊 196+ Technical indicators with RFE feature selection                    ║
║  💹 Advanced TP/SL management with trailing stops                           ║
║  🌐 Real-time web dashboard for monitoring                                  ║
║  🔄 4-hour timeframe with >70% confidence signals                           ║
║  💰 Demo and live trading support                                           ║
║                                                                              ║
║  Created for: amirsofali3/TB repository                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    app = TradingBotApplication()
    app.start()

if __name__ == "__main__":
    main()
