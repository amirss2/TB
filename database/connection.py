from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
import logging
from config.settings import DATABASE_CONFIG

# Try to import PyMySQL, fallback gracefully if not available
try:
    import pymysql
    pymysql.install_as_MySQLdb()
    PYMYSQL_AVAILABLE = True
except ImportError:
    logging.warning("PyMySQL not available, using SQLite only")
    PYMYSQL_AVAILABLE = False

class DatabaseConnection:
    def __init__(self):
        self.engine = None
        self.Session = None
        self.connection_string = self._build_connection_string()
        
    def _build_connection_string(self):
        """Build MySQL connection string"""
        try:
            return (f"mysql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
                    f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
                    f"/{DATABASE_CONFIG['database']}?charset={DATABASE_CONFIG['charset']}")
        except Exception as e:
            logging.warning(f"MySQL connection failed, falling back to SQLite: {e}")
            # Fallback to SQLite for testing/development
            return "sqlite:///trading_bot.db"
    
    def init_engine(self):
        """Initialize database engine"""
        try:
            # Try MySQL first if PyMySQL is available
            if PYMYSQL_AVAILABLE:
                mysql_connection_string = (f"mysql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}"
                                         f"@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}"
                                         f"/{DATABASE_CONFIG['database']}?charset={DATABASE_CONFIG['charset']}")
                
                try:
                    self.engine = create_engine(
                        mysql_connection_string,
                        pool_size=20,
                        max_overflow=0,
                        pool_pre_ping=True,
                        echo=False
                    )
                    
                    # Test the MySQL connection
                    with self.engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                        
                    logging.info("MySQL database engine initialized successfully")
                    self.Session = sessionmaker(bind=self.engine)
                    return True
                    
                except Exception as mysql_error:
                    logging.warning(f"MySQL connection failed: {mysql_error}")
            
            # Fallback to SQLite
            logging.info("Using SQLite database for development/testing...")
            
            sqlite_path = os.path.join(os.path.dirname(__file__), '..', 'trading_bot.db')
            self.engine = create_engine(
                f"sqlite:///{sqlite_path}",
                echo=False
            )
            
            # Test the SQLite connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            logging.info("SQLite database engine initialized successfully")
            self.Session = sessionmaker(bind=self.engine)
            return True
                
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            return False
    
    def get_session(self):
        """Get a new database session"""
        if not self.Session:
            self.init_engine()
        return self.Session()
    
    def test_connection(self):
        """Test database connection"""
        try:
            session = self.get_session()
            session.execute(text("SELECT 1"))
            session.close()
            return True
        except Exception as e:
            logging.error(f"Database connection test failed: {e}")
            return False

# Global database connection instance
db_connection = DatabaseConnection()