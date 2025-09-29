#!/usr/bin/env python3
"""
Database Setup Script for AI Trading Bot

This script creates the necessary database tables and handles initial setup.
"""

import os
import sys
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from database.connection import db_connection
from database.models import Base, SystemLog, Candle, TradingSignal, Position, ModelTraining, TradingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_database():
    """Create database if it doesn't exist"""
    from config.settings import DATABASE_CONFIG
    import pymysql
    
    try:
        # Connect without specifying database
        connection = pymysql.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            charset=DATABASE_CONFIG['charset']
        )
        
        with connection.cursor() as cursor:
            # Create database if it doesn't exist
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_CONFIG['database']}")
            logger.info(f"Database '{DATABASE_CONFIG['database']}' created or already exists")
        
        connection.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

def create_tables():
    """Create all database tables"""
    try:
        logger.info("Initializing database connection...")
        
        if not db_connection.init_engine():
            return False
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(db_connection.engine)
        
        logger.info("Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False

def test_database_connection():
    """Test database connection and basic operations"""
    try:
        logger.info("Testing database connection...")
        
        if not db_connection.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Test inserting a log entry
        session = db_connection.get_session()
        try:
            test_log = SystemLog(
                level='INFO',
                module='setup',
                message='Database setup test',
                details='Testing database functionality'
            )
            session.add(test_log)
            session.commit()
            
            # Query the log back
            log_count = session.query(SystemLog).filter(SystemLog.module == 'setup').count()
            logger.info(f"Database test successful. Found {log_count} test log entries.")
            
        finally:
            session.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting AI Trading Bot Database Setup")
    logger.info("=" * 50)
    
    # Step 1: Create database
    logger.info("Step 1: Creating database...")
    if not create_database():
        logger.error("Failed to create database. Please check your MySQL connection and credentials.")
        sys.exit(1)
    
    # Step 2: Create tables
    logger.info("Step 2: Creating tables...")
    if not create_tables():
        logger.error("Failed to create tables. Please check the database connection.")
        sys.exit(1)
    
    # Step 3: Test the setup
    logger.info("Step 3: Testing database setup...")
    if not test_database_connection():
        logger.error("Database test failed. Setup may be incomplete.")
        sys.exit(1)
    
    logger.info("=" * 50)
    logger.info("Database setup completed successfully!")
    logger.info("You can now run the trading bot with: python main.py")

if __name__ == "__main__":
    main()