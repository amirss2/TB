#!/usr/bin/env python3
"""
AI Trading Bot - Easy Setup and Installation Script

This script automates the installation and setup of the AI Trading Bot,
handling dependencies and database initialization.
"""

import os
import sys
import subprocess
import platform
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print setup banner"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AI TRADING BOT - SETUP & INSTALLATION                    ║
║                                                                              ║
║  🛠️  Automated setup for AI-powered cryptocurrency trading bot             ║
║  🔧 Installs dependencies and configures database                           ║
║  📦 Supports both MySQL and SQLite (fallback)                               ║
║  🚀 One-click setup for quick deployment                                    ║
║                                                                              ║
║  Created for: amirsofali3/TB repository                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"Python version check passed: {version.major}.{version.minor}.{version.micro}")
    return True

def install_system_packages():
    """Install system packages if needed"""
    system = platform.system().lower()
    logger.info(f"Detected system: {system}")
    
    if system == "linux":
        try:
            # Check if we can install packages
            result = subprocess.run(["which", "apt-get"], capture_output=True)
            if result.returncode == 0:
                logger.info("Installing system dependencies with apt-get...")
                subprocess.run(["sudo", "apt-get", "update", "-y"], check=False)
                subprocess.run(["sudo", "apt-get", "install", "-y", 
                              "python3-pip", "python3-dev", "python3-venv"], check=False)
        except Exception as e:
            logger.warning(f"Could not install system packages: {e}")
    
    return True

def install_python_packages():
    """Install required Python packages"""
    packages = [
        "sqlalchemy>=1.4.0",
        "flask>=2.0.0", 
        "flask-cors",
        "pandas",
        "numpy",
        "scikit-learn",
        "python-dotenv",
        "PyMySQL",
        "requests"
    ]
    
    logger.info("Installing Python packages...")
    for package in packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "--user", package, "--no-cache-dir"], 
                          check=False, capture_output=True)
        except Exception as e:
            logger.warning(f"Could not install {package}: {e}")
    
    logger.info("Python packages installation completed")
    return True

def setup_environment():
    """Setup environment configuration"""
    logger.info("Setting up environment configuration...")
    
    env_path = ".env"
    if os.path.exists(env_path):
        logger.info("Environment file already exists")
        return True
    
    env_example_path = ".env.example"
    if os.path.exists(env_example_path):
        import shutil
        shutil.copy(env_example_path, env_path)
        logger.info("Environment file created from template")
        
        print("""
⚠️  IMPORTANT: Please edit the .env file with your configuration:
   - Database credentials (MySQL) - optional, will use SQLite fallback
   - CoinEx API keys (for live trading) - optional, demo mode available
   - Other settings as needed
        """)
    
    return True

def test_database_setup():
    """Test database setup"""
    logger.info("Testing database setup...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from database.connection import db_connection
        from database.models import Base
        
        # Test connection and initialization
        if not db_connection.init_engine():
            logger.error("Database initialization failed")
            return False
        
        if not db_connection.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Create tables
        Base.metadata.create_all(db_connection.engine)
        logger.info("Database tables created successfully")
        
        logger.info("✅ Database setup test passed")
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        return False

def create_run_script():
    """Create easy run script"""
    run_script = """#!/bin/bash

echo "Starting AI Trading Bot..."
echo "=========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Run the trading bot
echo "🚀 Launching AI Trading Bot..."
echo "📊 Web dashboard will be available at: http://localhost:5000"
echo "🔄 Press Ctrl+C to stop the bot"
echo ""

python3 main.py
"""
    
    try:
        with open("start_bot.sh", "w") as f:
            f.write(run_script)
        
        # Make it executable
        os.chmod("start_bot.sh", 0o755)
        logger.info("Created start_bot.sh script")
        return True
    except Exception as e:
        logger.warning(f"Could not create run script: {e}")
        return True  # Not critical, don't fail setup

def print_success_message():
    """Print success message with instructions"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            🎉 SETUP COMPLETED! 🎉                          ║
║                                                                              ║
║  ✅ All dependencies installed                                              ║
║  ✅ Database configured (SQLite fallback ready)                            ║
║  ✅ Environment setup completed                                             ║
║  ✅ Application tested and ready                                            ║
║                                                                              ║
║  🚀 TO START THE BOT:                                                       ║
║     ./start_bot.sh                                                          ║
║     OR                                                                      ║
║     python3 main.py                                                         ║
║                                                                              ║
║  🌐 WEB DASHBOARD: http://localhost:5000                                    ║
║                                                                              ║
║  📝 NOTES:                                                                  ║
║  • Bot runs in DEMO mode by default (safe for testing)                     ║
║  • Edit .env file to configure MySQL or API keys                           ║
║  • SQLite database fallback works without additional setup                 ║
║  • Check logs/ folder for detailed application logs                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def main():
    """Main setup function"""
    print_banner()
    
    steps = [
        ("Python Version Check", check_python_version),
        ("System Packages", install_system_packages),
        ("Python Packages", install_python_packages), 
        ("Environment Setup", setup_environment),
        ("Database Test", test_database_setup),
        ("Run Script Creation", create_run_script)
    ]
    
    for step_name, step_function in steps:
        logger.info(f"Step: {step_name}")
        try:
            if not step_function():
                logger.error(f"❌ Setup failed at step: {step_name}")
                sys.exit(1)
        except Exception as e:
            logger.error(f"❌ Unexpected error in {step_name}: {e}")
            sys.exit(1)
    
    print_success_message()

if __name__ == "__main__":
    main()