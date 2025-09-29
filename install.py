#!/usr/bin/env python3
"""
AI Trading Bot Installation and Setup Script

This script helps set up the trading bot environment and validates the configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     AI TRADING BOT INSTALLATION                             ║
║                                                                              ║
║  This script will help you set up the AI Trading Bot environment           ║
║  and validate your configuration.                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is 3.8+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but you have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def check_pip():
    """Check if pip is available"""
    print("📦 Checking pip...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def create_virtual_environment():
    """Create virtual environment"""
    print("🔧 Creating virtual environment...")
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📥 Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = Path("venv/Scripts/pip")
    else:  # Unix/Linux/Mac
        pip_path = Path("venv/bin/pip")
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        
        # Install requirements
        subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['logs', 'models', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directories created")

def setup_environment_file():
    """Set up environment configuration file"""
    print("⚙️ Setting up environment configuration...")
    
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if env_path.exists():
        print("⚠️  .env file already exists")
        return True
    
    if env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        print("✅ .env file created from template")
        print("""
⚠️  IMPORTANT: Please edit the .env file with your actual configuration:
   - Database credentials (MySQL)
   - CoinEx API keys (for live trading)
   - Other settings as needed
        """)
        return True
    else:
        print("❌ .env.example file not found")
        return False

def validate_requirements():
    """Validate that all requirements are available"""
    print("🔍 Validating installation...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("❌ Virtual environment not found")
        return False
    
    # Check if .env exists
    if not Path('.env').exists():
        print("❌ .env file not found")
        return False
    
    print("✅ Installation validation passed")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                             INSTALLATION COMPLETE!                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎉 AI Trading Bot has been installed successfully!

📋 Next Steps:

1. 📝 Configure your environment:
   Edit the .env file with your database and API credentials:
   nano .env

2. 🗄️ Prepare your database:
   - Create MySQL database named 'TB'
   - Create 'candles' table with historical OHLCV data
   - Ensure you have data for: BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT

3. 🚀 Run the bot:
   ./run.sh
   
   OR manually:
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   python main.py

4. 🌐 Access the dashboard:
   Open http://localhost:5000 in your browser

⚠️  Important Notes:
- The bot starts in DEMO mode by default (safe for testing)
- Make sure your database has sufficient historical data
- For live trading, configure your CoinEx API credentials
- Monitor the web dashboard for training progress and signals

📚 Documentation:
- README.md for detailed information
- config/settings.py for configuration options
- Check logs/ directory for detailed logging

Happy Trading! 🚀
    """)

def main():
    """Main installation process"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Installation steps
    steps = [
        ("Virtual Environment", create_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("Environment File", setup_environment_file),
        ("Validation", validate_requirements)
    ]
    
    for step_name, step_function in steps:
        try:
            if not step_function():
                print(f"❌ Installation failed at step: {step_name}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")
            sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()