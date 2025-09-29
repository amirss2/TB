#!/bin/bash

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
