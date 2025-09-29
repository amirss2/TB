#!/bin/bash

# AI Trading Bot Run Script

echo "🤖 Starting AI Trading Bot..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create it from .env.example and configure your settings."
    echo "   cp .env.example .env"
    echo "   nano .env  # Edit with your database and API credentials"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs models temp

# Run the application
echo "🚀 Starting AI Trading Bot..."
echo "================================"
python main.py