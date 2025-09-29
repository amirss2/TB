# AI Trading Bot

This is a comprehensive AI-powered cryptocurrency trading bot built for automated trading with advanced features.

## Features

### 🤖 AI Model
- **Machine Learning**: RandomForest, GradientBoosting, and LogisticRegression models
- **Dynamic Feature Selection**: Intelligent importance-based correlation pruning and iterative optimization that dynamically determines optimal feature count based on macro F1 performance, replacing fixed 50-feature limits with adaptive selection
- **Traditional Feature Selection**: Alternative RFE/SHAP methods for compatibility
- **Training Data**: Uses 58,000+ historical OHLCV data points from MySQL database
- **Confidence Threshold**: Only executes trades with >70% model confidence
- **Auto-Retraining**: Continuous model improvement based on market conditions

### 📊 Technical Analysis
- **196+ Technical Indicators**: Complete suite including:
  - Moving Averages (SMA, EMA, various periods)
  - Momentum indicators (RSI, Stochastic, Williams %R, ROC, MFI)
  - Volatility indicators (Bollinger Bands, ATR)
  - Trend indicators (MACD, Ichimoku, SuperTrend)
  - Volume indicators (VWAP, Money Flow)
  - Custom combinations and derivatives
- **Real-time Calculation**: All indicators calculated on live market data
- **4-Hour Timeframe**: Optimized for 4H candlestick analysis

### 💹 Advanced Trading System
- **Position Management**: Sophisticated TP/SL system with:
  - 3-level Take Profit (TP1: 3%, TP2: 5%, TP3: 8%)
  - Trailing Stop Loss that moves up with profit levels
  - Emergency close on opposite high-confidence signals
- **Risk Management**: 2% risk per trade with proper position sizing
- **Multi-Symbol Support**: BTC, ETH, SOL, DOGE trading pairs
- **Demo Mode**: Risk-free testing with $100 demo balance

### 🌐 Real-Time Web Dashboard
- **Live Monitoring**: Real-time updates of all system components
- **Training Progress**: Visual feedback during model training and RFE
- **Position Tracking**: Real-time P&L and position management
- **Signal History**: Complete log of all trading signals with confidence scores
- **Market Overview**: Live price feeds and 24h change tracking
- **System Controls**: Start/stop trading, model retraining, position closure
- **Settings Management**: Adjustable demo balance and confidence thresholds

### 🔗 Exchange Integration
- **CoinEx API**: Full integration for live trading
- **Real-time Data**: Second-by-second price updates
- **Order Management**: Market and limit order support
- **Balance Tracking**: Real-time account balance monitoring

### 📈 Performance Analytics
- **Daily/Monthly P&L**: Comprehensive profit tracking
- **Win Rate Calculation**: Statistical analysis of trading performance
- **Portfolio Value Tracking**: Real-time portfolio valuation
- **Trade History**: Complete record of all executed trades

## Installation

### Prerequisites
- Python 3.8+
- MySQL Database with TB database and candles table
- CoinEx API credentials (for live trading)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/amirsofali3/TB.git
   cd TB
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your database and API credentials
   ```

4. **Database Setup:**
   Ensure your MySQL database has the `TB` database with a `candles` table containing historical OHLCV data for:
   - BTCUSDT
   - ETHUSDT  
   - SOLUSDT
   - DOGEUSDT

   Table structure:
   ```sql
   CREATE TABLE candles (
       id INT AUTO_INCREMENT PRIMARY KEY,
       symbol VARCHAR(20) NOT NULL,
       timestamp INT NOT NULL,
       open FLOAT NOT NULL,
       high FLOAT NOT NULL,
       low FLOAT NOT NULL,
       close FLOAT NOT NULL,
       volume FLOAT NOT NULL,
       created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
       INDEX idx_symbol_timestamp (symbol, timestamp)
   );
   ```

## Usage

### Quick Start
```bash
python main.py
```

The system will:
1. Initialize database connections
2. Start the web dashboard on http://localhost:5000
3. Begin model training with intelligent feature selection
4. Start real-time data fetching
5. Begin trading analysis (demo mode by default)

### Web Dashboard
Access the dashboard at `http://localhost:5000` to:
- Monitor training progress
- View real-time trading signals
- Track active positions and P&L
- Control trading start/stop
- Retrain the AI model
- Adjust settings

### Configuration
Key settings in `config/settings.py`:
- `TRADING_CONFIG`: Demo balance, symbols, timeframe, confidence threshold
- `TP_SL_CONFIG`: Take profit and stop loss levels
- `ML_CONFIG`: Training data size, dynamic feature selection, macro F1 optimization
- `COINEX_CONFIG`: API credentials for live trading

## Architecture

### Project Structure
```
TB/
├── main.py                 # Main application entry point
├── config/
│   └── settings.py        # Configuration management
├── database/
│   ├── connection.py      # Database connection handling
│   └── models.py         # SQLAlchemy ORM models
├── indicators/
│   ├── calculator.py     # Technical indicators calculation
│   └── definitions.py    # Indicator definitions loader
├── ml/
│   ├── model.py          # AI trading model
│   ├── trainer.py        # Model training pipeline
│   └── feature_selection.py # RFE feature selection
├── trading/
│   ├── engine.py         # Main trading engine
│   ├── position_manager.py # TP/SL position management
│   └── coinex_api.py     # Exchange API client
├── data/
│   └── fetcher.py        # Real-time data fetching
├── web/
│   ├── app.py            # Flask web application
│   └── templates/
│       └── dashboard.html # Web dashboard interface
└── requirements.txt       # Python dependencies
```

### Component Flow
1. **Data Layer**: Real-time price feeds + historical data from database
2. **Indicator Engine**: Calculates 196+ technical indicators
3. **AI Model**: Processes indicators to generate trading signals
4. **Trading Engine**: Executes trades based on AI signals
5. **Position Manager**: Handles TP/SL and risk management
6. **Web Dashboard**: Provides real-time monitoring and control

## Key Features Detail

### Dynamic Feature Selection
The system intelligently selects features using multiple approaches:

**Dynamic Mode (Performance-Driven)**:
- **Importance-Based Correlation Pruning**: When features are highly correlated (>95%), keeps the more important feature based on RandomForest importance scores
- **Adaptive Feature Count**: No fixed limit - dynamically determines optimal feature count through iterative pruning
- **Macro F1 Optimization**: Uses stratified cross-validation to optimize for macro F1 score rather than arbitrary feature counts
- **Safety Mechanisms**: Comprehensive fallbacks prevent over-pruning and ensure minimum feature requirements
- **Detailed Logging**: Complete iteration history with scores, improvements, and feature changes
- **History Persistence**: Saves selection history with optional truncation for storage efficiency

**Traditional Modes**:
- **RFE**: Recursive Feature Elimination with RandomForest
- **SHAP**: SHAP-based importance ranking
- **Hybrid**: Combination of RFE and statistical selection

**Configuration**:
```yaml
feature_selection:
  mode: "dynamic"  # or "rfe", "shap", "hybrid"
  dynamic:
    min_features: 20          # Minimum features to retain
    drop_fraction: 0.05       # Fraction to drop per iteration
    corr_threshold: 0.95      # Correlation threshold for pruning
    tolerance: 0.003          # Minimum improvement to continue
    metric: 'macro_f1'        # Optimization metric
    cv_splits: 3              # Cross-validation splits
    max_iterations: 50        # Maximum iteration limit
    max_history_length: 100   # History truncation for efficiency

data:
  min_fetch_interval_seconds: 30  # Throttling for API calls
```

See [docs/dynamic_feature_selection.md](docs/dynamic_feature_selection.md) for detailed documentation.

### TP/SL Management
Advanced position management with:
- **Initial SL**: 2% below entry (LONG) or above entry (SHORT)
- **TP1 at 3%**: Move SL to breakeven (entry price)
- **TP2 at 5%**: Move SL to TP1 level
- **TP3 at 8%**: Move SL to TP2 level
- **Emergency Close**: Close on opposite signal >70% confidence

### Real-Time Updates
- Price updates every second
- Historical data updates every 4 hours (timeframe boundary)
- Model predictions on timeframe completion
- Position monitoring continuous
- Dashboard updates every 5 seconds

## Safety Features

### Demo Mode
- Default operation in risk-free demo mode
- $100 virtual balance (configurable)
- All trading logic identical to live mode
- Safe testing environment for strategy validation

### Risk Management
- Maximum 2% risk per trade
- Position sizing based on available balance
- Maximum concurrent positions limit
- Confidence threshold enforcement
- Emergency stop mechanisms

### Error Handling
- Comprehensive error logging
- Graceful degradation on component failures
- Database connection recovery
- API rate limit handling
- System health monitoring

## Monitoring and Logging

### Web Dashboard Metrics
- Real-time system status
- Training progress with detailed stages
- Live position tracking with P&L
- Signal history with confidence scores
- Market overview with 24h changes
- Model performance metrics
- System logs and error tracking

### Database Logging
- All trading signals stored
- Complete position history
- Model training records
- Performance metrics tracking
- System event logging

## Development and Customization

### Adding New Indicators
1. Add indicator definition to `technical_indicators_only.csv`
2. Implement calculation in `indicators/calculator.py`
3. System will automatically include in RFE selection

### Model Customization
- Switch between RandomForest, GradientBoosting, LogisticRegression
- Adjust hyperparameters in `ml/model.py`
- Modify feature selection criteria in `ml/enhanced_feature_selection.py`
- Cross-validation optimization uses F1 macro metric by default

### Exchange Integration
- Implement new exchange by extending base API pattern
- Update configuration in `config/settings.py`
- Maintain consistent interface for trading engine

## Disclaimer

This trading bot is for educational and research purposes. Cryptocurrency trading involves significant risk. Always test thoroughly in demo mode before considering live trading. The authors are not responsible for any financial losses incurred through the use of this software.

## License

MIT License - see LICENSE file for details.

## Diagnostics and System Analysis

### Log Analysis Tool

The repository includes a diagnostic utility to analyze system logs and generate comprehensive reports:

```bash
python scripts/log_summary.py --input LOG.txt --out-json reports/diagnostic_stats.json --out-md reports/diagnostic_stats.md
```

**Features:**
- Parses LOG.txt and extracts key metrics
- Counts log levels (INFO, WARNING, ERROR)
- Identifies API errors and connection issues
- Tracks trading signals (BUY/SELL/HOLD) across symbols
- Analyzes latency patterns and performance metrics
- Generates both JSON and Markdown summary reports

### Diagnostic Prompts

For comprehensive system analysis, use the Persian diagnostic prompts in `docs/DIAGNOSTIC_PROMPTS.md`:

1. **تشخیص (Diagnostic)**: Generates a structured diagnostic report analyzing:
   - System architecture and performance
   - AI model effectiveness
   - Trading activity patterns
   - Error analysis and technical performance
   - Code quality and monitoring systems

2. **برنامه اصلاحات (Remediation Plan)**: Creates actionable improvement plans based on diagnostic findings

**Usage Workflow:**
1. Generate log statistics: `python scripts/log_summary.py --input LOG.txt --out-json reports/diagnostic_stats.json --out-md reports/diagnostic_stats.md`
2. Use diagnostic prompt with repository code + LOG.txt + generated stats
3. Apply remediation prompt to create improvement plan

### Configuration Templates

- `config/logging_example.yaml`: Example configuration for standardized JSON logging
- Future enhancement template for structured logging implementation

## Support

For issues and questions:
- Create GitHub issues for bugs and feature requests
- Check the logs directory for detailed error information
- Use the web dashboard system logs for real-time debugging
- Run diagnostic analysis using `scripts/log_summary.py` for comprehensive system insights