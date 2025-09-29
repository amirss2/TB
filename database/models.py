from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Candle(Base):
    """
    Candlestick data model - matches the existing database structure
    """
    __tablename__ = 'candles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(Integer, nullable=False, index=True)  # Unix timestamp
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Candle(symbol='{self.symbol}', timestamp={self.timestamp}, close={self.close})>"

class TradingSignal(Base):
    """
    AI model trading signals
    """
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=func.now())
    signal_type = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    price = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=True)
    indicators_used = Column(Text, nullable=True)  # JSON string of active indicators
    executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<TradingSignal(symbol='{self.symbol}', type='{self.signal_type}', confidence={self.confidence})>"

class Position(Base):
    """
    Trading positions
    """
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'LONG' or 'SHORT'
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    
    # TP/SL Management
    initial_sl = Column(Float, nullable=True)
    current_sl = Column(Float, nullable=True)
    tp1_price = Column(Float, nullable=True)
    tp2_price = Column(Float, nullable=True)
    tp3_price = Column(Float, nullable=True)
    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    
    # Status and tracking
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, STOPPED
    pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    
    # Timestamps
    opened_at = Column(DateTime, default=func.now())
    closed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', entry={self.entry_price}, status='{self.status}')>"

class ModelTraining(Base):
    """
    AI model training history and metrics
    """
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, unique=True)
    training_start = Column(DateTime, nullable=False)
    training_end = Column(DateTime, nullable=True)
    
    # Training data info
    total_samples = Column(Integer, nullable=True)
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    
    # Feature selection info
    total_indicators = Column(Integer, nullable=True)
    selected_indicators = Column(Integer, nullable=True)
    selected_features = Column(Text, nullable=True)  # JSON array of selected indicator names
    
    # Model performance
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default='TRAINING')  # TRAINING, COMPLETED, FAILED, ACTIVE
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<ModelTraining(version='{self.model_version}', status='{self.status}', accuracy={self.accuracy})>"

class TradingMetrics(Base):
    """
    Trading performance metrics
    """
    __tablename__ = 'trading_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Daily metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    daily_pnl_percentage = Column(Float, default=0.0)
    
    # Portfolio metrics
    portfolio_value = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    
    # Performance ratios
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<TradingMetrics(date='{self.date}', pnl={self.daily_pnl}, trades={self.total_trades})>"

class SystemLog(Base):
    """
    System operation logs
    """
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=func.now(), index=True)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON string for additional data
    
    def __repr__(self):
        return f"<SystemLog(level='{self.level}', module='{self.module}', time='{self.timestamp}')>"