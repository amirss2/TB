"""
Prediction Scheduler and Adaptive Threshold Management
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PredictionResult:
    """Structure for prediction results"""
    timestamp: datetime
    symbol: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    features_hash: str
    threshold_used: float

@dataclass
class SignalEvent:
    """Structure for trading signals"""
    id: str
    timestamp: datetime
    symbol: str
    direction: str
    confidence: float
    price: float
    threshold_used: float
    features_hash: str
    model_version: str

class AdaptiveThresholdManager:
    """Manages adaptive confidence thresholds to maintain target signal rates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('adaptive_threshold', {})
        self.enabled = self.config.get('enabled', True)
        
        # Threshold parameters
        self.target_signals_per_24h = self.config.get('target_signals_per_24h', 5)
        self.min_threshold = self.config.get('min_threshold', 0.5)
        self.max_threshold = self.config.get('max_threshold', 0.85)
        self.adjustment_rate = self.config.get('adjustment_rate', 0.05)
        self.evaluation_window = self.config.get('evaluation_window', 24)  # hours
        
        # Current state
        self.current_thresholds = {}  # per symbol
        self.signal_history = {}  # per symbol
        self.prediction_history = {}  # per symbol
        
        self.logger = logging.getLogger(__name__)
        
        if self.enabled:
            self.logger.info(f"Adaptive threshold manager initialized - target: {self.target_signals_per_24h} signals/24h")
        else:
            self.logger.info("Adaptive threshold manager disabled")
    
    def get_threshold(self, symbol: str) -> float:
        """Get current threshold for symbol"""
        if not self.enabled:
            return 0.7  # Default static threshold
        
        if symbol not in self.current_thresholds:
            # Initialize with middle value
            initial_threshold = (self.min_threshold + self.max_threshold) / 2
            self.current_thresholds[symbol] = initial_threshold
            self.signal_history[symbol] = []
            self.prediction_history[symbol] = []
            
            self.logger.info(f"Initialized threshold for {symbol}: {initial_threshold:.3f}")
        
        return self.current_thresholds[symbol]
    
    def record_prediction(self, symbol: str, confidence: float, prediction: str, 
                         timestamp: datetime = None):
        """Record a prediction for threshold adjustment"""
        if not self.enabled:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        
        self.prediction_history[symbol].append({
            'timestamp': timestamp,
            'confidence': confidence,
            'prediction': prediction
        })
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(hours=self.evaluation_window)
        self.prediction_history[symbol] = [
            pred for pred in self.prediction_history[symbol]
            if pred['timestamp'] > cutoff_time
        ]
    
    def record_signal(self, symbol: str, direction: str, confidence: float, 
                     timestamp: datetime = None):
        """Record a generated signal"""
        if not self.enabled:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append({
            'timestamp': timestamp,
            'direction': direction,
            'confidence': confidence
        })
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(hours=self.evaluation_window)
        self.signal_history[symbol] = [
            signal for signal in self.signal_history[symbol]
            if signal['timestamp'] > cutoff_time
        ]
        
        # Update threshold after recording signal
        self._update_threshold(symbol)
    
    def _update_threshold(self, symbol: str):
        """Update threshold based on recent signal rate"""
        if not self.enabled or symbol not in self.signal_history:
            return
        
        current_threshold = self.current_thresholds.get(symbol, 0.7)
        recent_signals = len(self.signal_history[symbol])
        
        # Calculate signal rate per 24h
        if recent_signals == 0:
            signal_rate_24h = 0
        else:
            # Scale to 24h rate based on evaluation window
            signal_rate_24h = recent_signals * (24 / self.evaluation_window)
        
        # Determine if we need adjustment
        target_rate = self.target_signals_per_24h
        rate_difference = signal_rate_24h - target_rate
        
        # Calculate adjustment
        if abs(rate_difference) > target_rate * 0.2:  # 20% tolerance
            if rate_difference > 0:
                # Too many signals - increase threshold
                new_threshold = min(
                    self.max_threshold,
                    current_threshold + self.adjustment_rate
                )
            else:
                # Too few signals - decrease threshold
                new_threshold = max(
                    self.min_threshold,
                    current_threshold - self.adjustment_rate
                )
            
            # Only update if change is significant
            if abs(new_threshold - current_threshold) > 0.01:
                self.current_thresholds[symbol] = new_threshold
                
                self.logger.info(
                    f"Threshold adjusted for {symbol}: {current_threshold:.3f} -> {new_threshold:.3f} "
                    f"(signals/24h: {signal_rate_24h:.1f}, target: {target_rate})"
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get adaptive threshold status"""
        status = {
            'enabled': self.enabled,
            'target_signals_per_24h': self.target_signals_per_24h,
            'evaluation_window_hours': self.evaluation_window,
            'symbols': {}
        }
        
        if self.enabled:
            for symbol in self.current_thresholds.keys():
                recent_signals = len(self.signal_history.get(symbol, []))
                recent_predictions = len(self.prediction_history.get(symbol, []))
                
                signal_rate_24h = recent_signals * (24 / self.evaluation_window) if recent_signals > 0 else 0
                
                status['symbols'][symbol] = {
                    'current_threshold': self.current_thresholds.get(symbol, 0.7),
                    'recent_signals': recent_signals,
                    'recent_predictions': recent_predictions,
                    'signal_rate_24h': signal_rate_24h,
                    'threshold_range': [self.min_threshold, self.max_threshold]
                }
        
        return status

class PredictionScheduler:
    """Schedules regular predictions and manages signal generation"""
    
    def __init__(self, config: Dict[str, Any], model, data_fetcher, threshold_manager: AdaptiveThresholdManager):
        self.config = config
        self.prediction_config = config.get('prediction', {})
        
        self.model = model
        self.data_fetcher = data_fetcher
        self.threshold_manager = threshold_manager
        
        # Configuration
        self.interval = self.prediction_config.get('scheduler_interval', 120)  # seconds
        self.intracandle_predictions = self.prediction_config.get('intracandle_predictions', True)
        self.min_data_points = self.prediction_config.get('min_data_points', 200)
        
        # State
        self.is_running = False
        self.scheduler_thread = None
        self.last_prediction_times = {}
        self.prediction_callbacks = []
        self.signal_callbacks = []
        
        # Statistics
        self.prediction_stats = {
            'total_predictions': 0,
            'signals_generated': 0,
            'last_run': None,
            'next_run': None,
            'errors': 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def add_prediction_callback(self, callback: Callable[[PredictionResult], None]):
        """Add callback for prediction events"""
        self.prediction_callbacks.append(callback)
    
    def add_signal_callback(self, callback: Callable[[SignalEvent], None]):
        """Add callback for signal events"""
        self.signal_callbacks.append(callback)
    
    def start(self):
        """Start the prediction scheduler"""
        if self.is_running:
            self.logger.warning("Prediction scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info(f"Prediction scheduler started - interval: {self.interval}s")
    
    def stop(self):
        """Stop the prediction scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("Prediction scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                self.prediction_stats['next_run'] = datetime.now() + timedelta(seconds=self.interval)
                
                # Run predictions for all symbols
                self._run_predictions()
                
                self.prediction_stats['last_run'] = datetime.now()
                
                # Sleep until next run
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in prediction scheduler loop: {e}")
                self.prediction_stats['errors'] += 1
                time.sleep(30)  # Wait before retrying
    
    def _run_predictions(self):
        """Run predictions for all configured symbols"""
        symbols = self.config.get('trading', {}).get('symbols', [])
        
        for symbol in symbols:
            try:
                self._predict_for_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Prediction failed for {symbol}: {e}")
    
    def _predict_for_symbol(self, symbol: str):
        """Generate prediction for a specific symbol"""
        try:
            # Get latest data
            latest_data = self.data_fetcher.get_latest_data_with_indicators(symbol)
            
            if latest_data is None or len(latest_data) < self.min_data_points:
                self.logger.warning(f"Insufficient data for {symbol}: {len(latest_data) if latest_data is not None else 0} points")
                return
            
            # Get features from the latest row
            if not hasattr(self.model, 'feature_names') or not self.model.feature_names:
                self.logger.warning(f"Model feature names not available for {symbol}")
                return
            
            # Prepare feature vector
            feature_vector = {}
            latest_row = latest_data.iloc[-1]
            
            for feature_name in self.model.feature_names:
                if feature_name in latest_data.columns:
                    feature_vector[feature_name] = latest_row[feature_name]
                else:
                    feature_vector[feature_name] = 0.0  # Default value for missing features
            
            # Generate prediction
            prediction_result = self.model.predict_single(feature_vector)
            
            if not prediction_result:
                self.logger.warning(f"No prediction result for {symbol}")
                return
            
            # Create feature hash for tracking
            feature_hash = str(hash(str(sorted(feature_vector.items()))))[:8]
            
            # Get adaptive threshold
            threshold = self.threshold_manager.get_threshold(symbol)
            
            # Create prediction result
            pred_result = PredictionResult(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction=prediction_result['signal'],
                confidence=prediction_result['confidence'],
                probabilities=prediction_result['probabilities'],
                features_hash=feature_hash,
                threshold_used=threshold
            )
            
            # Record prediction for threshold management
            self.threshold_manager.record_prediction(
                symbol, pred_result.confidence, pred_result.prediction, pred_result.timestamp
            )
            
            # Update statistics
            self.prediction_stats['total_predictions'] += 1
            
            # Log prediction event
            self.logger.info(
                f"PREDICTION: {symbol} -> {pred_result.prediction} "
                f"(confidence: {pred_result.confidence:.3f}, threshold: {threshold:.3f})"
            )
            
            # Call prediction callbacks
            for callback in self.prediction_callbacks:
                try:
                    callback(pred_result)
                except Exception as e:
                    self.logger.error(f"Prediction callback error: {e}")
            
            # Check if prediction meets threshold for signal generation
            if pred_result.confidence >= threshold and pred_result.prediction != 'HOLD':
                self._generate_signal(symbol, pred_result, latest_row.get('close', 0))
            
        except Exception as e:
            self.logger.error(f"Error in prediction for {symbol}: {e}")
    
    def _generate_signal(self, symbol: str, prediction: PredictionResult, price: float):
        """Generate trading signal from prediction"""
        try:
            # Create signal event
            signal = SignalEvent(
                id=f"{symbol}_{int(prediction.timestamp.timestamp())}",
                timestamp=prediction.timestamp,
                symbol=symbol,
                direction=prediction.prediction,
                confidence=prediction.confidence,
                price=price,
                threshold_used=prediction.threshold_used,
                features_hash=prediction.features_hash,
                model_version=getattr(self.model, 'model_version', 'unknown')
            )
            
            # Record signal for threshold management
            self.threshold_manager.record_signal(
                symbol, signal.direction, signal.confidence, signal.timestamp
            )
            
            # Update statistics
            self.prediction_stats['signals_generated'] += 1
            
            # Log signal generation
            self.logger.info(
                f"SIGNAL GENERATED: {symbol} {signal.direction} @ {price:.4f} "
                f"(confidence: {signal.confidence:.3f})"
            )
            
            # Call signal callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"Signal callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
    
    def force_prediction(self, symbol: str = None) -> Dict[str, Any]:
        """Force immediate prediction for debugging"""
        try:
            if symbol:
                symbols = [symbol]
            else:
                symbols = self.config.get('trading', {}).get('symbols', [])
            
            results = {}
            
            for sym in symbols:
                try:
                    self._predict_for_symbol(sym)
                    results[sym] = 'success'
                except Exception as e:
                    results[sym] = f'error: {str(e)}'
            
            return {
                'status': 'completed',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get prediction scheduler status"""
        return {
            'is_running': self.is_running,
            'interval_seconds': self.interval,
            'intracandle_predictions': self.intracandle_predictions,
            'min_data_points': self.min_data_points,
            'statistics': self.prediction_stats,
            'adaptive_threshold_status': self.threshold_manager.get_status()
        }