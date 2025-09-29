from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime, timedelta
import json

from database.connection import db_connection
from database.models import *
from config.settings import WEB_CONFIG

def create_app(trading_engine=None):
    """Create Flask application"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = WEB_CONFIG['secret_key']
    
    # Enable CORS for development
    CORS(app)
    
    # Store trading engine reference
    app.trading_engine = trading_engine
    
    @app.route('/')
    def dashboard():
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/api/system/status')
    def system_status():
        """Get system status"""
        try:
            if app.trading_engine:
                status = app.trading_engine.get_system_status()
                training_progress = app.trading_engine.trainer.get_training_progress()
                
                return jsonify({
                    'success': True,
                    'system_status': status,
                    'training_progress': training_progress
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Trading engine not initialized'
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/positions')
    def get_positions():
        """Get active positions"""
        try:
            if app.trading_engine:
                positions = app.trading_engine.position_manager.get_active_positions()
                return jsonify({
                    'success': True,
                    'positions': positions
                })
            else:
                return jsonify({'success': False, 'error': 'Trading engine not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/signals')
    def get_signals():
        """Get recent trading signals"""
        try:
            session = db_connection.get_session()
            
            # Get last 50 signals
            signals = session.query(TradingSignal).order_by(
                TradingSignal.timestamp.desc()
            ).limit(50).all()
            
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'id': signal.id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'timestamp': signal.timestamp.isoformat(),
                    'executed': signal.executed
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'signals': signal_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/metrics/daily')
    def get_daily_metrics():
        """Get daily trading metrics"""
        try:
            session = db_connection.get_session()
            
            # Get last 30 days of metrics
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            metrics = session.query(TradingMetrics).filter(
                TradingMetrics.date >= thirty_days_ago
            ).order_by(TradingMetrics.date.desc()).all()
            
            metrics_data = []
            for metric in metrics:
                metrics_data.append({
                    'date': metric.date.isoformat(),
                    'daily_pnl': metric.daily_pnl,
                    'daily_pnl_percentage': metric.daily_pnl_percentage,
                    'total_trades': metric.total_trades,
                    'winning_trades': metric.winning_trades,
                    'losing_trades': metric.losing_trades,
                    'win_rate': metric.win_rate,
                    'portfolio_value': metric.portfolio_value
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'metrics': metrics_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/model/info')
    def get_model_info():
        """Get model information and training history"""
        try:
            session = db_connection.get_session()
            
            # Get latest model training
            latest_training = session.query(ModelTraining).order_by(
                ModelTraining.training_start.desc()
            ).first()
            
            training_history = session.query(ModelTraining).order_by(
                ModelTraining.training_start.desc()
            ).limit(10).all()
            
            model_info = None
            if latest_training:
                model_info = {
                    'model_version': latest_training.model_version,
                    'training_start': latest_training.training_start.isoformat() if latest_training.training_start else None,
                    'training_end': latest_training.training_end.isoformat() if latest_training.training_end else None,
                    'status': latest_training.status,
                    'accuracy': latest_training.accuracy,
                    'precision': latest_training.precision,
                    'recall': latest_training.recall,
                    'f1_score': latest_training.f1_score,
                    'total_samples': latest_training.total_samples,
                    'selected_indicators': latest_training.selected_indicators,
                    'selected_features': json.loads(latest_training.selected_features) if latest_training.selected_features else []
                }
            
            history_data = []
            for training in training_history:
                history_data.append({
                    'model_version': training.model_version,
                    'training_start': training.training_start.isoformat() if training.training_start else None,
                    'status': training.status,
                    'accuracy': training.accuracy,
                    'selected_indicators': training.selected_indicators
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'current_model': model_info,
                'training_history': history_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/market/overview')
    def get_market_overview():
        """Get market overview"""
        try:
            if app.trading_engine and hasattr(app.trading_engine, 'data_fetcher'):
                overview = app.trading_engine.data_fetcher.get_market_overview()
                return jsonify({
                    'success': True,
                    'market_overview': overview
                })
            else:
                return jsonify({'success': False, 'error': 'Data fetcher not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/trading/start', methods=['POST'])
    def start_trading():
        """Start trading"""
        try:
            if app.trading_engine:
                if not app.trading_engine.is_running:
                    app.trading_engine.start_trading()
                    return jsonify({'success': True, 'message': 'Trading started'})
                else:
                    return jsonify({'success': False, 'error': 'Trading is already running'})
            else:
                return jsonify({'success': False, 'error': 'Trading engine not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/trading/stop', methods=['POST'])
    def stop_trading():
        """Stop trading"""
        try:
            if app.trading_engine:
                app.trading_engine.stop_trading()
                return jsonify({'success': True, 'message': 'Trading stopped'})
            else:
                return jsonify({'success': False, 'error': 'Trading engine not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/model/retrain', methods=['POST'])
    def retrain_model():
        """Retrain the AI model"""
        try:
            if app.trading_engine:
                # Start retraining in background
                import threading
                
                def retrain():
                    try:
                        app.trading_engine.train_model(retrain=True)
                    except Exception as e:
                        logging.error(f"Error retraining model: {e}")
                
                retrain_thread = threading.Thread(target=retrain, daemon=True)
                retrain_thread.start()
                
                return jsonify({'success': True, 'message': 'Model retraining started'})
            else:
                return jsonify({'success': False, 'error': 'Trading engine not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/position/<int:position_id>/close', methods=['POST'])
    def close_position(position_id):
        """Close a specific position"""
        try:
            if app.trading_engine:
                success = app.trading_engine.position_manager.close_position(
                    position_id, "Manual close via dashboard"
                )
                if success:
                    return jsonify({'success': True, 'message': 'Position closed'})
                else:
                    return jsonify({'success': False, 'error': 'Failed to close position'})
            else:
                return jsonify({'success': False, 'error': 'Trading engine not available'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/settings/update', methods=['POST'])
    def update_settings():
        """Update trading settings"""
        try:
            data = request.get_json()
            
            if 'demo_balance' in data and app.trading_engine:
                app.trading_engine.demo_balance = float(data['demo_balance'])
            
            if 'confidence_threshold' in data and app.trading_engine and app.trading_engine.model:
                app.trading_engine.model.set_confidence_threshold(float(data['confidence_threshold']))
            
            return jsonify({'success': True, 'message': 'Settings updated'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/logs')
    def get_logs():
        """Get system logs"""
        try:
            session = db_connection.get_session()
            
            logs = session.query(SystemLog).order_by(
                SystemLog.timestamp.desc()
            ).limit(100).all()
            
            log_data = []
            for log in logs:
                log_data.append({
                    'id': log.id,
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'module': log.module,
                    'message': log.message,
                    'details': json.loads(log.details) if log.details else None
                })
            
            session.close()
            
            return jsonify({
                'success': True,
                'logs': log_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app