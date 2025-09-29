import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime

# Import configuration
from config.settings import XGB_PRO_CONFIG

class TradingModel:
    """
    AI Trading Model for signal generation
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.is_trained = False
        self.model_version = None
        self.confidence_threshold = 0.7
        # Map عدد کلاس ها -> لیبل نهایی (طبق دیتای فعلی)
        self.label_map = {0: 'SELL', 1: 'BUY', 2: 'HOLD'}
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on specified type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'xgboost_professional':
            # Professional XGBoost configuration with configurable parameters
            xgb_config = XGB_PRO_CONFIG
            n_estimators = xgb_config.get('n_estimators', 8000)
            early_stopping = xgb_config.get('early_stopping_rounds', 300)
            
            # Log XGBoost configuration
            self.logger.info(f"XGBoost n_estimators={n_estimators}, max_depth={xgb_config.get('max_depth', 12)}, learning_rate={xgb_config.get('learning_rate', 0.01)}")
            self.logger.info(f"XGBoost early_stopping_rounds={early_stopping}, features={len(self.feature_names) if self.feature_names else 'unknown'}")
            
            # Handle early stopping configuration (0 means disabled)
            early_stopping_param = None if early_stopping == 0 else early_stopping
            
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=xgb_config.get('max_depth', 12),
                learning_rate=xgb_config.get('learning_rate', 0.01),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                colsample_bylevel=xgb_config.get('colsample_bylevel', 0.8),
                min_child_weight=xgb_config.get('min_child_weight', 3),
                gamma=xgb_config.get('gamma', 0.1),
                reg_alpha=xgb_config.get('reg_alpha', 0.1),
                reg_lambda=xgb_config.get('reg_lambda', 1.0),
                random_state=42,
                n_jobs=-1,
                tree_method=xgb_config.get('tree_method', 'hist'),
                enable_categorical=False,
                eval_metric='mlogloss',
                early_stopping_rounds=early_stopping_param,
                verbosity=1
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the trading model
        
        Args:
            X_train: Training features
            y_train: Training labels (0=SELL, 1=BUY, 2=HOLD)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            optimize_hyperparameters: Whether to perform hyperparameter optimization
        
        Returns:
            Dictionary with training metrics
        """
        try:
            self.feature_names = list(X_train.columns)
            
            if optimize_hyperparameters and self.model_type == 'random_forest':
                # Hyperparameter optimization
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [6, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                
                self.logger.info("Starting hyperparameter optimization...")
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=5, scoring='accuracy',
                    n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.logger.info(f"Best parameters: {grid_search.best_params_}")
            elif self.model_type == 'xgboost_professional':
                # Professional XGBoost training with configurable parameters
                xgb_config = XGB_PRO_CONFIG
                n_estimators = xgb_config.get('n_estimators', 8000)
                early_stopping = xgb_config.get('early_stopping_rounds', 300)
                self.logger.info(f"Training professional XGBoost model with {n_estimators} trees...")
                
                if X_val is not None and y_val is not None and early_stopping > 0:
                    # Train with early stopping using validation set
                    self.logger.info(f"Using validation set with early stopping rounds: {early_stopping}")
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=100  # Print every 100 iterations
                    )
                else:
                    # Train without early stopping or validation
                    if early_stopping == 0:
                        self.logger.info("Early stopping disabled - training all trees")
                    else:
                        self.logger.info("No validation set provided - training without early stopping")
                    self.model.fit(X_train, y_train, verbose=100)
                self.logger.info("Professional XGBoost model training completed")
            else:
                # Regular training
                self.model.fit(X_train, y_train)
            
            # NEW: log class mapping
            if hasattr(self.model, "classes_"):
                self.logger.info(f"MODEL CLASS ORDER: {self.model.classes_.tolist()}  (expected 0=SELL,1=BUY,2=HOLD)")
                self.logger.debug(f"LABEL MAP USED: {self.label_map}")

            # Calculate training metrics
            train_accuracy = self.model.score(X_train, y_train)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'feature_count': len(self.feature_names),
                'training_samples': len(X_train)
            }
            
            if X_val is not None and y_val is not None:
                val_accuracy = self.model.score(X_val, y_val)
                metrics['val_accuracy'] = val_accuracy
                
                # Get predictions with probabilities
                y_pred_proba = self.model.predict_proba(X_val)
                y_pred = self.model.predict(X_val)
                
                # Calculate precision, recall, F1 for each class
                from sklearn.metrics import classification_report, confusion_matrix
                report = classification_report(y_val, y_pred, output_dict=True)
                metrics.update({
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1_score': report['weighted avg']['f1-score'],
                    'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
                })
            
            self.is_trained = True
            self.model_version = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Store training metadata for validation
            self.training_metadata = metrics.copy()
            self.training_metadata['training_timestamp'] = datetime.now().isoformat()
            
            self.logger.info(f"Model trained successfully. Accuracy: {train_accuracy:.4f}")
            if X_val is not None and y_val is not None:
                self.logger.info(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trading predictions
        
        Args:
            X: Features for prediction
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(X)
            
            # Get class predictions
            predictions = self.model.predict(X)
            
            # Calculate confidence as maximum probability
            confidence_scores = np.max(probabilities, axis=1)
            
            # Apply balanced confidence calculation to maintain quality while allowing higher confidence
            # Reduce confidence if the probabilities are close to each other (high uncertainty)
            prob_std = np.std(probabilities, axis=1)
            uncertainty_penalty = prob_std * 0.8  # Reduced from 2.0 to 0.8 - less aggressive penalty
            
            # Also consider margin between top 2 predictions for each sample
            sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]  # Sort descending
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # Difference between 1st and 2nd
            low_margin_penalty = np.maximum(0, 0.15 - margins) * 1.5  # Reduced threshold from 0.3 to 0.15, multiplier from 2.0 to 1.5
            
            # Calculate balanced confidence with reduced penalties
            balanced_confidence = confidence_scores - uncertainty_penalty - low_margin_penalty
            
            # Ensure confidence doesn't go below 0.1 (10% minimum for valid signals)
            balanced_confidence = np.maximum(balanced_confidence, 0.1)
            
            return predictions, balanced_confidence
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict for a single sample
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Convert to DataFrame with correct column order
            X = pd.DataFrame([features])[self.feature_names]
            
            probs_row = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)  # ترتیب واقعی ستون‌های احتمال
            
            # ساخت دیکشنری احتمال با استفاده از کلاس‌های واقعی
            proba_dict = {}
            for cls_value, p in zip(classes, probs_row):
                label = self.label_map.get(cls_value, str(cls_value))
                proba_dict[label] = float(p)
            
            # پیدا کردن سیگنال برنده
            winner_index = int(np.argmax(probs_row))
            winner_class_value = classes[winner_index]
            signal = self.label_map.get(winner_class_value, 'HOLD')
            confidence = float(probs_row[winner_index])
            
            # Apply balanced confidence calculation to maintain quality while allowing higher confidence
            # Reduce confidence if probabilities are close (high uncertainty)
            prob_std = float(np.std(probs_row))
            
            # Also consider margin between top 2 predictions
            sorted_probs = sorted(probs_row, reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]  # Difference between 1st and 2nd
            
            # Balanced adjustments - less aggressive than before
            uncertainty_penalty = prob_std * 0.8  # Reduced from 2.0 to 0.8
            low_margin_penalty = max(0, 0.15 - margin) * 1.5  # Reduced threshold from 0.3 to 0.15, multiplier from 2.0 to 1.5
            
            # Calculate balanced confidence with reduced penalties
            balanced_confidence = max(0.1, confidence - uncertainty_penalty - low_margin_penalty)  # 10% minimum instead of 0%
            
            # Use balanced confidence for threshold check
            final_confidence = balanced_confidence
            
            # اطمینان از وجود همه کلیدها
            for k in ['BUY', 'SELL', 'HOLD']:
                if k not in proba_dict:
                    proba_dict[k] = 0.0
            
            # دیباگ (بعداً خواستی پاک کن)
            self.logger.debug(f"PRED_DEBUG classes={classes} probs={probs_row.tolist()} mapped={proba_dict} pick={signal} orig_conf={confidence:.3f} balanced_conf={final_confidence:.3f}")
            
            return {
                'signal': signal,
                'confidence': final_confidence,  # Use balanced confidence
                'probabilities': {
                    'BUY': proba_dict['BUY'],
                    'SELL': proba_dict['SELL'],
                    'HOLD': proba_dict['HOLD']
                },
                'meets_threshold': final_confidence >= self.confidence_threshold,
                'original_confidence': confidence,  # Keep original for debugging
                'uncertainty_penalty': uncertainty_penalty,
                'margin': margin,  # Margin between top predictions
                'low_margin_penalty': low_margin_penalty,
                'balanced_confidence': balanced_confidence  # Added for transparency
            }
            
        except Exception as e:
            self.logger.error(f"Error making single prediction: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning("Model does not support feature importance")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'model_version': self.model_version,
                'confidence_threshold': self.confidence_threshold,
                'is_trained': self.is_trained,
                'label_map': self.label_map
            }
            
            joblib.dump(model_data, filepath)
            
            # Log model file size
            if os.path.exists(filepath):
                file_size_bytes = os.path.getsize(filepath)
                file_size_mb = file_size_bytes / (1024 * 1024)
                self.logger.info(f"Model saved to {filepath}")
                self.logger.info(f"Model file size: {file_size_mb:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file"""
        try:
            if not os.path.exists(filepath):
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.model_version = model_data['model_version']
            self.confidence_threshold = model_data['confidence_threshold']
            self.is_trained = model_data['is_trained']
            
            if 'label_map' in model_data:
                self.label_map = model_data['label_map']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for signal generation"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            self.logger.info(f"Confidence threshold set to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': self.model_type,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'confidence_threshold': self.confidence_threshold,
            'feature_names': self.feature_names.copy(),
            'training_metadata': getattr(self, 'training_metadata', {})
        }
    
    def validate_model_performance(self, recent_signals_data: List[Dict], recent_trades_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate model performance against actual trading results
        
        Args:
            recent_signals_data: Recent trading signals with outcomes
            recent_trades_data: Recent completed trades
            
        Returns:
            Dict with real performance metrics vs claimed accuracy
        """
        try:
            if not recent_trades_data:
                return {
                    'real_accuracy': 0.0,
                    'claimed_accuracy': getattr(self, 'training_metadata', {}).get('val_accuracy', 0.0),
                    'performance_gap': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'message': 'No trade data available for validation'
                }
            
            # Calculate real trading performance
            total_trades = len(recent_trades_data)
            winning_trades = sum(1 for trade in recent_trades_data if trade.get('pnl', 0) > 0)
            real_accuracy = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Get claimed accuracy from training
            claimed_accuracy = getattr(self, 'training_metadata', {}).get('val_accuracy', 0.0)
            
            # Calculate performance gap
            performance_gap = abs(claimed_accuracy - real_accuracy)
            
            # Determine if model needs retraining
            needs_retraining = performance_gap > 0.3 or real_accuracy < 0.4  # 30% gap or <40% real accuracy
            
            # Generate insights
            insights = []
            if real_accuracy < 0.3:
                insights.append("Model performing very poorly in live trading")
            elif real_accuracy < claimed_accuracy - 0.2:
                insights.append("Significant overfitting detected - model accuracy much lower in practice")
            elif real_accuracy > claimed_accuracy + 0.1:
                insights.append("Model performing better than expected")
            
            if performance_gap > 0.4:
                insights.append("Large discrepancy between training and live performance")
            
            return {
                'real_accuracy': real_accuracy,
                'claimed_accuracy': claimed_accuracy,
                'performance_gap': performance_gap,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': real_accuracy,
                'needs_retraining': needs_retraining,
                'insights': insights,
                'model_health': 'good' if performance_gap < 0.15 and real_accuracy > 0.5 else 'poor'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating model performance: {e}")
            return {
                'error': str(e),
                'model_health': 'unknown'
            }
