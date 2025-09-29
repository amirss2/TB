import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from typing import List, Dict, Tuple, Any
import logging

class FeatureSelector:
    """
    Recursive Feature Elimination (RFE) and feature selection for trading indicators
    """
    
    def __init__(self, n_features_to_select: int = 50):
        self.n_features_to_select = n_features_to_select
        self.selected_features = []
        self.feature_rankings = {}
        self.feature_scores = {}
        self.logger = logging.getLogger(__name__)
        
        # Use RandomForest as estimator for RFE
        self.estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, 
                          must_include: List[str] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select best features using Recursive Feature Elimination
        
        Args:
            X: Feature matrix
            y: Target labels
            must_include: List of features that must be included (not subject to elimination)
        
        Returns:
            Tuple of (selected_features, selection_info)
        """
        try:
            self.logger.info(f"Starting RFE with {len(X.columns)} features, selecting {self.n_features_to_select}")
            
            # Separate must-include features from RFE candidates
            must_include = must_include or []
            rfe_candidates = [col for col in X.columns if col not in must_include]
            
            self.logger.info(f"Must include: {len(must_include)}, RFE candidates: {len(rfe_candidates)}")
            
            if len(must_include) >= self.n_features_to_select:
                self.logger.warning("Must-include features exceed target count, using all must-include features")
                selected_features = must_include[:self.n_features_to_select]
                
                selection_info = {
                    'method': 'must_include_only',
                    'total_features': len(X.columns),
                    'selected_count': len(selected_features),
                    'must_include_count': len(selected_features),
                    'rfe_selected_count': 0
                }
            else:
                # Number of features to select via RFE
                rfe_target = self.n_features_to_select - len(must_include)
                
                if len(rfe_candidates) > 0 and rfe_target > 0:
                    # Prepare data for RFE (only candidates)
                    X_rfe = X[rfe_candidates].copy()
                    
                    # Remove any columns with all NaN or constant values
                    X_rfe = self._clean_features(X_rfe)
                    
                    if len(X_rfe.columns) > rfe_target:
                        # Perform RFE
                        rfe = RFE(estimator=self.estimator, n_features_to_select=rfe_target)
                        rfe.fit(X_rfe, y)
                        
                        # Get selected features from RFE
                        rfe_selected = X_rfe.columns[rfe.support_].tolist()
                        
                        # Store rankings
                        for i, feature in enumerate(X_rfe.columns):
                            self.feature_rankings[feature] = rfe.ranking_[i]
                    
                    else:
                        # Use all available candidates
                        rfe_selected = X_rfe.columns.tolist()
                        self.feature_rankings = {feat: 1 for feat in rfe_selected}
                else:
                    rfe_selected = []
                
                # Combine must-include and RFE selected
                selected_features = must_include + rfe_selected
                
                selection_info = {
                    'method': 'rfe_with_must_include',
                    'total_features': len(X.columns),
                    'selected_count': len(selected_features),
                    'must_include_count': len(must_include),
                    'rfe_selected_count': len(rfe_selected),
                    'rfe_target': rfe_target
                }
            
            self.selected_features = selected_features
            
            self.logger.info(f"Feature selection completed: {len(selected_features)} features selected")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f"Error in RFE feature selection: {e}")
            raise
    
    def select_features_statistical(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, Any]]:
        """
        Select features using statistical methods (SelectKBest with f_classif)
        """
        try:
            self.logger.info(f"Starting statistical feature selection")
            
            # Clean features
            X_clean = self._clean_features(X.copy())
            
            # Use SelectKBest with f_classif
            k = min(self.n_features_to_select, len(X_clean.columns))
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_clean, y)
            
            # Get selected features and scores
            selected_mask = selector.get_support()
            selected_features = X_clean.columns[selected_mask].tolist()
            
            # Store scores
            for i, feature in enumerate(X_clean.columns):
                self.feature_scores[feature] = selector.scores_[i]
            
            self.selected_features = selected_features
            
            selection_info = {
                'method': 'statistical_selectkbest',
                'total_features': len(X.columns),
                'selected_count': len(selected_features),
                'k_value': k
            }
            
            self.logger.info(f"Statistical selection completed: {len(selected_features)} features selected")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f"Error in statistical feature selection: {e}")
            raise
    
    def select_features_hybrid(self, X: pd.DataFrame, y: pd.Series, 
                             must_include: List[str] = None,
                             rfe_ratio: float = 0.7) -> Tuple[List[str], Dict[str, Any]]:
        """
        Hybrid approach combining RFE and statistical selection
        
        Args:
            X: Feature matrix
            y: Target labels
            must_include: Features that must be included
            rfe_ratio: Ratio of features to select using RFE (rest using statistical)
        """
        try:
            must_include = must_include or []
            
            # Calculate split
            available_slots = self.n_features_to_select - len(must_include)
            rfe_slots = int(available_slots * rfe_ratio)
            stat_slots = available_slots - rfe_slots
            
            self.logger.info(f"Hybrid selection: {rfe_slots} RFE + {stat_slots} statistical + {len(must_include)} must-include")
            
            # Get RFE candidates (excluding must-include)
            rfe_candidates = [col for col in X.columns if col not in must_include]
            X_candidates = X[rfe_candidates].copy()
            X_candidates = self._clean_features(X_candidates)
            
            selected_features = must_include.copy()
            
            if rfe_slots > 0 and len(X_candidates.columns) > 0:
                # RFE selection
                k_rfe = min(rfe_slots, len(X_candidates.columns))
                rfe = RFE(estimator=self.estimator, n_features_to_select=k_rfe)
                rfe.fit(X_candidates, y)
                
                rfe_selected = X_candidates.columns[rfe.support_].tolist()
                selected_features.extend(rfe_selected)
                
                # Remove RFE selected from candidates for statistical selection
                remaining_candidates = [col for col in X_candidates.columns if col not in rfe_selected]
            else:
                rfe_selected = []
                remaining_candidates = X_candidates.columns.tolist()
            
            if stat_slots > 0 and len(remaining_candidates) > 0:
                # Statistical selection from remaining candidates
                X_remaining = X[remaining_candidates].copy()
                X_remaining = self._clean_features(X_remaining)
                
                if len(X_remaining.columns) > 0:
                    k_stat = min(stat_slots, len(X_remaining.columns))
                    selector = SelectKBest(score_func=f_classif, k=k_stat)
                    selector.fit(X_remaining, y)
                    
                    stat_selected = X_remaining.columns[selector.get_support()].tolist()
                    selected_features.extend(stat_selected)
                else:
                    stat_selected = []
            else:
                stat_selected = []
            
            self.selected_features = selected_features
            
            selection_info = {
                'method': 'hybrid_rfe_statistical',
                'total_features': len(X.columns),
                'selected_count': len(selected_features),
                'must_include_count': len(must_include),
                'rfe_selected_count': len(rfe_selected),
                'statistical_selected_count': len(stat_selected),
                'rfe_ratio': rfe_ratio
            }
            
            self.logger.info(f"Hybrid selection completed: {len(selected_features)} features selected")
            
            return selected_features, selection_info
            
        except Exception as e:
            self.logger.error(f"Error in hybrid feature selection: {e}")
            raise
    
    def _clean_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean features by removing invalid columns"""
        X_clean = X.copy()
        
        # Remove columns with all NaN
        X_clean = X_clean.dropna(axis=1, how='all')
        
        # Remove constant columns
        for col in X_clean.columns:
            if X_clean[col].nunique() <= 1:
                X_clean = X_clean.drop(columns=[col])
                self.logger.warning(f"Removed constant feature: {col}")
        
        # Fill remaining NaN with median (for numerical stability)
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            if X_clean[col].isna().any():
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
        
        return X_clean
    
    def get_feature_ranking_report(self) -> pd.DataFrame:
        """Get detailed feature ranking report"""
        if not self.feature_rankings:
            return pd.DataFrame()
        
        ranking_df = pd.DataFrame([
            {'feature': feature, 'rank': rank}
            for feature, rank in self.feature_rankings.items()
        ]).sort_values('rank')
        
        # Add scores if available
        if self.feature_scores:
            score_df = pd.DataFrame([
                {'feature': feature, 'score': score}
                for feature, score in self.feature_scores.items()
            ])
            ranking_df = ranking_df.merge(score_df, on='feature', how='left')
        
        return ranking_df
    
    def select_features_dynamic(self, X: pd.DataFrame, y: pd.Series,
                              must_include: List[str] = None,
                              min_features: int = 20,
                              drop_fraction: float = 0.05,
                              corr_threshold: float = 0.95,
                              tolerance: float = 0.003,
                              metric: str = 'macro_f1',
                              cv_splits: int = 3,
                              max_iterations: int = 50) -> Tuple[List[str], Dict[str, Any]]:
        """
        Dynamic iterative pruning method with macro F1 optimization
        
        Args:
            X: Feature matrix
            y: Target labels
            must_include: Features that must be included (base OHLCV etc.)
            min_features: Minimum number of features to retain
            drop_fraction: Fraction of features to drop each iteration
            corr_threshold: Correlation threshold for pruning
            tolerance: Minimum improvement to continue
            metric: Metric to optimize ('macro_f1' or 'balanced_accuracy')
            cv_splits: Number of cross-validation splits
            max_iterations: Maximum number of iterations
            
        Returns:
            Tuple of (selected_features, selection_info)
        """
        try:
            self.logger.info(f"Starting dynamic feature selection with {len(X.columns)} features")
            
            # Prepare must-include features
            must_include = must_include or []
            history = []
            
            # Clean initial features
            X_clean = self._clean_features(X.copy())
            
            # Step 1: Correlation pruning
            candidate_features = [col for col in X_clean.columns if col not in must_include]
            X_candidates = X_clean[candidate_features]
            
            # Remove highly correlated features using importance-based selection
            corr_matrix = X_candidates.corr().abs()
            
            # Get initial feature importance to guide correlation pruning
            temp_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            temp_estimator.fit(X_candidates, y)
            feature_importance = dict(zip(candidate_features, temp_estimator.feature_importances_))
            
            to_drop = set()
            # Find pairs of highly correlated features and drop the less important one
            for i, feature1 in enumerate(candidate_features):
                if feature1 in to_drop:
                    continue
                    
                for j, feature2 in enumerate(candidate_features[i+1:], i+1):
                    if feature2 in to_drop:
                        continue
                    
                    correlation = corr_matrix.iloc[i, j]
                    if abs(correlation) > corr_threshold:
                        # Drop the feature with lower importance
                        imp1 = feature_importance.get(feature1, 0)
                        imp2 = feature_importance.get(feature2, 0)
                        
                        if imp1 < imp2:
                            to_drop.add(feature1)
                            break  # Move to next feature1
                        else:
                            to_drop.add(feature2)
            
            # Update candidates by removing dropped features
            candidate_features = [col for col in candidate_features if col not in to_drop]
            
            self.logger.info(f"Correlation pruning: removed {len(to_drop)} features, {len(candidate_features)} candidates remain")
            
            # Safety check: ensure we have some candidate features
            if not candidate_features:
                self.logger.warning("No candidate features remain after correlation pruning, using all must_include features")
                current_features = must_include
            else:
                # Step 2: Initial feature set (must_include + all candidates)
                current_features = must_include + candidate_features
            
            current_features = [col for col in current_features if col in X_clean.columns]
            
            # Safety check: ensure we have minimum features
            if len(current_features) < min_features:
                self.logger.warning(f"Feature count ({len(current_features)}) below minimum ({min_features}), "
                                  f"adding top features by importance")
                # Add back some features if we dropped too many
                all_candidates = [col for col in X_clean.columns if col not in must_include]
                if all_candidates:
                    temp_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    temp_estimator.fit(X_clean[all_candidates], y)
                    feature_importance = dict(zip(all_candidates, temp_estimator.feature_importances_))
                    
                    # Add top features to reach minimum
                    needed = min_features - len(current_features)
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    for feat, _ in top_features[:needed]:
                        if feat not in current_features:
                            current_features.append(feat)
            
            # Prepare cross-validation
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Scorer based on metric
            if metric == 'macro_f1':
                scorer = make_scorer(f1_score, average='macro')
            else:
                scorer = 'balanced_accuracy'
            
            # Baseline score
            X_current = X_clean[current_features]
            baseline_scores = cross_val_score(estimator, X_current, y, cv=cv, scoring=scorer, n_jobs=-1)
            baseline_score = np.mean(baseline_scores[~np.isnan(baseline_scores)])
            
            self.logger.info(f"Baseline {metric}: {baseline_score:.4f} with {len(current_features)} features")
            
            history.append({
                'iteration': 0,
                'features_count': len(current_features),
                'score': baseline_score,
                'features': current_features.copy(),
                'action': 'baseline'
            })
            
            # Step 3: Iterative pruning
            best_score = baseline_score
            best_features = current_features.copy()
            iterations_without_improvement = 0
            
            for iteration in range(1, max_iterations + 1):
                if len(current_features) <= max(min_features, len(must_include)):
                    self.logger.info(f"Reached minimum features limit at iteration {iteration}")
                    break
                
                # Calculate how many features to drop
                droppable_features = [f for f in current_features if f not in must_include]
                if not droppable_features:
                    break
                
                n_to_drop = max(1, int(len(droppable_features) * drop_fraction))
                n_to_drop = min(n_to_drop, len(droppable_features) - max(0, min_features - len(must_include)))
                
                if n_to_drop <= 0:
                    break
                
                # Get feature importance for ranking
                temp_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                temp_estimator.fit(X_clean[current_features], y)
                
                feature_importance = pd.DataFrame({
                    'feature': current_features,
                    'importance': temp_estimator.feature_importances_
                }).sort_values('importance')
                
                # Find least important droppable features
                droppable_importance = feature_importance[
                    feature_importance['feature'].isin(droppable_features)
                ]
                features_to_drop = droppable_importance.head(n_to_drop)['feature'].tolist()
                
                # Test new feature set
                test_features = [f for f in current_features if f not in features_to_drop]
                X_test = X_clean[test_features]
                
                try:
                    test_scores = cross_val_score(estimator, X_test, y, cv=cv, scoring=scorer, n_jobs=-1)
                    test_score = np.mean(test_scores[~np.isnan(test_scores)])
                except Exception as e:
                    self.logger.warning(f"CV evaluation failed at iteration {iteration}: {e}")
                    test_score = 0.0
                
                self.logger.info(f"Iteration {iteration}: {metric} = {test_score:.4f} with {len(test_features)} features")
                
                history.append({
                    'iteration': iteration,
                    'features_count': len(test_features),
                    'score': test_score,
                    'features': test_features.copy(),
                    'action': f'dropped_{n_to_drop}_features'
                })
                
                # Check if improvement is significant
                if test_score > best_score + tolerance:
                    best_score = test_score
                    best_features = test_features.copy()
                    current_features = test_features.copy()
                    iterations_without_improvement = 0
                    self.logger.info(f"Improvement found: {test_score:.4f} > {best_score - tolerance:.4f}")
                else:
                    iterations_without_improvement += 1
                    # Continue with current set for next iteration
                    current_features = test_features.copy()
                    
                    if iterations_without_improvement >= 3:  # Early stopping
                        self.logger.info(f"Early stopping: no improvement for 3 iterations")
                        break
            
            # Final selection info
            selection_info = {
                'method': 'dynamic_iterative_pruning',
                'total_features': len(X.columns),
                'selected_count': len(best_features),
                'must_include_count': len(must_include),
                'baseline_score': baseline_score,
                'final_score': best_score,
                'improvement': best_score - baseline_score,
                'iterations': len(history) - 1,
                'metric': metric,
                'tolerance': tolerance,
                'correlation_removed': len(to_drop),
                'history': history
            }
            
            self.selected_features = best_features
            
            self.logger.info(f"Dynamic selection completed: {len(best_features)} features, "
                           f"{metric} improved from {baseline_score:.4f} to {best_score:.4f}")
            
            return best_features, selection_info
            
        except Exception as e:
            self.logger.error(f"Error in dynamic feature selection: {e}")
            # Fallback to RFE if dynamic fails
            self.logger.warning("Falling back to RFE selection")
            return self.select_features_rfe(X, y, must_include)

    def evaluate_feature_importance_on_recent_data(self, X: pd.DataFrame, y: pd.Series,
                                                 recent_samples: int = 1000) -> pd.DataFrame:
        """
        Evaluate feature importance on most recent data samples
        
        Args:
            X: Full feature matrix
            y: Full target labels  
            recent_samples: Number of recent samples to use for evaluation
            
        Returns:
            DataFrame with feature importance scores
        """
        try:
            # Use most recent samples
            if len(X) > recent_samples:
                X_recent = X.tail(recent_samples)
                y_recent = y.tail(recent_samples)
            else:
                X_recent = X
                y_recent = y
            
            self.logger.info(f"Evaluating feature importance on {len(X_recent)} recent samples")
            
            # Clean data
            X_clean = self._clean_features(X_recent)
            
            # Train temporary model for importance
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            temp_model.fit(X_clean, y_recent)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': X_clean.columns,
                'importance': temp_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error evaluating feature importance: {e}")
            return pd.DataFrame()