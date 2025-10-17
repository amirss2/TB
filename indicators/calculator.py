import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from indicators.definitions import IndicatorDefinitions

class IndicatorCalculator:
    """
    Calculates all technical indicators based on OHLCV data
    """
    
    def __init__(self):
        self.definitions = IndicatorDefinitions()
        self.logger = logging.getLogger(__name__)
        # Track indicators that have been warned about to avoid spam
        self._generic_warnings_logged = set()
        
        # Indicators that are expected to use generic calculation (no warning needed)
        self._expected_generic_indicators = {
            'Price_Channel_High', 'Price_Channel_Low', 'Williams_R',
            'Chaikin_Oscillator', 'Detrended_Price_Oscillator', 'Klinger_Oscillator',
            'Mass_Index', 'Negative_Volume_Index', 'Positive_Volume_Index',
            'Price_Volume_Trend', 'Volume_Price_Confirmation_Indicator'
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for the given OHLCV data
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with original data + all calculated indicators
        """
        result_df = df.copy()
        
        # First calculate prerequisites
        result_df = self._calculate_prerequisites(result_df)
        
        # Then calculate all other indicators
        for indicator_name, indicator_info in self.definitions.get_all_indicators().items():
            if indicator_info['category'].lower() != 'prereq':
                try:
                    result_df = self._calculate_indicator(result_df, indicator_name, indicator_info)
                except Exception as e:
                    self.logger.error(f"Error calculating {indicator_name}: {e}")
                    # Add NaN column for failed indicators
                    result_df[indicator_name] = np.nan
        
        return result_df
    
    def _calculate_prerequisites(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate prerequisite indicators first"""
        result_df = df.copy()
        
        # Previous values
        result_df['Prev Close'] = result_df['close'].shift(1)
        result_df['Prev High'] = result_df['high'].shift(1)
        result_df['Prev Low'] = result_df['low'].shift(1)
        
        # Price averages
        result_df['Typical Price (TP)'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
        result_df['Median Price (MP)'] = (result_df['high'] + result_df['low']) / 2
        result_df['HLC3'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
        result_df['OHLC4'] = (result_df['open'] + result_df['high'] + result_df['low'] + result_df['close']) / 4
        
        # Enhanced Feature Engineering
        result_df = self._add_enhanced_features(result_df)
        
        return result_df
    
    def _add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced features for better prediction:
        - Trend features
        - Volatility features  
        - Volume features
        - Price action features
        """
        result_df = df.copy()
        
        try:
            # === TREND FEATURES ===
            # Calculate SMA for trend features if not exists
            if 'SMA_50' not in result_df.columns:
                result_df['SMA_50'] = result_df['close'].rolling(window=50, min_periods=1).mean()
            
            # Calculate ATR for volatility-based features if not exists
            if 'ATR_14' not in result_df.columns:
                high_low = result_df['high'] - result_df['low']
                high_close = np.abs(result_df['high'] - result_df['close'].shift())
                low_close = np.abs(result_df['low'] - result_df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                result_df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
            
            # Trend strength: distance from SMA normalized by ATR
            result_df['Trend_Strength'] = (result_df['close'] - result_df['SMA_50']) / result_df['ATR_14'].replace(0, np.nan)
            
            # Trend direction over different periods
            result_df['Trend_5'] = result_df['close'].rolling(window=5, min_periods=1).apply(
                lambda x: 1 if len(x) > 0 and x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
            result_df['Trend_10'] = result_df['close'].rolling(window=10, min_periods=1).apply(
                lambda x: 1 if len(x) > 0 and x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
            
            # === VOLATILITY FEATURES ===
            # Volatility ratio: current ATR vs average ATR
            result_df['Volatility_Ratio'] = result_df['ATR_14'] / result_df['ATR_14'].rolling(window=50, min_periods=1).mean()
            
            # Bollinger Band width (if bands exist, otherwise calculate)
            if 'Bollinger_Bands_upper' in result_df.columns and 'Bollinger_Bands_lower' in result_df.columns:
                result_df['BB_Width'] = (result_df['Bollinger_Bands_upper'] - result_df['Bollinger_Bands_lower']) / result_df['close']
            else:
                # Calculate simple BB width
                sma_20 = result_df['close'].rolling(window=20, min_periods=1).mean()
                std_20 = result_df['close'].rolling(window=20, min_periods=1).std()
                bb_upper = sma_20 + (2 * std_20)
                bb_lower = sma_20 - (2 * std_20)
                result_df['BB_Width'] = (bb_upper - bb_lower) / result_df['close']
            
            # === VOLUME FEATURES ===
            # Volume ratio vs moving average
            result_df['Volume_SMA_20'] = result_df['volume'].rolling(window=20, min_periods=1).mean()
            result_df['Volume_Ratio'] = result_df['volume'] / result_df['Volume_SMA_20'].replace(0, np.nan)
            
            # Volume change rate
            result_df['Volume_Change'] = result_df['volume'].pct_change()
            
            # Volume trend
            result_df['Volume_Trend'] = result_df['volume'].rolling(window=5, min_periods=1).apply(
                lambda x: 1 if len(x) > 0 and x.iloc[-1] > x.iloc[0] else -1, raw=False
            )
            
            # === PRICE ACTION FEATURES ===
            # Price range (high-low normalized by close)
            result_df['Price_Range'] = (result_df['high'] - result_df['low']) / result_df['close']
            
            # Candle body ratio (absolute change / range)
            result_df['Candle_Body_Ratio'] = np.abs(result_df['close'] - result_df['open']) / (result_df['high'] - result_df['low']).replace(0, np.nan)
            
            # Upper shadow ratio
            result_df['Upper_Shadow'] = (result_df['high'] - result_df[['open', 'close']].max(axis=1)) / result_df['close']
            
            # Lower shadow ratio
            result_df['Lower_Shadow'] = (result_df[['open', 'close']].min(axis=1) - result_df['low']) / result_df['close']
            
            # Price momentum (rate of change)
            result_df['Price_Momentum_5'] = result_df['close'].pct_change(periods=5)
            result_df['Price_Momentum_10'] = result_df['close'].pct_change(periods=10)
            
            # Gap detection (difference between current open and previous close)
            result_df['Gap'] = (result_df['open'] - result_df['close'].shift(1)) / result_df['close'].shift(1)
            
            # Replace infinities with NaN
            result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            self.logger.debug("Enhanced features added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding enhanced features: {e}")
        
        return result_df
    
    def _calculate_indicator(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate a single indicator based on its definition"""
        result_df = df.copy()
        
        try:
            # Route to appropriate calculation method based on indicator name/type
            if 'SMA' in name:
                result_df = self._calculate_sma(result_df, name, info)
            elif 'EMA' in name:
                result_df = self._calculate_ema(result_df, name, info)
            elif 'RSI' in name:
                result_df = self._calculate_rsi(result_df, name, info)
            elif 'MACD' in name:
                result_df = self._calculate_macd(result_df, name, info)
            elif 'Bollinger' in name and 'Width' not in name:
                result_df = self._calculate_bollinger(result_df, name, info)
            elif 'Bollinger_Width' in name:
                result_df = self._calculate_bollinger_width(result_df, name, info)
            elif 'Stoch' in name:
                result_df = self._calculate_stochastic(result_df, name, info)
            elif 'Williams_%R' in name:
                result_df = self._calculate_williams_r(result_df, name, info)
            elif 'ROC' in name:
                result_df = self._calculate_roc(result_df, name, info)
            elif 'MFI' in name:
                result_df = self._calculate_mfi(result_df, name, info)
            elif 'Momentum' in name:
                result_df = self._calculate_momentum(result_df, name, info)
            elif 'ATR' in name:
                result_df = self._calculate_atr(result_df, name, info)
            elif 'Ichimoku' in name:
                result_df = self._calculate_ichimoku_component(result_df, name, info)
            elif 'SuperTrend' in name:
                result_df = self._calculate_supertrend(result_df, name, info)
            elif 'VWAP' in name:
                result_df = self._calculate_vwap(result_df, name, info)
            else:
                # Generic calculation attempt
                result_df = self._calculate_generic(result_df, name, info)
                
        except Exception as e:
            self.logger.error(f"Failed to calculate {name}: {e}")
            result_df[name] = np.nan
            
        return result_df
    
    def _calculate_sma(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Simple Moving Average"""
        period = info['parameters'].get('period', 20)
        source_col = 'close'  # Default to close, could be parameterized
        
        if source_col in df.columns:
            df[name] = df[source_col].rolling(window=period).mean()
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Exponential Moving Average"""
        period = info['parameters'].get('period', 20)
        source_col = 'close'
        
        if source_col in df.columns:
            df[name] = df[source_col].ewm(span=period).mean()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        period = info['parameters'].get('period', 14)
        source_col = 'close'
        
        if source_col in df.columns:
            delta = df[source_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[name] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD"""
        fast = info['parameters'].get('fast', 12)
        slow = info['parameters'].get('slow', 26)
        signal = info['parameters'].get('signal', 9)
        source_col = 'close'
        
        if source_col in df.columns:
            exp1 = df[source_col].ewm(span=fast).mean()
            exp2 = df[source_col].ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            if 'Signal' in name:
                df[name] = signal_line
            elif 'Histogram' in name:
                df[name] = histogram
            else:
                df[name] = macd_line
        
        return df
    
    def _calculate_bollinger(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        period = info['parameters'].get('period', 20)
        mult = info['parameters'].get('mult', 2.0)
        source_col = 'close'
        
        if source_col in df.columns:
            sma = df[source_col].rolling(window=period).mean()
            std = df[source_col].rolling(window=period).std()
            
            upper = sma + (mult * std)
            lower = sma - (mult * std)
            
            # Return all three bands as separate columns
            df[f"{name}_upper"] = upper
            df[f"{name}_middle"] = sma
            df[f"{name}_lower"] = lower
        
        return df
    
    def _calculate_bollinger_width(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Band Width"""
        period = info['parameters'].get('period', 20)
        mult = info['parameters'].get('mult', 2.0)
        source_col = 'close'
        
        # Extract period and multiplier from name if not in parameters
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)_x([\d.]+)', name)
            if match:
                period = int(match.group(1))
                mult = float(match.group(2))
        
        if source_col in df.columns:
            sma = df[source_col].rolling(window=period).mean()
            std = df[source_col].rolling(window=period).std()
            upper = sma + (mult * std)
            lower = sma - (mult * std)
            df[name] = (upper - lower) / sma
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        period = info['parameters'].get('period', 14)
        
        # Extract period from name if not in parameters
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)', name)
            if match:
                period = int(match.group(1))
        
        lowest_low = df['low'].rolling(window=period).min()
        highest_high = df['high'].rolling(window=period).max()
        df[name] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Williams %R"""
        period = info['parameters'].get('period', 14)
        
        # Extract period from name if not in parameters
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)', name)
            if match:
                period = int(match.group(1))
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        df[name] = -100 * (highest_high - df['close']) / (highest_high - lowest_low)
        
        return df
    
    def _calculate_roc(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Rate of Change"""
        period = info['parameters'].get('period', 1)
        
        # Extract period from name
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)', name)
            if match:
                period = int(match.group(1))
        
        df[name] = 100 * (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Money Flow Index"""
        period = info['parameters'].get('period', 14)
        
        # Extract period from name
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)', name)
            if match:
                period = int(match.group(1))
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfr = positive_mf / negative_mf
        df[name] = 100 - (100 / (1 + mfr))
        
        return df
    
    def _calculate_momentum(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Momentum"""
        period = info['parameters'].get('period', 10)
        
        # Extract period from name
        if 'period' not in info['parameters']:
            import re
            match = re.search(r'_(\d+)', name)
            if match:
                period = int(match.group(1))
        
        df[name] = df['close'] - df['close'].shift(period)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Average True Range"""
        period = info['parameters'].get('period', 14)
        
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        df[name] = true_range.rolling(window=period).mean()
        
        return df
    
    def _calculate_ichimoku_component(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Ichimoku components"""
        if 'Tenkan' in name:
            period = info['parameters'].get('period', 9)
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df[name] = (highest_high + lowest_low) / 2
        elif 'Kijun' in name:
            period = info['parameters'].get('period', 26)
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df[name] = (highest_high + lowest_low) / 2
        elif 'SenkouB' in name:
            period = info['parameters'].get('period', 52)
            shift = info['parameters'].get('shift', 26)
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df[name] = ((highest_high + lowest_low) / 2).shift(shift)
        elif 'Chikou' in name:
            shift = info['parameters'].get('shift', 26)
            df[name] = df['close'].shift(-shift)
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate SuperTrend"""
        period = info['parameters'].get('period', 10)
        multiplier = info['parameters'].get('multiplier', 3.0)
        
        # Calculate ATR if not already present
        if 'ATR' not in df.columns:
            df = self._calculate_atr(df, 'ATR', {'parameters': {'period': period}})
        
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * df['ATR'])
        lower_band = hl2 - (multiplier * df['ATR'])
        
        # SuperTrend calculation logic
        supertrend = np.zeros(len(df))
        direction = np.ones(len(df))
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= lower_band.iloc[i-1]:
                direction[i] = -1
            elif df['close'].iloc[i] >= upper_band.iloc[i-1]:
                direction[i] = 1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = lower_band.iloc[i]
            else:
                supertrend[i] = upper_band.iloc[i]
        
        df[name] = supertrend
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df[name] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def _calculate_generic(self, df: pd.DataFrame, name: str, info: Dict[str, Any]) -> pd.DataFrame:
        """Generic calculation for simple indicators"""
        # Only warn for indicators not in the expected generic list, and only once
        if name not in self._expected_generic_indicators and name not in self._generic_warnings_logged:
            self.logger.debug(f"Generic calculation used for {name} - may need specific implementation")
            self._generic_warnings_logged.add(name)
        
        # For now, set to NaN - could be extended based on specific formulas
        df[name] = np.nan
        
        return df