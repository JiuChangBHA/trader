"""
Bollinger Bands trading strategy implementation.
"""

import pandas as pd
import numpy as np
# Try to import from TA-Lib-Precompiled first
try:
    import talib
    USE_TALIB = True
except ImportError:
    # Fall back to pandas-ta if available
    try:
        import pandas_ta as ta
        USE_TALIB = False
    except ImportError:
        raise ImportError("Neither TA-Lib nor pandas-ta is installed. Please install one of them.")
from typing import Dict, Tuple

from .base_strategy import Strategy

class BollingerBandsStrategy(Strategy):
    """
    Trading strategy based on Bollinger Bands indicator.
    
    Buy signal when price crosses below lower band and then back above it.
    Sell signal when price crosses above upper band and then back below it.
    """
    
    def __init__(self, symbol: str, parameters: Dict, risk_multiplier: float = 1.0,
                trade_direction: str = "both"):
        """
        Initialize the Bollinger Bands strategy.
        
        Parameters:
            timeperiod: Period for moving average calculation
            nbdevup: Standard deviation multiplier for upper band
            nbdevdn: Standard deviation multiplier for lower band
            matype: Moving average type
        """
        super().__init__(symbol, parameters, risk_multiplier, trade_direction)
        self.was_below_lower = False
        self.was_above_upper = False
    
    def calculate_signal(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate trading signal based on Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (signal, metadata)
        """
        # Extract parameters
        timeperiod = self.parameters.get("timeperiod", 20)
        nbdevup = self.parameters.get("nbdevup", 2.0)
        nbdevdn = self.parameters.get("nbdevdn", 2.0)
        matype = self.parameters.get("matype", 0)
        
        # Calculate Bollinger Bands
        if USE_TALIB:
            upper, middle, lower = talib.BBANDS(
                data['close'].values,
                timeperiod=timeperiod,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
                matype=matype
            )
        else:
            # Use pandas-ta
            bbands = ta.bbands(data['close'], length=timeperiod, std=nbdevup)
            # pandas-ta returns a DataFrame with columns: BBL, BBM, BBU, etc.
            lower = bbands['BBL_' + str(timeperiod) + '_' + str(float(nbdevup))].values
            middle = bbands['BBM_' + str(timeperiod) + '_' + str(float(nbdevup))].values
            upper = bbands['BBU_' + str(timeperiod) + '_' + str(float(nbdevup))].values
        
        # Get latest values
        latest_close = data['close'].iloc[-1]
        latest_upper = upper[-1]
        latest_lower = lower[-1]
        latest_middle = middle[-1]
        
        # Calculate percentage from middle band
        pct_from_middle = (latest_close - latest_middle) / latest_middle * 100
        
        # Initialize signal as neutral
        signal = 0
        
        # Check for buy signal (price crosses from below lower band to above)
        if latest_close <= latest_lower:
            self.was_below_lower = True
        elif self.was_below_lower and latest_close > latest_lower:
            signal = 1
            self.was_below_lower = False
        
        # Check for sell signal (price crosses from above upper band to below)
        if latest_close >= latest_upper:
            self.was_above_upper = True
        elif self.was_above_upper and latest_close < latest_upper:
            signal = -1
            self.was_above_upper = False
        
        # Prepare metadata
        metadata = {
            "indicator": "bollinger_bands",
            "upper_band": latest_upper,
            "middle_band": latest_middle,
            "lower_band": latest_lower,
            "price": latest_close,
            "pct_from_middle": pct_from_middle,
            "bb_width": (latest_upper - latest_lower) / latest_middle
        }
        
        return signal, metadata
