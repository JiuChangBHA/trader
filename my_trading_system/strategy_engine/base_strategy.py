
"""
Base strategy class that all specific strategies inherit from.
"""

import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal

class Strategy(ABC):
    """
    Base strategy class that defines the interface for all trading strategies.
    """
    
    def __init__(self, symbol: str, parameters: Dict, risk_multiplier: float = 1.0,
                trade_direction: Literal["long", "short", "both"] = "both"):
        """
        Initialize the strategy.
        
        Args:
            symbol: Trading symbol
            parameters: Strategy-specific parameters
            risk_multiplier: Risk multiplier to adjust position size
            trade_direction: Trading direction (long, short, or both)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.symbol = symbol
        self.parameters = parameters
        self.risk_multiplier = risk_multiplier
        self.trade_direction = trade_direction
        self.current_position = 0
        self.current_signal = 0  # -1 for sell, 0 for neutral, 1 for buy
    
    @abstractmethod
    def calculate_signal(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Calculate trading signal based on the strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (signal, metadata):
                signal: -1 for sell, 0 for neutral, 1 for buy
                metadata: Dictionary with additional signal information
        """
        pass
    
    def filter_signal(self, signal: int) -> int:
        """
        Filter signal based on trade direction setting.
        
        Args:
            signal: Raw signal (-1, 0, or 1)
            
        Returns:
            Filtered signal
        """
        if self.trade_direction == "long" and signal < 0:
            return 0
        elif self.trade_direction == "short" and signal > 0:
            return 0
        return signal
    
    def process_data(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Process new data and generate trading signal if needed.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Signal info dictionary if signal changed, None otherwise
        """
        try:
            raw_signal, metadata = self.calculate_signal(data)
            
            # Filter signal based on trade direction
            signal = self.filter_signal(raw_signal)
            
            # Check if signal changed
            if signal != self.current_signal:
                self.logger.info(f"Signal changed for {self.symbol}: {self.current_signal} -> {signal}")
                self.current_signal = signal
                
                return {
                    "symbol": self.symbol,
                    "signal": signal,
                    "strategy": self.__class__.__name__,
                    "risk_multiplier": self.risk_multiplier,
                    **metadata
                }
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error processing data for {self.symbol}: {e}")
            return None
