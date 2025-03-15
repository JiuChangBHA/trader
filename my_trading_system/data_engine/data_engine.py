
"""
DataEngine module for handling market data ingestion and processing.
"""

import logging
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import alpaca_trade_api as tradeapi

class DataEngine:
    """
    Handles real-time and historical market data ingestion from Alpaca API.
    """
    
    def __init__(self, api: tradeapi.REST, symbols: List[str], timeframe: str = "1Min", 
                 lookback_days: int = 30):
        """
        Initialize the DataEngine.
        
        Args:
            api: Alpaca REST API instance
            symbols: List of symbols to track
            timeframe: Bar timeframe (e.g. "1Min", "5Min", "1D")
            lookback_days: Number of days of historical data to load
        """
        self.logger = logging.getLogger(__name__)
        self.api = api
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.bars: Dict[str, pd.DataFrame] = {}
        self.latest_quotes: Dict[str, dict] = {}
        self.latest_trades: Dict[str, dict] = {}
        self._running = False
        self._data_callbacks = []
    
    async def start(self):
        """Start the data engine"""
        self.logger.info(f"Starting DataEngine for symbols: {', '.join(self.symbols)}")
        self._running = True
        
        # Load historical data
        await self._load_historical_data()
        
        # Start real-time data stream
        asyncio.create_task(self._stream_data())
    
    async def stop(self):
        """Stop the data engine"""
        self.logger.info("Stopping DataEngine")
        self._running = False
    
    def register_data_callback(self, callback):
        """Register a callback to be called when new data arrives"""
        self._data_callbacks.append(callback)
    
    async def _load_historical_data(self):
        """Load historical bar data for all symbols"""
        self.logger.info(f"Loading {self.lookback_days} days of historical data")
        
        for symbol in self.symbols:
            try:
                # Get historical bars
                bars = self.api.get_bars(
                    symbol,
                    self.timeframe,
                    limit=None,
                    start=pd.Timestamp.now(tz='America/New_York').floor('D') - pd.Timedelta(days=self.lookback_days),
                    end=pd.Timestamp.now(tz='America/New_York')
                ).df
                
                if len(bars) > 0:
                    self.bars[symbol] = bars
                    self.logger.info(f"Loaded {len(bars)} bars for {symbol}")
                else:
                    self.logger.warning(f"No historical data found for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Error loading historical data for {symbol}: {e}")
    
    async def _stream_data(self):
        """Stream real-time data updates"""
        self.logger.info("Starting real-time data stream")
        
        # This is just a placeholder for the real implementation
        # In a real implementation, you would use Alpaca's websocket API
        while self._running:
            try:
                for symbol in self.symbols:
                    # Get latest bar
                    latest_bar = self.api.get_latest_bar(symbol)
                    
                    # Update data
                    if symbol in self.bars:
                        # Create a DataFrame with the new bar and append it
                        new_bar = pd.DataFrame([{
                            'open': latest_bar.o,
                            'high': latest_bar.h,
                            'low': latest_bar.l,
                            'close': latest_bar.c,
                            'volume': latest_bar.v,
                            'trade_count': latest_bar.n,
                            'vwap': latest_bar.vw
                        }], index=[pd.Timestamp(latest_bar.t)])
                        
                        # Append only if the timestamp is new
                        if latest_bar.t not in self.bars[symbol].index:
                            self.bars[symbol] = pd.concat([self.bars[symbol], new_bar])
                            
                            # Notify callbacks
                            for callback in self._data_callbacks:
                                callback(symbol, 'bar', new_bar)
                
                # Wait for next update
                await asyncio.sleep(60)  # Update every minute
            
            except Exception as e:
                self.logger.error(f"Error in data stream: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    def get_bars(self, symbol: str, lookback: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback: Number of bars to return (None for all)
            
        Returns:
            DataFrame with bar data
        """
        if symbol not in self.bars:
            return pd.DataFrame()
                    
        df = self.bars[symbol]
        if lookback is not None:
            return df.tail(lookback)
        return df
