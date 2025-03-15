#!/usr/bin/env python3
"""
Alpaca Trading System Initialization Script
==========================================
This script creates the necessary directory structure and config files for a new Alpaca Trading System project.
"""

import os
import sys
import json
import argparse
import getpass
import shutil
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Initialize Alpaca Trading System Project')
    
    parser.add_argument('-d', '--directory', dest='project_dir', default='.',
                        help='Project directory (default: current directory)')
    parser.add_argument('-n', '--name', dest='project_name', default='alpaca_trading_system',
                        help='Project name (default: alpaca_trading_system)')
    parser.add_argument('--api-key', dest='api_key',
                        help='Alpaca API Key (will prompt if not provided)')
    parser.add_argument('--api-secret', dest='api_secret',
                        help='Alpaca API Secret (will prompt if not provided)')
    parser.add_argument('--paper', dest='paper_trading', action='store_true', default=True,
                        help='Use paper trading API (default: True)')
    parser.add_argument('--live', dest='paper_trading', action='store_false',
                        help='Use live trading API')
    
    return parser.parse_args()

def create_directory_structure(project_dir):
    """Create the necessary directory structure for the project"""
    directories = [
        'logs',
        'data',
        'config',
        'backtest_results',
        'reports'
    ]
    
    for directory in directories:
        dir_path = os.path.join(project_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

def create_config_file(project_dir, api_key, api_secret, paper_trading):
    """Create the default configuration file"""
    config = {
        "alpaca_api_key": api_key,
        "alpaca_api_secret": api_secret,
        "alpaca_base_url": "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets",
        "alpaca_data_url": "https://data.alpaca.markets",
        "trading_mode": "paper" if paper_trading else "live",
        "log_level": "INFO",
        "default_symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META"],
        "risk_management": {
            "max_position_size_percent": 5.0,
            "max_total_positions": 10,
            "max_drawdown_percent": 10.0,
            "stop_loss_percent": 2.0
        },
        "trading_hours": {
            "start": "09:30",
            "end": "16:00",
            "timezone": "America/New_York"
        },
        "data_settings": {
            "bar_timeframe": "1Min",
            "lookback_days": 30,
            "update_interval_seconds": 60
        },
        "web_portal": {
            "enabled": True,
            "port": 8080,
            "host": "localhost"
        },
        "notifications": {
            "email": {
                "enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "from_address": "",
                "to_address": "",
                "username": "",
                "password": ""
            },
            "webhook": {
                "enabled": False,
                "url": ""
            }
        }
    }
    
    config_path = os.path.join(project_dir, 'config', 'config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created configuration file: {config_path}")

def create_strategy_configs(project_dir):
    """Create default strategy configuration files"""
    strategy_configs = {
        "bollinger_bands": {
            "enabled": True,
            "symbols": ["AAPL", "MSFT"],
            "parameters": {
                "timeperiod": 20,
                "nbdevup": 2.0,
                "nbdevdn": 2.0,
                "matype": 0
            },
            "risk_multiplier": 1.0,
            "trade_direction": "both"  # both, long, short
        },
        "moving_average_crossover": {
            "enabled": True,
            "symbols": ["AMZN", "GOOGL"],
            "parameters": {
                "fast_period": 9,
                "slow_period": 21,
                "signal_period": 9
            },
            "risk_multiplier": 0.8,
            "trade_direction": "long"
        },
        "rsi": {
            "enabled": True,
            "symbols": ["META"],
            "parameters": {
                "timeperiod": 14,
                "overbought": 70,
                "oversold": 30
            },
            "risk_multiplier": 0.7,
            "trade_direction": "both"
        }
    }
    
    strategy_dir = os.path.join(project_dir, 'config', 'strategies')
    os.makedirs(strategy_dir, exist_ok=True)
    
    for strategy_name, config in strategy_configs.items():
        config_path = os.path.join(strategy_dir, f"{strategy_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Created strategy configuration: {config_path}")

def create_module_structure(project_dir, project_name):
    """Create the Python module structure for the project"""
    module_dir = os.path.join(project_dir, project_name)
    os.makedirs(module_dir, exist_ok=True)
    
    modules = [
        "data_engine",
        "strategy_engine",
        "risk_manager",
        "execution_engine",
        "performance_tracker",
        "market_health",
        "alerts",
        "state_manager",
        "task_manager",
        "web_portal",
        "utils"
    ]
    
    # Create __init__.py in main module
    with open(os.path.join(module_dir, "__init__.py"), 'w') as f:
        f.write(f'"""Main package for {project_name}."""\n\n__version__ = "0.1.0"\n')
    
    # Create module subdirectories with __init__.py
    for module in modules:
        module_path = os.path.join(module_dir, module)
        os.makedirs(module_path, exist_ok=True)
        
        with open(os.path.join(module_path, "__init__.py"), 'w') as f:
            f.write(f'"""{module.replace("_", " ").title()} module."""\n')
        
        print(f"Created module: {module_path}")

def create_main_script(project_dir, project_name):
    """Create the main script to run the trading system"""
    main_script = f"""#!/usr/bin/env python3
\"\"\"
{project_name.replace("_", " ").title()}
=======================================
Main entry point for the trading system.
\"\"\"

import os
import sys
import json
import logging
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='{project_name.replace("_", " ").title()}')
    parser.add_argument('-c', '--config', dest='config_path', default='config/config.json',
                        help='Path to configuration file')
    parser.add_argument('-l', '--log-level', dest='log_level', default=None,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (overrides config file)')
    parser.add_argument('--backtest', dest='backtest', action='store_true',
                        help='Run in backtest mode')
    return parser.parse_args()

def setup_logging(log_level):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, '{project_name}.log')),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    log_level = args.log_level or config.get('log_level', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger('{project_name}')
    logger.info('Starting {project_name.replace("_", " ").title()}')
    
    # TODO: Initialize system components
    # from {project_name}.data_engine import DataEngine
    # from {project_name}.strategy_engine import StrategyEngine
    # from {project_name}.risk_manager import RiskManager
    # from {project_name}.execution_engine import ExecutionEngine
    # from {project_name}.web_portal import WebPortal
    
    # TODO: Start system
    
    logger.info('{project_name.replace("_", " ").title()} started')

if __name__ == '__main__':
    main()
"""
    
    script_path = os.path.join(project_dir, f"run_{project_name}.py")
    with open(script_path, 'w') as f:
        f.write(main_script)
    
    # Make executable
    os.chmod(script_path, os.stat(script_path).st_mode | 0o111)
    
    print(f"Created main script: {script_path}")

def create_example_components(project_dir, project_name):
    """Create example components to help get started"""
    
    # Create DataEngine example
    data_engine_path = os.path.join(project_dir, project_name, "data_engine", "data_engine.py")
    data_engine_code = """
\"\"\"
DataEngine module for handling market data ingestion and processing.
\"\"\"

import logging
import asyncio
from typing import Dict, List, Optional
import pandas as pd
import alpaca_trade_api as tradeapi

class DataEngine:
    \"\"\"
    Handles real-time and historical market data ingestion from Alpaca API.
    \"\"\"
    
    def __init__(self, api: tradeapi.REST, symbols: List[str], timeframe: str = "1Min", 
                 lookback_days: int = 30):
        \"\"\"
        Initialize the DataEngine.
        
        Args:
            api: Alpaca REST API instance
            symbols: List of symbols to track
            timeframe: Bar timeframe (e.g. "1Min", "5Min", "1D")
            lookback_days: Number of days of historical data to load
        \"\"\"
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
        \"\"\"Start the data engine\"\"\"
        self.logger.info(f"Starting DataEngine for symbols: {', '.join(self.symbols)}")
        self._running = True
        
        # Load historical data
        await self._load_historical_data()
        
        # Start real-time data stream
        asyncio.create_task(self._stream_data())
    
    async def stop(self):
        \"\"\"Stop the data engine\"\"\"
        self.logger.info("Stopping DataEngine")
        self._running = False
    
    def register_data_callback(self, callback):
        \"\"\"Register a callback to be called when new data arrives\"\"\"
        self._data_callbacks.append(callback)
    
    async def _load_historical_data(self):
        \"\"\"Load historical bar data for all symbols\"\"\"
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
        \"\"\"Stream real-time data updates\"\"\"
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
        \"\"\"
        Get historical bar data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            lookback: Number of bars to return (None for all)
            
        Returns:
            DataFrame with bar data
        \"\"\"
        if symbol not in self.bars:
return pd.DataFrame()
        
        df = self.bars[symbol]
        if lookback is not None:
            return df.tail(lookback)
        return df
"""

    os.makedirs(os.path.dirname(data_engine_path), exist_ok=True)
    with open(data_engine_path, 'w') as f:
        f.write(data_engine_code)
    
    # Create strategy base class example
    strategy_base_path = os.path.join(project_dir, project_name, "strategy_engine", "base_strategy.py")
    strategy_base_code = """
\"\"\"
Base strategy class that all specific strategies inherit from.
\"\"\"

import logging
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal

class Strategy(ABC):
    \"\"\"
    Base strategy class that defines the interface for all trading strategies.
    \"\"\"
    
    def __init__(self, symbol: str, parameters: Dict, risk_multiplier: float = 1.0,
                trade_direction: Literal["long", "short", "both"] = "both"):
        \"\"\"
        Initialize the strategy.
        
        Args:
            symbol: Trading symbol
            parameters: Strategy-specific parameters
            risk_multiplier: Risk multiplier to adjust position size
            trade_direction: Trading direction (long, short, or both)
        \"\"\"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.symbol = symbol
        self.parameters = parameters
        self.risk_multiplier = risk_multiplier
        self.trade_direction = trade_direction
        self.current_position = 0
        self.current_signal = 0  # -1 for sell, 0 for neutral, 1 for buy
    
    @abstractmethod
    def calculate_signal(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        \"\"\"
        Calculate trading signal based on the strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (signal, metadata):
                signal: -1 for sell, 0 for neutral, 1 for buy
                metadata: Dictionary with additional signal information
        \"\"\"
        pass
    
    def filter_signal(self, signal: int) -> int:
        \"\"\"
        Filter signal based on trade direction setting.
        
        Args:
            signal: Raw signal (-1, 0, or 1)
            
        Returns:
            Filtered signal
        \"\"\"
        if self.trade_direction == "long" and signal < 0:
            return 0
        elif self.trade_direction == "short" and signal > 0:
            return 0
        return signal
    
    def process_data(self, data: pd.DataFrame) -> Optional[Dict]:
        \"\"\"
        Process new data and generate trading signal if needed.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Signal info dictionary if signal changed, None otherwise
        \"\"\"
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
"""

    os.makedirs(os.path.dirname(strategy_base_path), exist_ok=True)
    with open(strategy_base_path, 'w') as f:
        f.write(strategy_base_code)

    # Create example strategy implementation
    bollinger_strategy_path = os.path.join(project_dir, project_name, "strategy_engine", "bollinger_bands_strategy.py")
    bollinger_strategy_code = """\"\"\"
Bollinger Bands trading strategy implementation.
\"\"\"

import pandas as pd
import numpy as np
import talib
from typing import Dict, Tuple

from .base_strategy import Strategy

class BollingerBandsStrategy(Strategy):
    \"\"\"
    Trading strategy based on Bollinger Bands indicator.
    
    Buy signal when price crosses below lower band and then back above it.
    Sell signal when price crosses above upper band and then back below it.
    \"\"\"
    
    def __init__(self, symbol: str, parameters: Dict, risk_multiplier: float = 1.0,
                trade_direction: str = "both"):
        \"\"\"
        Initialize the Bollinger Bands strategy.
        
        Parameters:
            timeperiod: Period for moving average calculation
            nbdevup: Standard deviation multiplier for upper band
            nbdevdn: Standard deviation multiplier for lower band
            matype: Moving average type
        \"\"\"
        super().__init__(symbol, parameters, risk_multiplier, trade_direction)
        self.was_below_lower = False
        self.was_above_upper = False
    
    def calculate_signal(self, data: pd.DataFrame) -> Tuple[int, Dict]:
        \"\"\"
        Calculate trading signal based on Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (signal, metadata)
        \"\"\"
        # Extract parameters
        timeperiod = self.parameters.get("timeperiod", 20)
        nbdevup = self.parameters.get("nbdevup", 2.0)
        nbdevdn = self.parameters.get("nbdevdn", 2.0)
        matype = self.parameters.get("matype", 0)
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        )
        
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
"""

    os.makedirs(os.path.dirname(bollinger_strategy_path), exist_ok=True)
    with open(bollinger_strategy_path, 'w') as f:
        f.write(bollinger_strategy_code)

    print(f"Created example components in {project_name} package")

def create_requirements_file(project_dir):
    """Create a requirements.txt file with necessary dependencies"""
    requirements = """
# API and Data
alpaca-trade-api>=2.0.0
websocket-client>=1.0.0
websockets>=10.0
pandas>=1.3.0
numpy>=1.20.0

# Technical Analysis
ta-lib>=0.4.0

# Web Interface
flask>=2.0.0
flask-cors>=3.0.0
flask-socketio>=5.0.0

# Utilities
python-dotenv>=0.19.0
schedule>=1.0.0
tqdm>=4.62.0

# Visualization
matplotlib>=3.4.0
plotly>=5.0.0

# Testing
pytest>=6.0.0
pytest-cov>=2.0.0
"""
    
    requirements_path = os.path.join(project_dir, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements.strip())
    
    print(f"Created requirements file: {requirements_path}")

def create_readme(project_dir, project_name):
    """Create a README.md file for the project"""
    readme = f"""# {project_name.replace("_", " ").title()}

An algorithmic trading system using the Alpaca API.

## Core Components

1. **Data Engine**: Handles real-time and historical market data ingestion, managing streams of bars, quotes, and trades for multiple symbols.
2. **Strategy Engine**: Implements a modular approach to trading strategies with a base strategy class that specific strategies inherit from.
3. **Risk Manager**: Controls position sizing and enforces risk limits.
4. **Execution Engine**: Manages order submission, tracking, and updates through the Alpaca API.
5. **Performance Tracker**: Monitors and reports on system performance metrics.
6. **Market Health Monitor**: Analyzes overall market conditions to adjust trading behavior.
7. **Alert Manager**: Sends notifications when system events or alerts occur.
8. **System State Manager**: Saves and loads system state for continuity across restarts.
9. **Scheduled Task Manager**: Handles recurring tasks and scheduled operations.
10. **Web Portal**: Provides a simple web interface for monitoring and controlling the trading system.

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your Alpaca API keys in `config/config.json` or use environment variables.

3. Customize the strategy configurations in `config/strategies/`.

4. Run the system:
   ```
   ./run_{project_name}.py
   ```

## Configuration

The system is configured through JSON files in the `config` directory. You can customize:

- API credentials
- Trading parameters
- Risk management settings
- Strategy-specific parameters
- Notification settings

## Strategies

Currently implemented strategies:

- Bollinger Bands Strategy
- Moving Average Crossover Strategy
- RSI (Relative Strength Index) Strategy

## Contributing

Feel free to fork and contribute to this project.
"""
    
    readme_path = os.path.join(project_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"Created README: {readme_path}")

def main():
    args = parse_arguments()
    
    project_dir = os.path.abspath(args.project_dir)
    project_name = args.project_name
    
    print(f"Initializing Alpaca Trading System project '{project_name}' in {project_dir}")
    
    # Ensure directory exists
    os.makedirs(project_dir, exist_ok=True)
    
    # Get API credentials if not provided
    api_key = args.api_key
    if not api_key:
        api_key = getpass.getpass("Enter Alpaca API Key: ")
    
    api_secret = args.api_secret
    if not api_secret:
        api_secret = getpass.getpass("Enter Alpaca API Secret: ")
    
    # Create directory structure
    create_directory_structure(project_dir)
    
    # Create configuration files
    create_config_file(project_dir, api_key, api_secret, args.paper_trading)
    create_strategy_configs(project_dir)
    
    # Create module structure
    create_module_structure(project_dir, project_name)
    
    # Create example components
    create_example_components(project_dir, project_name)
    
    # Create main script
    create_main_script(project_dir, project_name)
    
    # Create requirements.txt
    create_requirements_file(project_dir)
    
    # Create README.md
    create_readme(project_dir, project_name)
    
    print(f"\nProject initialization complete!")
    print(f"To get started, run:")
    print(f"  cd {project_dir}")
    print(f"  pip install -r requirements.txt")
    print(f"  ./run_{project_name}.py")

if __name__ == "__main__":
    main()