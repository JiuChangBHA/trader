"""
Alpaca-based Algorithmic Trading System
=======================================

A comprehensive, scalable trading system that supports multiple stocks
with individualized strategies and parameters.

Main modules:
- Data Engine: Real-time and historical data ingestion
- Strategy Engine: Strategy implementation and signal generation
- Risk Manager: Position sizing and risk controls
- Execution Engine: Order placement and management
- Monitoring: Logging, performance tracking, and alerts
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# Alpaca API integration
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.rest import REST, TimeFrame

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration loader
class Config:
    def __init__(self, config_path: str = "config.json"):
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Extract API credentials
            self.api_key = self.config.get('alpaca_api_key') or os.environ.get('ALPACA_API_KEY')
            self.api_secret = self.config.get('alpaca_api_secret') or os.environ.get('ALPACA_API_SECRET')
            self.base_url = self.config.get('alpaca_base_url', 'https://api.alpaca.markets')
            
            # Trading parameters
            self.symbols = self.config.get('symbols', [])
            self.strategies = self.config.get('strategies', {})
            self.risk_params = self.config.get('risk_parameters', {})
            self.market_hours = self.config.get('market_hours', {})
            
            # System settings
            self.data_retention = self.config.get('data_retention_days', 30)
            self.backtest_mode = self.config.get('backtest_mode', False)
            
            # Trading mode (paper or live)
            self.trading_mode = self.config.get('trading_mode', 'paper')
            
            # Backtest settings
            self.initial_equity = self.config.get('initial_equity', 100000.0)  # Default $100,000
            
            # Local data settings
            self.use_local_data = self.config.get('use_local_data', False)
            self.local_data_dir = self.config.get('local_data_dir', '')
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def set_backtest_mode(self, enabled: bool = True):
        """Enable or disable backtest mode"""
        self.backtest_mode = enabled
        logger.info(f"Backtest mode {'enabled' if enabled else 'disabled'}")
        
    def set_backtest_dates(self, start_date: str, end_date: str):
        """Set backtest date range"""
        self.backtest_start_date = start_date
        self.backtest_end_date = end_date
        logger.info(f"Backtest date range set: {start_date} to {end_date}")
        
    def set_local_data_options(self, use_local_data: bool, local_data_dir: str = None):
        """Set options for using local data files"""
        self.use_local_data = use_local_data
        if local_data_dir:
            self.local_data_dir = local_data_dir
        logger.info(f"Local data {'enabled' if use_local_data else 'disabled'}" + 
                   (f" with directory: {local_data_dir}" if local_data_dir else ""))

# Data Engine: Handles market data ingestion and processing
class DataEngine:
    def __init__(self, api: REST, symbols: List[str], backtest_mode: bool = False):
        self.api = api
        self.symbols = symbols
        self.bars = {symbol: pd.DataFrame() for symbol in symbols}
        self.quotes = {symbol: None for symbol in symbols}
        self.trades = {symbol: None for symbol in symbols}
        self.data_queue = queue.Queue()
        self.running = False
        self.data_thread = None
        self.backtest_mode = backtest_mode
        
    def start(self):
        """Start the data engine"""
        self.running = True
        
        # Skip data polling in backtest mode
        if not self.backtest_mode:
            self.data_thread = threading.Thread(target=self._poll_data)
            self.data_thread.daemon = True
            self.data_thread.start()
            logger.info("Data Engine started with live data polling")
        else:
            logger.info("Data Engine started in backtest mode (no live data polling)")
        
    def stop(self):
        """Stop the data engine"""
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=5)
        logger.info("Data Engine stopped")
    
    def _poll_data(self):
        """Poll data from Alpaca API instead of streaming to avoid connection limits"""
        logger.info("Starting data polling")
        
        while self.running:
            try:
                # Poll for latest bars for each symbol
                for symbol in self.symbols:
                    try:
                        # Get the latest bar
                        latest_bar = self.api.get_latest_bar(symbol)
                        
                        # Create a DataFrame with the new bar
                        new_bar = pd.DataFrame({
                            'timestamp': [pd.Timestamp(latest_bar.t)],
                            'open': [latest_bar.o],
                            'high': [latest_bar.h],
                            'low': [latest_bar.l],
                            'close': [latest_bar.c],
                            'volume': [latest_bar.v]
                        })
                        new_bar.set_index('timestamp', inplace=True)
                        
                        # Update our bars dictionary
                        if symbol in self.bars:
                            # Only add if it's a new bar or we have no data yet
                            if len(self.bars[symbol]) == 0:
                                self.bars[symbol] = new_bar
                                self.data_queue.put(('BAR', symbol, latest_bar))
                            elif len(self.bars[symbol].index) > 0:
                                # Ensure consistent timezone handling
                                last_timestamp = self.bars[symbol].index[-1]
                                new_timestamp = new_bar.index[0]
                                
                                # Make both timestamps timezone-aware for comparison
                                if last_timestamp.tz is None and new_timestamp.tz is not None:
                                    last_timestamp = last_timestamp.tz_localize(new_timestamp.tz)
                                elif last_timestamp.tz is not None and new_timestamp.tz is None:
                                    new_timestamp = new_timestamp.tz_localize(last_timestamp.tz)
                                
                                if new_timestamp > last_timestamp:
                                    self.bars[symbol] = pd.concat([self.bars[symbol], new_bar]).tail(1000)
                                    self.data_queue.put(('BAR', symbol, latest_bar))
                        
                        # Get latest quote
                        latest_quote = self.api.get_latest_quote(symbol)
                        self.quotes[symbol] = latest_quote
                        self.data_queue.put(('QUOTE', symbol, latest_quote))
                        
                        # Get latest trade
                        latest_trade = self.api.get_latest_trade(symbol)
                        self.trades[symbol] = latest_trade
                        self.data_queue.put(('TRADE', symbol, latest_trade))
                        
                        # Add a small delay between symbols to avoid rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Error polling data for {symbol}: {str(e)}")
                
                # Wait before polling again
                time.sleep(5)  # Poll every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in data polling: {str(e)}")
                time.sleep(10)  # Wait longer before retrying after an error
    
    def get_historical_data(self, symbol: str, timeframe: TimeFrame = TimeFrame.Day, 
                           limit: int = 100) -> pd.DataFrame:
        """Fetch historical bar data for a given symbol"""
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            if not bars.empty:
                bars.index = pd.to_datetime(bars.index)
                # Store in our bars dictionary with the most recent data
                self.bars[symbol] = bars
            return bars
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

# Base Strategy class that all strategies will inherit from
class BaseStrategy:
    def __init__(self, symbol: str, parameters: Dict[str, Any]):
        self.symbol = symbol
        self.parameters = parameters
        self.name = self.__class__.__name__
        self.position = 0
        
    def process_bar(self, bar_data: pd.DataFrame) -> Dict[str, Any]:
        """Process new bar data and generate trading signals"""
        raise NotImplementedError("Subclasses must implement process_bar method")
    
    def get_min_bars_required(self) -> int:
        """Return the minimum number of bars required for this strategy"""
        raise NotImplementedError("Subclasses must implement get_min_bars_required method")

# Sample strategy implementations
class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any]):
        super().__init__(symbol, parameters)
        # Default parameters if not provided
        self.period = parameters.get('period', 20)
        self.std_dev_multiplier = parameters.get('stdDevMultiplier', 2.0)
    
    def get_min_bars_required(self) -> int:
        return self.period + 1
    
    def process_bar(self, bar_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on Bollinger Bands"""
        if len(bar_data) < self.period:
            return {'action': 'HOLD', 'reason': 'Insufficient data'}
        
        # Calculate Bollinger Bands
        closes = bar_data['close'].values
        sma = np.mean(closes[-self.period:])
        std_dev = np.std(closes[-self.period:])
        upper_band = sma + (std_dev * self.std_dev_multiplier)
        lower_band = sma - (std_dev * self.std_dev_multiplier)
        current_close = closes[-1]
        
        # Generate signals
        signal = {'action': 'HOLD', 'price': current_close, 'sma': sma, 
                 'upper_band': upper_band, 'lower_band': lower_band}
        
        if current_close < lower_band and self.position <= 0:
            signal['action'] = 'BUY'
            signal['reason'] = 'Price below lower Bollinger Band'
        elif current_close > upper_band and self.position >= 0:
            signal['action'] = 'SELL'
            signal['reason'] = 'Price above upper Bollinger Band'
            
        return signal

class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any]):
        super().__init__(symbol, parameters)
        self.short_period = parameters.get('short_period', 9)
        self.long_period = parameters.get('long_period', 21)
    
    def get_min_bars_required(self) -> int:
        return max(self.short_period, self.long_period) + 1
    
    def process_bar(self, bar_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on Moving Average Crossover"""
        if len(bar_data) < self.long_period:
            return {'action': 'HOLD', 'reason': 'Insufficient data'}
        
        # Calculate moving averages
        closes = bar_data['close'].values
        short_ma = np.mean(closes[-self.short_period:])
        long_ma = np.mean(closes[-self.long_period:])
        prev_short_ma = np.mean(closes[-self.short_period-1:-1])
        prev_long_ma = np.mean(closes[-self.long_period-1:-1])
        
        # Determine crossover
        signal = {'action': 'HOLD', 'price': closes[-1], 
                 'short_ma': short_ma, 'long_ma': long_ma}
        
        # Golden Cross (short crosses above long)
        if prev_short_ma <= prev_long_ma and short_ma > long_ma and self.position <= 0:
            signal['action'] = 'BUY'
            signal['reason'] = 'Golden Cross'
        # Death Cross (short crosses below long)
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma and self.position >= 0:
            signal['action'] = 'SELL'
            signal['reason'] = 'Death Cross'
            
        return signal

class RSIStrategy(BaseStrategy):
    def __init__(self, symbol: str, parameters: Dict[str, Any]):
        super().__init__(symbol, parameters)
        self.period = parameters.get('period', 14)
        self.overbought = parameters.get('overbought', 70)
        self.oversold = parameters.get('oversold', 30)
    
    def get_min_bars_required(self) -> int:
        return self.period + 1
    
    def process_bar(self, bar_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on RSI (Relative Strength Index)"""
        if len(bar_data) < self.period + 1:
            return {'action': 'HOLD', 'reason': 'Insufficient data'}
        
        # Calculate price changes
        closes = bar_data['close'].values
        price_diff = np.diff(closes)
        
        # Separate gains and losses
        gains = np.where(price_diff > 0, price_diff, 0)
        losses = np.where(price_diff < 0, -price_diff, 0)
        
        # Calculate RSI
        avg_gain = np.mean(gains[-self.period:])
        avg_loss = np.mean(losses[-self.period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        signal = {'action': 'HOLD', 'price': closes[-1], 'rsi': rsi}
        
        # Generate signals
        if rsi < self.oversold and self.position <= 0:
            signal['action'] = 'BUY'
            signal['reason'] = f'RSI oversold ({rsi:.2f})'
        elif rsi > self.overbought and self.position >= 0:
            signal['action'] = 'SELL'
            signal['reason'] = f'RSI overbought ({rsi:.2f})'
            
        return signal

# Factory for creating strategy instances
class StrategyFactory:
    _strategies = {
        'BollingerBands': BollingerBandsStrategy,
        'MovingAverageCrossover': MovingAverageCrossoverStrategy,
        'RSI': RSIStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str, symbol: str, parameters: Dict[str, Any]) -> BaseStrategy:
        """Create a strategy instance based on the strategy type"""
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return cls._strategies[strategy_type](symbol, parameters)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class) -> None:
        """Register a new strategy class"""
        cls._strategies[name] = strategy_class

# Risk Manager: Handles position sizing and risk controls
class RiskManager:
    def __init__(self, api: REST, config: Config):
        self.api = api
        self.max_position_size = config.risk_params.get('max_position_size', 0.05)  # 5% of portfolio per position
        self.max_drawdown = config.risk_params.get('max_drawdown', 0.10)  # 10% max drawdown
        self.max_open_positions = config.risk_params.get('max_open_positions', 10)
        self.backtest_mode = config.backtest_mode
        self.positions = {}
        
        # Initialize equity values
        self.starting_equity = config.initial_equity if hasattr(config, 'initial_equity') else 100000.0  # Use config value or default
        self.current_equity = self.starting_equity
        self.current_drawdown = 0.0
        
        # If not in backtest mode, try to get real account equity
        if not self.backtest_mode:
            try:
                account = self.api.get_account()
                self.starting_equity = float(account.equity)
                self.current_equity = self.starting_equity
            except Exception as e:
                logger.warning(f"Could not get account equity, using default: {str(e)}")
        
        logger.info(f"Risk Manager initialized with equity: ${self.current_equity:.2f}")
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get risk metrics for dashboard display"""
        try:
            # Update account info to ensure we have the latest data
            self.update_account_info()
            
            # Calculate volatility based on equity changes if we have performance tracker data
            volatility = 0.0
            var_95 = 0.0
            expected_shortfall = 0.0
            beta = 0.0
            correlation_spy = 0.0
            
            # Return the metrics
            return {
                'max_drawdown': self.current_drawdown * 100,  # Convert to percentage
                'volatility': volatility,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'beta': beta,
                'correlation_spy': correlation_spy
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'var_95': 0.0,
                'expected_shortfall': 0.0,
                'beta': 0.0,
                'correlation_spy': 0.0
            }
    
    def update_account_info(self) -> None:
        """Update account equity information"""
        try:
            if self.backtest_mode:
                # In backtest mode, we use the simulated equity
                # This would normally be updated by the backtest engine based on P&L
                # Make sure we're using the starting equity value
                if self.current_equity == 0:
                    self.current_equity = self.starting_equity
                
                # No need to update anything, just log the current values
                logger.info(f"Account equity: ${self.current_equity:.2f}, Drawdown: {self.current_drawdown:.2%}")
                return
                
            # Live mode - get real account info
            account = self.api.get_account()
            current_equity = float(account.equity)
            
            self.current_equity = current_equity
            
            # Calculate current drawdown
            peak_equity = max(self.starting_equity, current_equity)
            if peak_equity > 0:  # Avoid division by zero
                self.current_drawdown = 1 - (current_equity / peak_equity)
            
            logger.info(f"Account equity: ${current_equity:.2f}, Drawdown: {self.current_drawdown:.2%}")
        except Exception as e:
            logger.error(f"Error updating account info: {str(e)}")
    
    def check_risk_limits(self) -> bool:
        """Check if we've exceeded any risk limits"""
        # Update account info
        self.update_account_info()
        
        # Check max drawdown
        if self.current_drawdown > self.max_drawdown:
            logger.warning(f"Max drawdown exceeded: {self.current_drawdown:.2%} > {self.max_drawdown:.2%}")
            return False
        
        # Check number of open positions
        try:
            if self.backtest_mode:
                # In backtest mode, use our internal position tracking
                num_positions = len([p for p in self.positions.values() if p != 0])
            else:
                positions = self.api.list_positions()
                num_positions = len(positions)
                
            if num_positions >= self.max_open_positions:
                logger.warning(f"Max open positions reached: {num_positions} >= {self.max_open_positions}")
                return False
        except Exception as e:
            logger.error(f"Error checking positions: {str(e)}")
            # Continue with other checks
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float = 1.0) -> int:
        """Calculate position size based on account equity and risk parameters"""
        try:
            # Use the correct equity value
            equity = self.current_equity
            
            # If equity is 0 or None, use the initial equity
            if not equity or equity == 0:
                equity = self.starting_equity
                logger.warning(f"Using initial equity ${equity:.2f} for position sizing")
            
            # Calculate position value based on max position size
            max_position_value = equity * self.max_position_size
            
            # Adjust by signal strength (0.0 to 1.0)
            adjusted_position_value = max_position_value * signal_strength
            
            # Calculate number of shares
            if price > 0:  # Avoid division by zero
                shares = int(adjusted_position_value / price)
            else:
                shares = 0
            
            # Log the calculation
            logger.info(f"Position size for {symbol}: {shares} shares (${shares * price:.2f})")
            
            return shares
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def get_current_position(self, symbol: str) -> int:
        """Get current position for a symbol"""
        try:
            if self.backtest_mode:
                # Use internal position tracking in backtest mode
                return self.positions.get(symbol, 0)
                
            position = self.api.get_position(symbol)
            return int(position.qty)
        except:
            # No position exists
            return 0
    
    def should_exit_all_positions(self) -> bool:
        """Determine if we should exit all positions due to risk limits"""
        # Check if drawdown limit has been breached
        if self.current_drawdown > self.max_drawdown:
            return True
        
        # Add other exit criteria here...
        return False

# Execution Engine: Handles order submission and management
class ExecutionEngine:
    def __init__(self, api: REST):
        self.api = api
        self.orders = {}  # Track orders by symbol
        
    def submit_order(self, symbol: str, qty: int, side: str, 
                    order_type: str = 'market', time_in_force: str = 'day',
                    limit_price: float = None, stop_price: float = None) -> str:
        """Submit an order to Alpaca"""
        try:
            if qty <= 0:
                logger.warning(f"Invalid quantity for {symbol}: {qty}")
                return None
                
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            order_id = order.id
            self.orders[order_id] = {
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'status': order.status,
                'submitted_at': datetime.now()
            }
            
            logger.info(f"Order submitted for {symbol}: {side} {qty} shares as {order_type} order. Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            self.api.cancel_order(order_id)
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'canceled'
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all pending orders"""
        try:
            self.api.cancel_all_orders()
            for order_id in self.orders:
                if self.orders[order_id]['status'] in ['new', 'accepted', 'pending_new']:
                    self.orders[order_id]['status'] = 'canceled'
            logger.info("All orders canceled")
            return True
        except Exception as e:
            logger.error(f"Error canceling all orders: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get status of an order"""
        try:
            order = self.api.get_order(order_id)
            
            # Update our local tracking
            if order_id in self.orders:
                self.orders[order_id]['status'] = order.status
                
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'filled_qty': order.filled_qty,
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'avg_fill_price': order.filled_avg_price
            }
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {str(e)}")
            return {}
    
    def update_orders(self) -> None:
        """Update status of all tracked orders"""
        for order_id in list(self.orders.keys()):
            order_info = self.orders[order_id]
            if order_info['status'] not in ['filled', 'canceled', 'expired', 'rejected']:
                self.get_order_status(order_id)
                
    def get_recent_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent orders for display in dashboard"""
        try:
            # In backtest mode, use our internal order tracking
            if hasattr(self, 'orders') and self.orders:
                # Convert internal order format to display format
                recent_orders = []
                for order_id, order_info in list(self.orders.items())[-limit:]:
                    recent_orders.append({
                        "symbol": order_info.get('symbol', ''),
                        "side": order_info.get('side', ''),
                        "qty": order_info.get('qty', 0),
                        "type": order_info.get('type', 'market'),
                        "status": order_info.get('status', ''),
                        "created_at": order_info.get('submitted_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                    })
                return recent_orders
            
            # In live mode, get from API
            try:
                orders = self.api.list_orders(status='all', limit=limit)
                return [{
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": order.qty,
                    "type": order.type,
                    "status": order.status,
                    "created_at": order.submitted_at.strftime('%Y-%m-%d %H:%M:%S') if order.submitted_at else ''
                } for order in orders]
            except Exception as e:
                logger.error(f"Error getting orders from API: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting recent orders: {str(e)}")
            return []
            
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions for display in dashboard"""
        try:
            # Check if we're in backtest mode by looking for positions attribute
            if hasattr(self, 'positions') and isinstance(self.positions, dict):
                # Convert internal position format to display format
                position_list = []
                for symbol, position_info in self.positions.items():
                    if position_info.get('qty', 0) != 0:  # Only include non-zero positions
                        # Calculate unrealized P&L if we have current price
                        unrealized_pl = 0
                        unrealized_plpc = 0
                        avg_entry_price = position_info.get('avg_entry_price', 0)
                        current_price = position_info.get('current_price', 0)
                        qty = position_info.get('qty', 0)
                        
                        if avg_entry_price > 0 and current_price > 0 and qty != 0:
                            unrealized_pl = (current_price - avg_entry_price) * qty
                            unrealized_plpc = (current_price / avg_entry_price - 1) * 100
                            
                        position_list.append({
                            "symbol": symbol,
                            "qty": qty,
                            "avg_entry_price": avg_entry_price,
                            "current_price": current_price,
                            "unrealized_pl": unrealized_pl,
                            "unrealized_plpc": unrealized_plpc
                        })
                return position_list
            
            # In live mode, get from API
            try:
                positions = self.api.list_positions()
                return [{
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc)
                } for p in positions]
            except Exception as e:
                logger.error(f"Error getting positions from API: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

# Performance Tracker: Tracks and reports on system performance
class PerformanceTracker:
    def __init__(self, api: REST, db_path: str = "performance.db", backtest_mode: bool = False):
        self.api = api
        self.db_path = db_path
        self.start_time = datetime.now()
        self.trades = []
        self.daily_returns = []
        self.equity_curve = []
        self.initial_equity = 100000.0  # Default initial equity for backtesting
        self.backtest_mode = backtest_mode
        
        # Initialize tracking
        self.update_performance()
    
    def update_performance(self, backtest_date=None) -> None:
        """Update performance metrics"""
        try:
            # Get account info
            try:
                if self.backtest_mode:
                    # In backtest mode, use the initial equity value
                    equity = self.initial_equity
                    logger.debug(f"Using initial equity for backtest: ${equity:.2f}")
                else:
                    # In live mode, get from API
                    account = self.api.get_account()
                    equity = float(account.equity)
            except Exception as e:
                # If API fails, use the last equity value or initial value
                if self.equity_curve:
                    equity = self.equity_curve[-1]['equity']
                else:
                    equity = self.initial_equity
                logger.debug(f"Using fallback equity value: ${equity:.2f}")
                    
            # Record equity point with the appropriate timestamp
            # Use backtest_date if provided and in backtest mode, otherwise use current time
            timestamp = backtest_date if self.backtest_mode and backtest_date else datetime.now()
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity
            })
            
            # Calculate daily return if we have more than one data point
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                if prev_equity > 0:  # Avoid division by zero
                    daily_return = (equity - prev_equity) / prev_equity
                    self.daily_returns.append({
                        'date': timestamp.date(),
                        'return': daily_return
                    })
            
            # Get closed positions (completed trades)
            try:
                if not self.backtest_mode:
                    closed_orders = self.api.list_orders(status='closed', limit=100)
                    
                    # Process new trades
                    for order in closed_orders:
                        if order.filled_at and order.id not in [t['order_id'] for t in self.trades]:
                            self.trades.append({
                                'order_id': order.id,
                                'symbol': order.symbol,
                                'side': order.side,
                                'qty': int(order.qty),
                                'filled_price': float(order.filled_avg_price),
                                'filled_at': order.filled_at,
                                'profit_loss': None  # Will calculate later when we have both sides
                            })
            except Exception as e:
                # In backtest mode or if API fails, we might not have access to orders
                logger.debug(f"Could not retrieve orders: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.equity_curve:
            return {}
            
        try:
            # Calculate returns
            starting_equity = self.equity_curve[0]['equity']
            ending_equity = self.equity_curve[-1]['equity']
            
            if starting_equity > 0:  # Avoid division by zero
                total_return = (ending_equity - starting_equity) / starting_equity
            else:
                total_return = 0
            
            # Calculate Sharpe ratio if we have daily returns
            sharpe_ratio = None
            if self.daily_returns:
                returns = [r['return'] for r in self.daily_returns]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:  # Avoid division by zero
                    sharpe_ratio = (avg_return / std_return) * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0
            
            # Calculate max drawdown
            peak = 0
            max_drawdown = 0
            for point in self.equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']
                if peak > 0:  # Avoid division by zero
                    drawdown = (peak - point['equity']) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
            # Calculate win rate
            win_count = sum(1 for trade in self.trades if trade['profit_loss'] and trade['profit_loss'] > 0)
            total_trades = len([t for t in self.trades if t['profit_loss'] is not None])
            win_rate = win_count / total_trades if total_trades > 0 else 0
                
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def generate_report(self) -> str:
        """Generate a performance report"""
        self.update_performance()
        metrics = self.calculate_metrics()
        
        report = "=== Performance Report ===\n"
        report += f"Report Date: {datetime.now()}\n"
        report += f"Trading Duration: {datetime.now() - self.start_time}\n\n"
        
        # Add performance metrics
        report += "Performance Metrics:\n"
        
        # Format metrics with proper handling of None values
        total_return = metrics.get('total_return')
        if total_return is not None:
            report += f"- Total Return: {total_return:.2%}\n"
        else:
            report += f"- Total Return: N/A\n"
            
        sharpe_ratio = metrics.get('sharpe_ratio')
        if sharpe_ratio is not None:
            report += f"- Sharpe Ratio: {sharpe_ratio:.2f}\n"
        else:
            report += f"- Sharpe Ratio: N/A\n"
            
        max_drawdown = metrics.get('max_drawdown')
        if max_drawdown is not None:
            report += f"- Max Drawdown: {max_drawdown:.2%}\n"
        else:
            report += f"- Max Drawdown: N/A\n"
            
        win_rate = metrics.get('win_rate')
        if win_rate is not None:
            report += f"- Win Rate: {win_rate:.2%}\n"
        else:
            report += f"- Win Rate: N/A\n"
            
        total_trades = metrics.get('total_trades')
        if total_trades is not None:
            report += f"- Total Trades: {total_trades}\n"
        else:
            report += f"- Total Trades: N/A\n"
        
        return report
        
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio statistics for dashboard display"""
        self.update_performance()
        metrics = self.calculate_metrics()
        
        # Convert metrics to percentages for display
        stats = {}
        
        # Total return
        if 'total_return' in metrics and metrics['total_return'] is not None:
            stats['total_return'] = metrics['total_return'] * 100
        else:
            stats['total_return'] = 0.0
            
        # Calculate daily, monthly, annual returns if we have equity data
        if len(self.equity_curve) > 1:
            # Get first and last equity points
            first_equity = self.equity_curve[0]['equity']
            last_equity = self.equity_curve[-1]['equity']
            
            # Calculate trading days
            trading_days = (self.equity_curve[-1]['timestamp'] - self.equity_curve[0]['timestamp']).days
            if trading_days == 0:
                trading_days = 1  # Avoid division by zero
                
            # Calculate returns
            if first_equity > 0:
                total_return = (last_equity / first_equity) - 1
                
                # Annualize returns
                daily_return = (1 + total_return) ** (1 / trading_days) - 1
                monthly_return = (1 + daily_return) ** 21 - 1  # Approx 21 trading days in a month
                annual_return = (1 + daily_return) ** 252 - 1  # Approx 252 trading days in a year
                
                stats['daily_return'] = daily_return * 100
                stats['monthly_return'] = monthly_return * 100
                stats['annual_return'] = annual_return * 100
            else:
                stats['daily_return'] = 0.0
                stats['monthly_return'] = 0.0
                stats['annual_return'] = 0.0
        else:
            stats['daily_return'] = 0.0
            stats['monthly_return'] = 0.0
            stats['annual_return'] = 0.0
            
        # Add other metrics
        if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] is not None:
            stats['sharpe_ratio'] = metrics['sharpe_ratio']
        else:
            stats['sharpe_ratio'] = 0.0
            
        if 'win_rate' in metrics and metrics['win_rate'] is not None:
            stats['win_rate'] = metrics['win_rate'] * 100
        else:
            stats['win_rate'] = 0.0
            
        if 'max_drawdown' in metrics and metrics['max_drawdown'] is not None:
            stats['max_drawdown'] = metrics['max_drawdown'] * 100
        else:
            stats['max_drawdown'] = 0.0
            
        # Add current equity
        if self.equity_curve:
            stats['current_equity'] = self.equity_curve[-1]['equity']
        else:
            stats['current_equity'] = self.initial_equity
            
        return stats
        
    def get_equity_history(self) -> pd.DataFrame:
        """Get equity curve data for charting"""
        if not self.equity_curve:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'equity'])
            
        # Convert to DataFrame
        df = pd.DataFrame(self.equity_curve)
        
        return df
    
    def check_market_hours(self) -> bool:
        """Check if market is open"""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {str(e)}")
            return False

# Strategy Manager: Manages all strategies and processes signals
class StrategyManager:
    def __init__(self, api: REST, config: Config, data_engine: DataEngine, 
                risk_manager: RiskManager, execution_engine: ExecutionEngine):
        self.api = api
        self.config = config
        self.data_engine = data_engine
        self.risk_manager = risk_manager
        self.execution_engine = execution_engine
        self.strategies = {}
        
        # Initialize strategies from config
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Initialize strategies from configuration"""
        for symbol, strategy_configs in self.config.strategies.items():
            self.strategies[symbol] = []
            
            # Create strategy instances for each symbol
            for strategy_config in strategy_configs:
                strategy_type = strategy_config['type']
                parameters = strategy_config['parameters']
                
                try:
                    strategy = StrategyFactory.create_strategy(strategy_type, symbol, parameters)
                    self.strategies[symbol].append(strategy)
                    logger.info(f"Initialized {strategy_type} strategy for {symbol}")
                except Exception as e:
                    logger.error(f"Error initializing {strategy_type} strategy for {symbol}: {str(e)}")
        
        # Load historical data for each symbol
        for symbol in self.strategies:
            self.data_engine.get_historical_data(symbol)
    
    def process_market_data(self, data_type: str, symbol: str, data) -> None:
        """Process incoming market data and execute strategies"""
        # Only process bar data for now
        if data_type != 'BAR' or symbol not in self.strategies:
            return
            
        # Check if we have sufficient data for all strategies
        bar_data = self.data_engine.bars[symbol]
        if bar_data.empty:
            return
            
        # Process each strategy for this symbol
        for strategy in self.strategies[symbol]:
            # Make sure we have enough data
            min_bars = strategy.get_min_bars_required()
            if len(bar_data) < min_bars:
                continue
                
            # Get strategy signal
            signal = strategy.process_bar(bar_data)
            
            # Execute signal if it's a BUY or SELL
            if signal['action'] in ['BUY', 'SELL']:
                self._execute_signal(strategy, symbol, signal)
    
    def _execute_signal(self, strategy: BaseStrategy, symbol: str, signal: Dict[str, Any]) -> None:
        """Execute a trading signal"""
        # Check risk limits first
        if not self.risk_manager.check_risk_limits():
            logger.warning(f"Risk limits exceeded. Not executing {signal['action']} signal for {symbol}")
            return
            
        # Get current position for this symbol
        current_position = self.risk_manager.get_current_position(symbol)
        
        # Determine order side and quantity
        side = signal['action'].lower()
        current_price = signal['price']
        
        if side == 'buy':
            # If already long, don't increase position
            if current_position > 0:
                logger.info(f"Already long {current_position} shares of {symbol}. Not buying more.")
                return
                
            # Calculate position size
            qty = self.risk_manager.calculate_position_size(symbol, current_price)
            
            # If short, close position first
            if current_position < 0:
                close_qty = abs(current_position)
                logger.info(f"Closing short position of {close_qty} shares for {symbol}")
                self.execution_engine.submit_order(symbol, close_qty, 'buy')
                qty = max(0, qty - close_qty)  # Adjust remaining quantity
                
        elif side == 'sell':
            # If already short, don't increase position
            if current_position < 0:
                logger.info(f"Already short {abs(current_position)} shares of {symbol}. Not selling more.")
                return
                
            # Calculate position size
            qty = self.risk_manager.calculate_position_size(symbol, current_price)
            
            # If long, close position first
            if current_position > 0:
                close_qty = current_position
                logger.info(f"Closing long position of {close_qty} shares for {symbol}")
                self.execution_engine.submit_order(symbol, close_qty, 'sell')
                qty = max(0, qty - close_qty)  # Adjust remaining quantity
        
        # Submit order if quantity is valid
        if qty > 0:
            logger.info(f"Executing {side} signal for {symbol}: {signal['reason']}")
            order_id = self.execution_engine.submit_order(symbol, qty, side)
            
            # Update strategy's position tracking
            if order_id:
                if side == 'buy':
                    strategy.position += qty
                else:
                    strategy.position -= qty

# Main Trading System class
class AlpacaTradingSystem:
    def __init__(self, config_path: str = "config.json"):
        # Load configuration
        self.config = Config(config_path)
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=self.config.api_key,
            secret_key=self.config.api_secret,
            base_url=self.config.base_url
        )
        
        # Create system components
        self.data_engine = DataEngine(self.api, self.config.symbols, self.config.backtest_mode)
        self.risk_manager = RiskManager(self.api, self.config)
        self.execution_engine = ExecutionEngine(self.api)
        self.strategy_manager = StrategyManager(
            self.api, self.config, self.data_engine, 
            self.risk_manager, self.execution_engine
        )
        
        # Initialize performance tracker with correct initial equity
        self.performance_tracker = PerformanceTracker(self.api, backtest_mode=self.config.backtest_mode)
        self.performance_tracker.initial_equity = self.config.initial_equity if hasattr(self.config, 'initial_equity') else 100000.0
        
        # System state
        self.running = False
        self.main_thread = None
        self.start_time = None
        
        # Log storage for dashboard
        self.log_buffer = []
        self.max_log_entries = 1000  # Maximum number of log entries to store
        
        # Set up log handler to capture logs
        self._setup_log_capture()
        
    def _setup_log_capture(self):
        """Set up log capture for dashboard display"""
        class LogCaptureHandler(logging.Handler):
            def __init__(self, trading_system):
                super().__init__()
                self.trading_system = trading_system
                
            def emit(self, record):
                # Use backtest date for timestamp if in backtest mode
                if hasattr(self.trading_system, 'config') and self.trading_system.config.backtest_mode and hasattr(self.trading_system, 'backtest_current_date'):
                    # Use the current backtest date for the log timestamp
                    timestamp = self.trading_system.backtest_current_date.strftime("%Y-%m-%d %H:%M:%S") if self.trading_system.backtest_current_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                log_entry = {
                    "timestamp": timestamp,
                    "level": record.levelname,
                    "message": record.getMessage()
                }
                
                # Add to log buffer, maintaining max size
                self.trading_system.log_buffer.append(log_entry)
                if len(self.trading_system.log_buffer) > self.trading_system.max_log_entries:
                    self.trading_system.log_buffer.pop(0)
        
        # Add our custom handler to the logger
        log_handler = LogCaptureHandler(self)
        logger.addHandler(log_handler)
        
    def get_logs(self, level="all"):
        """Get system logs filtered by level"""
        if level.lower() == "all":
            return self.log_buffer
        else:
            # Filter logs by level
            return [log for log in self.log_buffer if log["level"].lower() == level.lower()]
    
    def start(self):
        """Start the trading system"""
        if self.running:
            logger.warning("Trading system is already running")
            return
        
        logger.info("Starting Alpaca Trading System")
        self.start_time = datetime.now()
        
        # Start data engine
        self.data_engine.start()
        
        # Start main trading loop in a separate thread
        self.running = True
        self.main_thread = threading.Thread(target=self._trading_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        logger.info("Trading system started successfully")
        
    def stop(self):
        """Stop the trading system"""
        if not self.running:
            logger.warning("Trading system is not running")
            return
        
        logger.info("Stopping Alpaca Trading System")
        
        # Set flag to stop the main loop
        self.running = False
        
        # Wait for main thread to finish
        if self.main_thread:
            self.main_thread.join(timeout=10)
        
        # Stop data engine
        try:
            self.data_engine.stop()
        except Exception as e:
            logger.error(f"Error stopping data engine: {str(e)}")
        
        # Cancel all pending orders
        try:
            self.execution_engine.cancel_all_orders()
        except Exception as e:
            logger.error(f"Error canceling orders: {str(e)}")
        
        logger.info("Trading system stopped successfully")
        
    def run_backtest(self, start_date: str, end_date: str, use_local_data: bool = False, local_data_dir: str = None):
        """Run system in backtest mode"""
        if self.running:
            logger.warning("Cannot start backtest while system is running")
            return
            
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Enable backtest mode
        self.config.set_backtest_mode(True)
        self.config.set_backtest_dates(start_date, end_date)
        
        # Configure local data options if provided
        if use_local_data:
            self.config.set_local_data_options(True, local_data_dir)
            logger.info(f"Using local data from: {local_data_dir}")
        
        # Set initial equity for backtesting
        initial_equity = self.config.initial_equity if hasattr(self.config, 'initial_equity') else 100000.0
        
        # Reinitialize components with backtest mode
        self.data_engine = DataEngine(self.api, self.config.symbols, backtest_mode=True)
        
        # Reinitialize the risk manager with backtest mode enabled
        self.risk_manager = RiskManager(self.api, self.config)
        self.risk_manager.starting_equity = initial_equity
        self.risk_manager.current_equity = initial_equity
        
        # Reinitialize the performance tracker with backtest mode enabled
        self.performance_tracker = PerformanceTracker(self.api, backtest_mode=True)
        self.performance_tracker.initial_equity = initial_equity
        
        # Force update the equity curve with the initial equity at the backtest start date
        self.performance_tracker.equity_curve = []
        start_date_dt = pd.to_datetime(start_date)
        self.performance_tracker.update_performance(backtest_date=start_date_dt)
        
        logger.info(f"Backtest initial equity: ${initial_equity:.2f}")
        
        # Setup backtest data
        self._setup_backtest(start_date, end_date)
        
        # Run backtest
        self.start()
        
        # Wait for completion
        max_wait_time = 300  # Maximum wait time in seconds (5 minutes)
        start_wait_time = time.time()
        last_report_time = time.time()
        report_interval = 60  # Generate report every 60 seconds
        
        while self.running:
            # Check if we've exceeded the maximum wait time
            if time.time() - start_wait_time > max_wait_time:
                logger.warning("Backtest exceeded maximum wait time, stopping")
                self.stop()
                break
                
            # Check if backtest is complete
            if hasattr(self, 'backtest_data_processed') and self.backtest_data_processed:
                logger.info("Backtest data processing complete")
                break
            
            # Periodically generate report during backtest
            current_time = time.time()
            if current_time - last_report_time > report_interval:
                # Generate interim report
                self.performance_tracker.generate_report()
                last_report_time = current_time
                
            time.sleep(1)
            
        # Generate final report
        report = self.performance_tracker.generate_report()
        logger.info(f"Backtest completed. Report:\n{report}")
        
        # Stop the data engine to prevent further polling
        self.data_engine.stop()
        
        # Restore original trading loop if we're not already stopped
        if hasattr(self, '_original_trading_loop'):
            self._trading_loop = self._original_trading_loop
        
        return report
        
    def _trading_loop(self):
        """Main trading loop"""
        logger.info("Trading loop started")
        
        while self.running:
            try:
                # Check market hours
                is_market_open = self.strategy_manager.check_market_hours()
                
                if is_market_open or self.config.backtest_mode:
                    # Process data from queue
                    while not self.data_engine.data_queue.empty():
                        data_type, symbol, data = self.data_engine.data_queue.get()
                        self.strategy_manager.process_market_data(data_type, symbol, data)
                    
                    # Update order status
                    self.execution_engine.update_orders()
                    
                    # Update performance metrics
                    self.performance_tracker.update_performance()
                    
                    # Check for risk threshold breaches
                    if self.risk_manager.should_exit_all_positions():
                        logger.warning("Risk thresholds exceeded - exiting all positions")
                        # TODO: Implement exit all positions function
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                if self.config.backtest_mode:
                    # In backtest mode, we stop on errors
                    self.running = False
                
        logger.info("Trading loop ended")
        
    def _setup_backtest(self, start_date: str, end_date: str):
        """Setup data for backtesting"""
        logger.info("Loading historical data for backtest")
        
        # Convert dates to datetime objects
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Store backtest end date for checking completion
        self.backtest_end_date = end
        self.backtest_current_date = start
        self.backtest_data_processed = False
        
        # Track symbols with data
        symbols_with_data = []
        
        # Get historical data for each symbol
        for symbol in self.config.symbols:
            try:
                # Check if we should use local data
                if self.config.use_local_data and self.config.local_data_dir:
                    # Load data from local CSV file
                    bars = self._load_data_from_csv(symbol, start, end)
                else:
                    # Get daily bars for the backtest period from Alpaca API
                    bars = self.api.get_bars(
                        symbol, 
                        TimeFrame.Day, 
                        start.strftime('%Y-%m-%d'),
                        end.strftime('%Y-%m-%d'),
                        adjustment='raw'
                    ).df
                
                # Store in data engine
                if not bars.empty:
                    self.data_engine.bars[symbol] = bars
                    logger.info(f"Loaded {len(bars)} bars for {symbol}")
                    symbols_with_data.append(symbol)
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error loading backtest data for {symbol}: {str(e)}")
        
        # Update the symbols list to only include those with data
        if not symbols_with_data:
            logger.error("No data available for any symbols. Cannot run backtest.")
            return
            
        logger.info(f"Backtest will use data for {len(symbols_with_data)} symbols: {', '.join(symbols_with_data)}")
                
        # Modify the _trading_loop method to process backtest data day by day
        self._original_trading_loop = self._trading_loop
        self._trading_loop = self._backtest_trading_loop
        
    def _load_data_from_csv(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """Load historical data from a local CSV file"""
        try:
            # Construct the file path
            file_path = os.path.join(self.config.local_data_dir, f"{symbol}_data.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"CSV file not found for {symbol}: {file_path}")
                return pd.DataFrame()
                
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Convert date column to datetime and set as index
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Make timestamps timezone-aware (use 'America/New_York' for market data)
            df['Date'] = df['Date'].dt.tz_localize('America/New_York')
            
            df = df.set_index('Date')
            
            # Rename columns to match Alpaca API format
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # Filter by date range
            # Make sure start_date and end_date are also timezone-aware
            if start_date.tz is None:
                start_date = start_date.tz_localize('America/New_York')
            if end_date.tz is None:
                end_date = end_date.tz_localize('America/New_York')
                
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Sort by date
            df = df.sort_index()
            
            logger.info(f"Loaded {len(df)} rows from CSV for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _backtest_trading_loop(self):
        """Trading loop for backtest mode"""
        logger.info("Trading loop started")
        
        # Process each day in the backtest period
        current_date = self.backtest_current_date
        end_date = self.backtest_end_date
        
        while self.running and current_date <= end_date:
            try:
                logger.info(f"Processing backtest data for {current_date.strftime('%Y-%m-%d')}")
                
                # Process data for current date
                for symbol in self.config.symbols:
                    if symbol in self.data_engine.bars:
                        # Get data for current date
                        bar_data = self.data_engine.bars[symbol]
                        
                        # Convert current_date to string for comparison
                        current_date_str = current_date.strftime('%Y-%m-%d')
                        
                        # Filter bars for the current date
                        day_data = pd.DataFrame()
                        for idx, row in bar_data.iterrows():
                            # Handle timezone-aware and timezone-naive timestamps
                            idx_date_str = idx.strftime('%Y-%m-%d')
                            if idx_date_str == current_date_str:
                                # Use concat instead of append
                                new_row = pd.DataFrame([row], index=[idx])
                                day_data = pd.concat([day_data, new_row]) if not day_data.empty else new_row
                        
                        if not day_data.empty:
                            # Process each bar for this day
                            for idx, bar in day_data.iterrows():
                                # Create a bar object similar to what would come from the API
                                bar_obj = type('obj', (object,), {
                                    't': idx,
                                    'o': bar['open'],
                                    'h': bar['high'],
                                    'l': bar['low'],
                                    'c': bar['close'],
                                    'v': bar['volume'],
                                    'symbol': symbol
                                })
                                
                                # Add to data queue
                                self.data_engine.data_queue.put(('BAR', symbol, bar_obj))
                
                # Process data from queue
                while not self.data_engine.data_queue.empty():
                    data_type, symbol, data = self.data_engine.data_queue.get()
                    self.strategy_manager.process_market_data(data_type, symbol, data)
                
                # Update order status
                self.execution_engine.update_orders()
                
                # Update performance metrics with the current backtest date
                self.performance_tracker.update_performance(backtest_date=current_date)
                
                # Check for risk threshold breaches
                if self.risk_manager.should_exit_all_positions():
                    logger.warning("Risk thresholds exceeded - exiting all positions")
                    # TODO: Implement exit all positions function
                
                # Move to next day
                current_date += timedelta(days=1)
                self.backtest_current_date = current_date
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in backtest trading loop: {str(e)}")
                self.running = False
        
        # Backtest complete
        logger.info("Backtest complete - all historical data processed")
        self.backtest_data_processed = True
        self.running = False
        logger.info("Trading loop ended")

class MarketHealthMonitor:
    """Monitors overall market health to adjust trading behavior"""
    def __init__(self, api: REST, market_index: str = "SPY"):
        self.api = api
        self.market_index = market_index
        self.vix_symbol = "^VIX"  # Volatility index
        self.market_index_data = pd.DataFrame()
        self.vix_data = pd.DataFrame()
        self.market_trends = {
            'short_term': None,  # Up, Down, Sideways
            'medium_term': None,
            'long_term': None
        }
        self.volatility_state = None  # Low, Medium, High
        self.last_update = None
        
    def update(self) -> None:
        """Update market health indicators"""
        now = datetime.now()
        
        # Only update once per hour
        if self.last_update and (now - self.last_update).seconds < 3600:
            return
            
        try:
            # Get market index data (e.g., S&P 500)
            market_bars = self.api.get_bars(
                self.market_index,
                TimeFrame.Day,
                limit=100
            ).df
            
            if not market_bars.empty:
                self.market_index_data = market_bars
                
                # Calculate moving averages for trend determination
                self.market_index_data['sma20'] = self.market_index_data['close'].rolling(window=20).mean()
                self.market_index_data['sma50'] = self.market_index_data['close'].rolling(window=50).mean()
                self.market_index_data['sma200'] = self.market_index_data['close'].rolling(window=200).mean()
                
                # Determine trends
                latest = self.market_index_data.iloc[-1]
                
                # Short-term trend (5-day vs 20-day)
                short_term_ma = self.market_index_data['close'].tail(5).mean()
                if short_term_ma > latest['sma20'] * 1.01:
                    self.market_trends['short_term'] = 'Up'
                elif short_term_ma < latest['sma20'] * 0.99:
                    self.market_trends['short_term'] = 'Down'
                else:
                    self.market_trends['short_term'] = 'Sideways'
                
                # Medium-term trend (20-day vs 50-day)
                if latest['sma20'] > latest['sma50'] * 1.01:
                    self.market_trends['medium_term'] = 'Up'
                elif latest['sma20'] < latest['sma50'] * 0.99:
                    self.market_trends['medium_term'] = 'Down'
                else:
                    self.market_trends['medium_term'] = 'Sideways'
                
                # Long-term trend (50-day vs 200-day)
                if latest['sma50'] > latest['sma200'] * 1.01:
                    self.market_trends['long_term'] = 'Up'
                elif latest['sma50'] < latest['sma200'] * 0.99:
                    self.market_trends['long_term'] = 'Down'
                else:
                    self.market_trends['long_term'] = 'Sideways'
            
            # Update volatility state
            try:
                # This requires access to market data for VIX
                vix_bars = self.api.get_bars("UVXY", TimeFrame.Day, limit=20).df
                if not vix_bars.empty:
                    self.vix_data = vix_bars
                    avg_vix = vix_bars['close'].mean()
                    
                    if avg_vix < 15:
                        self.volatility_state = 'Low'
                    elif avg_vix > 25:
                        self.volatility_state = 'High'
                    else:
                        self.volatility_state = 'Medium'
            except:
                # VIX data not available, use standard deviation of returns instead
                returns = self.market_index_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                if volatility < 0.10:  # 10%
                    self.volatility_state = 'Low'
                elif volatility > 0.20:  # 20%
                    self.volatility_state = 'High'
                else:
                    self.volatility_state = 'Medium'
            
            self.last_update = now
            logger.info(f"Market health updated: ST: {self.market_trends['short_term']}, MT: {self.market_trends['medium_term']}, LT: {self.market_trends['long_term']}, Vol: {self.volatility_state}")
            
        except Exception as e:
            logger.error(f"Error updating market health: {str(e)}")
    
    def adjust_position_size(self, base_size: float) -> float:
        """Adjust position size based on market conditions"""
        # Default multiplier is 1.0
        multiplier = 1.0
        
        # Adjust based on market trend alignment
        trend_count = sum(1 for trend in self.market_trends.values() if trend == 'Up')
        
        if trend_count == 3:  # All trends up
            multiplier *= 1.2
        elif trend_count == 0:  # All trends down
            multiplier *= 0.5
        elif self.market_trends['medium_term'] == 'Down':
            multiplier *= 0.8
            
        # Adjust based on volatility
        if self.volatility_state == 'High':
            multiplier *= 0.7
        elif self.volatility_state == 'Low':
            multiplier *= 1.1
            
        return base_size * multiplier
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get a summary of current market state"""
        self.update()  # Ensure data is current
        
        return {
            'market_trends': self.market_trends,
            'volatility': self.volatility_state,
            'timestamp': self.last_update
        }
    
class AlertManager:
    """Manages system alerts and notifications"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_levels = {
            'INFO': 0,
            'WARNING': 1,
            'ERROR': 2,
            'CRITICAL': 3
        }
        self.min_alert_level = self.alert_levels.get(
            config.get('min_alert_level', 'WARNING'), 1)
        
        # Configure notification channels
        self.channels = {
            'email': config.get('email_alerts', False),
            'sms': config.get('sms_alerts', False),
            'webhook': config.get('webhook_alerts', False)
        }
        
        # Track active alerts
        self.active_alerts = []
        
    def send_alert(self, level: str, message: str, data: Dict[str, Any] = None) -> None:
        """Send an alert through configured channels"""
        # Check alert level threshold
        if self.alert_levels.get(level, 0) < self.min_alert_level:
            return
            
        # Create alert object
        alert = {
            'level': level,
            'message': message,
            'data': data or {},
            'timestamp': datetime.now()
        }
        
        # Log the alert
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(f"ALERT: {message}")
        
        # Add to active alerts
        self.active_alerts.append(alert)
        
        # Send through configured channels
        if self.channels['email'] and level in ['ERROR', 'CRITICAL']:
            self._send_email_alert(alert)
            
        if self.channels['sms'] and level == 'CRITICAL':
            self._send_sms_alert(alert)
            
        if self.channels['webhook']:
            self._send_webhook_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via email"""
        if 'email_config' not in self.config:
            return
            
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            email_config = self.config['email_config']
            
            # Create message
            subject = f"Trading Alert [{alert['level']}]: {alert['message'][:50]}..."
            body = f"""
            Trading System Alert
            ---------------------
            Level: {alert['level']}
            Time: {alert['timestamp']}
            Message: {alert['message']}
            
            Additional Data:
            {json.dumps(alert['data'], indent=2)}
            """
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = email_config['from_address']
            msg['To'] = email_config['to_address']
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                if email_config.get('use_tls', False):
                    server.starttls()
                if 'username' in email_config and 'password' in email_config:
                    server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
                
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def _send_sms_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via SMS"""
        if 'sms_config' not in self.config:
            return
            
        try:
            # Using Twilio as an example SMS provider
            from twilio.rest import Client
            
            sms_config = self.config['sms_config']
            
            # Create Twilio client
            client = Client(sms_config['account_sid'], sms_config['auth_token'])
            
            # Craft message (limited to essential info due to SMS constraints)
            message_text = f"ALERT [{alert['level']}]: {alert['message']}"
            
            # Send SMS
            message = client.messages.create(
                body=message_text,
                from_=sms_config['from_number'],
                to=sms_config['to_number']
            )
            
            logger.info(f"SMS alert sent: {message.sid}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {str(e)}")

    def _send_webhook_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert via webhook"""
        if 'webhook_config' not in self.config:
            return
            
        try:
            import requests
            
            webhook_config = self.config['webhook_config']
            
            # Prepare payload
            payload = {
                'level': alert['level'],
                'message': alert['message'],
                'timestamp': alert['timestamp'].isoformat(),
                'data': alert['data']
            }
            
            # Send webhook request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers
            )
            
            # Check response
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Webhook alert sent successfully")
            else:
                logger.error(f"Webhook alert failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")

class SystemStateManager:
    """Manages saving and loading system state for continuity across restarts"""
    def __init__(self, state_file: str = "system_state.json"):
        self.state_file = state_file
        self.state = {
            'positions': {},
            'strategies': {},
            'performance': {
                'equity_points': [],
                'trades': []
            },
            'last_update': None
        }
        
        # Try to load existing state
        self.load_state()
        
    def save_state(self, positions: Dict[str, Any], strategy_states: Dict[str, Any], 
                 performance_data: Dict[str, Any]) -> None:
        """Save current system state to file"""
        try:
            # Update state with current data
            self.state['positions'] = positions
            self.state['strategies'] = strategy_states
            self.state['performance'] = performance_data
            self.state['last_update'] = datetime.now().isoformat()
            
            # Save to file
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
                
            logger.info(f"System state saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
    
    def load_state(self) -> Dict[str, Any]:
        """Load system state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                    
                logger.info(f"System state loaded from {self.state_file}")
                
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
            
        return self.state
    
    def get_position_state(self) -> Dict[str, Any]:
        """Get saved position data"""
        return self.state.get('positions', {})
    
    def get_strategy_state(self) -> Dict[str, Any]:
        """Get saved strategy states"""
        return self.state.get('strategies', {})
    
    def get_performance_data(self) -> Dict[str, Any]:
        """Get saved performance data"""
        return self.state.get('performance', {})
    
    def clear_state(self) -> None:
        """Clear stored state"""
        self.state = {
            'positions': {},
            'strategies': {},
            'performance': {
                'equity_points': [],
                'trades': []
            },
            'last_update': datetime.now().isoformat()
        }
        
        # Remove state file if it exists
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            logger.info(f"System state file {self.state_file} removed")

class ScheduledTaskManager:
    """Manages scheduled and recurring tasks for the trading system"""
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.thread = None
        
    def add_task(self, name: str, func: Callable, interval: int, 
                args: Tuple = None, kwargs: Dict[str, Any] = None) -> None:
        """
        Add a scheduled task
        
        Args:
            name: Task name
            func: Function to call
            interval: Interval in seconds
            args: Function arguments
            kwargs: Function keyword arguments
        """
        self.tasks[name] = {
            'func': func,
            'interval': interval,
            'last_run': 0,  # Start immediately
            'args': args or (),
            'kwargs': kwargs or {}
        }
        logger.info(f"Scheduled task added: {name}, interval: {interval}s")
    
    def remove_task(self, name: str) -> bool:
        """Remove a scheduled task"""
        if name in self.tasks:
            del self.tasks[name]
            logger.info(f"Scheduled task removed: {name}")
            return True
        return False
    
    def start(self) -> None:
        """Start the task scheduler"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Task scheduler started")
    
    def stop(self) -> None:
        """Stop the task scheduler"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Task scheduler stopped")
    
    def _run_scheduler(self) -> None:
        """Main scheduler loop"""
        while self.running:
            now = time.time()
            
            for name, task in self.tasks.items():
                # Check if task should run
                if now - task['last_run'] >= task['interval']:
                    try:
                        task['func'](*task['args'], **task['kwargs'])
                        task['last_run'] = now
                        
                    except Exception as e:
                        logger.error(f"Error executing scheduled task {name}: {str(e)}")
            
            # Sleep for a bit to avoid high CPU usage
            time.sleep(1)

class WebPortal:
    """Simple web interface for monitoring and controlling the trading system"""
    def __init__(self, trading_system, host: str = 'localhost', port: int = 8080):
        self.trading_system = trading_system
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        
        # Web framework imports
        from flask import Flask, jsonify, render_template, request
        self.app = Flask(__name__)
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register web routes"""
        @self.app.route('/')
        def home():
            """Dashboard home page"""
            # In a real implementation, you'd load a proper HTML template
            return """
            <html>
                <head>
                    <title>Trading System Dashboard</title>
                    <meta http-equiv="refresh" content="30">
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #333; }
                        .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
                        .btn { padding: 5px 10px; margin-right: 10px; cursor: pointer; }
                    </style>
                </head>
                <body>
                    <h1>Trading System Dashboard</h1>
                    <div class="card">
                        <h2>System Status</h2>
                        <p id="status">Loading...</p>
                        <button class="btn" onclick="fetch('/api/system/start', {method: 'POST'}).then(() => location.reload())">Start</button>
                        <button class="btn" onclick="fetch('/api/system/stop', {method: 'POST'}).then(() => location.reload())">Stop</button>
                    </div>
                    <div class="card">
                        <h2>Positions</h2>
                        <div id="positions">Loading...</div>
                    </div>
                    <div class="card">
                        <h2>Performance</h2>
                        <div id="performance">Loading...</div>
                    </div>
                    
                    <script>
                        // Simple JavaScript to load data
                        fetch('/api/system/status').then(r => r.json()).then(data => {
                            document.getElementById('status').innerHTML = 
                                `<strong>Status:</strong> ${data.running ? 'Running' : 'Stopped'}<br>` +
                                `<strong>Uptime:</strong> ${data.uptime}<br>`;
                        });
                        
                        fetch('/api/positions').then(r => r.json()).then(data => {
                            let html = '<table border="1" cellpadding="5">' +
                                '<tr><th>Symbol</th><th>Quantity</th><th>Entry Price</th><th>Current Price</th><th>P&L</th></tr>';
                            
                            for (const pos of data) {
                                html += `<tr>
                                    <td>${pos.symbol}</td>
                                    <td>${pos.qty}</td>
                                    <td>$${pos.avg_entry_price}</td>
                                    <td>$${pos.current_price}</td>
                                    <td>$${pos.unrealized_pl} (${pos.unrealized_plpc}%)</td>
                                </tr>`;
                            }
                            
                            html += '</table>';
                            document.getElementById('positions').innerHTML = html;
                        });
                        
                        fetch('/api/performance').then(r => r.json()).then(data => {
                            document.getElementById('performance').innerHTML = 
                                `<strong>Total Return:</strong> ${data.total_return}%<br>` +
                                `<strong>Sharpe Ratio:</strong> ${data.sharpe_ratio}<br>` +
                                `<strong>Win Rate:</strong> ${data.win_rate}%<br>` +
                                `<strong>Max Drawdown:</strong> ${data.max_drawdown}%<br>`;
                        });
                    </script>
                </body>
            </html>
            """
        
        @self.app.route('/api/system/status')
        def system_status():
            """Get system status"""
            uptime = "N/A"
            if hasattr(self.trading_system, 'start_time') and self.trading_system.start_time:
                uptime = str(datetime.now() - self.trading_system.start_time)
                
            return jsonify({
                'running': self.trading_system.running,
                'uptime': uptime,
                'backtest_mode': self.trading_system.config.backtest_mode
            })
        
        @self.app.route('/api/system/start', methods=['POST'])
        def start_system():
            """Start the trading system"""
            if not self.trading_system.running:
                self.trading_system.start()
            return jsonify({'status': 'ok', 'message': 'System started'})
        
        @self.app.route('/api/system/stop', methods=['POST'])
        def stop_system():
            """Stop the trading system"""
            if self.trading_system.running:
                self.trading_system.stop()
            return jsonify({'status': 'ok', 'message': 'System stopped'})
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            try:
                positions = self.trading_system.api.list_positions()
                return jsonify([{
                    'symbol': p.symbol,
                    'qty': p.qty,
                    'avg_entry_price': p.avg_entry_price,
                    'current_price': p.current_price,
                    'unrealized_pl': p.unrealized_pl,
                    'unrealized_plpc': p.unrealized_plpc
                } for p in positions])
            except:
                return jsonify([])
        
        @self.app.route('/api/performance')
        def get_performance():
            """Get performance metrics"""
            metrics = self.trading_system.performance_tracker.calculate_metrics()
            
            # Format for display
            return jsonify({
                'total_return': f"{metrics.get('total_return', 0) * 100:.2f}",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'win_rate': f"{metrics.get('win_rate', 0) * 100:.2f}",
                'max_drawdown': f"{metrics.get('max_drawdown', 0) * 100:.2f}",
                'total_trades': metrics.get('total_trades', 0)
            })
    
    def start(self):
        """Start the web server in a background thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Web portal is already running")
            return
        
        # Start in a separate thread
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Web portal started at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the web server"""
        # Shutdown Flask server (if running with Werkzeug)
        if self.server:
            self.server.shutdown()
        
        if self.thread:
            self.thread.join(timeout=5)
            
        logger.info("Web portal stopped")
    
    def _run_server(self):
        """Run the Flask web server"""
        try:
            # Run with minimum output (no debug mode)
            import logging as flask_logging
            flask_log = flask_logging.getLogger('werkzeug')
            flask_log.setLevel(flask_logging.ERROR)
            
            self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)
        except Exception as e:
            logger.error(f"Error running web portal: {str(e)}")

