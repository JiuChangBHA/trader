# My Trading System

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
   ./run_my_trading_system.py
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
