#!/usr/bin/env python3
"""
My Trading System
=======================================
Main entry point for the trading system.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='My Trading System')
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
            logging.FileHandler(os.path.join(log_dir, 'my_trading_system.log')),
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
    
    logger = logging.getLogger('my_trading_system')
    logger.info('Starting My Trading System')
    
    # TODO: Initialize system components
    from my_trading_system.data_engine import DataEngine
    from my_trading_system.strategy_engine import StrategyEngine
    from my_trading_system.risk_manager import RiskManager
    from my_trading_system.execution_engine import ExecutionEngine
    from my_trading_system.web_portal import WebPortal
    
    # Create Alpaca API connection
    api = tradeapi.REST(
        config['alpaca_api_key'],
        config['alpaca_api_secret'],
        base_url=config['alpaca_base_url']
    )
    
    # Create and start the trading system
    trading_system = TradingSystem(api, config, backtest_mode=args.backtest)
    trading_system.start()
    
    logger.info('My Trading System started')
    
    # Keep the main thread running
    try:
        # If in backtest mode, run and exit
        if args.backtest:
            trading_system.run_backtest()
            trading_system.stop()
            return
            
        # Otherwise, keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        trading_system.stop()
        logger.info("Trading system stopped")
        
if __name__ == '__main__':
    main()
