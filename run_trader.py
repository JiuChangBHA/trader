import os
import sys
import json
import logging
import argparse
from pathlib import Path
import alpaca_trade_api as tradeapi
from trader import AlpacaTradingSystem
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Run the trading system')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
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
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup logging
    log_level = args.log_level or config.get('log_level', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger('my_trading_system')
    logger.info('Starting My Trading System')

    trader = AlpacaTradingSystem()
    trader.start()

    api = tradeapi.REST(
        config['alpaca_api_key'],
        config['alpaca_api_secret'],
        base_url=config['alpaca_base_url']
    )

    # Keep the main thread running
    try:
        # If in backtest mode, run and exit
        if args.backtest:
            trader.run_backtest()
            trader.stop()
            return
            
        # Otherwise, keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        trader.stop()
        logger.info("Trading system stopped")

if __name__ == "__main__":
    main()





