#!/usr/bin/env python3
"""
Alpaca Trading System - Main Entry Point
=======================================

This module serves as the main entry point for the Alpaca Trading System.
It provides CLI options for starting the system in various modes and with different configurations.
"""

import os
import sys
import argparse
import logging
import json
import signal
from datetime import datetime, timedelta

# Import the trading system components
from alpaca_trading_system import (
    AlpacaTradingSystem,
    Config,
    ScheduledTaskManager,
    AlertManager,
    SystemStateManager,
    MarketHealthMonitor,
    WebPortal
)

# Configure argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Alpaca Trading System')
    
    # General options
    parser.add_argument('-c', '--config', dest='config_path', default='config.json',
                        help='Path to configuration file (default: config.json)')
    parser.add_argument('-m', '--mode', dest='mode', choices=['live', 'paper', 'backtest'],
                        default='paper', help='Trading mode (default: paper)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Enable verbose logging')
    
    # Backtest options
    backtest_group = parser.add_argument_group('backtest options')
    backtest_group.add_argument('--start-date', dest='start_date',
                          help='Backtest start date (YYYY-MM-DD)')
    backtest_group.add_argument('--end-date', dest='end_date',
                          help='Backtest end date (YYYY-MM-DD)')
    backtest_group.add_argument('--output', dest='output_file',
                          help='Output file for backtest results')
    
    # Web interface options
    web_group = parser.add_argument_group('web interface options')
    web_group.add_argument('--web', dest='web_enabled', action='store_true',
                      help='Enable web interface')
    web_group.add_argument('--host', dest='web_host', default='localhost',
                      help='Web interface host (default: localhost)')
    web_group.add_argument('--port', dest='web_port', type=int, default=8080,
                      help='Web interface port (default: 8080)')
    
    return parser.parse_args()

# Validate configuration
def validate_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required fields
        required_fields = ['alpaca_api_key', 'alpaca_api_secret', 'symbols']
        for field in required_fields:
            if field not in config and not os.environ.get(f'ALPACA_{field.upper()}'):
                logging.error(f"Configuration missing required field: {field}")
                return False
        
        # Validate strategies
        if 'strategies' not in config:
            logging.error("No strategies defined in configuration")
            return False
        
        for symbol, strategies in config['strategies'].items():
            if not strategies:
                logging.warning(f"No strategies defined for symbol: {symbol}")
        
        return True
    except Exception as e:
        logging.error(f"Error validating configuration: {str(e)}")
        return False

# Handle signals for clean shutdown
def setup_signal_handlers(trading_system):
    def signal_handler(sig, frame):
        logging.info(f"Received signal {sig}, shutting down...")
        trading_system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# Schedule routine tasks
def setup_scheduled_tasks(task_manager, trading_system, market_monitor, system_state_mgr):
    # Update account information every 15 minutes
    task_manager.add_task(
        "update_account",
        trading_system.risk_manager.update_account_info,
        interval=900
    )
    
    # Check market health every 30 minutes
    task_manager.add_task(
        "market_health",
        market_monitor.update,
        interval=1800
    )
    
    # Save system state every hour
    def save_state_task():
        positions = {pos.symbol: {
            'qty': pos.qty,
            'avg_price': pos.avg_entry_price
        } for pos in trading_system.api.list_positions()}
        
        strategy_states = {}
        for symbol, strategies in trading_system.strategy_manager.strategies.items():
            strategy_states[symbol] = [{
                'name': strategy.name,
                'position': strategy.position
            } for strategy in strategies]
        
        performance_data = trading_system.performance_tracker.equity_curve
        
        system_state_mgr.save_state(positions, strategy_states, performance_data)
    
    task_manager.add_task(
        "save_state",
        save_state_task,
        interval=3600
    )
    
    # Generate performance report daily at midnight
    def generate_daily_report():
        current_time = datetime.now()
        if current_time.hour == 0 and current_time.minute < 5:
            report = trading_system.performance_tracker.generate_report()
            logging.info(f"Daily Performance Report:\n{report}")
    
    task_manager.add_task(
        "daily_report",
        generate_daily_report,
        interval=300  # Check every 5 minutes
    )

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_system.log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting Alpaca Trading System")
    
    # Validate configuration
    if not validate_config(args.config_path):
        logging.error("Invalid configuration. Exiting.")
        sys.exit(1)
    
    # Create trading system
    config = Config(args.config_path)
    
    # Update config based on command line arguments
    if args.mode == 'live':
        config.base_url = "https://api.alpaca.markets"
    elif args.mode == 'paper':
        config.base_url = "https://paper-api.alpaca.markets"
    elif args.mode == 'backtest':
        config.backtest_mode = True
    
    trading_system = AlpacaTradingSystem(args.config_path)
    
    # Set up signal handlers for clean shutdown
    setup_signal_handlers(trading_system)
    
    # Create additional system components
    market_monitor = MarketHealthMonitor(trading_system.api)
    system_state_mgr = SystemStateManager()
    task_manager = ScheduledTaskManager()
    
    # Configure alert manager from config
    alert_config = config.config.get('alerts', {})
    alert_manager = AlertManager(alert_config)
    
    # Set up scheduled tasks
    setup_scheduled_tasks(task_manager, trading_system, market_monitor, system_state_mgr)
    
    # Start web portal if enabled
    web_portal = None
    if args.web_enabled:
        web_portal = WebPortal(trading_system, host=args.web_host, port=args.web_port)
        web_portal.start()
    
    # Start task manager
    task_manager.start()
    
    try:
        if args.mode == 'backtest':
            # Run in backtest mode
            if not args.start_date or not args.end_date:
                logging.error("Start date and end date are required for backtest mode")
                sys.exit(1)
            
            report = trading_system.run_backtest(args.start_date, args.end_date)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    f.write(report)
                logging.info(f"Backtest results saved to {args.output_file}")
            
            # Exit after backtest
            sys.exit(0)
        else:
            # Start trading system in live or paper mode
            trading_system.start()
            
            # Keep the main thread alive
            logging.info(f"Trading system running in {args.mode} mode. Press Ctrl+C to exit.")
            while trading_system.running:
                import time
                time.sleep(1)
    
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
    finally:
        # Clean shutdown
        logging.info("Shutting down trading system...")
        
        if trading_system.running:
            trading_system.stop()
        
        task_manager.stop()
        
        if web_portal:
            web_portal.stop()
        
        logging.info("Trading system shutdown complete.")

if __name__ == "__main__":
    main()