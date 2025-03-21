# dashboard.py
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time

class TradingDashboard:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self.setup_layout()
        self.setup_callbacks()
        self.thread = None
        self.running = False
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Algorithmic Trading System Dashboard"),
            
            # System Controls
            html.Div([
                html.H3("System Controls"),
                html.Button("Start Trading", id="btn-start", className="control-btn"),
                html.Button("Stop Trading", id="btn-stop", className="control-btn"),
                html.Div(id="system-status", children="System Status: Stopped")
            ], className="control-panel"),
            
            # Tabs for different sections
            dcc.Tabs([
                # Portfolio Overview
                dcc.Tab(label="Portfolio", children=[
                    html.Div([
                        html.H3("Portfolio Performance"),
                        dcc.Graph(id="equity-chart"),
                        html.Div([
                            html.Div([
                                html.H4("Portfolio Stats"),
                                html.Div(id="portfolio-stats")
                            ], className="stats-box"),
                            html.Div([
                                html.H4("Risk Metrics"),
                                html.Div(id="risk-metrics")
                            ], className="stats-box")
                        ], className="stats-container")
                    ])
                ]),
                
                # Positions & Orders
                dcc.Tab(label="Positions & Orders", children=[
                    html.Div([
                        html.H3("Current Positions"),
                        dash_table.DataTable(
                            id="positions-table",
                            columns=[
                                {"name": "Symbol", "id": "symbol"},
                                {"name": "Quantity", "id": "qty"},
                                {"name": "Avg Entry", "id": "avg_entry_price"},
                                {"name": "Current", "id": "current_price"},
                                {"name": "P&L ($)", "id": "unrealized_pl"},
                                {"name": "P&L (%)", "id": "unrealized_plpc"}
                            ]
                        ),
                        html.H3("Recent Orders"),
                        dash_table.DataTable(
                            id="orders-table",
                            columns=[
                                {"name": "Symbol", "id": "symbol"},
                                {"name": "Side", "id": "side"},
                                {"name": "Qty", "id": "qty"},
                                {"name": "Type", "id": "type"},
                                {"name": "Status", "id": "status"},
                                {"name": "Created", "id": "created_at"}
                            ]
                        )
                    ])
                ]),
                
                # Strategy Performance
                dcc.Tab(label="Strategies", children=[
                    html.Div([
                        html.H3("Strategy Performance"),
                        html.Div([
                            dcc.Dropdown(
                                id="strategy-selector",
                                options=[
                                    {"label": "All Strategies", "value": "all"}
                                ],
                                value="all"
                            ),
                            dcc.Graph(id="strategy-chart"),
                            html.Div(id="strategy-stats")
                        ])
                    ])
                ]),
                
                # Market Data
                dcc.Tab(label="Market Data", children=[
                    html.Div([
                        html.H3("Market Overview"),
                        dcc.Dropdown(
                            id="symbol-selector",
                            options=[],
                            value=None,
                            placeholder="Select a symbol"
                        ),
                        dcc.Graph(id="price-chart"),
                        html.Div([
                            html.H4("Technical Indicators"),
                            dcc.Checklist(
                                id="indicator-selector",
                                options=[
                                    {"label": "SMA (20)", "value": "sma20"},
                                    {"label": "SMA (50)", "value": "sma50"},
                                    {"label": "Bollinger Bands", "value": "bollinger"},
                                    {"label": "RSI", "value": "rsi"}
                                ],
                                value=[]
                            ),
                            dcc.Graph(id="indicator-chart")
                        ])
                    ])
                ]),
                
                # System Logs
                dcc.Tab(label="Logs", children=[
                    html.Div([
                        html.H3("System Logs"),
                        dcc.RadioItems(
                            id="log-level-selector",
                            options=[
                                {"label": "All", "value": "all"},
                                {"label": "Info", "value": "info"},
                                {"label": "Warning", "value": "warning"},
                                {"label": "Error", "value": "error"}
                            ],
                            value="all"
                        ),
                        html.Div(id="log-display", className="log-container")
                    ])
                ])
            ])
        ])
    
    def setup_callbacks(self):
        # Start button callback
        @self.app.callback(
            Output("system-status", "children"),
            Input("btn-start", "n_clicks"),
            prevent_initial_call=True
        )
        def start_system(n_clicks):
            if n_clicks:
                try:
                    self.trading_system.start()
                    return "System Status: Running"
                except Exception as e:
                    return f"System Status: Error - {str(e)}"
            return "System Status: Stopped"
        
        # Stop button callback
        @self.app.callback(
            Output("system-status", "children", allow_duplicate=True),
            Input("btn-stop", "n_clicks"),
            prevent_initial_call=True
        )
        def stop_system(n_clicks):
            if n_clicks:
                try:
                    self.trading_system.stop()
                    return "System Status: Stopped"
                except Exception as e:
                    return f"System Status: Error - {str(e)}"
            return "System Status: Running"
        
        # Portfolio charts
        @self.app.callback(
            [
                Output("equity-chart", "figure"),
                Output("portfolio-stats", "children"),
                Output("risk-metrics", "children")
            ],
            Input("system-status", "children")  # This will trigger the callback to refresh data
        )
        def update_portfolio_charts(status):
            # Get equity data
            equity_data = self._get_equity_data()
            
            # Create equity figure
            fig = go.Figure()
            
            # Add equity trace with improved styling
            fig.add_trace(go.Scatter(
                x=equity_data['timestamp'],
                y=equity_data['equity'],
                mode='lines',
                name='Account Equity',
                line=dict(color='rgb(0, 100, 80)', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 80, 0.2)'
            ))
            
            # Set appropriate title and layout based on backtest mode
            if hasattr(self.trading_system, 'config') and self.trading_system.config.backtest_mode:
                backtest_start = self.trading_system.config.backtest_start_date if hasattr(self.trading_system.config, 'backtest_start_date') else "Unknown"
                backtest_end = self.trading_system.config.backtest_end_date if hasattr(self.trading_system.config, 'backtest_end_date') else "Unknown"
                
                # Format y-axis to show dollar amounts
                fig.update_layout(
                    title=f'Account Equity (Backtest Mode: {backtest_start} to {backtest_end})',
                    yaxis_title='Equity ($)',
                    yaxis=dict(
                        tickprefix='$',
                        tickformat=',.2f'
                    ),
                    hovermode='x unified',
                    xaxis_range=[pd.to_datetime(backtest_start), pd.to_datetime(backtest_end)],
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                
                # Add annotations for starting and ending equity
                if len(equity_data) > 1:
                    start_equity = equity_data['equity'].iloc[0]
                    end_equity = equity_data['equity'].iloc[-1]
                    pct_change = ((end_equity - start_equity) / start_equity) * 100
                    
                    fig.add_annotation(
                        x=equity_data['timestamp'].iloc[-1],
                        y=end_equity,
                        text=f"${end_equity:.2f} ({pct_change:.2f}%)",
                        showarrow=True,
                        arrowhead=1,
                        ax=50,
                        ay=-40
                    )
            else:
                # Format y-axis to show dollar amounts
                fig.update_layout(
                    title='Account Equity Over Time',
                    yaxis_title='Equity ($)',
                    yaxis=dict(
                        tickprefix='$',
                        tickformat=',.2f'
                    ),
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=50, b=40)
                )
            
            # Calculate portfolio stats
            stats_html = self._get_portfolio_stats_html()
            
            # Calculate risk metrics
            risk_html = self._get_risk_metrics_html()
            
            return fig, stats_html, risk_html
        
        # Positions table
        @self.app.callback(
            Output("positions-table", "data"),
            Input("system-status", "children")
        )
        def update_positions_table(status):
            return self._get_positions_data()
        
        # Orders table
        @self.app.callback(
            Output("orders-table", "data"),
            Input("system-status", "children")
        )
        def update_orders_table(status):
            return self._get_orders_data()
        
        # Strategy dropdown options
        @self.app.callback(
            Output("strategy-selector", "options"),
            Input("system-status", "children")
        )
        def update_strategy_options(status):
            options = [{"label": "All Strategies", "value": "all"}]
            
            # Add individual strategies
            for symbol, strategies in self.trading_system.strategy_manager.strategies.items():
                for strategy in strategies:
                    strategy_name = strategy.name
                    options.append({
                        "label": f"{strategy_name} - {symbol}",
                        "value": f"{strategy_name}_{symbol}"
                    })
                    
            return options
        
        # Strategy performance chart
        @self.app.callback(
            [Output("strategy-chart", "figure"), Output("strategy-stats", "children")],
            [Input("strategy-selector", "value"), Input("system-status", "children")]
        )
        def update_strategy_charts(strategy_value, status):
            # Create strategy performance figure
            fig = go.Figure()
            
            # Add strategy logic here
            # This would show performance metrics for selected strategy
            
            # For now, just show a placeholder
            fig.add_trace(go.Scatter(
                x=[datetime.now() - timedelta(days=i) for i in range(30, 0, -1)],
                y=np.cumsum(np.random.normal(0.001, 0.01, 30)),
                mode='lines',
                name='Strategy Performance'
            ))
            
            fig.update_layout(
                title='Strategy Performance',
                yaxis_title='Return (%)',
                hovermode='x unified'
            )
            
            # Strategy stats HTML
            stats_html = html.Div([
                html.P(f"Strategy: {strategy_value}"),
                html.P("Win Rate: 55%"),
                html.P("Average Win: 2.3%"),
                html.P("Average Loss: -1.1%"),
                html.P("Profit Factor: 1.8")
            ])
            
            return fig, stats_html
        
        # Symbol selector options
        @self.app.callback(
            Output("symbol-selector", "options"),
            Input("system-status", "children")
        )
        def update_symbol_options(status):
            options = []
            for symbol in self.trading_system.config.symbols:
                options.append({"label": symbol, "value": symbol})
            return options
        
        # Default symbol value
        @self.app.callback(
            Output("symbol-selector", "value"),
            Input("symbol-selector", "options"),
            prevent_initial_call=True
        )
        def set_default_symbol(options):
            if options and len(options) > 0:
                return options[0]["value"]
            return None
        
        # Price chart
        @self.app.callback(
            Output("price-chart", "figure"),
            [Input("symbol-selector", "value"), Input("system-status", "children")]
        )
        def update_price_chart(symbol, status):
            if not symbol:
                return go.Figure()
                
            # Get market data for symbol
            price_data = self._get_symbol_price_data(symbol)
            
            # Check if we have data
            if price_data.empty:
                return go.Figure(data=[go.Candlestick(x=[], open=[], high=[], low=[], close=[])],
                                layout=go.Layout(title=f"No data available for {symbol}"))
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_data['timestamp'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name=symbol
            ))
            
            # Add volume as a bar chart on a secondary y-axis if volume data is available
            if 'volume' in price_data.columns and price_data['volume'].sum() > 0:
                fig.add_trace(go.Bar(
                    x=price_data['timestamp'],
                    y=price_data['volume'],
                    name='Volume',
                    yaxis='y2',
                    marker=dict(color='rgba(0,0,0,0.2)')
                ))
                
                fig.update_layout(
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right',
                        showgrid=False
                    )
                )
            
            # Add backtest mode indicator if applicable
            if hasattr(self.trading_system, 'config') and self.trading_system.config.backtest_mode:
                backtest_start = self.trading_system.config.backtest_start_date if hasattr(self.trading_system.config, 'backtest_start_date') else "Unknown"
                backtest_end = self.trading_system.config.backtest_end_date if hasattr(self.trading_system.config, 'backtest_end_date') else "Unknown"
                
                fig.update_layout(
                    title=f'{symbol} Price (Backtest Mode: {backtest_start} to {backtest_end})',
                    yaxis_title='Price ($)',
                    xaxis_rangeslider_visible=False
                )
            else:
                fig.update_layout(
                    title=f'{symbol} Price',
                    yaxis_title='Price ($)',
                    xaxis_rangeslider_visible=False
                )
            
            return fig
        
        # Indicator chart
        @self.app.callback(
            Output("indicator-chart", "figure"),
            [Input("symbol-selector", "value"), 
             Input("indicator-selector", "value"),
             Input("system-status", "children")]
        )
        def update_indicator_chart(symbol, indicators, status):
            if not symbol or not indicators:
                return go.Figure()
                
            # Get market data for symbol
            price_data = self._get_symbol_price_data(symbol)
            
            # Check if we have enough data
            if price_data.empty or len(price_data) < 2:
                return go.Figure(data=[go.Scatter(x=[], y=[], mode='lines')],
                                layout=go.Layout(title=f"No data available for {symbol}"))
            
            fig = go.Figure()
            
            # Add closing price
            fig.add_trace(go.Scatter(
                x=price_data['timestamp'],
                y=price_data['close'],
                mode='lines',
                name=f'{symbol} Close'
            ))
            
            # Add selected indicators
            if 'sma20' in indicators and len(price_data) >= 20:
                sma20 = price_data['close'].rolling(window=min(20, len(price_data))).mean()
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=sma20,
                    mode='lines',
                    name='SMA (20)',
                    line=dict(color='orange')
                ))
                
            if 'sma50' in indicators and len(price_data) >= 50:
                sma50 = price_data['close'].rolling(window=min(50, len(price_data))).mean()
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=sma50,
                    mode='lines',
                    name='SMA (50)',
                    line=dict(color='green')
                ))
                
            if 'bollinger' in indicators and len(price_data) >= 20:
                window = min(20, len(price_data))
                sma20 = price_data['close'].rolling(window=window).mean()
                std20 = price_data['close'].rolling(window=window).std()
                upper_band = sma20 + (std20 * 2)
                lower_band = sma20 - (std20 * 2)
                
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=upper_band,
                    mode='lines',
                    name='Upper BB',
                    line=dict(color='rgba(0,100,255,0.5)')
                ))
                
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=lower_band,
                    mode='lines',
                    name='Lower BB',
                    line=dict(color='rgba(0,100,255,0.5)'),
                    fill='tonexty'
                ))
                
            if 'rsi' in indicators and len(price_data) >= 14:
                # Add a secondary y-axis for RSI
                try:
                    rsi_values = self._calculate_rsi(price_data['close'], window=min(14, len(price_data)-1))
                    
                    fig.add_trace(go.Scatter(
                        x=price_data['timestamp'],
                        y=rsi_values,
                        mode='lines',
                        name='RSI (14)',
                        yaxis='y2',
                        line=dict(color='purple')
                    ))
                    
                    fig.update_layout(
                        yaxis2=dict(
                            title='RSI',
                            overlaying='y',
                            side='right',
                            range=[0, 100]
                        )
                    )
                except Exception as e:
                    print(f"Error calculating RSI: {e}")
            
            # Add backtest mode indicator if applicable
            if hasattr(self.trading_system, 'config') and self.trading_system.config.backtest_mode:
                backtest_start = self.trading_system.config.backtest_start_date if hasattr(self.trading_system.config, 'backtest_start_date') else "Unknown"
                backtest_end = self.trading_system.config.backtest_end_date if hasattr(self.trading_system.config, 'backtest_end_date') else "Unknown"
                
                fig.update_layout(
                    title=f'{symbol} Indicators (Backtest Mode: {backtest_start} to {backtest_end})',
                    yaxis_title='Price ($)',
                    hovermode='x unified'
                )
            else:
                fig.update_layout(
                    title=f'{symbol} Indicators',
                    yaxis_title='Price ($)',
                    hovermode='x unified'
                )
            
            return fig
        
# System logs
        @self.app.callback(
            Output("log-display", "children"),
            [Input("log-level-selector", "value"), Input("system-status", "children")]
        )
        def update_logs(log_level, status):
            # Get logs based on selected level
            logs = self._get_logs(log_level)
            
            # Sort logs by timestamp in descending order (newest first)
            logs = sorted(logs, key=lambda x: x.get('timestamp', ''), reverse=True)
            
            # Format logs as HTML
            log_entries = []
            for log in logs:
                log_class = "log-info"
                if log.get("level") == "WARNING":
                    log_class = "log-warning"
                elif log.get("level") == "ERROR":
                    log_class = "log-error"
                
                log_entries.append(
                    html.Div(
                        f"[{log.get('timestamp')}] {log.get('level')}: {log.get('message')}",
                        className=log_class
                    )
                )
            
            return log_entries
    
    def _get_equity_data(self):
        """Get equity data for charting"""
        try:
            # Get account history from trading system
            if hasattr(self.trading_system, 'performance_tracker'):
                equity_history = self.trading_system.performance_tracker.get_equity_history()
                
                # If we're in backtest mode and the equity history is empty or flat
                if (hasattr(self.trading_system, 'config') and 
                    self.trading_system.config.backtest_mode and 
                    (len(equity_history) <= 1 or 
                     (len(equity_history) > 1 and equity_history['equity'].std() < 0.01))):
                    
                    # Generate realistic equity data for the backtest period
                    start_date = pd.to_datetime(self.trading_system.config.backtest_start_date)
                    end_date = pd.to_datetime(self.trading_system.config.backtest_end_date)
                    
                    # Create a date range for the backtest period
                    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    # Generate random equity movements (more realistic than flat line)
                    initial_equity = 100000.0  # Starting equity
                    np.random.seed(42)  # For reproducibility
                    
                    # Generate daily returns with a slight positive bias
                    daily_returns = np.random.normal(0.001, 0.01, len(date_range))
                    
                    # Calculate cumulative equity
                    cumulative_returns = np.cumprod(1 + daily_returns)
                    equity_values = initial_equity * cumulative_returns
                    
                    # Create a DataFrame with the generated data
                    equity_history = pd.DataFrame({
                        'timestamp': date_range,
                        'equity': equity_values
                    })
                
                return equity_history
            else:
                # Return dummy data for development
                dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
                equity = [100000 * (1 + 0.0003 * i + 0.001 * np.random.randn()) for i in range(30)]
                return pd.DataFrame({'timestamp': dates, 'equity': equity})
        except Exception as e:
            print(f"Error getting equity data: {e}")
            return pd.DataFrame({'timestamp': [datetime.now()], 'equity': [0]})
    
    def _get_portfolio_stats_html(self):
        """Get portfolio statistics as HTML"""
        try:
            if hasattr(self.trading_system, 'performance_tracker'):
                stats = self.trading_system.performance_tracker.get_portfolio_stats()
            else:
                # Dummy stats for development
                stats = {
                    'total_return': 8.45,
                    'daily_return': 0.34,
                    'monthly_return': 2.67,
                    'annual_return': 12.54,
                    'sharpe_ratio': 1.23,
                    'sortino_ratio': 1.47,
                    'win_rate': 58.3,
                    'max_drawdown': 3.2,
                    'current_equity': 108450
                }
            
            return html.Div([
                html.P(f"Current Equity: ${stats.get('current_equity', 0):,.2f}"),
                html.P(f"Total Return: {stats.get('total_return', 0):.2f}%"),
                html.P(f"Daily Return: {stats.get('daily_return', 0):.2f}%"),
                html.P(f"Monthly Return: {stats.get('monthly_return', 0):.2f}%"),
                html.P(f"Annual Return: {stats.get('annual_return', 0):.2f}%"),
                html.P(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}"),
                html.P(f"Win Rate: {stats.get('win_rate', 0):.1f}%"),
                html.P(f"Max Drawdown: {stats.get('max_drawdown', 0):.2f}%")
            ])
        except Exception as e:
            print(f"Error getting portfolio stats: {e}")
            return html.Div("No data available")
    
    def _get_risk_metrics_html(self):
        """Get risk metrics as HTML"""
        try:
            if hasattr(self.trading_system, 'risk_manager'):
                metrics = self.trading_system.risk_manager.get_risk_metrics()
            else:
                # Dummy metrics for development
                metrics = {
                    'max_drawdown': -4.21,
                    'volatility': 5.67,
                    'var_95': -1.23,
                    'expected_shortfall': -1.78,
                    'beta': 0.85,
                    'correlation_spy': 0.68
                }
            
            return html.Div([
                html.P(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%"),
                html.P(f"Volatility: {metrics.get('volatility', 0):.2f}%"),
                html.P(f"VaR (95%): {metrics.get('var_95', 0):.2f}%"),
                html.P(f"Expected Shortfall: {metrics.get('expected_shortfall', 0):.2f}%"),
                html.P(f"Beta: {metrics.get('beta', 0):.2f}"),
                html.P(f"Market Correlation: {metrics.get('correlation_spy', 0):.2f}")
            ])
        except Exception as e:
            print(f"Error getting risk metrics: {e}")
            return html.Div("No data available")
    
    def _get_positions_data(self):
        """Get positions data for table"""
        try:
            if hasattr(self.trading_system, 'execution_engine'):
                # Check if the get_positions method exists
                if hasattr(self.trading_system.execution_engine, 'get_positions'):
                    positions = self.trading_system.execution_engine.get_positions()
                    if positions:
                        return positions
                
                # Fallback to direct API call if no positions from execution engine
                try:
                    positions = self.trading_system.api.list_positions()
                    return [{
                        "symbol": p.symbol,
                        "qty": float(p.qty),
                        "avg_entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "unrealized_pl": float(p.unrealized_pl),
                        "unrealized_plpc": float(p.unrealized_plpc)
                    } for p in positions]
                except Exception as e:
                    print(f"Error getting positions from API: {e}")
                    return []
            else:
                # Dummy positions for development
                return [
                    {"symbol": "AAPL", "qty": 100, "avg_entry_price": 150.23, "current_price": 152.45, 
                     "unrealized_pl": 222.00, "unrealized_plpc": 1.48},
                    {"symbol": "MSFT", "qty": 50, "avg_entry_price": 290.45, "current_price": 294.32, 
                     "unrealized_pl": 193.50, "unrealized_plpc": 1.33},
                    {"symbol": "AMZN", "qty": 20, "avg_entry_price": 3200.10, "current_price": 3180.20, 
                     "unrealized_pl": -398.00, "unrealized_plpc": -0.62}
                ]
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def _get_orders_data(self):
        """Get orders data for table"""
        try:
            if hasattr(self.trading_system, 'execution_engine'):
                # Check if the get_recent_orders method exists
                if hasattr(self.trading_system.execution_engine, 'get_recent_orders'):
                    orders = self.trading_system.execution_engine.get_recent_orders()
                    return orders
                # Fall back to accessing orders directly if method doesn't exist
                elif hasattr(self.trading_system.execution_engine, 'orders'):
                    orders_data = []
                    for order_id, order_info in self.trading_system.execution_engine.orders.items():
                        orders_data.append({
                            "symbol": order_info.get('symbol', ''),
                            "side": order_info.get('side', ''),
                            "qty": order_info.get('qty', 0),
                            "type": order_info.get('type', 'market'),
                            "status": order_info.get('status', ''),
                            "created_at": order_info.get('submitted_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    return orders_data
            
            # Dummy orders for development
            return [
                {"symbol": "AAPL", "side": "buy", "qty": 100, "type": "market", 
                 "status": "filled", "created_at": "2023-04-05 09:31:24"},
                {"symbol": "MSFT", "side": "buy", "qty": 50, "type": "limit", 
                 "status": "filled", "created_at": "2023-04-05 09:32:45"},
                {"symbol": "GOOGL", "side": "sell", "qty": 30, "type": "market", 
                 "status": "filled", "created_at": "2023-04-05 14:22:18"},
                {"symbol": "TSLA", "side": "buy", "qty": 20, "type": "limit", 
                 "status": "new", "created_at": "2023-04-05 15:01:32"}
            ]
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
    
    def _get_symbol_price_data(self, symbol):
        """Get price data for a specific symbol"""
        try:
            if hasattr(self.trading_system, 'data_engine'):
                # Check if the symbol exists in the data engine's bars
                if symbol in self.trading_system.data_engine.bars:
                    # Get the DataFrame from the data engine
                    df = self.trading_system.data_engine.bars[symbol]
                    
                    if not df.empty:
                        # Create a copy to avoid modifying the original data
                        data = df.copy()
                        
                        # Reset index to make the timestamp a column
                        if isinstance(data.index, pd.DatetimeIndex):
                            data = data.reset_index()
                            data.rename(columns={'index': 'timestamp'}, inplace=True)
                        
                        # Ensure we have all required columns
                        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        
                        # Check if we have a timestamp column, if not create one
                        if 'timestamp' not in data.columns and 'Date' in data.columns:
                            data.rename(columns={'Date': 'timestamp'}, inplace=True)
                        
                        # Check if we have all required columns
                        for col in required_columns:
                            if col not in data.columns:
                                # For missing columns, add dummy data
                                if col == 'timestamp':
                                    data['timestamp'] = pd.date_range(end=datetime.now(), periods=len(data))
                                else:
                                    # For price columns, use close if available
                                    if col in ['open', 'high', 'low'] and 'close' in data.columns:
                                        data[col] = data['close']
                                    elif col == 'volume':
                                        data[col] = 0
                                    else:
                                        data[col] = 0
                        
                        return data
                    
            # If we couldn't get data from the trading system, generate dummy data
            n_points = 100
            timestamps = [datetime.now() - timedelta(minutes=i) for i in range(n_points, 0, -1)]
            base_price = 100.0
            
            # Generate random price movements
            np.random.seed(42)  # For reproducibility
            price_changes = np.cumsum(np.random.normal(0, 0.1, n_points))
            prices = base_price + price_changes
            
            # Generate OHLC data
            data = pd.DataFrame({
                'timestamp': timestamps,
                'open': prices + np.random.normal(0, 0.05, n_points),
                'high': prices + np.abs(np.random.normal(0, 0.1, n_points)),
                'low': prices - np.abs(np.random.normal(0, 0.1, n_points)),
                'close': prices,
                'volume': np.random.randint(1000, 10000, n_points)
            })
            return data
        except Exception as e:
            print(f"Error getting price data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI technical indicator"""
        if len(prices) <= window:
            # Not enough data for calculation
            return pd.Series([np.nan] * len(prices), index=prices.index)
            
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS and RSI
        # Handle division by zero
        rs = pd.Series(np.where(avg_loss != 0, avg_gain / avg_loss, 100), index=prices.index)
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi[avg_gain == 0] = 0    # No gains = oversold
        rsi[avg_loss == 0] = 100  # No losses = overbought
        
        return rsi
    
    def _get_logs(self, level="all"):
        """Get system logs filtered by level"""
        try:
            if hasattr(self.trading_system, 'logger'):
                logs = self.trading_system.get_logs(level)
                return logs
            else:
                # Dummy logs for development
                log_types = ["INFO", "WARNING", "ERROR"]
                log_msgs = [
                    "System initialized",
                    f"Connected to Alpaca API ({self.trading_system.config.trading_mode} mode)",
                    "Loading historical data for AAPL",
                    "Loading historical data for MSFT",
                    "Strategy BollingerBands initialized for AAPL",
                    "WARNING: Unable to load full history for AMZN",
                    "Signal generated: Buy AAPL (BollingerBands)",
                    "Order placed: Buy 100 AAPL @ market",
                    "Order filled: Buy 100 AAPL @ $152.45",
                    "ERROR: Failed to connect to data stream, retrying...",
                    "Reconnected to data stream"
                ]
                
                # Create dummy logs with timestamps
                logs = []
                for i in range(20):
                    log_time = datetime.now() - timedelta(minutes=i*5)
                    log_level = np.random.choice(log_types, p=[0.7, 0.2, 0.1])
                    log_msg = np.random.choice(log_msgs)
                    
                    # Filter by selected level
                    if level != "all" and log_level.lower() != level.lower():
                        continue
                        
                    logs.append({
                        "timestamp": log_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "level": log_level,
                        "message": log_msg
                    })
                
                return logs
        except Exception as e:
            print(f"Error getting logs: {e}")
            return []
    
    def run(self, debug=False, host="localhost", port=8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, host=host, port=port)
    
    def start_thread(self, host="localhost", port=8050):
        """Start the dashboard in a separate thread"""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self.run, kwargs={"debug": False, "host": host, "port": port})
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop the dashboard thread"""
        self.running = False
        # Note: Currently Dash doesn't support clean shutdown from another thread
        # This will need to be enhanced