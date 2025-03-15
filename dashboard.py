# dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
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
            fig.add_trace(go.Scatter(
                x=equity_data['timestamp'],
                y=equity_data['equity'],
                mode='lines',
                name='Account Equity'
            ))
            
            fig.update_layout(
                title='Account Equity Over Time',
                yaxis_title='Equity ($)',
                hovermode='x unified'
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
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=price_data['timestamp'],
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name=symbol
            ))
            
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
            
            fig = go.Figure()
            
            # Add closing price
            fig.add_trace(go.Scatter(
                x=price_data['timestamp'],
                y=price_data['close'],
                mode='lines',
                name=f'{symbol} Close'
            ))
            
            # Add selected indicators
            if 'sma20' in indicators:
                sma20 = price_data['close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=sma20,
                    mode='lines',
                    name='SMA (20)',
                    line=dict(color='orange')
                ))
                
            if 'sma50' in indicators:
                sma50 = price_data['close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(
                    x=price_data['timestamp'],
                    y=sma50,
                    mode='lines',
                    name='SMA (50)',
                    line=dict(color='green')
                ))
                
            if 'bollinger' in indicators:
                sma20 = price_data['close'].rolling(window=20).mean()
                std20 = price_data['close'].rolling(window=20).std()
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
                
            if 'rsi' in indicators:
                # Add a secondary y-axis for RSI
                rsi_values = self._calculate_rsi(price_data['close'])
                
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
            
            fig.update_layout(
                title=f'{symbol} Indicators',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            
            return fig
        
        # System logs
        @self.app.callback(
            Output("log-display", "children"),
            [Input("log-level-selector