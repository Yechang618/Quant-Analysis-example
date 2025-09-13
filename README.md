This is the back-testing engine developed by Chang Ye

============ **Update records** ============

**Version** 2 @ 09-05-2025

Implement Kronos (from https://github.com/shiyu-coder/Kronos, thanks for the great work from Shi Yu's team!!!) for back-testing. And formalized data fetching, preprocessing and generalized the strategy function for machine learning models.


**Version** 1 @ 09-05-2025

Develop the example for backtest and time series analysis

**Stock Backtesting Engine**
A comprehensive Python-based backtesting framework for developing and testing quantitative trading strategies using real-world market data.

**Features**
Real-time Data Fetching: Automatically retrieves OHLCV data from Yahoo Finance

**Multiple Trading Strategies**
5 built-in strategies with customizable parameters, including:

1. Trend Following
Logic: Buy when price above 200-day SMA and 20-day SMA > 50-day SMA

Best for: Trending markets, momentum plays

2. Mean Reversion
Logic: Buy when price 2% below lower Bollinger Band, sell when 2% above upper band

Best for: Range-bound markets, contrarian strategies

3. MACD Crossover
Logic: Buy when MACD crosses above signal line and both positive

Best for: Momentum and trend confirmation

4. RSI Momentum
Logic: Buy when RSI crosses above 30 from below, sell when crosses below 70 from above

Best for: Overbought/oversold conditions

5. Breakout
Logic: Buy when price breaks above 20-day high, sell when breaks below 20-day low

Best for: Volatility breakouts, range expansion

**Technical Indicators** 
The script calculates over 20 technical indicators:

Moving Averages: SMA (20, 50, 200), EMA (12, 26)

Oscillators: MACD, RSI, Stochastic %K/%D

Volatility: Bollinger Bands, ATR, ATR Percentage

Volume: Volume SMA, Volume Ratio

Momentum: 5-day and 20-day momentum

**Risk Management** 
Built-in position sizing, stop losses, and commission/slippage modeling

**Performance Metrics** 
Comprehensive analytics including Sharpe ratio, Sortino ratio, drawdown analysis, and benchmark comparison

**Visualization** 
Interactive plots showing equity curves, drawdowns, signals, and technical indicators

**Benchmark Comparison** 
Compare strategy performance against major indices (S&P 500, NASDAQ, etc.)
