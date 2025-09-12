import yfinance as yf
import pandas as pd
import numpy as np
# from model import Kronos, KronosTokenizer, KronosPredictor
import torch

# Download data from Yahoo Finance
def download_data(ticker, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    # print(data.info())
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['open', 'high', 'low', 'close', 'volume']  # Rename for Kronos
    data['amount'] = data['close'] * data['volume']  # Calculate amount (approximate)
    ### data.index: DateTimeIndex
    return data

# Preprocess data for Kronos
def preprocess_data(data, lookback=400, pred_len = 120):
    """
    Prepare data for Kronos prediction.
    """
    df = data.copy()
    df = df.dropna()
    df = df.reset_index()
    # x_df = data[['open', 'high', 'low', 'close', 'volume', 'amount']].iloc[-lookback:]
    # x_timestamp = data.loc[-lookback:, 'timestamps']

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']

    # x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    # x_timestamp = df.loc[:lookback-1, 'timestamps']
    # y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']

    return x_df, x_timestamp


# Generate predictions
def generate_predictions(predictor, x_df, data, x_timestamp, lookback = 400, pred_len=30):
    """
    Generate forecasts using Kronos.
    """
    # Create future timestamps (assuming daily data)
    # last_date = x_timestamp[-1]
    # future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq='D')
    df = data.copy().reset_index()
    future_dates = df.loc[lookback:lookback+pred_len-1, 'timestamps']
    # print(future_dates)
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=future_dates,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1
    )
    return pred_df, future_dates

# Backtest strategy
def backtest_strategy(historical_data, predictions, initial_capital=10000, commission=0.001):
    """
    Simple backtesting strategy: Buy if predicted close > current close, else sell.
    """
    capital = initial_capital
    position = 0
    equity_curve = []
    trades = []

    # Align predictions with historical data (for the backtest period)
    combined_data = predictions.copy()
    combined_data['true_close'] = np.nan
    combined_data.loc[predictions.index, 'true_close'] = historical_data.loc[predictions.index, 'close']

    for i in range(1, len(combined_data)):
        current_price = combined_data['true_close'].iloc[i]
        pred_price = combined_data['close'].iloc[i]

        if np.isnan(pred_price):
            continue  # Skip if no prediction

        # Signal: Buy if predicted price > current price
        if pred_price > current_price and position == 0:
            # Buy
            shares_to_buy = (capital * (1 - commission)) / current_price
            cost = shares_to_buy * current_price * (1 + commission)
            capital -= cost
            position = shares_to_buy
            trades.append(('Buy', combined_data.index[i], current_price, shares_to_buy))
        elif pred_price <= current_price and position > 0:
            # Sell
            revenue = position * current_price * (1 - commission)
            capital += revenue
            trades.append(('Sell', combined_data.index[i], current_price, position))
            position = 0

        # Update equity curve
        equity = capital + (position * current_price)
        equity_curve.append(equity)

    # Calculate performance metrics
    equity_series = pd.Series(equity_curve, index=combined_data.index[1:len(equity_curve)+1])
    returns = equity_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    max_drawdown = (equity_series / equity_series.expanding().max() - 1).min()

    return equity_series, trades, sharpe_ratio, max_drawdown, combined_data

    