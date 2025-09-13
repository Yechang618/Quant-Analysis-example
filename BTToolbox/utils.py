import yfinance as yf
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor
import torch

#
def load_kronos_model(model_name="NeoQuasar/Kronos-small"):
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu", max_context=512)
    return predictor
#
class backTestData(object):
    """
    Data processor for backtest
        self.data: original data with time index, pandas dataframe. 
            -> columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            -> index, DateTimeIndex
        self.df: 
            -> columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'timestamps']
            -> index, Reseted       
    
    """
    def __init__(self, ticker, start_date, end_date, interval='1d', lookback = 200):
        self.ticker = ticker
        self.start_date = start_date
        self.end_data = end_date
        self.lookback = lookback
        self.interval = interval
        self.data = self.fetch_data(ticker, start_date, end_date)
        self.nDates_all = len(self.data)
        self.df = self.data.copy()
        self.df['timestamps'] = pd.to_datetime(self.data.index)
        self.df = self.df.reset_index()
    
        self.df_lookback, self.ts_lookback, self.df_target, self.ts_target = self.split_data()
        self.nDates_target = len(self.df_target)
        print(f"Preprocessed data with {len(self.df_lookback)} lookbacks and {len(self.df_target)} targets...")
        
    def fetch_data(self, ticker, start_date, end_date, interval='1d'):
        """
        Fetch real-world data from Yahoo Finance
        INPUTS:
            ticker
            start_date
            end_date
            interval (default, '1d')
    
        OUTPUT:
            data: pandas dataframe. 
                -> columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                -> index, DateTimeIndex
            
        """
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        # try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {ticker} in the specified date range")

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        # Clean data
        data = data.dropna()
        data = data[~data.index.duplicated(keep='first')]
        # print(data.head(5))
        # print(data.info())
        data.columns = data.columns.droplevel(-1)
        data.columns = ['open', 'high', 'low', 'close', 'volume']  # Rename for Kronos        
        data['amount'] = data['close'] * data['volume']
        print(f"Data fetched: {len(data)} records from {data.index[0]} to {data.index[-1]}")
        return data

    def getData(self):
        return self.data
        
    def getData(self):
        return self.data
        
    def split_data(self):
        """
        
        """
        lookback = self.lookback
        assert self.nDates_all > lookback
        df = self.df.copy()
        df_lookback = df.loc[: lookback, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        ts_lookback = df.loc[: lookback, 'timestamps']   
        df_target = df.loc[lookback: self.nDates_all, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        ts_target = df.loc[lookback: self.nDates_all, 'timestamps']   
        
        return df_lookback, ts_lookback, df_target, ts_target
        # timestamps_x = df.loc[curr - lookback: curr, 'timestamps']        


    def get_data_subset(self, curr, lookback = 10, pred_end = 1):
        """
        INPUTs:
            curr: index, int
            lookback: int
            pred_end: int
        OUTPUTs:
            df_x: lookback data, from curr - lookback to curr - 1
            timestamps_x: lookback timestamps, from curr - lookback to curr - 1
            close_y: close price to predict
            timestamps_y: ... 
            
        """
        assert curr - lookback >= 0
        df = self.df.copy()
        df_x = df.loc[curr-lookback: curr, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        timestamps_x = df.loc[curr - lookback: curr, 'timestamps']
        close_y = df.loc[curr: curr + pred_end, 'close']
        timestamps_y = df.loc[curr: curr + pred_end, 'timestamps']
        return df_x, timestamps_x, close_y, timestamps_y

    ###        
    def generate_kronos_predictions(self, verbose_interval = 100):
        """
        Generate forecasts using Kronos.

        df: dataframe with no index, with column timestamps
        
        OUTPUT:
            df_pred: 
                columns: close
                index: DateTimeIndex, consistent with ts_target
        """

        # Load Kronos model
        predictor = load_kronos_model()        
        df = self.df.copy()
        # df_target = self.df_target.copy()
        df_pred = self.df_target.copy() # Indexed from lookback to nDate_all
        ts_target = self.ts_target.copy()
        df_pred['timestamps'] = ts_target
        df_pred['close_pred'] = np.nan
        lookback = self.lookback

        print(f"Generating Kronos predictions from {df_pred['timestamps'].iloc[0]}  to {df_pred['timestamps'].iloc[-1]}")
        
        for i in range(self.nDates_target):
            index_target = df.loc[i : i + lookback, 'timestamps']
            df_input = df.loc[i : i + lookback, ['open', 'high', 'low', 'close', 'volume', 'amount']]
            ts_pred = df.loc[i + lookback : i + lookback, 'timestamps']
            # print(ts_pred.info())
            pred_df = predictor.predict(
                df=df_input,
                x_timestamp=index_target,
                y_timestamp=ts_pred,
                pred_len=1,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            pred_date = pd.to_datetime(pred_df.index)[0]
            # print(pred_date)
            # print(df_pred.info())
            assert df_pred.loc[i + lookback, 'timestamps'] == pred_date
            df_pred.loc[i + lookback, 'close_pred'] = pred_df.loc[pred_date, 'close']
            if verbose_interval > 0 and i % verbose_interval == 0:
                print(f"Got Kronos prediction at {pred_date}, {self.nDates_target - i} left.")
            # pred_df has DateTimeIndex
        print(f"Predictions generated as: ")
        df_pred.index = ts_target
        print(df_pred.info())
        return df_pred

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
def preprocess_data(data, lookback=400):
    """
    Prepare data for Kronos prediction.
    """
    df = data.copy()
    df = df.dropna()
    df = df.reset_index()

    x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
    x_timestamp = df.loc[:lookback-1, 'timestamps']

    return x_df, x_timestamp


# Generate predictions
def generate_predictions(predictor, x_df, data, x_timestamp, lookback = 400, pred_len=30):
    """
    Generate forecasts using Kronos.
    """
    # Create future timestamps (assuming daily data)
    # last_date = x_timestamp[-1]
    # future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq='D')
    df = data.copy()
    df['timestamps'] = pd.to_datetime(df.index)
    df = df.reset_index()
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

    