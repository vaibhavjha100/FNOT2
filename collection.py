'''
Collect structured and unstructured data equity data from various sources.
'''

import pandas as pd
import numpy as np
import yfinance as yf

def collect_ohlcv(ticker, period='max', interval='1d'):
    """
    Collect OHLCV data for a given ticker using yfinance.

    Parameters:
    ticker (str): The stock ticker symbol.
    period (str): The period for which to collect data (default is 'max').
    interval (str): The interval between data points (default is '1d').

    Returns:
    pd.DataFrame: A DataFrame containing the OHLCV data.
    """
    df = yf.download(ticker, period=period, interval=interval, multi_level_index=False)
    df.sort_index(ascending=True, inplace=True)
    return df

def get_data(ticker, period='max', interval='1d'):
    """
    Get all types of data for a given ticker.

    Parameters:
    ticker (str): The stock ticker symbol.
    period (str): The period for which to collect data (default is 'max').
    interval (str): The interval between data points (default is '1d').

    Returns:
    pd.DataFrame: A DataFrame containing the data.
    """
    # Get today's date for file naming
    today = pd.Timestamp.now().strftime('%Y%m%d')
    # Collect data
    ohlcv = collect_ohlcv(ticker, period, interval)
    ohlcv.to_csv(f'data/{ticker}_ohlcv{today}.csv')
    print(f"Data collected and saved to: data/{ticker}_ohlcv{today}.csv")
    return ohlcv

if __name__ == "__main__":
    ticker = '^NSEI'
    data = get_data(ticker, period='max', interval='1d')
    print(data.head())
    print(data.info())