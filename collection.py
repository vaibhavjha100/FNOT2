'''
Collect structured and unstructured data equity data from various sources.
'''

import pandas as pd
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv
load_dotenv()

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
    df.to_csv(f'data/{ticker}_ohlcv.csv')
    print(f"Data collected and saved to: data/{ticker}_ohlcv.csv")
    return df


def collect_news(start_date=None, end_date=None):
    """
    Collect news data from start_date to end_date.

    Parameters:
    start_date (str): The start date for news collection in 'YYYY-MM-DD' format.
    end_date (str): The end date for news collection in 'YYYY-MM-DD' format.
    Returns:
    pd.DataFrame: A DataFrame containing the news data.
    """




if __name__ == "__main__":
    ticker = os.getenv('TICKER', '^NSEI')  # Default to Nifty50 if TICKER is not set in .env
    data = collect_ohlcv(ticker, period='max', interval='1d')
    print(data.head())
    print(data.info())