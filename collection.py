'''
Collect structured and unstructured data equity data from various sources.
'''

import time
import pandas as pd
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv
import sys

from marketminer import scrape_economic_times, scrape_fundamentals
import requests
load_dotenv()

def collect_ohlcv(ticker, start, end):
    """
    Collect OHLCV data for a given ticker using yfinance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.

    Returns:
    pd.DataFrame: A DataFrame containing the OHLCV data.
    """
    # Make end date inclusive
    end = pd.to_datetime(end) + pd.Timedelta(days=1)
    ticker_yf =ticker + '.NS' if not ticker.endswith('.NS') else ''
    df = yf.download(ticker_yf, start=start, end=end, multi_level_index=False)
    df.sort_index(ascending=True, inplace=True)
    df.to_csv(f'data/{ticker}_ohlcv.csv')
    print(f"Data collected and saved to: data/{ticker}_ohlcv.csv")
    return df

def collect_news(start, end):
    '''
    Collect news data from economic times using marketminer.
    Parameters:
    start (str): The start date for the news collection.
    end (str): The end date for the news collection.
    Returns:
    pd.DataFrame: A DataFrame containing the news data.
    '''
    news_df = scrape_economic_times(start, end)
    news_df.to_csv('data/news.csv', index=True)
    print("News data collected and saved to: data/news.csv")
    return news_df

def collect_fundamentals(ticker):
    """
    Collect fundamental data for a given ticker using marketminer.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    dict: A dictionary containing various types of fundamental data.
    """
    fund_data = scrape_fundamentals(ticker)
    for key, value in fund_data.items():
        value.to_csv(f'data/{ticker}_{key}.csv', index=True)
        print(f"{key} data saved to: data/{ticker}_{key}.csv")
    return fund_data

def get_data(ticker, start, end):
    """
    Run all functions in this module to collect various types of data.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.

    Returns:
    None: This function saves dataframes to CSV files in the 'data' directory.

    Ouptput:
    All data dataframes are saved to the 'data' directory.
    """
    ohlcv = collect_ohlcv(ticker, start=start, end=end)
    print(ohlcv.head())
    print(ohlcv.info())
    time.sleep(2)
    news = collect_news(start=start, end=end)
    print(news.head())
    print(news.info())
    time.sleep(2)
    fund = collect_fundamentals(ticker)
    print(fund.keys())
    for key, value in fund.items():
        print(f"{key} DataFrame:")
        print(value.head())
        print(value.info())




if __name__ == "__main__":
    ticker = os.getenv('TICKER')
    start = os.getenv('START_DATE')
    end = os.getenv('END_DATE')
    get_data(ticker, start, end)
