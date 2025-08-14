'''
Collect structured and unstructured data equity data from various sources.
'''

import pandas as pd
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv
import sys

# Show which interpreter is running this script (helps detect mismatch)
print(f"Using Python: {sys.executable}")

from marketminer import scrape_economic_times
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
    ticker += '.NS'
    df = yf.download(ticker, start=start, end=end)
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

def get_data(api_key, ticker, start, end):
    """
    Run all functions in this module to collect various types of data.

    Parameters:
    api_key (str): The API key for Alpha Vantage.
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
    news = collect_news(start=start, end=end)
    print(news.head())
    print(news.info())




if __name__ == "__main__":
    ticker = os.getenv('TICKER')
    api_key = os.getenv('API_KEY')
    start = os.getenv('START_DATE')
    end = os.getenv('END_DATE')
    get_data(api_key, ticker, start, end)
