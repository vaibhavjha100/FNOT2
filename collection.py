'''
Collect structured and unstructured data equity data from various sources.
'''

import pandas as pd
import numpy as np
import yfinance as yf
import os
from dotenv import load_dotenv
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

def collect_stock_news(api_key, ticker, start, end):
    """
    Collect news data for a given ticker using Alpha Vantage API.
    Parameters:
    api_key (str): The API key for Alpha Vantage.
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.
    Returns:
    pd.DataFrame: A DataFrame containing the news data.
    """
    #ticker += '.BSE'
    # Time format is YYYYMMDDTHHMM
    # Example: 20230101T0000
    start_date = str(pd.to_datetime(start).strftime('%Y%m%d'))+"T0000"
    end_date = str(pd.to_datetime(end).strftime('%Y%m%d'))+"T2359"
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "topics": "earnings",
        "time_from": start_date,
        "time_to": end_date,
        "limit": 100,
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(data)
    if 'feed' not in data:
        raise ValueError("No news data found for the given ticker and date range.")
    news_data = pd.DataFrame(data['feed'])
    news_data['ticker'] = ticker
    news_data['time_published'] = pd.to_datetime(news_data['time_published'])
    # Set the index to the time_published column
    news_data.set_index('time_published', inplace=True)
    news_data.sort_index(ascending=True, inplace=True)
    # news_data.to_csv(f'data/{ticker}_news.csv', index=False)
    print(f"News data collected for {ticker} from {start} to {end}.")
    return news_data


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
    stock_news_data = collect_stock_news(api_key, ticker, start, end)
    print(stock_news_data.head())
    print(stock_news_data.info())




if __name__ == "__main__":
    ticker = os.getenv('TICKER')
    api_key = os.getenv('API_KEY')
    start = os.getenv('START_DATE')
    end = os.getenv('END_DATE')
    # get_data(api_key, ticker, start, end)
    # Test news data collection
    news_data = collect_stock_news(api_key, ticker, start, end)
    print(news_data.head())
    print(news_data.info())