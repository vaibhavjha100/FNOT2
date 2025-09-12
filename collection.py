'''
Collect structured and unstructured data equity data from various sources.
'''

import time
import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv

from marketminer import scrape_economic_times, scrape_fundamentals, scrape_macro_india
load_dotenv()

def collect_ohlcv(ticker, start, end, datadir='data'):
    """
    Collect OHLCV data for a given ticker using yfinance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    pd.DataFrame: A DataFrame containing the OHLCV data.
    """
    # Make end date inclusive
    end = pd.to_datetime(end) + pd.Timedelta(days=1)
    ticker_yf =ticker + '.NS' if not ticker.endswith('.NS') else ''
    df = yf.download(ticker_yf, start=start, end=end, multi_level_index=False)
    df.sort_index(ascending=True, inplace=True)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    df.to_csv(f'{datadir}/{ticker}_ohlcv.csv', index=True)
    return df

def collect_news(start, end, datadir='data'):
    '''
    Collect news data from economic times using marketminer.

    Parameters:
    start (str): The start date for the news collection.
    end (str): The end date for the news collection.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    pd.DataFrame: A DataFrame containing the news data.
    '''
    news_df = scrape_economic_times(start, end)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    news_df.to_csv(f'{datadir}/news.csv', index=False)
    print(f"✅ News data collected and saved to: {datadir}/news.csv")
    return news_df

def collect_fundamentals(ticker, datadir='data'):
    """
    Collect fundamental data for a given ticker using marketminer.

    Parameters:
    ticker (str): The stock ticker symbol.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    dict: A dictionary containing various types of fundamental data.
    """
    fund_data = scrape_fundamentals(ticker)
    for key, value in fund_data.items():
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        value.to_csv(f'{datadir}/{ticker}_{key}.csv', index=True)
        print(f"✅ {key} data saved to: {datadir}/{ticker}_{key}.csv")
    return fund_data

def collect_macro_india(start, end, datadir='data'):
    """
    Collect macroeconomic data for India using marketminer.

    Parameters:
    start (str): The start date for the macroeconomic data collection.
    end (str): The end date for the macroeconomic data collection.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    dict: A dictionary containing various types of macroeconomic data.
    """
    m_dict = scrape_macro_india(start, end)
    for key, value in m_dict.items():
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        value.to_csv(f'{datadir}/{key}_macro.csv', index=True)
        print(f"✅ {key} data saved to: {datadir}/{key}_macro.csv")
    return m_dict

def get_actions(ticker, start, end, datadir='data'):
    """
    Collect stock actions data for a given ticker using yfinance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    pd.DataFrame: A DataFrame containing the stock actions data.
    """
    ticker_yf =ticker + '.NS' if not ticker.endswith('.NS') else ''
    stock = yf.Ticker(ticker_yf)
    actions = stock.actions
    actions = actions[(actions.index >= start) & (actions.index <= end)]
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    actions.to_csv(f'{datadir}/{ticker}_actions.csv', index=True)
    print(f"✅ Actions data collected and saved to: {datadir}/{ticker}_actions.csv")
    return actions

def get_data(ticker, start, end, datadir='data'):
    """
    Run all functions in this module to collect various types of data.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for the data collection.
    end (str): The end date for the data collection.
    datadir (str): The directory to save the data. Default is 'data'.

    Returns:
    None: This function saves dataframes to CSV files in the 'data' directory.

    Ouptput:
    All data dataframes are saved to the 'data' directory.
    """
    ohlcv = collect_ohlcv(ticker, start=start, end=end, datadir=datadir)
    print(ohlcv.head())
    print(ohlcv.info())
    time.sleep(2)
    news = collect_news(start=start, end=end, datadir=datadir)
    print(news.head())
    print(news.info())
    time.sleep(2)
    fund = collect_fundamentals(ticker, datadir=datadir)
    print(fund.keys())
    for key, value in fund.items():
        print(f"{key} DataFrame:")
        print(value.head())
        print(value.info())
    macro = collect_macro_india(start=start, end=end, datadir=datadir)
    print(macro.keys())
    for key, value in macro.items():
        print(f"{key} DataFrame:")
        print(value.head())
        print(value.info())
    actions = get_actions(ticker, start=start, end=end, datadir=datadir)
    print(actions.head())
    print(actions.info())



if __name__ == "__main__":
    ticker = os.getenv('TICKER')
    start = os.getenv('START_DATE')
    end = os.getenv('END_DATE')
    get_data(ticker, start, end)
