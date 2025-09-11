"""
Module for preprocessing features data for reinforcement learning models.

Our normalization strategy: z-score normalization for most features (done at individual feature level).

List of preprocessing files:
1. OHLCV data preprocessing
2. Sentiment Analysis using finsenti -> Sentiment data preprocessing
3. Actions data preprocessing
4. Fundamentals data preprocessing (Balance Sheet, Profit Loss, Cash Flow)
5. Macroeconomic data preprocessing (Daily/Weekly/Fortnightly/Monthly/Quarterly)
6. Ratio data preprocessing
7. Trading data preprocessing (execution price, trading price)

After preprocessing, merge all dataframes on Date index.

The merged dataframe will be used to PCA and build sequences for LSTM-DDPG model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

load_dotenv()

DATADIR = 'data/'


def get_trading_data(ticker, spread_coeff=0.1, sigma_noise=0.001):
    """
    Prepare trading data for a given ticker.
    We will create realistic trading prices based on OHLCV data.
    Execution price = mid price + (spread/2) + slippage noise
    No need to normalize prices, as this will be used for trading actions.

    Parameters:
    ticker (str): The stock ticker symbol.
    spread_coeff (float): Coefficient to estimate the bid-ask spread.
    sigma_noise (float): Standard deviation of slippage noise.
    Returns:
    pd.DataFrame: A DataFrame containing the trading data.
    """

    file_path = os.path.join(DATADIR, f"{ticker}_ohlcv.csv")
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()

    trading_df = pd.DataFrame(index=df.index)
    trading_df['mid'] = (df['High'] + df['Low']) / 2
    trading_df['spread_est'] = spread_coeff * (df['High'] - df['Low']) / trading_df['mid']

    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, sigma_noise, size=len(df))

    trading_df['execution_price'] = trading_df['mid'] + (trading_df['spread_est']/2) + noise
    trading_df['execution_price'] = trading_df['execution_price'].clip(lower=0)

    trading_df['trading_price'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4

    return trading_df[['execution_price']]

def get_ohlcv_index(ticker):
    """
    Get the OHLCV data index for a given ticker.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DatetimeIndex: The index of the OHLCV DataFrame.
    """
    file_path = os.path.join(DATADIR, f"{ticker}_ohlcv.csv")
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date').sort_index()
    return df.index

def preprocess_ohclv(ticker):
    """
    Preprocess OHLCV data for a given ticker.
    Create technical indicators as features.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DataFrame: A preprocessed DataFrame containing the OHLCV data.
    """
    file_path = os.path.join(DATADIR, f"{ticker}_ohlcv.csv")
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date').sort_index()

    # Compute returns
    df['return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # Technical Indicators
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()

    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    df['std_20'] = df['Close'].rolling(window=20).std()
    df['upper_bb'] = df['sma_20'] + 2 * df['std_20']
    df['lower_bb'] = df['sma_20'] - 2 * df['std_20']

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-8)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()  # 3-day SMA of %K

    # Williams %R
    df['williams_r'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-8)

    # Keltner Channels
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    atr10 = (df['High'] - df['Low']).rolling(10).mean()
    df['kc_middle'] = ema20
    df['kc_upper'] = ema20 + 2 * atr10
    df['kc_lower'] = ema20 - 2 * atr10

    # Volume indicators
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-8)
    mfv = mfm * df['Volume']
    df['cmf'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
    short_ema_vol = df['Volume'].ewm(span=12, adjust=False).mean()
    long_ema_vol = df['Volume'].ewm(span=26, adjust=False).mean()
    df['vol_osc'] = (short_ema_vol - long_ema_vol) / (long_ema_vol + 1e-8)

    # Support and Resistance Levels
    df['support_20'] = df['Low'].rolling(20).min()
    df['resistance_20'] = df['High'].rolling(20).max()

    # ATR (Average True Range)
    df['tr'] = df[['High', 'Low', 'Close']].apply(
        lambda x: max(x['High'] - x['Low'],
                      abs(x['High'] - x['Close']),
                      abs(x['Low'] - x['Close'])), axis=1)
    df['atr_14'] = df['tr'].rolling(window=14).mean()

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Normalize features with Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(zscore)

    return df

def get_sentiment(ticker: str, df: pd.DataFrame=None, text_column: str='body', gemini_api_key: str=None, start_date: str|datetime|date=None, end_date: str|datetime|date=None, aggregation_method: str='weighted_mean', sentiment_col: str='compound'):
    """
    Perform sentiment analysis using finsenti.
    Parameters:
    ticker (str): The stock ticker symbol.
    df (pd.DataFrame): DataFrame containing news data. If None, data will be loaded from 'data/news.csv'.
    text_column (str): The column name containing the news text.
    gemini_api_key (str): API key for finsenti.
    start_date (str|datetime|date): Start date for filtering news data.
    end_date (str|datetime|date): End date for filtering news data.
    aggregation_method (str): Method to aggregate sentiment scores.
    sentiment_col (str): The column name for sentiment scores in the finsenti output.
    Returns:
    pd.DataFrame: A preprocessed DataFrame containing the sentiment scores.
    - The returned DataFrame has Date as index and sentiment score column only.
    """
    from finsenti import finsenti_pipeline

    if df is None:
        file_path = os.path.join(DATADIR, 'news.csv')
        df = pd.read_csv(file_path, parse_dates=['date'], index_col='date').sort_index()

    # Filter by date range
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    tickers = [ticker]

    sent_df = finsenti_pipeline(tickers=tickers, df=df, text_column=text_column, gemini_api_key=gemini_api_key, aggregation_method=aggregation_method, sentiment_col=sentiment_col)

    # Save to CSV
    sent_df.to_csv(os.path.join(DATADIR, f"{ticker}_sentiment.csv"), index=True)
    print(f"✅ Sentiment data saved to: {os.path.join(DATADIR, f'{ticker}_sentiment.csv')}")
    return sent_df

def preprocess_sentiment(ticker):
    """
    Preprocess sentiment data.
    Create additional features if needed.

    Parameters:
    ticker (str): The stock ticker symbol.
    Returns:
    pd.DataFrame: A preprocessed DataFrame containing the sentiment data.
    """
    file_path = os.path.join(DATADIR, f"{ticker}_sentiment.csv")
    df = pd.read_csv(file_path).sort_index()

    # Check if 'date' column exists for setting index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    # Check if Unnamed: 0 column exists and set it as index if it does and rename it to 'date'
    elif 'Unnamed: 0' in df.columns:
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
        df.set_index('Unnamed: 0', inplace=True)
        df.index.name = 'date'

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Smooth sentiment scores with rolling mean
    df['sentiment_score_smoothed'] = df['sentiment_score'].ewm(span=5, adjust=False).mean()

    # Normalize all numeric columns with Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(zscore)

    # Set to OHLCV date index
    ohlcv_index = get_ohlcv_index(ticker)
    df = df.reindex(ohlcv_index).fillna(0)

    return df

def preprocess_actions(ticker):
    """
    Preprocess actions data for a given ticker.
    Actions data is assumed to be in 'data/{ticker}_actions.csv'.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DataFrame: A preprocessed DataFrame containing the actions data.
    """
    file_path = os.path.join(DATADIR, f"{ticker}_actions.csv")
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date').sort_index()

    idx = get_ohlcv_index(ticker)

    # Ensure index is timezone-naive to match OHLCV
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    actions = pd.DataFrame(0, index=idx, columns=df.columns)

    # Map each announcement date to next trading day using 'bfill' logic (get_indexer)
    # positions array gives integer positions in ohlcv_idx; -1 means no suitable index (event after last trading day)
    positions = idx.get_indexer(df.index, method='bfill')

    for ann_date, pos in zip(df.index, positions):
        if pos == -1:
            continue
        trade_date = idx[pos]
        actions.loc[trade_date] += df.loc[ann_date]
    actions.fillna(0, inplace=True)

    actions['Dividend_Event'] = (actions['Dividends'] > 0).astype(int)
    actions['Stock_Split_Event'] = (actions['Stock Splits'] > 0).astype(int)

    return actions

def preprocess_fundamentals(ticker):
    """
    Preprocess balance sheet, profit loss and cash flow data for a given ticker.
    Balance sheet data is assumed to be in 'data/{ticker}_balance_sheet.csv'.
    Profit loss data is assumed to be in 'data/{ticker}_profit_loss.csv'.
    Cash flow data is assumed to be in 'data/{ticker}_cash_flows.csv'.

    Parameters:
    ticker (str): The stock ticker symbol.

    Returns:
    pd.DataFrame: A preprocessed DataFrame containing the fundamentals data.
    """
    file_path_bs = os.path.join(DATADIR, f"{ticker}_balance_sheet.csv")
    df_bs = pd.read_csv(file_path_bs).sort_index()

    file_path_pl = os.path.join(DATADIR, f"{ticker}_profit_loss.csv")
    df_pl = pd.read_csv(file_path_pl).sort_index()

    file_path_cf = os.path.join(DATADIR, f"{ticker}_cash_flows.csv")
    df_cf = pd.read_csv(file_path_cf).sort_index()


    # Check if Unnamed: 0 column exists and set it as index if it does and rename it to 'Date'
    # For fundamentals data, the dates are years, not in datetime format (for example, 2020)
    if 'Unnamed: 0' in df_bs.columns:
        df_bs['Unnamed: 0'] = pd.to_datetime(df_bs['Unnamed: 0'], format='%Y')
        df_bs.set_index('Unnamed: 0', inplace=True)
        df_bs.index.name = 'Date'

    if 'Unnamed: 0' in df_pl.columns:
        df_pl['Unnamed: 0'] = pd.to_datetime(df_pl['Unnamed: 0'], format='%Y')
        df_pl.set_index('Unnamed: 0', inplace=True)
        df_pl.index.name = 'Date'

    if 'Unnamed: 0' in df_cf.columns:
        df_cf['Unnamed: 0'] = pd.to_datetime(df_cf['Unnamed: 0'], format='%Y')
        df_cf.set_index('Unnamed: 0', inplace=True)
        df_cf.index.name = 'Date'

    df_bs.index = pd.to_datetime(df_bs.index)
    df_bs.sort_index(inplace=True)

    df_pl.index = pd.to_datetime(df_pl.index)
    df_pl.sort_index(inplace=True)

    df_cf.index = pd.to_datetime(df_cf.index)
    df_cf.sort_index(inplace=True)

    # Merge all three dataframes on Date index
    df = pd.merge(df_bs, df_pl, left_index=True, right_index=True, how='outer', suffixes=('_bs', '_pl'))
    df = pd.merge(df, df_cf, left_index=True, right_index=True, how='outer', suffixes=('', '_cf'))

    # Resample to daily frequency using forward fill
    df = df.resample('D').ffill()

    # Set to OHLCV date index
    ohlcv_index = get_ohlcv_index(ticker)
    df = df.reindex(ohlcv_index, method='ffill')

    # Add Ratios as features
    ratios = {}
    eps = 1e-9
    if "NetProfit" in df.columns and "Sales" in df.columns:
        ratios["NetMargin"] = df["NetProfit"] / (df["Sales"] + eps)
    if "OperatingProfit" in df.columns and "Sales" in df.columns:
        ratios["OperatingMargin"] = df["OperatingProfit"] / (df["Sales"] + eps)
    if "Borrowings" in df.columns and "EquityCapital" in df.columns:
        ratios["Leverage"] = df["Borrowings"] / (df["EquityCapital"] + eps)
    if "CashEquivalents" in df.columns and "TotalLiabilities" in df.columns:
        ratios["CashToDebt"] = df["CashEquivalents"] / (df["TotalLiabilities"] + eps)
    if "Sales" in df.columns and "TotalAssets" in df.columns:
        ratios["AssetTurnover"] = df["Sales"] / (df["TotalAssets"] + eps)

    ratios_df = pd.DataFrame(ratios, index=df.index)
    df = pd.concat([df, ratios_df], axis=1)

    # Normalize all numeric columns with Z-score
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(zscore)

    return df

if __name__ == "__main__":
    # Load environment variables
    ticker = os.getenv('TICKER')
    start_date = os.getenv('START_DATE')
    end_date = os.getenv('END_DATE')
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    # EDA
    df = preprocess_fundamentals(ticker)
    print(df.head())
    print(df.info())
    print(df.describe().T)
    # Null values
    print("Null values in each column:")
    print(df.isnull().sum())
    # Duplicate rows
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    # Date range
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    # Are dates sorted?
    print(f"Dates sorted: {df.index.is_monotonic_increasing}")
    # Outliers detection using Z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number]), nan_policy='omit'))
    outliers = (z_scores > 3).any(axis=1)
    print(f"Number of outlier rows (Z-score > 3): {outliers.sum()}")

    # Distribution plots for TotalLiabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalLiabilities'].dropna(), bins=30, kde=True)
    plt.title('Distribution of TotalLiabilities')
    plt.xlabel('TotalLiabilities')
    plt.ylabel('Frequency')
    plt.show()


    # # Correlation matrix
    # corr_matrix = df.corr()
    # print("Correlation matrix of OHLCV data:")
    # # Partial display of correlation matrix
    # print(corr_matrix.iloc[:10, :10])
    # # Check and make pairs for highly correlated features
    # threshold = 0.8
    # correlated_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns
    #                     if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > threshold]
    # print(f"Highly correlated feature pairs (|correlation| > {threshold}):")
    # for pair in correlated_pairs:
    #     print(pair)
    #
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    # plt.title("Feature Correlation Heatmap")
    # plt.show()
