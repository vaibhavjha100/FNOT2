"""
Module for scheduling of collection and preprocessing of recent data.
This module provides the functionality to schedule data collection and preprocessing tasks
at specified intervals using Windows Task Scheduler so that model inference can be done on
the most recent data.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from collection import get_data
from preprocessing import get_sentiment, preprocess_data
from dotenv import load_dotenv
load_dotenv()

temp_dir = 'temp'
temp_datadir = os.path.join(temp_dir, 'data')

fresh_dir = 'fresh'
fresh_datadir = os.path.join(fresh_dir, 'data')
fresh_featdir = os.path.join(fresh_dir, 'features')
modeldir = 'models'

def collection_job(ticker, start, end, datadir, gemini_api_key=None):
    """
    Job to collect recent data for the given ticker and date range.
    We will also generate sentiment data here to complete the data collection step.
    Data will be stored in temp directory.

    Parameters:
    ticker (str): The stock ticker symbol.
    start (str): The start date for data collection in 'YYYY-MM-DD' format.
    end (str): The end date for data collection in 'YYYY-MM-DD' format.
    datadir (str): The directory to save the collected data.
    gemini_api_key (str): API key for Gemini to fetch sentiment data. Default is None.

    Returns:
    None: This function saves collected data to the specified directory.
    """

    print(f"📊 Collecting data for {ticker} from {start} to {end}...")
    get_data(ticker=ticker, start=start, end=end, datadir=datadir)

    print("📰 Generating sentiment data...")
    get_sentiment(ticker=ticker, start_date=start, end_date=end, datadir=datadir, gemini_api_key=gemini_api_key)

    print("✅ Data collection job completed.")

def merge_data_job(ticker, temp_datadir, fresh_datadir):
    """
    Job to merge newly collected data with existing fresh data.

    Parameters:
    ticker (str): The stock ticker symbol.
    temp_datadir (str): The directory where newly collected data is stored.
    fresh_datadir (str): The directory where existing fresh data is stored.

    Returns:
    None: This function merges data files and saves them in the fresh data directory.
    """

    # Check if fresh_datadir is empty
    # If it is empty, simply move all files from temp_datadir to fresh_datadir
    if not os.path.exists(fresh_datadir):
        os.makedirs(fresh_datadir)

    if not os.listdir(fresh_datadir):
        print(f"📂 Fresh data directory is empty. Moving all files from temp to fresh for {ticker}...")
        for file in os.listdir(temp_datadir):
            temp_file_path = os.path.join(temp_datadir, file)
            fresh_file_path = os.path.join(fresh_datadir, file)
            os.rename(temp_file_path, fresh_file_path)
            print(f"✅ Moved: {fresh_file_path}")
        print("✅ Data merging job completed.")
        return

    print(f"🔄 Merging new data into fresh data for {ticker}...")

    # List of data files is all files in temp_datadir
    data_files = [f for f in os.listdir(temp_datadir) if f.endswith('.csv')]

    for file in data_files:
        temp_file_path = os.path.join(temp_datadir, file)
        fresh_file_path = os.path.join(fresh_datadir, file)

        if os.path.exists(temp_file_path):
            new_data = pd.read_csv(temp_file_path, parse_dates=True, index_col=0)
            if os.path.exists(fresh_file_path):
                existing_data = pd.read_csv(fresh_file_path, parse_dates=True, index_col=0)
                combined_data = pd.concat([existing_data, new_data]).drop_duplicates().sort_index()
            else:
                combined_data = new_data

            combined_data.to_csv(fresh_file_path)
            print(f"✅ Merged and saved: {fresh_file_path}")
        else:
            print(f"⚠️ Warning: {temp_file_path} does not exist and will be skipped.")

    print("✅ Data merging job completed.")

def preprocessing_job(ticker, fresh_datadir, fresh_featdir, modeldir, n_components=0.95, new_sentiment=False, gemini_api_key=None, start_date=None, end_date=None):
    """
    Job to preprocess the merged fresh data and generate features for the model.

    Parameters:
    ticker (str): The stock ticker symbol.
    fresh_datadir (str): The directory where merged fresh data is stored.
    fresh_featdir (str): The directory to save the generated features.
    modeldir (str): The directory where models are stored.
    n_components (float): The number of PCA components to retain. Default is 0.95.
    new_sentiment (bool): Whether to generate new sentiment data. Default is False.
    gemini_api_key (str): API key for Gemini to fetch sentiment data if new_sentiment is True. Default is None.
    start_date (str): The start date for sentiment data generation if new_sentiment is True. Default is None.
    end_date (str): The end date for sentiment data generation if new_sentiment is True

    Returns:
    None: This function saves preprocessed features to the specified directory.
    """

    print(f"⚙️ Preprocessing data for {ticker}...")

    preprocess_data(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        datadir=fresh_datadir,
        featdir=fresh_featdir,
        modeldir=modeldir,
        n_components=n_components,
        new_sentiment=new_sentiment,
        gemini_api_key=gemini_api_key,
        pca_new=False
    )

    print("✅ Data preprocessing job completed.")

def run_daily_pipeline(ticker, gemini_api_key=None):
    """
    Run the daily data collection and preprocessing pipeline for the given ticker.

    Parameters:
    ticker (str): The stock ticker symbol.
    gemini_api_key (str): API key for Gemini to fetch sentiment data. Default is None.

    Returns:
    None
    """

    # Check if fresh_datadir exists, if not create it
    if not os.path.exists(fresh_datadir):
        os.makedirs(fresh_datadir)

    if not os.path.exists(fresh_featdir):
        os.makedirs(fresh_featdir)

    if not os.path.exists(temp_datadir):
        os.makedirs(temp_datadir)

    # End date is yesterday
    end_date = datetime.now() - timedelta(days=1)
    end_str = end_date.strftime('%Y-%m-%d')

    # If fresh_datadir is empty, collect last 200 days of data
    if not os.listdir(fresh_datadir):
        start_date = end_date - timedelta(days=200)
    else:
        # Start date is the day after the last date in fresh_datadir
        existing_files = [f for f in os.listdir(fresh_datadir) if f.endswith('.csv')]
        if not existing_files:
            start_date = end_date - timedelta(days=200)
        else:
            latest_date = None
            for file in existing_files:
                df = pd.read_csv(os.path.join(fresh_datadir, file), parse_dates=True, index_col=0)
                if not df.empty:
                    file_latest_date = df.index.max()
                    if latest_date is None or file_latest_date > latest_date:
                        latest_date = file_latest_date
            if latest_date is None:
                start_date = end_date - timedelta(days=200)
            else:
                start_date = latest_date + timedelta(days=1)

    start_str = start_date.strftime('%Y-%m-%d')

    # Step 1: Collect recent data and generate sentiment
    collection_job(ticker=ticker, start=start_str, end=end_str, datadir=temp_datadir, gemini_api_key=gemini_api_key)

    # Step 2: Merge new data with existing fresh data
    merge_data_job(ticker=ticker, temp_datadir=temp_datadir, fresh_datadir=fresh_datadir)

    # Step 3: Preprocess merged fresh data to generate features
    preprocessing_job(ticker=ticker, fresh_datadir=fresh_datadir, fresh_featdir=fresh_featdir, modeldir=modeldir, n_components=0.95, new_sentiment=False)

    # Clean up temp_datadir
    for file in os.listdir(temp_datadir):
        file_path = os.path.join(temp_datadir, file)
        os.remove(file_path)
    print("🧹 Cleaned up temporary data.")
    print("🎉 Daily pipeline run completed.")


if __name__ == "__main__":
    ticker = os.getenv('TICKER')
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if ticker is None:
        raise ValueError("TICKER environment variable not set.")

    run_daily_pipeline(ticker=ticker, gemini_api_key=gemini_api_key)