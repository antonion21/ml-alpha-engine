import yfinance as yf
import pandas as pd
import os

def fetch_and_save_data(ticker, start_date, end_date, output_dir="data"):
    # fetches historical daily data for a given ticker and saves to CSV
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    
    # create data directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        print("Warning: No data fetched.")
        return
    
    # save to CSV
    output_path = os.path.join(output_dir, f"{ticker}_daily.csv")
    df.to_csv(output_path)
    print(f"Data successfully saved to {output_path}")

if __name__ == "__main__":
    # test pipeline with SPY (S&P 500 ETF)
    fetch_and_save_data(
        ticker="SPY", 
        start_date="2015-01-01", 
        end_date="2024-01-01"
    )