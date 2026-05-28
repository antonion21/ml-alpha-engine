import numpy as np
import pandas as pd

def engineer_features(df):
    # adds quantitative features to df
    # ensures no lookahead bias by only using historical rolling windows
    df = df.copy()
    
    # log returns (QF standard for stationarity)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # rolling Volatility (20-day standard deviation of returns)
    df['volatility_20d'] = df['log_return'].rolling(window=20).std()
    
    # momentum (20day Z-score)
    # measures how far the current price is from the 20day mean in standard deviations
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['z_score_20d'] = (df['Close'] - sma_20) / std_20
    
    # target: next days direction (1 if UP, 0 if DOWN)
    # shift by -1 to align TOMORROWS return with TODAYS features
    df['target_1d'] = (df['log_return'].shift(-1) > 0).astype(int)
    
    # drop NaNs created by rolling windows and shifting
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    import os
    
    file_path = "data/SPY_daily.csv"
    
    if os.path.exists(file_path):
        print("Loading data...")
        data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        print("Engineering features...")
        featured_data = engineer_features(data)
        
        print("Feature engineering successful. New shape:", featured_data.shape)
        print("\nLatest feature rows:")
        print(featured_data[['Close', 'log_return', 'volatility_20d', 'z_score_20d', 'target_1d']].tail())
    else:
        print(f"File not found: {file_path}. Run data_loader.py first.")