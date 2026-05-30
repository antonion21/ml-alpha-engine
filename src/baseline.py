import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# import previous functions
from data_loader import fetch_and_save_data
from features import engineer_features

def train_baseline_model(df):
    # trains simple Logistic Regression model as a baseline
    # uses simple chronological split for training and testing
    print("Preparing data for baseline model...")
    
    # define features and target
    feature_cols = ['log_return', 'volatility_20d', 'z_score_20d']
    X = df[feature_cols]
    y = df['target_1d']
    
    # chronological train/test split (80% train 20% test)
    # important!! no random splitting in time series!!
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")
    
    # train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # predict probabilities and classes
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # evaluate
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n--- Baseline Model Performance ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("----------------------------------\n")
    
    return model

if __name__ == "__main__":
    file_path = "data/SPY_daily.csv"
    
    if not os.path.exists(file_path):
        print("Data not found. Fetching now...")
        fetch_and_save_data("SPY", "2015-01-01", "2024-01-01")
        
    df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df_features = engineer_features(df_raw)
    
    train_baseline_model(df_features)