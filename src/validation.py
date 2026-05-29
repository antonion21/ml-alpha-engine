import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import os

# import previous functions
from data_loader import fetch_and_save_data
from features import engineer_features

def walk_forward_validation(X, y, n_splits=5):
    # performs walk-forward validation using TimeSeriesSplit
    # this ensures we never train on future data to predict the past,
    # which is the most common mistake in quantitative modeling
    print(f"Starting Walk-Forward Validation with {n_splits} splits...\n")
    
    # TimeSeriesSplit creates expanding training windows
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold = 1
    accuracies = []
    aucs = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # initialize and train baseline model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # predict on the out-of-sample window
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # evaluate
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        accuracies.append(acc)
        aucs.append(auc)
        
        print(f"Fold {fold}: Train Size={len(X_train)}, Test Size={len(X_test)} | Acc: {acc:.4f}, AUC: {auc:.4f}")
        fold += 1
        
    print("\n--- Cross-Validation Results ---")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (Std: {np.std(accuracies):.4f})")
    print(f"Mean AUC:      {np.mean(aucs):.4f} (Std: {np.std(aucs):.4f})")
    print("--------------------------------\n")
    
    return np.mean(accuracies), np.mean(aucs)

if __name__ == "__main__":
    file_path = "data/SPY_daily.csv"
    
    if not os.path.exists(file_path):
        print("Data not found. Fetching now...")
        fetch_and_save_data("SPY", "2015-01-01", "2024-01-01")
        
    df_raw = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df_features = engineer_features(df_raw)
    
    # define features and target exactly as in baseline.py
    feature_cols = ['log_return', 'volatility_20d', 'z_score_20d']
    X = df_features[feature_cols]
    y = df_features['target_1d']
    
    walk_forward_validation(X, y)